#include "svc_framing.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <sstream>
#include <utility>

#include "core/bitstream/bitwriter.h"

namespace av2enc::svc {
namespace {

bool fail(std::string& error, std::string message) {
  error = std::move(message);
  return false;
}

unsigned type_value(ObuType type) { return static_cast<unsigned>(type); }

void put_uvlc(BitWriter& writer, std::uint32_t value) {
  unsigned leading_zeros = 0;
  while (value >= ((1ULL << (leading_zeros + 1U)) - 1ULL)) {
    ++leading_zeros;
    if (leading_zeros >= 32U) {
      break;
    }
  }
  writer.put_bits(0U, leading_zeros);
  writer.put_bit(true);
  const std::uint64_t payload =
      static_cast<std::uint64_t>(value) - ((1ULL << leading_zeros) - 1ULL);
  writer.put_bits(payload, leading_zeros);
}

void put_trailing_bits(BitWriter& writer) {
  writer.put_bit(true);
  writer.align_to_byte();
}

std::vector<std::uint8_t> take_aligned(BitWriter& writer) {
  writer.align_to_byte();
  return writer.take_bytes();
}

std::vector<const layers::ExtendedLayerConfig*> sorted_layers(
    const layers::LayerConfigRecord& record) {
  std::vector<const layers::ExtendedLayerConfig*> layers;
  layers.reserve(record.layers.size());
  for (const layers::ExtendedLayerConfig& layer : record.layers) {
    layers.push_back(&layer);
  }
  std::stable_sort(layers.begin(), layers.end(),
                   [](const layers::ExtendedLayerConfig* a,
                      const layers::ExtendedLayerConfig* b) {
                     return a->xlayer_id < b->xlayer_id;
                   });
  return layers;
}

const layers::EmbeddedLayerConfig* find_embedded(
    const layers::ExtendedLayerConfig& layer, int mlayer_id) {
  for (const layers::EmbeddedLayerConfig& embedded : layer.embedded) {
    if (static_cast<int>(embedded.mlayer_id) == mlayer_id) {
      return &embedded;
    }
  }
  return nullptr;
}

// --- §5.4.1 seq_lcr_id rewriting -------------------------------------------
//
// compose_layered_stream() wraps already-encoded single-layer sequence headers
// byte-for-byte.  A layered plan associates every sequence header with the
// global LCR through seq_lcr_id, so the composed header must carry the plan's
// id instead of the single-layer encoder's 0.  The rewrite is bit-exact: the
// field width is fixed at f(3), so no length change and no re-serialisation of
// the rest of the header is needed.

bool read_payload_bit(std::span<const std::uint8_t> bytes, std::size_t bit_pos,
                      bool* bit) {
  if (bit == nullptr || bit_pos >= bytes.size() * 8U) return false;
  const std::size_t byte = bit_pos >> 3U;
  const unsigned shift = 7U - static_cast<unsigned>(bit_pos & 7U);
  *bit = ((bytes[byte] >> shift) & 1U) != 0U;
  return true;
}

bool read_payload_bits(std::span<const std::uint8_t> bytes,
                       std::size_t* bit_pos, unsigned count,
                       std::uint32_t* value) {
  if (bit_pos == nullptr || value == nullptr) return false;
  std::uint32_t out = 0;
  for (unsigned i = 0; i < count; ++i) {
    bool bit = false;
    if (!read_payload_bit(bytes, *bit_pos, &bit)) return false;
    out = (out << 1U) | (bit ? 1U : 0U);
    ++*bit_pos;
  }
  *value = out;
  return true;
}

bool read_payload_uvlc(std::span<const std::uint8_t> bytes,
                       std::size_t* bit_pos, std::uint32_t* value) {
  if (bit_pos == nullptr || value == nullptr) return false;
  unsigned leading_zeros = 0;
  bool bit = false;
  do {
    if (!read_payload_bit(bytes, *bit_pos, &bit)) return false;
    ++*bit_pos;
    if (bit) break;
    ++leading_zeros;
  } while (leading_zeros <= 31U);
  if (leading_zeros > 31U) return false;
  if (leading_zeros == 0) {
    *value = 0;
    return true;
  }
  std::uint32_t payload = 0;
  if (!read_payload_bits(bytes, bit_pos, leading_zeros, &payload)) {
    return false;
  }
  *value = ((1U << leading_zeros) - 1U) + payload;
  return true;
}

bool write_payload_bits(std::vector<std::uint8_t>& bytes, std::size_t bit_pos,
                        unsigned count, std::uint32_t value) {
  if (bit_pos + count > bytes.size() * 8U) return false;
  for (unsigned i = 0; i < count; ++i) {
    const unsigned shift = count - 1U - i;
    const bool bit = ((value >> shift) & 1U) != 0U;
    const std::size_t pos = bit_pos + i;
    const std::size_t byte = pos >> 3U;
    const unsigned out_shift = 7U - static_cast<unsigned>(pos & 7U);
    if (bit) {
      bytes[byte] = static_cast<std::uint8_t>(bytes[byte] |
                                              (1U << out_shift));
    } else {
      bytes[byte] = static_cast<std::uint8_t>(bytes[byte] &
                                              ~(1U << out_shift));
    }
  }
  return true;
}

bool rewrite_sequence_lcr_id(std::span<const std::uint8_t> payload,
                             std::uint8_t seq_lcr_id,
                             std::vector<std::uint8_t>& out,
                             std::string& error) {
  out.assign(payload.begin(), payload.end());
  std::size_t bit_pos = 0;
  std::uint32_t value = 0;
  if (!read_payload_uvlc(payload, &bit_pos, &value)) {
    return fail(error, "sequence header: truncated seq_header_id");
  }
  if (!read_payload_bits(payload, &bit_pos, 5U, &value)) {
    return fail(error, "sequence header: truncated seq_profile_idc");
  }
  bool single_picture = false;
  if (!read_payload_bit(payload, bit_pos, &single_picture)) {
    return fail(error, "sequence header: truncated single_picture_header_flag");
  }
  ++bit_pos;
  if (!read_payload_bits(payload, &bit_pos, 5U, &value)) {
    return fail(error, "sequence header: truncated seq_level_idx");
  }
  const std::uint32_t seq_level_idx = value;
  if (seq_level_idx > 3U && !single_picture) {
    bool seq_tier = false;
    if (!read_payload_bit(payload, bit_pos, &seq_tier)) {
      return fail(error, "sequence header: truncated seq_tier");
    }
    ++bit_pos;
  }
  if (!read_payload_uvlc(payload, &bit_pos, &value)) {
    return fail(error, "sequence header: truncated chroma_format_idc");
  }
  if (!read_payload_uvlc(payload, &bit_pos, &value)) {
    return fail(error, "sequence header: truncated bit_depth_idc");
  }

  if (single_picture) {
    if (seq_lcr_id != 0U) {
      return fail(error,
                  "sequence header has single_picture_header_flag = 1, which "
                  "forces seq_lcr_id = 0, but the plan associates an LCR");
    }
    return true;  // seq_lcr_id is inferred 0; nothing to rewrite.
  }
  if (seq_lcr_id > 7U) {
    return fail(error, "seq_lcr_id is outside f(3)");
  }
  if (!write_payload_bits(out, bit_pos, 3U, seq_lcr_id)) {
    return fail(error, "sequence header: truncated seq_lcr_id");
  }
  return true;
}

bool write_xlayer_info(BitWriter& writer, bool is_global,
                       const layers::ExtendedLayerConfig& layer,
                       const layers::LayerDependencyMaps& dependencies,
                       std::string& error) {
  if (layer.mlayer_map == 0 || (layer.mlayer_map & ~0xFFU) != 0) {
    return fail(error, "LCR xlayer needs a non-empty 8-bit mlayer_map");
  }
  // No representation/crop/purpose/color payloads: each extended layer has its
  // own sequence header with the layer's dimensions.
  writer.put_bit(false);  // lcr_rep_info_present_flag
  writer.put_bit(false);  // lcr_xlayer_purpose_present_flag
  writer.put_bit(false);  // lcr_xlayer_color_info_present_flag
  writer.put_bit(true);   // lcr_embedded_layer_info_present_flag
  writer.align_to_byte();

  writer.put_bits(layer.mlayer_map, 8);
  for (int m = 0; m < layers::kEmbeddedLayerCount; ++m) {
    if ((layer.mlayer_map & (1U << static_cast<unsigned>(m))) == 0) {
      continue;
    }
    const std::uint8_t tlayer_map =
        layer.tlayer_map[static_cast<std::size_t>(m)];
    if (tlayer_map == 0 || (tlayer_map & ~0x0FU) != 0) {
      std::ostringstream out;
      out << "LCR embedded layer m" << m << " needs a non-empty 4-bit tlayer_map";
      return fail(error, out.str());
    }
    writer.put_bits(tlayer_map, 4);

    // lcr_global_atlas_id_present_flag / local atlas flag are 0 in this
    // pipeline, so no atlas id / priority / rendering method is written.
    writer.put_bits(0U, 8);  // lcr_layer_type = TEXTURE_LAYER
    writer.put_bits(1U, 8);  // lcr_view_type = VIEW_CENTER
    if (m > 0) {
      // lcr_dependent_layer_map f(m), bit ref = embedded layer ref.
      for (int ref = 0; ref < m; ++ref) {
        writer.put_bit(dependencies.mlayer[static_cast<std::size_t>(m)]
                                              [static_cast<std::size_t>(ref)]);
      }
    }

    const layers::EmbeddedLayerConfig* embedded = find_embedded(layer, m);
    const bool same_resolution =
        embedded == nullptr ? true : embedded->same_sh_max_resolution;
    writer.put_bit(same_resolution);
    if (!same_resolution) {
      const std::uint32_t width = embedded->max_expected_width;
      const std::uint32_t height = embedded->max_expected_height;
      if (width == 0 || height == 0) {
        return fail(error,
                    "LCR embedded layer needs dimensions when it differs from "
                    "its sequence header");
      }
      put_uvlc(writer, width);
      put_uvlc(writer, height);
    }
    writer.align_to_byte();
  }
  (void)is_global;
  return true;
}

std::uint32_t xlayer_dependency_mask(const layers::ExtendedLayerConfig& layer) {
  std::uint32_t mask = 0;
  for (std::uint8_t dep : layer.dependent_xlayer_ids) {
    if (dep < layers::kExtendedLayerCount) {
      mask |= 1U << dep;
    }
  }
  return mask;
}

bool validate_ops_maps(const layers::OperatingPoint& point, int x,
                       std::string& error) {
  const std::uint32_t mlayer_map = point.mlayer_map[static_cast<std::size_t>(x)];
  if (mlayer_map == 0 || (mlayer_map & ~0xFFU) != 0) {
    std::ostringstream out;
    out << "operating point " << static_cast<int>(point.op_index)
        << " needs a non-empty 8-bit mlayer_map for xlayer " << x;
    return fail(error, out.str());
  }
  for (int m = 0; m < layers::kEmbeddedLayerCount; ++m) {
    if ((mlayer_map & (1U << static_cast<unsigned>(m))) == 0) {
      continue;
    }
    const std::uint8_t tlayer_map =
        point.tlayer_map[static_cast<std::size_t>(x)]
                        [static_cast<std::size_t>(m)];
    if (tlayer_map == 0 || (tlayer_map & ~0x0FU) != 0) {
      std::ostringstream out;
      out << "operating point " << static_cast<int>(point.op_index)
          << " needs a non-empty 4-bit tlayer_map for xlayer " << x
          << " mlayer " << m;
      return fail(error, out.str());
    }
  }
  return true;
}

void write_mlayer_info(BitWriter& writer, std::uint32_t mlayer_map,
                       const std::array<std::uint8_t, layers::kEmbeddedLayerCount>&
                           tlayer_map) {
  writer.put_bits(mlayer_map, 8);
  for (int m = 0; m < layers::kEmbeddedLayerCount; ++m) {
    if ((mlayer_map & (1U << static_cast<unsigned>(m))) == 0) {
      continue;
    }
    writer.put_bits(tlayer_map[static_cast<std::size_t>(m)], 4);
  }
}

bool append_leb128_bytes(BitWriter& writer, std::uint32_t value) {
  std::vector<std::uint8_t> bytes;
  append_leb128(bytes, value);
  for (std::uint8_t byte : bytes) {
    writer.put_bits(byte, 8);
  }
  return true;
}

}  // namespace

bool operator==(const ObuHeader& a, const ObuHeader& b) {
  return a.type == b.type && a.tlayer_id == b.tlayer_id &&
         a.mlayer_id == b.mlayer_id && a.xlayer_id == b.xlayer_id;
}

bool operator!=(const ObuHeader& a, const ObuHeader& b) { return !(a == b); }

bool is_temporal_delimiter(ObuType type) {
  return type == ObuType::kTemporalDelimiter;
}

bool is_global_layer_type(ObuType type) {
  return type == ObuType::kTemporalDelimiter || type == ObuType::kMsdo;
}

bool is_frame_obu(ObuType type) {
  switch (type) {
    case ObuType::kClosedLoopKey:
    case ObuType::kOpenLoopKey:
    case ObuType::kLeadingTileGroup:
    case ObuType::kRegularTileGroup:
    case ObuType::kSwitch:
    case ObuType::kLeadingSef:
    case ObuType::kRegularSef:
    case ObuType::kLeadingTip:
    case ObuType::kRegularTip:
    case ObuType::kRasFrame:
    case ObuType::kBridgeFrame:
      return true;
    default:
      return false;
  }
}

bool is_config_obu(ObuType type) {
  return type == ObuType::kSequenceHeader ||
         type == ObuType::kLayerConfigRecord ||
         type == ObuType::kOperatingPointSet ||
         type == ObuType::kAtlasSegment;
}

bool encode_obu_header(const ObuHeader& header, std::vector<std::uint8_t>& out,
                       std::string& error) {
  out.clear();
  const unsigned type = type_value(header.type);
  if (type > 31U) {
    return fail(error, "obu_type is outside f(5)");
  }
  if (header.tlayer_id > 3U) {
    return fail(error, "obu_tlayer_id is outside f(2)");
  }
  if (header.mlayer_id > 7U) {
    return fail(error, "obu_mlayer_id is outside f(3)");
  }
  if (header.xlayer_id > 31U) {
    return fail(error, "obu_xlayer_id is outside f(5)");
  }
  if (is_global_layer_type(header.type)) {
    if (header.xlayer_id != layers::kGlobalXLayerId) {
      return fail(error,
                  "OBU_TEMPORAL_DELIMITER/OBU_MSDO require "
                  "obu_xlayer_id = GLOBAL_XLAYER_ID");
    }
    if (header.mlayer_id != 0) {
      return fail(error,
                  "GLOBAL_XLAYER_ID requires obu_mlayer_id = 0");
    }
  }

  const bool needs_extension = !is_global_layer_type(header.type) &&
                               (header.mlayer_id != 0 ||
                                header.xlayer_id != 0);
  BitWriter writer;
  writer.put_bit(needs_extension);
  writer.put_bits(type, 5);
  writer.put_bits(header.tlayer_id, 2);
  if (needs_extension) {
    writer.put_bits(header.mlayer_id, 3);
    writer.put_bits(header.xlayer_id, 5);
  }
  out = take_aligned(writer);
  return true;
}

bool decode_obu_header(std::span<const std::uint8_t> bytes, ObuHeader& header,
                       std::size_t& consumed, std::string& error) {
  consumed = 0;
  if (bytes.empty()) {
    return fail(error, "truncated OBU header");
  }
  const std::uint8_t first = bytes[0];
  const bool extension = (first & 0x80U) != 0;
  const unsigned type = (static_cast<unsigned>(first) >> 2U) & 0x1FU;
  header.type = static_cast<ObuType>(type);
  header.tlayer_id = static_cast<std::uint8_t>(first & 0x03U);
  if (extension) {
    if (bytes.size() < 2) {
      return fail(error, "truncated OBU header extension byte");
    }
    header.mlayer_id = static_cast<std::uint8_t>((bytes[1] >> 5U) & 0x07U);
    header.xlayer_id = static_cast<std::uint8_t>(bytes[1] & 0x1FU);
    consumed = 2;
  } else {
    header.mlayer_id = 0;
    header.xlayer_id = static_cast<std::uint8_t>(
        is_global_layer_type(header.type) ? layers::kGlobalXLayerId : 0);
    consumed = 1;
  }
  return true;
}

void append_leb128(std::vector<std::uint8_t>& out, std::uint32_t value) {
  do {
    std::uint8_t byte = static_cast<std::uint8_t>(value & 0x7FU);
    value >>= 7U;
    if (value != 0) {
      byte = static_cast<std::uint8_t>(byte | 0x80U);
    }
    out.push_back(byte);
  } while (value != 0);
}

std::size_t leb128_size(std::uint32_t value) {
  std::size_t bytes = 1;
  while (value >= 0x80U) {
    value >>= 7U;
    ++bytes;
  }
  return bytes;
}

bool frame_obu(const ObuHeader& header, std::span<const std::uint8_t> payload,
               std::vector<std::uint8_t>& out, std::string& error) {
  std::vector<std::uint8_t> header_bytes;
  if (!encode_obu_header(header, header_bytes, error)) {
    return false;
  }
  const std::uint64_t size = header_bytes.size() + payload.size();
  if (size > std::numeric_limits<std::uint32_t>::max()) {
    return fail(error, "OBU is larger than the leb128 size field");
  }
  const std::size_t start = out.size();
  append_leb128(out, static_cast<std::uint32_t>(size));
  out.insert(out.end(), header_bytes.begin(), header_bytes.end());
  out.insert(out.end(), payload.begin(), payload.end());
  (void)start;
  return true;
}

bool parse_obus(std::span<const std::uint8_t> bytes,
                std::vector<ParsedObu>& obus, std::string& error) {
  obus.clear();
  std::size_t offset = 0;
  while (offset < bytes.size()) {
    const std::size_t obu_start = offset;
    std::uint64_t size_value = 0;
    unsigned shift = 0;
    unsigned leb_bytes = 0;
    while (true) {
      if (offset >= bytes.size()) {
        return fail(error, "truncated leb128 OBU size");
      }
      const std::uint8_t byte = bytes[offset++];
      if (leb_bytes >= 5U) {
        return fail(error, "leb128 OBU size exceeds 32 bits");
      }
      size_value |= static_cast<std::uint64_t>(byte & 0x7FU) << shift;
      ++leb_bytes;
      if ((byte & 0x80U) == 0) {
        break;
      }
      shift += 7U;
    }
    if (size_value > std::numeric_limits<std::uint32_t>::max()) {
      return fail(error, "leb128 OBU size exceeds 32 bits");
    }
    const std::uint32_t size = static_cast<std::uint32_t>(size_value);
    if (size == 0) {
      return fail(error, "OBU size is zero");
    }
    if (size > bytes.size() - offset) {
      return fail(error, "OBU size exceeds the remaining input");
    }
    ParsedObu obu;
    const auto body = bytes.subspan(offset, size);
    std::size_t header_bytes = 0;
    if (!decode_obu_header(body, obu.header, header_bytes, error)) {
      return false;
    }
    obu.payload.assign(body.begin() + static_cast<std::ptrdiff_t>(header_bytes),
                       body.end());
    obu.raw.assign(bytes.begin() + static_cast<std::ptrdiff_t>(obu_start),
                   bytes.begin() +
                       static_cast<std::ptrdiff_t>(offset + size));
    offset += size;
    obus.push_back(std::move(obu));
  }
  return true;
}

bool write_layer_config_record_payload(
    const layers::LayerConfigRecord& record,
    const layers::LayerDependencyMaps& dependencies,
    std::vector<std::uint8_t>& out, std::string& error) {
  out.clear();
  if (record.global_config_record_id > 7) {
    return fail(error, "lcr_global_config_record_id is outside f(3)");
  }
  const std::vector<const layers::ExtendedLayerConfig*> layers =
      sorted_layers(record);

  BitWriter writer;
  bool dependent = false;
  if (record.is_global) {
    if (record.xlayer_id != layers::kGlobalXLayerId) {
      return fail(error, "global LCR must use obu_xlayer_id = 31");
    }
    if ((record.xlayer_map & ~0x7FFFFFFFU) != 0) {
      return fail(error, "lcr_xlayer_map has bits outside f(31)");
    }
    if (layers.empty()) {
      return fail(error, "global LCR describes no extended layer");
    }
    for (std::size_t i = 0; i < layers.size(); ++i) {
      const layers::ExtendedLayerConfig* layer = layers[i];
      if (layer->xlayer_id > layers::kMaxExtendedLayerId) {
        return fail(error, "global LCR contains the GLOBAL xlayer id");
      }
      if ((record.xlayer_map & (1U << layer->xlayer_id)) == 0) {
        return fail(error, "global LCR layer is absent from lcr_xlayer_map");
      }
      if (i > 0 && layers[i - 1]->xlayer_id == layer->xlayer_id) {
        return fail(error, "global LCR repeats an extended layer");
      }
      dependent = dependent || layer->dependent_xlayers;
    }
    // Every set bit must be described: this writer emits one complete record.
    for (int x = 0; x < layers::kMaxExtendedLayerId; ++x) {
      if ((record.xlayer_map & (1U << x)) == 0) {
        continue;
      }
      const bool found = std::any_of(
          layers.begin(), layers.end(), [x](const layers::ExtendedLayerConfig* l) {
            return static_cast<int>(l->xlayer_id) == x;
          });
      if (!found) {
        std::ostringstream out_message;
        out_message << "global LCR omits extended layer x" << x
                    << " selected by lcr_xlayer_map";
        return fail(error, out_message.str());
      }
    }

    writer.put_bits(record.global_config_record_id, 3);
    writer.put_bits(record.xlayer_map, 31);
    writer.put_bit(record.aggregate.present);
    writer.put_bit(false);  // lcr_seq_profile_tier_level_info_present_flag
    writer.put_bit(true);   // lcr_global_payload_present_flag
    writer.put_bit(dependent);
    writer.put_bit(false);  // lcr_global_atlas_id_present_flag
    writer.put_bits(0U, 7);  // lcr_global_purpose_id
    writer.put_bit(false);   // lcr_doh_constraint_flag
    writer.put_bit(false);   // lcr_enforce_tile_alignment_flag
    writer.put_bits(0U, 3);  // atlas id / reserved
    writer.put_bits(0U, 5);  // reserved
    if (record.aggregate.present) {
      const layers::AggregateInfo& aggregate = record.aggregate;
      if (aggregate.config_idc > 63 || aggregate.level_idx > 31 ||
          aggregate.max_interop > 15) {
        return fail(error, "LCR aggregate info is outside its descriptor");
      }
      writer.put_bits(aggregate.config_idc, 6);
      writer.put_bits(aggregate.level_idx, 5);
      writer.put_bit(aggregate.max_tier_flag);
      writer.put_bits(aggregate.max_interop, 4);
    }

    for (const layers::ExtendedLayerConfig* layer : layers) {
      BitWriter payload;
      if (dependent && layer->xlayer_id > 0) {
        payload.put_bits(xlayer_dependency_mask(*layer), layer->xlayer_id);
      }
      if (!write_xlayer_info(payload, true, *layer, dependencies, error)) {
        return false;
      }
      const std::vector<std::uint8_t> bytes = take_aligned(payload);
      append_leb128_bytes(writer, static_cast<std::uint32_t>(bytes.size()));
      for (std::uint8_t byte : bytes) {
        writer.put_bits(byte, 8);
      }
    }
  } else {
    if (record.xlayer_id > layers::kMaxExtendedLayerId) {
      return fail(error, "local LCR must use obu_xlayer_id 0..30");
    }
    if (record.xlayer_map != 0) {
      return fail(error, "local LCR must not carry lcr_xlayer_map");
    }
    if (layers.size() != 1 ||
        layers.front()->xlayer_id != record.xlayer_id) {
      return fail(error,
                  "local LCR must describe exactly its own extended layer");
    }
    const layers::ExtendedLayerConfig* layer = layers.front();
    dependent = layer->dependent_xlayers;

    writer.put_bits(0U, 3);  // lcr_global_id
    writer.put_bits(record.global_config_record_id, 3);  // lcr_local_id
    writer.put_bit(false);  // profile tier level info present
    writer.put_bit(false);  // local atlas id present
    writer.put_bits(0U, 3);  // reserved
    writer.put_bits(0U, 5);  // reserved
    if (!write_xlayer_info(writer, false, *layer, dependencies, error)) {
      return false;
    }
  }

  writer.put_bit(false);  // obu_extension_flag
  put_trailing_bits(writer);
  out = take_aligned(writer);
  return true;
}

bool write_operating_point_set_payload(const layers::OperatingPointSet& set,
                                       std::vector<std::uint8_t>& out,
                                       std::string& error) {
  out.clear();
  if (set.xlayer_id >= layers::kExtendedLayerCount) {
    return fail(error, "OPS obu_xlayer_id is outside 0..31");
  }
  if (set.ops_id > 15) {
    return fail(error, "ops_id is outside f(4)");
  }
  if (set.points.empty() || set.points.size() > 7) {
    return fail(error, "ops_cnt must be 1..7");
  }
  const bool is_global = set.xlayer_id == layers::kGlobalXLayerId;

  BitWriter writer;
  writer.put_bit(true);  // ops_reset_flag
  writer.put_bits(set.ops_id, 4);
  writer.put_bits(static_cast<std::uint32_t>(set.points.size()), 3);
  writer.put_bits(0U, 4);  // ops_priority
  writer.put_bits(0U, 7);  // ops_intent
  writer.put_bit(false);   // ops_intent_present_flag
  writer.put_bit(false);   // ops_ptl_present_flag
  writer.put_bit(false);   // ops_color_info_present_flag
  if (is_global) {
    writer.put_bits(1U, 2);  // ops_mlayer_info_idc = 1 (explicit maps)
  } else {
    writer.put_bits(0U, 2);  // ops_reserved_2bits
  }

  for (std::size_t i = 0; i < set.points.size(); ++i) {
    const layers::OperatingPoint& point = set.points[i];
    if (point.op_index != i) {
      return fail(error, "operating point op_index is not its OPS position");
    }
    if ((point.xlayer_map & ~0x7FFFFFFFU) != 0) {
      return fail(error, "ops_xlayer_map has bits outside f(31)");
    }
    if (is_global) {
      if (point.xlayer_map == 0) {
        return fail(error, "global operating point selects no extended layer");
      }
      for (int x = 0; x < layers::kMaxExtendedLayerId; ++x) {
        if ((point.xlayer_map & (1U << x)) == 0) {
          continue;
        }
        if (!validate_ops_maps(point, x, error)) {
          return false;
        }
      }
    } else {
      const std::uint32_t self = 1U << set.xlayer_id;
      if (point.xlayer_map != self) {
        return fail(error,
                    "local operating point must select exactly its own xlayer");
      }
      if (!validate_ops_maps(point, set.xlayer_id, error)) {
        return false;
      }
    }

    BitWriter payload;
    payload.put_bit(false);  // ops_decoder_model_info_present_flag
    payload.put_bit(false);  // ops_initial_display_delay_present_flag
    if (is_global) {
      payload.put_bits(point.xlayer_map, 31);
      for (int x = 0; x < layers::kMaxExtendedLayerId; ++x) {
        if ((point.xlayer_map & (1U << x)) == 0) {
          continue;
        }
        write_mlayer_info(payload,
                          point.mlayer_map[static_cast<std::size_t>(x)],
                          point.tlayer_map[static_cast<std::size_t>(x)]);
      }
    } else {
      write_mlayer_info(
          payload, point.mlayer_map[static_cast<std::size_t>(set.xlayer_id)],
          point.tlayer_map[static_cast<std::size_t>(set.xlayer_id)]);
    }
    const std::vector<std::uint8_t> bytes = take_aligned(payload);
    append_leb128_bytes(writer, static_cast<std::uint32_t>(bytes.size()));
    for (std::uint8_t byte : bytes) {
      writer.put_bits(byte, 8);
    }
  }

  writer.put_bit(false);  // obu_extension_flag
  put_trailing_bits(writer);
  out = take_aligned(writer);
  return true;
}

bool compose_layered_stream(const LayerPlan& plan,
                            const std::vector<LayerInput>& inputs,
                            const ComposeOptions& options,
                            std::vector<std::uint8_t>& out,
                            std::string& error) {
  out.clear();
  if (!validate_layer_plan(plan, error)) {
    return false;
  }
  const std::size_t expected = plan.config.spatial_layers.size();
  if (inputs.size() != expected) {
    std::ostringstream message;
    message << "compose needs one single-layer input per spatial layer ("
            << expected << "), got " << inputs.size();
    return fail(error, message.str());
  }

  struct ParsedInput {
    const LayerInput* input = nullptr;
    std::vector<ParsedObu> obus;
    std::size_t temporal_delimiter_count = 0;
    std::size_t sequence_count = 0;
    std::size_t frame_count = 0;
  };
  std::vector<ParsedInput> parsed(expected);

  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const SpatialLayerSpec& spatial = plan.config.spatial_layers[i];
    const LayerInput& input = inputs[i];
    if (input.layer.xlayer_id != spatial.xlayer_id ||
        input.layer.mlayer_id != 0 ||
        input.layer.tlayer_id != 0) {
      std::ostringstream message;
      message << "input " << i << " must be layer x"
              << static_cast<int>(spatial.xlayer_id) << "/m0/t0";
      return fail(error, message.str());
    }
    if (!layer_is_present(plan, input.layer)) {
      return fail(error, "input layer is not present in the plan");
    }
    ParsedInput entry;
    entry.input = &input;
    if (!parse_obus(input.obus, entry.obus, error)) {
      return fail(error, std::string("input ") + std::to_string(i) + ": " +
                                error);
    }
    if (entry.obus.empty()) {
      return fail(error, "input has no OBUs");
    }
    for (const ParsedObu& obu : entry.obus) {
      if (obu.header.type == ObuType::kSequenceHeader) {
        ++entry.sequence_count;
      } else if (obu.header.type == ObuType::kTemporalDelimiter) {
        ++entry.temporal_delimiter_count;
      } else if (is_frame_obu(obu.header.type)) {
        ++entry.frame_count;
      } else if (obu.header.type == ObuType::kLayerConfigRecord ||
                 obu.header.type == ObuType::kOperatingPointSet ||
                 obu.header.type == ObuType::kAtlasSegment) {
        return fail(error,
                    "input already carries global layer metadata; pass a raw "
                    "single-layer temporal unit");
      }
    }
    if (entry.temporal_delimiter_count != 1 ||
        entry.obus.front().header.type != ObuType::kTemporalDelimiter) {
      return fail(error,
                  "input must start with exactly one temporal delimiter");
    }
    if (entry.sequence_count != 1) {
      return fail(error, "input needs exactly one sequence header");
    }
    if (entry.frame_count != 1) {
      return fail(error, "input needs exactly one coded frame");
    }
    parsed[i] = std::move(entry);
  }

  if (options.emit_temporal_delimiter) {
    ObuHeader td;
    td.type = ObuType::kTemporalDelimiter;
    td.xlayer_id = static_cast<std::uint8_t>(layers::kGlobalXLayerId);
    if (!frame_obu(td, std::span<const std::uint8_t>{}, out, error)) {
      return false;
    }
  }

  if (options.emit_lcr) {
    std::vector<std::uint8_t> payload;
    if (!write_layer_config_record_payload(
            plan.lcr, plan.sequences.front().dependencies, payload, error)) {
      return fail(error, std::string("LCR: ") + error);
    }
    ObuHeader header;
    header.type = ObuType::kLayerConfigRecord;
    header.xlayer_id = static_cast<std::uint8_t>(layers::kGlobalXLayerId);
    if (!frame_obu(header, payload, out, error)) {
      return false;
    }
  }

  if (options.emit_ops) {
    std::vector<std::uint8_t> payload;
    if (!write_operating_point_set_payload(plan.ops, payload, error)) {
      return fail(error, std::string("OPS: ") + error);
    }
    ObuHeader header;
    header.type = ObuType::kOperatingPointSet;
    header.xlayer_id = static_cast<std::uint8_t>(layers::kGlobalXLayerId);
    if (!frame_obu(header, payload, out, error)) {
      return false;
    }
  }

  for (std::size_t i = 0; i < parsed.size(); ++i) {
    const ParsedInput& entry = parsed[i];
    const std::uint8_t xlayer = entry.input->layer.xlayer_id;
    for (const ParsedObu& obu : entry.obus) {
      if (obu.header.type == ObuType::kTemporalDelimiter) {
        continue;  // one global TD covers the temporal unit
      }
      ObuHeader header;
      header.type = obu.header.type;
      header.xlayer_id = xlayer;
      if (obu.header.type == ObuType::kSequenceHeader) {
        header.mlayer_id = 0;
        header.tlayer_id = 0;
        // Associate the composed sequence header with the plan's global LCR.
        // Without an emitted LCR the header must claim no association (0).
        const std::uint8_t target_lcr_id =
            options.emit_lcr ? plan.sequences[i].seq_lcr_id : 0U;
        std::vector<std::uint8_t> sequence_payload;
        if (!rewrite_sequence_lcr_id(obu.payload, target_lcr_id,
                                     sequence_payload, error)) {
          return fail(error, std::string("input ") + std::to_string(i) + ": " +
                                    error);
        }
        if (!frame_obu(header, sequence_payload, out, error)) {
          return false;
        }
        continue;
      }
      header.mlayer_id = entry.input->layer.mlayer_id;
      header.tlayer_id = entry.input->layer.tlayer_id;
      if (!frame_obu(header, obu.payload, out, error)) {
        return false;
      }
    }
  }
  return true;
}

bool obu_retained(const layers::RetentionMap& retention,
                  const ObuHeader& header, bool& retained,
                  std::string& error) {
  retained = false;
  if (is_temporal_delimiter(header.type)) {
    retained = true;
    return true;
  }
  if (header.xlayer_id >= layers::kExtendedLayerCount ||
      header.mlayer_id >= layers::kEmbeddedLayerCount ||
      header.tlayer_id >= layers::kTemporalLayerCount) {
    return fail(error, "OBU layer ids are outside the retention map");
  }
  const int x = header.xlayer_id;
  const int m = header.mlayer_id;
  const int t = header.tlayer_id;
  if (x == layers::kGlobalXLayerId) {
    retained = m == 0 && t == 0 &&
               retention.retain[static_cast<std::size_t>(x)][0][0];
    return true;
  }
  bool selected = retention.xlayer_selected[static_cast<std::size_t>(x)];
  if (!selected) {
    for (int mm = 0; mm < layers::kEmbeddedLayerCount && !selected; ++mm) {
      for (int tt = 0; tt < layers::kTemporalLayerCount; ++tt) {
        if (retention.retain[static_cast<std::size_t>(x)]
                            [static_cast<std::size_t>(mm)]
                            [static_cast<std::size_t>(tt)]) {
          selected = true;
          break;
        }
      }
    }
  }
  if (!selected) {
    retained = false;
    return true;
  }
  if (retention.retain[static_cast<std::size_t>(x)]
                      [static_cast<std::size_t>(m)]
                      [static_cast<std::size_t>(t)]) {
    retained = true;
    return true;
  }
  // Annex F.3.2 Step 2: at mId == tId == 0, configuration OBUs survive even
  // when their exact pair is not retained, as long as the xlayer is selected.
  if (m == 0 && t == 0 && is_config_obu(header.type)) {
    retained = true;
    return true;
  }
  retained = false;
  return true;
}

bool extract_substream(std::span<const std::uint8_t> input,
                       const layers::RetentionMap& retention,
                       std::vector<std::uint8_t>& output, std::string& error) {
  output.clear();
  std::vector<ParsedObu> obus;
  if (!parse_obus(input, obus, error)) {
    return false;
  }
  for (const ParsedObu& obu : obus) {
    bool keep = false;
    if (!obu_retained(retention, obu.header, keep, error)) {
      return false;
    }
    if (keep) {
      output.insert(output.end(), obu.raw.begin(), obu.raw.end());
    }
  }
  return true;
}

}  // namespace av2enc::svc
