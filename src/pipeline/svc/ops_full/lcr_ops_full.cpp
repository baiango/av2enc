#include "lcr_ops_full.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <sstream>
#include <utility>

namespace av2enc::svc::ops_full {
namespace {

bool fail(std::string& error, std::string message) {
  error = std::move(message);
  return false;
}

// --- §4.11 descriptors -----------------------------------------------------

void put_uvlc(BitWriter& writer, std::uint32_t value) {
  unsigned leading_zeros = 0;
  while (value >= ((1ULL << (leading_zeros + 1U)) - 1ULL)) {
    ++leading_zeros;
    if (leading_zeros >= 32U) break;
  }
  writer.put_bits(0U, leading_zeros);
  writer.put_bit(true);
  const std::uint64_t payload =
      static_cast<std::uint64_t>(value) - ((1ULL << leading_zeros) - 1ULL);
  writer.put_bits(payload, leading_zeros);
}

void put_rg(BitWriter& writer, std::uint32_t value, unsigned n) {
  const std::uint32_t q = value >> n;
  for (std::uint32_t i = 0; i < q; ++i) writer.put_bit(true);
  writer.put_bit(false);
  writer.put_bits(value & ((1U << n) - 1U), n);
}

void put_trailing_bits(BitWriter& writer) {
  writer.put_bit(true);
  writer.align_to_byte();
}

std::vector<std::uint8_t> take_aligned(BitWriter& writer) {
  writer.align_to_byte();
  return writer.take_bytes();
}

void append_leb128(std::vector<std::uint8_t>& out, std::uint32_t value) {
  do {
    std::uint8_t byte = static_cast<std::uint8_t>(value & 0x7FU);
    value >>= 7U;
    if (value != 0) byte = static_cast<std::uint8_t>(byte | 0x80U);
    out.push_back(byte);
  } while (value != 0);
}

void append_leb128_bytes(BitWriter& writer, std::uint32_t value) {
  std::vector<std::uint8_t> bytes;
  append_leb128(bytes, value);
  for (std::uint8_t byte : bytes) writer.put_bits(byte, 8);
}

// --- MSB-first cursor ------------------------------------------------------

class Cursor {
 public:
  Cursor() = default;
  explicit Cursor(std::span<const std::uint8_t> data) : data_(data) {}

  bool read_bit(bool& value, std::string& error) {
    if (bit_position_ >= data_.size() * 8U) {
      return fail(error, "truncated syntax: ran past the payload");
    }
    const std::size_t byte = bit_position_ >> 3U;
    const unsigned shift = 7U - static_cast<unsigned>(bit_position_ & 7U);
    value = ((data_[byte] >> shift) & 1U) != 0U;
    ++bit_position_;
    return true;
  }

  bool read_bits(unsigned count, std::uint32_t& value, std::string& error) {
    std::uint32_t out = 0;
    for (unsigned i = 0; i < count; ++i) {
      bool bit = false;
      if (!read_bit(bit, error)) return false;
      out = (out << 1U) | (bit ? 1U : 0U);
    }
    value = out;
    return true;
  }

  bool read_uvlc(std::uint32_t& value, std::string& error) {
    unsigned leading_zeros = 0;
    bool bit = false;
    do {
      if (!read_bit(bit, error)) return false;
      if (bit) break;
      ++leading_zeros;
    } while (leading_zeros < 32U);
    if (leading_zeros >= 32U) {
      return fail(error, "uvlc() has 32 leading zero bits");
    }
    if (leading_zeros == 0) {
      value = 0;
      return true;
    }
    std::uint32_t payload = 0;
    if (!read_bits(leading_zeros, payload, error)) return false;
    value = ((1U << leading_zeros) - 1U) + payload;
    return true;
  }

  bool read_rg(unsigned n, std::uint32_t& value, std::string& error) {
    unsigned q = 0;
    for (; q < 32U; ++q) {
      bool bit = false;
      if (!read_bit(bit, error)) return false;
      if (!bit) {
        std::uint32_t remainder = 0;
        if (n > 0 && !read_bits(n, remainder, error)) return false;
        value = (q << n) | remainder;
        return true;
      }
    }
    return fail(error, "rg() prefix exceeds 31 one bits");
  }

  bool align_to_byte(std::string& error) {
    while ((bit_position_ & 7U) != 0U) {
      bool bit = false;
      if (!read_bit(bit, error)) return false;
    }
    return true;
  }

  bool read_leb128(std::uint32_t& value, std::string& error) {
    if ((bit_position_ & 7U) != 0U) {
      return fail(error, "leb128() is not byte-aligned");
    }
    std::uint64_t out = 0;
    unsigned shift = 0;
    unsigned count = 0;
    while (true) {
      std::uint32_t byte = 0;
      if (!read_bits(8, byte, error)) return false;
      ++count;
      if (count > 5U) return fail(error, "leb128() exceeds 32 bits");
      out |= static_cast<std::uint64_t>(byte & 0x7FU) << shift;
      if ((byte & 0x80U) == 0) break;
      shift += 7U;
    }
    if (out > std::numeric_limits<std::uint32_t>::max()) {
      return fail(error, "leb128() exceeds 32 bits");
    }
    value = static_cast<std::uint32_t>(out);
    return true;
  }

  bool skip_bytes(std::size_t count, std::string& error) {
    if ((bit_position_ & 7U) != 0U) {
      return fail(error, "internal: attempted a byte skip while unaligned");
    }
    if (count > bits_remaining() / 8U) {
      return fail(error, "lcr_data_size exceeds the payload");
    }
    bit_position_ += count * 8U;
    return true;
  }

  std::size_t bit_position() const { return bit_position_; }
  std::size_t bits_remaining() const {
    const std::size_t total = data_.size() * 8U;
    return bit_position_ >= total ? 0U : total - bit_position_;
  }
  bool byte_aligned() const { return (bit_position_ & 7U) == 0U; }
  std::span<const std::uint8_t> data() const { return data_; }

 private:
  std::span<const std::uint8_t> data_{};
  std::size_t bit_position_ = 0;
};

// Reads the extensible-OBU tail: obu_extension_flag == 0, then trailing_bits()
// (a single one bit, zero-padded to the end of the payload).
bool read_extension_and_trailing_bits(Cursor& cursor, std::string& error) {
  bool bit = false;
  if (!cursor.read_bit(bit, error)) return false;
  if (bit) {
    return fail(error, "obu_extension_flag must be 0 for LCR/OPS (§6.2.1)");
  }
  if (cursor.bits_remaining() == 0U) {
    return fail(error, "missing trailing_bits() stop bit");
  }
  if (!cursor.read_bit(bit, error)) return false;
  if (!bit) return fail(error, "trailing_bits() must start with a one bit");
  while (cursor.bits_remaining() != 0U) {
    if (!cursor.read_bit(bit, error)) return false;
    if (bit) return fail(error, "non-zero bit after the trailing_bits() stop");
  }
  return true;
}

// ---------------------------------------------------------------------------
// LCR writers
// ---------------------------------------------------------------------------

bool write_lcr_global_info(BitWriter& writer, const GlobalLcrInfo& info,
                           std::string& error);

bool write_lcr_local_info(BitWriter& writer, const LocalLcrInfo& info,
                          std::string& error);

bool check_uvlc_range(std::uint32_t value, const char* name,
                      std::string& error) {
  if (value >= 0xFFFFFFFEU) {
    std::ostringstream out;
    out << name << " is outside uvlc()";
    return fail(error, out.str());
  }
  return true;
}

}  // namespace

std::size_t popcount32(std::uint32_t value) {
  std::size_t count = 0;
  while (value != 0) {
    value &= value - 1U;
    ++count;
  }
  return count;
}

bool global_xlayer(const OperatingPointSet& set) {
  return set.xlayer_id == static_cast<std::uint8_t>(kGlobalXLayerId);
}

bool global_xlayer(const LayerConfigRecord& record) { return record.is_global; }

OpsSummary summarize_ops(const OperatingPointSet& set) {
  OpsSummary summary;
  summary.xlayer_id = set.xlayer_id;
  summary.ops_id = set.ops_id;
  summary.ops_cnt = static_cast<std::uint8_t>(set.points.size());
  return summary;
}

// ---------------------------------------------------------------------------
// LCR sub-structure writers
// ---------------------------------------------------------------------------

bool write_lcr_aggregate_info(BitWriter& writer, const AggregateInfo& aggregate,
                              std::string& error) {
  if (aggregate.config_idc > 63U) {
    return fail(error, "lcr_config_idc is outside f(6)");
  }
  if (aggregate.level_idx > 31U) {
    return fail(error, "lcr_aggregate_level_idx is outside f(5)");
  }
  if (aggregate.max_interop > 15U) {
    return fail(error, "lcr_max_interop is outside f(4)");
  }
  writer.put_bits(aggregate.config_idc, 6);
  writer.put_bits(aggregate.level_idx, 5);
  writer.put_bit(aggregate.max_tier_flag);
  writer.put_bits(aggregate.max_interop, 4);
  return true;
}

bool write_lcr_seq_profile_tier_level_info(
    BitWriter& writer, const ProfileTierLevelInfo& info, std::string& error) {
  if (info.seq_profile_idc > 31U) {
    return fail(error, "lcr_seq_profile_idc is outside f(5)");
  }
  if (info.max_level_idx > 31U) {
    return fail(error, "lcr_max_level_idx is outside f(5)");
  }
  if (info.max_mlayer_count > 7U) {
    return fail(error, "lcr_max_mlayer_count is outside f(3)");
  }
  writer.put_bits(info.seq_profile_idc, 5);
  writer.put_bits(info.max_level_idx, 5);
  writer.put_bit(info.tier_flag);
  writer.put_bits(info.max_mlayer_count, 3);
  writer.put_bits(0U, 2);  // lsptli_reserved_2bits
  return true;
}

bool write_lcr_xlayer_color_info(BitWriter& writer,
                                 const ColorDescription& color,
                                 std::string& error) {
  if (color.description_idc > 3U) {
    return fail(error,
                "layer_color_description_idc is outside rg(2); Table 6.13 "
                "values 4 and 5 are not representable by the rg(2) descriptor");
  }
  put_rg(writer, color.description_idc, 2);
  if (color.description_idc == 0U) {
    writer.put_bits(color.primaries, 8);
    writer.put_bits(color.transfer, 8);
    writer.put_bits(color.matrix, 8);
  }
  writer.put_bit(color.full_range);
  return true;
}

bool write_lcr_rep_info(BitWriter& writer, const RepInfo& rep,
                        std::string& error) {
  if (!check_uvlc_range(rep.max_pic_width, "lcr_max_pic_width", error) ||
      !check_uvlc_range(rep.max_pic_height, "lcr_max_pic_height", error)) {
    return false;
  }
  put_uvlc(writer, rep.max_pic_width);
  put_uvlc(writer, rep.max_pic_height);
  writer.put_bit(rep.format_info_present);
  writer.put_bit(rep.cropping_window_present);
  if (rep.format_info_present) {
    if (!check_uvlc_range(rep.bit_depth_idc, "lcr_bit_depth_idc", error) ||
        !check_uvlc_range(rep.chroma_format_idc, "lcr_chroma_format_idc",
                          error)) {
      return false;
    }
    put_uvlc(writer, rep.bit_depth_idc);
    put_uvlc(writer, rep.chroma_format_idc);
  }
  if (rep.cropping_window_present) {
    if (!check_uvlc_range(rep.crop_left, "lcr_cropping_win_left_offset",
                          error) ||
        !check_uvlc_range(rep.crop_right, "lcr_cropping_win_right_offset",
                          error) ||
        !check_uvlc_range(rep.crop_top, "lcr_cropping_win_top_offset",
                          error) ||
        !check_uvlc_range(rep.crop_bottom, "lcr_cropping_win_bottom_offset",
                          error)) {
      return false;
    }
    put_uvlc(writer, rep.crop_left);
    put_uvlc(writer, rep.crop_right);
    put_uvlc(writer, rep.crop_top);
    put_uvlc(writer, rep.crop_bottom);
  }
  return true;
}

bool write_lcr_embedded_layer_info(BitWriter& writer,
                                   const EmbeddedLayerInfo& embedded,
                                   bool atlas_segment_present,
                                   std::string& error) {
  const int j = embedded.mlayer_id;
  if (j < 0 || j >= kMaxEmbeddedLayers) {
    return fail(error, "embedded layer id is outside 0..7");
  }
  if ((embedded.tlayer_map & ~0x0FU) != 0U || embedded.tlayer_map == 0U) {
    return fail(error, "lcr_tlayer_map must be a non-empty 4-bit map");
  }
  if ((embedded.dependent_layer_mask >> j) != 0U) {
    return fail(error, "lcr_dependent_layer_map references a layer >= j");
  }
  writer.put_bits(embedded.tlayer_map, 4);
  if (atlas_segment_present) {
    writer.put_bits(embedded.atlas_segment_id, 8);
    writer.put_bits(embedded.priority_order, 8);
    writer.put_bits(embedded.rendering_method, 8);
  }
  writer.put_bits(embedded.layer_type, 8);
  if (embedded.layer_type == kLayerAux) {
    writer.put_bits(embedded.auxiliary_type, 8);
  }
  writer.put_bits(embedded.view_type, 8);
  if (embedded.view_type == kViewExplicit) {
    writer.put_bits(embedded.view_id, 8);
  }
  if (j > 0) {
    writer.put_bits(embedded.dependent_layer_mask, static_cast<unsigned>(j));
  }
  writer.put_bit(embedded.same_sh_max_resolution);
  if (!embedded.same_sh_max_resolution) {
    if (embedded.max_expected_width == 0U ||
        embedded.max_expected_height == 0U) {
      return fail(error,
                  "lcr_max_expected_width/height must be non-zero when they "
                  "differ from the sequence header");
    }
    if (!check_uvlc_range(embedded.max_expected_width,
                          "lcr_max_expected_width", error) ||
        !check_uvlc_range(embedded.max_expected_height,
                          "lcr_max_expected_height", error)) {
      return false;
    }
    put_uvlc(writer, embedded.max_expected_width);
    put_uvlc(writer, embedded.max_expected_height);
  }
  writer.align_to_byte();
  return true;
}

bool write_lcr_xlayer_info(BitWriter& writer, const XLayerInfo& info,
                           bool is_global, bool atlas_id_present,
                           std::string& error) {
  writer.put_bit(info.rep_info_present);
  writer.put_bit(info.purpose_present);
  writer.put_bit(info.color_info_present);
  writer.put_bit(info.embedded_layer_info_present);
  if (info.rep_info_present) {
    if (!write_lcr_rep_info(writer, info.rep, error)) return false;
  }
  if (info.purpose_present) {
    if (info.purpose_id > 127U) {
      return fail(error, "lcr_xlayer_purpose_id is outside f(7)");
    }
    writer.put_bits(info.purpose_id, 7);
  }
  if (info.color_info_present) {
    if (!write_lcr_xlayer_color_info(writer, info.color, error)) return false;
  }
  writer.align_to_byte();
  if (info.embedded_layer_info_present) {
    if (info.mlayer_map == 0U) {
      return fail(error, "lcr_mlayer_map must be non-empty when present");
    }
    if (popcount32(info.mlayer_map) != info.embedded.size()) {
      return fail(error,
                  "lcr_mlayer_map must describe exactly the embedded layers "
                  "given in the record");
    }
    writer.put_bits(info.mlayer_map, 8);
    for (int j = 0; j < kMaxEmbeddedLayers; ++j) {
      if ((info.mlayer_map & (1U << static_cast<unsigned>(j))) == 0U) continue;
      const EmbeddedLayerInfo* embedded = nullptr;
      for (const EmbeddedLayerInfo& candidate : info.embedded) {
        if (candidate.mlayer_id == j) {
          embedded = &candidate;
          break;
        }
      }
      if (embedded == nullptr) {
        std::ostringstream out;
        out << "lcr_mlayer_map selects m" << j
            << " but no descriptor was supplied";
        return fail(error, out.str());
      }
      if (!write_lcr_embedded_layer_info(writer, *embedded,
                                         atlas_id_present, error)) {
        return false;
      }
    }
  } else if (is_global && atlas_id_present) {
    writer.put_bits(info.xlayer_atlas_segment_id, 8);
    writer.put_bits(info.xlayer_priority_order, 8);
    writer.put_bits(info.xlayer_rendering_method, 8);
  }
  return true;
}

namespace {

// §5.8.5 payload bits for one extended layer (no lcr_data_size prefix).
bool write_lcr_global_payload(BitWriter& writer, const GlobalPayload& payload,
                              bool is_global, bool dependent_xlayers,
                              bool atlas_id_present, std::string& error) {
  const int x = payload.xlayer_id;
  if (x < 0 || x > kMaxExtendedLayerId) {
    return fail(error, "global LCR payload xlayer id is outside 0..30");
  }
  if (dependent_xlayers && x > 0) {
    if ((payload.dependent_xlayer_mask >> x) != 0U) {
      return fail(error,
                  "lcr_num_dependent_xlayer_map references a layer >= its own "
                  "id");
    }
    writer.put_bits(payload.dependent_xlayer_mask, static_cast<unsigned>(x));
  }
  return write_lcr_xlayer_info(writer, payload.info, is_global,
                               atlas_id_present, error);
}

bool write_lcr_global_info(BitWriter& writer, const GlobalLcrInfo& info,
                           std::string& error) {
  if (info.config_record_id < 1U || info.config_record_id > 7U) {
    return fail(error,
                "lcr_global_config_record_id must be 1..7 (§6.8.2)");
  }
  if (info.xlayer_map == 0U || (info.xlayer_map & ~0x7FFFFFFFU) != 0U) {
    return fail(error, "lcr_xlayer_map must be non-zero and fit f(31)");
  }
  if (info.dependent_xlayers_flag) {
    return fail(error,
                "lcr_dependent_xlayers_flag must be 0 (§6.8.2); per-xlayer "
                "dependency maps are reserved in this version");
  }
  writer.put_bits(info.config_record_id, 3);
  writer.put_bits(info.xlayer_map, 31);
  writer.put_bit(info.aggregate_info_present);
  writer.put_bit(info.profile_tier_level_info_present);
  writer.put_bit(info.global_payload_present);
  writer.put_bit(info.dependent_xlayers_flag);
  writer.put_bit(info.global_atlas_id_present);
  if (info.global_purpose_id > 127U) {
    return fail(error, "lcr_global_purpose_id is outside f(7)");
  }
  writer.put_bits(info.global_purpose_id, 7);
  writer.put_bit(info.doh_constraint_flag);
  writer.put_bit(info.enforce_tile_alignment_flag);
  if (info.global_atlas_id_present) {
    if (info.global_atlas_id > 7U) {
      return fail(error, "lcr_global_atlas_id is outside f(3)");
    }
    writer.put_bits(info.global_atlas_id, 3);
  } else {
    writer.put_bits(0U, 3);  // lcr_global_reserved_zero_3bits
  }
  writer.put_bits(0U, 5);  // lcr_global_reserved_zero_5bits

  if (info.aggregate_info_present) {
    if (!write_lcr_aggregate_info(writer, info.aggregate, error)) return false;
  }
  if (info.profile_tier_level_info_present) {
    for (int x = 0; x <= kMaxExtendedLayerId; ++x) {
      if ((info.xlayer_map & (1U << static_cast<unsigned>(x))) == 0U) continue;
      if (!write_lcr_seq_profile_tier_level_info(
              writer, info.ptl[static_cast<std::size_t>(x)], error)) {
        return false;
      }
    }
  }
  if (info.global_payload_present) {
    int previous = -1;
    for (const GlobalPayload& payload : info.payloads) {
      if (payload.xlayer_id <= previous) {
        return fail(error,
                    "LCR payloads must be unique and in ascending xlayer "
                    "order");
      }
      previous = payload.xlayer_id;
      if ((info.xlayer_map & (1U << payload.xlayer_id)) == 0U) {
        return fail(error,
                    "LCR payload xlayer is not selected by lcr_xlayer_map");
      }
      BitWriter payload_writer;
      if (!write_lcr_global_payload(payload_writer, payload, true,
                                    info.dependent_xlayers_flag,
                                    info.global_atlas_id_present, error)) {
        return false;
      }
      const std::vector<std::uint8_t> bytes = take_aligned(payload_writer);
      std::uint32_t size = static_cast<std::uint32_t>(bytes.size());
      if (payload.declared_size_bytes != 0U) {
        if (payload.declared_size_bytes < size) {
          return fail(error,
                      "lcr_data_size is smaller than the payload syntax");
        }
        size = payload.declared_size_bytes;
      }
      append_leb128_bytes(writer, size);
      for (std::uint8_t byte : bytes) {
        writer.put_bits(byte, 8);
      }
      // lcr_remaining_payload_bit: zero to lcr_data_size bytes. The syntax
      // always ends byte-aligned, so the remainder is whole bytes.
      for (std::uint32_t pad = static_cast<std::uint32_t>(bytes.size());
           pad < size; ++pad) {
        writer.put_bits(0U, 8);
      }
    }
    if (info.payloads.size() != popcount32(info.xlayer_map)) {
      return fail(error,
                  "lcr_global_payload_present = 1 requires one payload per "
                  "layer selected by lcr_xlayer_map");
    }
  }
  return true;
}

bool write_lcr_local_info(BitWriter& writer, const LocalLcrInfo& info,
                          std::string& error) {
  if (info.local_id < 1U || info.local_id > 7U) {
    return fail(error, "lcr_local_id must be 1..7 (§6.8.3)");
  }
  if (info.global_id > 7U) {
    return fail(error, "lcr_global_id is outside f(3)");
  }
  if (info.xlayer_id > kMaxExtendedLayerId) {
    return fail(error, "local LCR obu_xlayer_id is outside 0..30");
  }
  if (info.local_atlas_id_present && info.local_atlas_id > 7U) {
    return fail(error, "lcr_local_atlas_id is outside f(3)");
  }
  writer.put_bits(info.global_id, 3);
  writer.put_bits(info.local_id, 3);
  writer.put_bit(info.profile_tier_level_info_present);
  writer.put_bit(info.local_atlas_id_present);
  if (info.profile_tier_level_info_present) {
    if (!write_lcr_seq_profile_tier_level_info(writer, info.ptl, error)) {
      return false;
    }
  }
  if (info.local_atlas_id_present) {
    writer.put_bits(info.local_atlas_id, 3);
  } else {
    writer.put_bits(0U, 3);  // lcr_local_reserved_zero_3bits
  }
  writer.put_bits(0U, 5);  // lcr_local_reserved_zero_5bits
  return write_lcr_xlayer_info(writer, info.xlayer, false,
                               info.local_atlas_id_present, error);
}

}  // namespace

// ---------------------------------------------------------------------------
// LCR payload (public)
// ---------------------------------------------------------------------------

bool write_lcr_payload(const LayerConfigRecord& record,
                       std::vector<std::uint8_t>& out, std::string& error) {
  out.clear();
  BitWriter writer;
  if (record.is_global) {
    if (!write_lcr_global_info(writer, record.global, error)) return false;
  } else {
    if (!write_lcr_local_info(writer, record.local, error)) return false;
  }
  writer.put_bit(false);  // obu_extension_flag
  put_trailing_bits(writer);
  out = take_aligned(writer);
  return true;
}

// ---------------------------------------------------------------------------
// OPS sub-structure writers
// ---------------------------------------------------------------------------

bool write_ops_aggregate_info(BitWriter& writer,
                              const OpsAggregateInfo& aggregate,
                              std::string& error) {
  if (aggregate.config_idc > 63U) {
    return fail(error, "ops_config_idc is outside f(6)");
  }
  if (aggregate.level_idx > 31U) {
    return fail(error, "ops_aggregate_level_idx is outside f(5)");
  }
  if (aggregate.max_interop > 15U) {
    return fail(error, "ops_max_interop is outside f(4)");
  }
  writer.put_bits(aggregate.config_idc, 6);
  writer.put_bits(aggregate.level_idx, 5);
  writer.put_bit(aggregate.max_tier_flag);
  writer.put_bits(aggregate.max_interop, 4);
  return true;
}

bool write_ops_seq_profile_tier_level_info(
    BitWriter& writer, const OpsProfileTierLevelInfo& info,
    std::string& error) {
  if (info.seq_profile_idc > 31U) {
    return fail(error, "ops_seq_profile_idc is outside f(5)");
  }
  if (info.level_idx > 31U) {
    return fail(error, "ops_level_idx is outside f(5)");
  }
  if (info.mlayer_count > 7U) {
    return fail(error, "ops_mlayer_count is outside f(3)");
  }
  writer.put_bits(info.seq_profile_idc, 5);
  writer.put_bits(info.level_idx, 5);
  writer.put_bit(info.tier_flag);
  writer.put_bits(info.mlayer_count, 3);
  writer.put_bits(0U, 2);  // ops_ptl_reserved_2bits
  return true;
}

bool write_ops_decoder_model_info(BitWriter& writer,
                                  const OpsDecoderModelInfo& info,
                                  std::string& error) {
  if (!check_uvlc_range(info.decoder_buffer_delay,
                        "ops_decoder_buffer_delay", error) ||
      !check_uvlc_range(info.encoder_buffer_delay,
                        "ops_encoder_buffer_delay", error)) {
    return false;
  }
  put_uvlc(writer, info.decoder_buffer_delay);
  put_uvlc(writer, info.encoder_buffer_delay);
  writer.put_bit(info.low_delay_mode_flag);
  return true;
}

bool write_ops_color_info(BitWriter& writer, const ColorDescription& color,
                          std::string& error) {
  if (color.description_idc > 3U) {
    return fail(error,
                "ops_color_description_idc is outside rg(2); Table 6.13 "
                "values 4 and 5 are not representable by the rg(2) descriptor");
  }
  put_rg(writer, color.description_idc, 2);
  if (color.description_idc == 0U) {
    writer.put_bits(color.primaries, 8);
    writer.put_bits(color.transfer, 8);
    writer.put_bits(color.matrix, 8);
  }
  writer.put_bit(color.full_range);
  return true;
}

bool write_ops_mlayer_info(BitWriter& writer, const OpsLayerMlayerInfo& info,
                           std::string& error) {
  if (info.mlayer_map == 0U) {
    return fail(error, "ops_mlayer_map must be non-empty");
  }
  writer.put_bits(info.mlayer_map, 8);
  for (int j = 0; j < kMaxEmbeddedLayers; ++j) {
    if ((info.mlayer_map & (1U << static_cast<unsigned>(j))) == 0U) continue;
    const std::uint8_t tlayer_map =
        info.tlayer_map[static_cast<std::size_t>(j)];
    if (tlayer_map == 0U || (tlayer_map & ~0x0FU) != 0U) {
      std::ostringstream out;
      out << "ops_tlayer_map[m" << j << "] must be a non-empty 4-bit map";
      return fail(error, out.str());
    }
    writer.put_bits(tlayer_map, 4);
  }
  return true;
}

namespace {

bool write_operating_point_payload(BitWriter& writer,
                                   const OperatingPointSet& set,
                                   const OpsPoint& point, std::string& error) {
  const bool is_global = global_xlayer(set);
  if (set.intent_present) {
    if (point.op_intent > 127U) {
      return fail(error, "ops_op_intent is outside f(7)");
    }
    writer.put_bits(point.op_intent, 7);
  }
  if (set.ptl_present) {
    if (is_global) {
      if (!write_ops_aggregate_info(writer, point.aggregate, error)) {
        return false;
      }
    } else if (!write_ops_seq_profile_tier_level_info(
                   writer, point.ptl[static_cast<std::size_t>(set.xlayer_id)],
                   error)) {
      return false;
    }
  }
  if (set.color_info_present) {
    if (!write_ops_color_info(writer, point.color, error)) return false;
  }
  writer.put_bit(point.decoder_model_info_present);
  if (point.decoder_model_info_present) {
    if (!write_ops_decoder_model_info(writer, point.decoder_model, error)) {
      return false;
    }
  }
  writer.put_bit(point.initial_display_delay_present);
  if (point.initial_display_delay_present) {
    if (point.initial_display_delay_minus_1 > 15U) {
      return fail(error, "ops_initial_display_delay_minus_1 is outside f(4)");
    }
    writer.put_bits(point.initial_display_delay_minus_1, 4);
  }

  if (is_global) {
    if ((point.xlayer_map & ~0x7FFFFFFFU) != 0U) {
      return fail(error, "ops_xlayer_map is outside f(31)");
    }
    writer.put_bits(point.xlayer_map, 31);
    for (int j = 0; j <= kMaxExtendedLayerId; ++j) {
      if ((point.xlayer_map & (1U << static_cast<unsigned>(j))) == 0U) continue;
      if (set.ptl_present) {
        if (!write_ops_seq_profile_tier_level_info(
                writer, point.ptl[static_cast<std::size_t>(j)], error)) {
          return false;
        }
      }
      if (set.mlayer_info_idc == 1U) {
        if (!write_ops_mlayer_info(
                writer, point.layer_info[static_cast<std::size_t>(j)], error)) {
          return false;
        }
      } else if (set.mlayer_info_idc == 2U) {
        const OpsLayerMlayerInfo& layer =
            point.layer_info[static_cast<std::size_t>(j)];
        writer.put_bit(layer.explicit_info);
        if (layer.explicit_info) {
          if (!write_ops_mlayer_info(writer, layer, error)) return false;
        } else {
          if (layer.embedded_ops_id > 15U) {
            return fail(error, "ops_embedded_ops_id is outside f(4)");
          }
          if (layer.embedded_op_index > 7U) {
            return fail(error, "ops_embedded_op_index is outside f(3)");
          }
          writer.put_bits(layer.embedded_ops_id, 4);
          writer.put_bits(layer.embedded_op_index, 3);
        }
      }
    }
  } else {
    if (!write_ops_mlayer_info(
            writer, point.layer_info[static_cast<std::size_t>(set.xlayer_id)],
            error)) {
      return false;
    }
  }
  writer.align_to_byte();
  return true;
}

}  // namespace

bool write_ops_payload(const OperatingPointSet& set,
                       std::vector<std::uint8_t>& out, std::string& error) {
  out.clear();
  if (set.xlayer_id > kGlobalXLayerId) {
    return fail(error, "OPS obu_xlayer_id is outside 0..31");
  }
  if (set.ops_id > kMaxOpsId) {
    return fail(error, "ops_id is outside f(4)");
  }
  if (set.points.size() > static_cast<std::size_t>(kMaxOpsCnt)) {
    return fail(error, "ops_cnt must be 0..7");
  }
  const bool is_global = global_xlayer(set);
  if (is_global) {
    if (set.mlayer_info_idc == 3U) {
      return fail(error, "ops_mlayer_info_idc must not be 3 (§6.10.2)");
    }
  } else if (set.reserved_2bits != 0U) {
    return fail(error, "ops_reserved_2bits must be 0");
  }

  BitWriter writer;
  writer.put_bit(set.reset_flag);
  writer.put_bits(set.ops_id, 4);
  writer.put_bits(static_cast<std::uint32_t>(set.points.size()), 3);
  if (!set.points.empty()) {
    if (set.priority > 15U) {
      return fail(error, "ops_priority is outside f(4)");
    }
    if (set.intent > 127U) {
      return fail(error, "ops_intent is outside f(7)");
    }
    writer.put_bits(set.priority, 4);
    writer.put_bits(set.intent, 7);
    writer.put_bit(set.intent_present);
    writer.put_bit(set.ptl_present);
    writer.put_bit(set.color_info_present);
    if (is_global) {
      writer.put_bits(set.mlayer_info_idc, 2);
    } else {
      writer.put_bits(set.reserved_2bits, 2);
    }
    for (std::size_t i = 0; i < set.points.size(); ++i) {
      const OpsPoint& point = set.points[i];
      BitWriter payload;
      if (!write_operating_point_payload(payload, set, point, error)) {
        return false;
      }
      const std::vector<std::uint8_t> bytes = take_aligned(payload);
      append_leb128_bytes(writer, static_cast<std::uint32_t>(bytes.size()));
      for (std::uint8_t byte : bytes) writer.put_bits(byte, 8);
    }
  }
  writer.put_bit(false);  // obu_extension_flag
  put_trailing_bits(writer);
  out = take_aligned(writer);
  return true;
}

// ---------------------------------------------------------------------------
// LCR reader
// ---------------------------------------------------------------------------

namespace {

bool read_color_description(Cursor& cursor, ColorDescription& color,
                            std::string& error) {
  std::uint32_t idc = 0;
  if (!cursor.read_rg(2, idc, error)) return false;
  color.description_idc = static_cast<std::uint8_t>(idc);
  if (idc == 0U) {
    std::uint32_t value = 0;
    if (!cursor.read_bits(8, value, error)) return false;
    color.primaries = static_cast<std::uint8_t>(value);
    if (!cursor.read_bits(8, value, error)) return false;
    color.transfer = static_cast<std::uint8_t>(value);
    if (!cursor.read_bits(8, value, error)) return false;
    color.matrix = static_cast<std::uint8_t>(value);
  }
  bool full_range = false;
  if (!cursor.read_bit(full_range, error)) return false;
  color.full_range = full_range;
  return true;
}

bool read_lcr_embedded_layer_info(Cursor& cursor,
                                  bool atlas_segment_present, int j,
                                  EmbeddedLayerInfo& embedded,
                                  std::string& error) {
  std::uint32_t value = 0;
  embedded.mlayer_id = static_cast<std::uint8_t>(j);
  if (!cursor.read_bits(4, value, error)) return false;
  embedded.tlayer_map = static_cast<std::uint8_t>(value);
  if (atlas_segment_present) {
    if (!cursor.read_bits(8, value, error)) return false;
    embedded.atlas_segment_id = static_cast<std::uint8_t>(value);
    if (!cursor.read_bits(8, value, error)) return false;
    embedded.priority_order = static_cast<std::uint8_t>(value);
    if (!cursor.read_bits(8, value, error)) return false;
    embedded.rendering_method = static_cast<std::uint8_t>(value);
  }
  if (!cursor.read_bits(8, value, error)) return false;
  embedded.layer_type = static_cast<std::uint8_t>(value);
  if (embedded.layer_type == kLayerAux) {
    if (!cursor.read_bits(8, value, error)) return false;
    embedded.auxiliary_type = static_cast<std::uint8_t>(value);
  }
  if (!cursor.read_bits(8, value, error)) return false;
  embedded.view_type = static_cast<std::uint8_t>(value);
  if (embedded.view_type == kViewExplicit) {
    if (!cursor.read_bits(8, value, error)) return false;
    embedded.view_id = static_cast<std::uint8_t>(value);
  }
  if (j > 0) {
    std::uint32_t mask = 0;
    if (!cursor.read_bits(static_cast<unsigned>(j), mask, error)) return false;
    embedded.dependent_layer_mask = mask;
  }
  bool same = false;
  if (!cursor.read_bit(same, error)) return false;
  embedded.same_sh_max_resolution = same;
  if (!same) {
    if (!cursor.read_uvlc(embedded.max_expected_width, error)) return false;
    if (!cursor.read_uvlc(embedded.max_expected_height, error)) return false;
  }
  return cursor.align_to_byte(error);
}

bool read_lcr_xlayer_info(Cursor& cursor, bool is_global,
                          bool atlas_id_present, XLayerInfo& info,
                          std::string& error) {
  bool flag = false;
  if (!cursor.read_bit(flag, error)) return false;
  info.rep_info_present = flag;
  if (!cursor.read_bit(flag, error)) return false;
  info.purpose_present = flag;
  if (!cursor.read_bit(flag, error)) return false;
  info.color_info_present = flag;
  if (!cursor.read_bit(flag, error)) return false;
  info.embedded_layer_info_present = flag;

  if (info.rep_info_present) {
    RepInfo& rep = info.rep;
    if (!cursor.read_uvlc(rep.max_pic_width, error)) return false;
    if (!cursor.read_uvlc(rep.max_pic_height, error)) return false;
    if (!cursor.read_bit(rep.format_info_present, error)) return false;
    if (!cursor.read_bit(rep.cropping_window_present, error)) return false;
    if (rep.format_info_present) {
      std::uint32_t value = 0;
      if (!cursor.read_uvlc(value, error)) return false;
      rep.bit_depth_idc = static_cast<std::uint8_t>(value);
      if (!cursor.read_uvlc(value, error)) return false;
      rep.chroma_format_idc = static_cast<std::uint8_t>(value);
    }
    if (rep.cropping_window_present) {
      if (!cursor.read_uvlc(rep.crop_left, error)) return false;
      if (!cursor.read_uvlc(rep.crop_right, error)) return false;
      if (!cursor.read_uvlc(rep.crop_top, error)) return false;
      if (!cursor.read_uvlc(rep.crop_bottom, error)) return false;
    }
  }
  if (info.purpose_present) {
    std::uint32_t value = 0;
    if (!cursor.read_bits(7, value, error)) return false;
    info.purpose_id = static_cast<std::uint8_t>(value);
  }
  if (info.color_info_present) {
    if (!read_color_description(cursor, info.color, error)) return false;
  }
  if (!cursor.align_to_byte(error)) return false;

  if (info.embedded_layer_info_present) {
    std::uint32_t mlayer_map = 0;
    if (!cursor.read_bits(8, mlayer_map, error)) return false;
    info.mlayer_map = static_cast<std::uint8_t>(mlayer_map);
    for (int j = 0; j < kMaxEmbeddedLayers; ++j) {
      if ((mlayer_map & (1U << static_cast<unsigned>(j))) == 0U) continue;
      EmbeddedLayerInfo embedded;
      if (!read_lcr_embedded_layer_info(cursor, atlas_id_present, j, embedded,
                                        error)) {
        return false;
      }
      info.embedded.push_back(embedded);
    }
  } else if (is_global && atlas_id_present) {
    std::uint32_t value = 0;
    if (!cursor.read_bits(8, value, error)) return false;
    info.xlayer_atlas_segment_id = static_cast<std::uint8_t>(value);
    if (!cursor.read_bits(8, value, error)) return false;
    info.xlayer_priority_order = static_cast<std::uint8_t>(value);
    if (!cursor.read_bits(8, value, error)) return false;
    info.xlayer_rendering_method = static_cast<std::uint8_t>(value);
  }
  return true;
}

bool read_lcr_seq_profile_tier_level_info(Cursor& cursor,
                                          ProfileTierLevelInfo& info,
                                          std::string& error) {
  std::uint32_t value = 0;
  if (!cursor.read_bits(5, value, error)) return false;
  info.seq_profile_idc = static_cast<std::uint8_t>(value);
  if (!cursor.read_bits(5, value, error)) return false;
  info.max_level_idx = static_cast<std::uint8_t>(value);
  bool tier = false;
  if (!cursor.read_bit(tier, error)) return false;
  info.tier_flag = tier;
  if (!cursor.read_bits(3, value, error)) return false;
  info.max_mlayer_count = static_cast<std::uint8_t>(value);
  if (!cursor.read_bits(2, value, error)) return false;  // reserved
  return true;
}

bool read_lcr_aggregate_info(Cursor& cursor, AggregateInfo& aggregate,
                             std::string& error) {
  std::uint32_t value = 0;
  if (!cursor.read_bits(6, value, error)) return false;
  aggregate.config_idc = static_cast<std::uint8_t>(value);
  if (!cursor.read_bits(5, value, error)) return false;
  aggregate.level_idx = static_cast<std::uint8_t>(value);
  bool tier = false;
  if (!cursor.read_bit(tier, error)) return false;
  aggregate.max_tier_flag = tier;
  if (!cursor.read_bits(4, value, error)) return false;
  aggregate.max_interop = static_cast<std::uint8_t>(value);
  return true;
}

}  // namespace

bool read_lcr_payload(std::span<const std::uint8_t> payload, bool is_global,
                      LayerConfigRecord& record, std::string& error) {
  record = LayerConfigRecord{};
  record.is_global = is_global;
  record.global = GlobalLcrInfo{};
  record.local = LocalLcrInfo{};
  Cursor cursor(payload);

  if (is_global) {
    GlobalLcrInfo& info = record.global;
    std::uint32_t value = 0;
    if (!cursor.read_bits(3, value, error)) return false;
    info.config_record_id = static_cast<std::uint8_t>(value);
    if (!cursor.read_bits(31, value, error)) return false;
    info.xlayer_map = value;
    bool flag = false;
    if (!cursor.read_bit(flag, error)) return false;
    info.aggregate_info_present = flag;
    if (!cursor.read_bit(flag, error)) return false;
    info.profile_tier_level_info_present = flag;
    if (!cursor.read_bit(flag, error)) return false;
    info.global_payload_present = flag;
    if (!cursor.read_bit(flag, error)) return false;
    info.dependent_xlayers_flag = flag;
    if (!cursor.read_bit(flag, error)) return false;
    info.global_atlas_id_present = flag;
    if (!cursor.read_bits(7, value, error)) return false;
    info.global_purpose_id = static_cast<std::uint8_t>(value);
    if (!cursor.read_bit(flag, error)) return false;
    info.doh_constraint_flag = flag;
    if (!cursor.read_bit(flag, error)) return false;
    info.enforce_tile_alignment_flag = flag;
    if (!cursor.read_bits(3, value, error)) return false;
    info.global_atlas_id = static_cast<std::uint8_t>(value);
    if (!info.global_atlas_id_present && value != 0U) {
      return fail(error, "lcr_global_reserved_zero_3bits must be 0");
    }
    if (!cursor.read_bits(5, value, error)) return false;
    if (value != 0U) {
      return fail(error, "lcr_global_reserved_zero_5bits must be 0");
    }
    if (info.aggregate_info_present &&
        !read_lcr_aggregate_info(cursor, info.aggregate, error)) {
      return false;
    }
    if (info.profile_tier_level_info_present) {
      for (int x = 0; x <= kMaxExtendedLayerId; ++x) {
        if ((info.xlayer_map & (1U << static_cast<unsigned>(x))) == 0U) {
          continue;
        }
        if (!read_lcr_seq_profile_tier_level_info(
                cursor, info.ptl[static_cast<std::size_t>(x)], error)) {
          return false;
        }
      }
    }
    if (info.global_payload_present) {
      for (int x = 0; x <= kMaxExtendedLayerId; ++x) {
        if ((info.xlayer_map & (1U << static_cast<unsigned>(x))) == 0U) {
          continue;
        }
        std::uint32_t size = 0;
        if (!cursor.read_leb128(size, error)) return false;
        if (size == 0U) {
          return fail(error, "lcr_data_size must not be 0");
        }
        if (!cursor.byte_aligned()) {
          return fail(error, "internal: lcr_data_size is not byte-aligned");
        }
        if (size > cursor.bits_remaining() / 8U) {
          return fail(error, "lcr_data_size exceeds the LCR payload");
        }
        const std::size_t start_byte = cursor.bit_position() / 8U;
        const auto slice =
            cursor.data().subspan(start_byte, static_cast<std::size_t>(size));
        Cursor payload_cursor(slice);
        GlobalPayload entry;
        entry.xlayer_id = static_cast<std::uint8_t>(x);
        if (info.dependent_xlayers_flag && x > 0) {
          std::uint32_t mask = 0;
          if (!payload_cursor.read_bits(static_cast<unsigned>(x), mask,
                                        error)) {
            return false;
          }
          entry.dependent_xlayer_mask = mask;
        }
        if (!read_lcr_xlayer_info(payload_cursor, true,
                                  info.global_atlas_id_present, entry.info,
                                  error)) {
          return false;
        }
        const std::size_t used_bits = payload_cursor.bit_position();
        if (used_bits > static_cast<std::size_t>(size) * 8U) {
          return fail(error, "lcr_data_size is smaller than its syntax");
        }
        entry.declared_size_bytes = size;
        info.payloads.push_back(std::move(entry));
        if (!cursor.skip_bytes(size, error)) return false;
      }
    }
  } else {
    LocalLcrInfo& info = record.local;
    std::uint32_t value = 0;
    if (!cursor.read_bits(3, value, error)) return false;
    info.global_id = static_cast<std::uint8_t>(value);
    if (!cursor.read_bits(3, value, error)) return false;
    info.local_id = static_cast<std::uint8_t>(value);
    bool flag = false;
    if (!cursor.read_bit(flag, error)) return false;
    info.profile_tier_level_info_present = flag;
    if (!cursor.read_bit(flag, error)) return false;
    info.local_atlas_id_present = flag;
    if (info.profile_tier_level_info_present &&
        !read_lcr_seq_profile_tier_level_info(cursor, info.ptl, error)) {
      return false;
    }
    if (!cursor.read_bits(3, value, error)) return false;
    info.local_atlas_id = static_cast<std::uint8_t>(value);
    if (!info.local_atlas_id_present && value != 0U) {
      return fail(error, "lcr_local_reserved_zero_3bits must be 0");
    }
    if (!cursor.read_bits(5, value, error)) return false;
    if (value != 0U) {
      return fail(error, "lcr_local_reserved_zero_5bits must be 0");
    }
    if (!read_lcr_xlayer_info(cursor, false, info.local_atlas_id_present,
                              info.xlayer, error)) {
      return false;
    }
  }
  return read_extension_and_trailing_bits(cursor, error);
}

// ---------------------------------------------------------------------------
// OPS reader
// ---------------------------------------------------------------------------

namespace {

bool read_ops_mlayer_info(Cursor& cursor, OpsLayerMlayerInfo& info,
                          std::string& error) {
  std::uint32_t mlayer_map = 0;
  if (!cursor.read_bits(8, mlayer_map, error)) return false;
  info.mlayer_map = static_cast<std::uint8_t>(mlayer_map);
  if (mlayer_map == 0U) return fail(error, "ops_mlayer_map must be non-empty");
  for (int j = 0; j < kMaxEmbeddedLayers; ++j) {
    if ((mlayer_map & (1U << static_cast<unsigned>(j))) == 0U) continue;
    std::uint32_t tlayer_map = 0;
    if (!cursor.read_bits(4, tlayer_map, error)) return false;
    info.tlayer_map[static_cast<std::size_t>(j)] =
        static_cast<std::uint8_t>(tlayer_map);
  }
  return true;
}

bool read_ops_aggregate_info(Cursor& cursor, OpsAggregateInfo& aggregate,
                             std::string& error) {
  std::uint32_t value = 0;
  if (!cursor.read_bits(6, value, error)) return false;
  aggregate.config_idc = static_cast<std::uint8_t>(value);
  if (!cursor.read_bits(5, value, error)) return false;
  aggregate.level_idx = static_cast<std::uint8_t>(value);
  bool tier = false;
  if (!cursor.read_bit(tier, error)) return false;
  aggregate.max_tier_flag = tier;
  if (!cursor.read_bits(4, value, error)) return false;
  aggregate.max_interop = static_cast<std::uint8_t>(value);
  return true;
}

bool read_ops_seq_profile_tier_level_info(Cursor& cursor,
                                          OpsProfileTierLevelInfo& info,
                                          std::string& error) {
  std::uint32_t value = 0;
  if (!cursor.read_bits(5, value, error)) return false;
  info.seq_profile_idc = static_cast<std::uint8_t>(value);
  if (!cursor.read_bits(5, value, error)) return false;
  info.level_idx = static_cast<std::uint8_t>(value);
  bool tier = false;
  if (!cursor.read_bit(tier, error)) return false;
  info.tier_flag = tier;
  if (!cursor.read_bits(3, value, error)) return false;
  info.mlayer_count = static_cast<std::uint8_t>(value);
  if (!cursor.read_bits(2, value, error)) return false;  // reserved
  return true;
}

bool read_ops_decoder_model_info(Cursor& cursor, OpsDecoderModelInfo& info,
                                 std::string& error) {
  if (!cursor.read_uvlc(info.decoder_buffer_delay, error)) return false;
  if (!cursor.read_uvlc(info.encoder_buffer_delay, error)) return false;
  bool low_delay = false;
  if (!cursor.read_bit(low_delay, error)) return false;
  info.low_delay_mode_flag = low_delay;
  return true;
}

}  // namespace

bool read_ops_payload(std::span<const std::uint8_t> payload,
                      std::uint8_t xlayer_id, OperatingPointSet& set,
                      std::string& error) {
  set = OperatingPointSet{};
  set.xlayer_id = xlayer_id;
  Cursor cursor(payload);
  bool flag = false;
  std::uint32_t value = 0;
  if (!cursor.read_bit(flag, error)) return false;
  set.reset_flag = flag;
  if (!cursor.read_bits(4, value, error)) return false;
  set.ops_id = static_cast<std::uint8_t>(value);
  if (!cursor.read_bits(3, value, error)) return false;
  const unsigned ops_cnt = value;
  if (ops_cnt != 0U) {
    if (!cursor.read_bits(4, value, error)) return false;
    set.priority = static_cast<std::uint8_t>(value);
    if (!cursor.read_bits(7, value, error)) return false;
    set.intent = static_cast<std::uint8_t>(value);
    if (!cursor.read_bit(flag, error)) return false;
    set.intent_present = flag;
    if (!cursor.read_bit(flag, error)) return false;
    set.ptl_present = flag;
    if (!cursor.read_bit(flag, error)) return false;
    set.color_info_present = flag;
    if (!cursor.read_bits(2, value, error)) return false;
    if (global_xlayer(set)) {
      set.mlayer_info_idc = static_cast<std::uint8_t>(value);
     
    } else {
      set.reserved_2bits = static_cast<std::uint8_t>(value);
    }
    for (unsigned i = 0; i < ops_cnt; ++i) {
      std::uint32_t size = 0;
      if (!cursor.read_leb128(size, error)) return false;
      if (size == 0U) return fail(error, "ops_data_size must not be 0");
      if (size > cursor.bits_remaining() / 8U) {
        return fail(error, "ops_data_size exceeds the OPS payload");
      }
      const std::size_t start_byte = cursor.bit_position() / 8U;
      const auto slice =
          cursor.data().subspan(start_byte, static_cast<std::size_t>(size));
      Cursor payload_cursor(slice);
      OpsPoint point;
      point.op_intent_present = set.intent_present;
      if (set.intent_present) {
        if (!payload_cursor.read_bits(7, value, error)) return false;
        point.op_intent = static_cast<std::uint8_t>(value);
      }
      if (set.ptl_present) {
        if (global_xlayer(set)) {
          if (!read_ops_aggregate_info(payload_cursor, point.aggregate,
                                       error)) {
            return false;
          }
        } else if (!read_ops_seq_profile_tier_level_info(
                       payload_cursor,
                       point.ptl[static_cast<std::size_t>(xlayer_id)], error)) {
          return false;
        }
      }
      if (set.color_info_present &&
          !read_color_description(payload_cursor, point.color, error)) {
        return false;
      }
      if (!payload_cursor.read_bit(flag, error)) return false;
      point.decoder_model_info_present = flag;
      if (flag && !read_ops_decoder_model_info(payload_cursor,
                                               point.decoder_model, error)) {
        return false;
      }
      if (!payload_cursor.read_bit(flag, error)) return false;
      point.initial_display_delay_present = flag;
      if (flag) {
        if (!payload_cursor.read_bits(4, value, error)) return false;
        point.initial_display_delay_minus_1 =
            static_cast<std::uint8_t>(value);
      }
      if (global_xlayer(set)) {
        if (!payload_cursor.read_bits(31, value, error)) return false;
        point.xlayer_map = value;
        for (int j = 0; j <= kMaxExtendedLayerId; ++j) {
          if ((point.xlayer_map & (1U << static_cast<unsigned>(j))) == 0U) {
            continue;
          }
          if (set.ptl_present &&
              !read_ops_seq_profile_tier_level_info(
                  payload_cursor, point.ptl[static_cast<std::size_t>(j)],
                  error)) {
            return false;
          }
          OpsLayerMlayerInfo& layer =
              point.layer_info[static_cast<std::size_t>(j)];
          if (set.mlayer_info_idc == 1U) {
            if (!read_ops_mlayer_info(payload_cursor, layer, error)) {
              return false;
            }
          } else if (set.mlayer_info_idc == 2U) {
            bool explicit_info = false;
            if (!payload_cursor.read_bit(explicit_info, error)) return false;
            layer.explicit_info = explicit_info;
            if (explicit_info) {
              if (!read_ops_mlayer_info(payload_cursor, layer, error)) {
                return false;
              }
            } else {
              if (!payload_cursor.read_bits(4, value, error)) return false;
              layer.embedded_ops_id = static_cast<std::uint8_t>(value);
              if (!payload_cursor.read_bits(3, value, error)) return false;
              layer.embedded_op_index = static_cast<std::uint8_t>(value);
            }
          }
        }
      } else {
        if (!read_ops_mlayer_info(
                payload_cursor, point.layer_info[static_cast<std::size_t>(
                                    xlayer_id)],
                error)) {
          return false;
        }
      }
      if (!payload_cursor.align_to_byte(error)) return false;
      if (payload_cursor.bit_position() !=
          static_cast<std::size_t>(size) * 8U) {
        return fail(error,
                    "opsBytes does not equal ops_data_size (§6.10.2)");
      }
      set.points.push_back(std::move(point));
      if (!cursor.skip_bytes(size, error)) return false;
    }
  }
  return read_extension_and_trailing_bits(cursor, error);
}

// ---------------------------------------------------------------------------
// Legacy-compatible construction (svc_framing merge path)
// ---------------------------------------------------------------------------

namespace {

std::vector<const layers::ExtendedLayerConfig*> sorted_legacy_layers(
    const layers::LayerConfigRecord& record) {
  std::vector<const layers::ExtendedLayerConfig*> out;
  out.reserve(record.layers.size());
  for (const layers::ExtendedLayerConfig& layer : record.layers) {
    out.push_back(&layer);
  }
  std::stable_sort(out.begin(), out.end(),
                   [](const layers::ExtendedLayerConfig* a,
                      const layers::ExtendedLayerConfig* b) {
                     return a->xlayer_id < b->xlayer_id;
                   });
  return out;
}

const layers::EmbeddedLayerConfig* find_legacy_embedded(
    const layers::ExtendedLayerConfig& layer, int mlayer_id) {
  for (const layers::EmbeddedLayerConfig& embedded : layer.embedded) {
    if (static_cast<int>(embedded.mlayer_id) == mlayer_id) return &embedded;
  }
  return nullptr;
}

XLayerInfo xlayer_info_from_legacy(const layers::ExtendedLayerConfig& layer,
                                   const layers::LayerDependencyMaps& deps) {
  XLayerInfo info;
  info.rep_info_present = false;
  info.purpose_present = false;
  info.color_info_present = false;
  info.embedded_layer_info_present = true;
  info.mlayer_map = static_cast<std::uint8_t>(layer.mlayer_map);
  for (int m = 0; m < kMaxEmbeddedLayers; ++m) {
    if ((layer.mlayer_map & (1U << static_cast<unsigned>(m))) == 0U) continue;
    EmbeddedLayerInfo embedded;
    embedded.mlayer_id = static_cast<std::uint8_t>(m);
    embedded.tlayer_map = layer.tlayer_map[static_cast<std::size_t>(m)];
    embedded.layer_type = kLayerTexture;
    embedded.view_type = kViewCenter;
    std::uint32_t mask = 0;
    for (int ref = 0; ref < m; ++ref) {
      if (deps.mlayer[static_cast<std::size_t>(m)]
                     [static_cast<std::size_t>(ref)]) {
        mask |= 1U << static_cast<unsigned>(ref);
      }
    }
    embedded.dependent_layer_mask = mask;
    const layers::EmbeddedLayerConfig* legacy =
        find_legacy_embedded(layer, m);
    const bool same =
        legacy == nullptr ? true : legacy->same_sh_max_resolution;
    embedded.same_sh_max_resolution = same;
    if (!same && legacy != nullptr) {
      embedded.max_expected_width = legacy->max_expected_width;
      embedded.max_expected_height = legacy->max_expected_height;
    }
    info.embedded.push_back(embedded);
  }
  return info;
}

}  // namespace

LayerConfigRecord lcr_from_legacy(const layers::LayerConfigRecord& record,
                                 const layers::LayerDependencyMaps& deps,
                                 bool include_full_info) {
  LayerConfigRecord out;
  out.is_global = record.is_global;
  if (!record.is_global) {
    LocalLcrInfo& local = out.local;
    local.xlayer_id = record.xlayer_id;
    local.global_id = 0;
    local.local_id = record.global_config_record_id;
    local.profile_tier_level_info_present = false;
    local.local_atlas_id_present = false;
    if (!record.layers.empty()) {
      local.xlayer = xlayer_info_from_legacy(record.layers.front(), deps);
    }
    if (include_full_info) {
      local.profile_tier_level_info_present = true;
      local.ptl = ProfileTierLevelInfo{};
      local.ptl.max_level_idx = 7;
    }
    return out;
  }

  GlobalLcrInfo& global = out.global;
  global.config_record_id = record.global_config_record_id;
  global.xlayer_map = record.xlayer_map & 0x7FFFFFFFU;
  global.aggregate_info_present = record.aggregate.present;
  global.aggregate.config_idc = record.aggregate.config_idc;
  global.aggregate.level_idx = record.aggregate.level_idx;
  global.aggregate.max_tier_flag = record.aggregate.max_tier_flag;
  global.aggregate.max_interop = record.aggregate.max_interop;
  global.profile_tier_level_info_present = include_full_info;
  global.global_payload_present = true;
  global.dependent_xlayers_flag = false;
  global.global_atlas_id_present = false;
  global.global_purpose_id = 0;
  global.doh_constraint_flag = false;
  global.enforce_tile_alignment_flag = false;
  if (include_full_info) {
    // Conservative defaults; a caller that wants real values must fill them.
    ProfileTierLevelInfo ptl;
    ptl.max_level_idx = 7;
    global.ptl.fill(ptl);
    global.aggregate_info_present = true;
    global.aggregate.level_idx = 7;
    global.global_purpose_id = 1;  // Stereoscopic viewports
  }

  for (const layers::ExtendedLayerConfig* layer : sorted_legacy_layers(record)) {
    global.dependent_xlayers_flag =
        global.dependent_xlayers_flag || layer->dependent_xlayers;
    GlobalPayload payload;
    payload.xlayer_id = layer->xlayer_id;
    std::uint32_t mask = 0;
    for (std::uint8_t dep : layer->dependent_xlayer_ids) {
      if (dep < kMaxExtendedLayerId) mask |= 1U << dep;
    }
    payload.dependent_xlayer_mask = mask;
    payload.info = xlayer_info_from_legacy(*layer, deps);
    if (include_full_info) {
      // The legacy record carries no coded dimensions, so lcr_rep_info()
      // stays absent; a caller that wants it must fill RepInfo itself.
      payload.info.rep_info_present = false;
      payload.info.purpose_present = true;
      payload.info.purpose_id = 1;
      payload.info.color_info_present = true;
      payload.info.color = ColorDescription{};
    }
    global.payloads.push_back(std::move(payload));
  }
  return out;
}

OperatingPointSet ops_from_legacy(const layers::OperatingPointSet& set,
                                  bool include_full_info) {
  OperatingPointSet out;
  out.xlayer_id = set.xlayer_id;
  out.ops_id = set.ops_id;
  out.reset_flag = true;
  out.priority = 0;
  out.intent = 0;
  out.intent_present = false;
  out.ptl_present = false;
  out.color_info_present = false;
  out.mlayer_info_idc = global_xlayer(out) ? 1U : 0U;
  out.reserved_2bits = 0;
  if (include_full_info) {
    out.intent = 1;  // OPSI_SCALABILITY
    out.intent_present = true;
    out.ptl_present = true;
    out.color_info_present = true;
    out.mlayer_info_idc = global_xlayer(out) ? 2U : 0U;
  }
  for (std::size_t i = 0; i < set.points.size(); ++i) {
    const layers::OperatingPoint& legacy = set.points[i];
    OpsPoint point;
    point.op_intent_present = out.intent_present;
    point.op_intent = out.intent;
    point.xlayer_map = legacy.xlayer_map & 0x7FFFFFFFU;
    point.color = ColorDescription{};
    if (out.ptl_present) {
      OpsAggregateInfo aggregate;
      aggregate.level_idx = 7;
      point.aggregate = aggregate;
      OpsProfileTierLevelInfo ptl;
      ptl.level_idx = 7;
      point.ptl.fill(ptl);
    }
    if (include_full_info) {
      point.decoder_model_info_present = (i == 0);
      point.initial_display_delay_present = true;
      point.initial_display_delay_minus_1 =
          static_cast<std::uint8_t>(i > 15 ? 15 : i);
    }
    for (int x = 0; x <= kMaxExtendedLayerId; ++x) {
      OpsLayerMlayerInfo& info = point.layer_info[static_cast<std::size_t>(x)];
      info.explicit_info = true;
      info.mlayer_map =
          static_cast<std::uint8_t>(legacy.mlayer_map[static_cast<std::size_t>(
              x)]);
      for (int m = 0; m < kMaxEmbeddedLayers; ++m) {
        info.tlayer_map[static_cast<std::size_t>(m)] =
            legacy.tlayer_map[static_cast<std::size_t>(x)]
                             [static_cast<std::size_t>(m)];
      }
    }
    out.points.push_back(std::move(point));
  }
  return out;
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

namespace {

const XLayerSequenceInfo* find_sequence(const StreamContext& context, int x) {
  for (const XLayerSequenceInfo& info : context.sequences) {
    if (static_cast<int>(info.xlayer_id) == x) return &info;
  }
  return nullptr;
}

std::string xlayer_name(int x) { return "x" + std::to_string(x); }

bool check_ptl_bounds(const char* what, int x,
                      const XLayerSequenceInfo& sequence,
                      std::uint8_t profile_idc, std::uint8_t level_idx,
                      bool tier_flag, std::uint8_t mlayer_count,
                      ValidationReport& report) {
  if (sequence.seq_profile_idc > profile_idc) {
    std::ostringstream out;
    out << what << ' ' << xlayer_name(x) << ": seq_profile_idc "
        << static_cast<int>(sequence.seq_profile_idc) << " exceeds the limit "
        << static_cast<int>(profile_idc);
    report.add(out.str());
  }
  if (sequence.seq_level_idx > level_idx) {
    std::ostringstream out;
    out << what << ' ' << xlayer_name(x) << ": seq_level_idx "
        << static_cast<int>(sequence.seq_level_idx) << " exceeds the limit "
        << static_cast<int>(level_idx);
    report.add(out.str());
  }
  if (sequence.seq_tier && !tier_flag) {
    std::ostringstream out;
    out << what << ' ' << xlayer_name(x)
        << ": seq_tier is high but the limit is main tier";
    report.add(out.str());
  }
  if (sequence.seq_max_mlayer_cnt > mlayer_count) {
    std::ostringstream out;
    out << what << ' ' << xlayer_name(x) << ": seq_max_mlayer_cnt "
        << static_cast<int>(sequence.seq_max_mlayer_cnt)
        << " exceeds the limit " << static_cast<int>(mlayer_count);
    report.add(out.str());
  }
  return true;
}

void check_embedded_maps(const char* what, int x, std::uint32_t mlayer_map,
                         std::span<const std::uint8_t> tlayer_map,
                         const XLayerSequenceInfo& sequence,
                         const layers::LayerDependencyMaps& dependencies,
                         ValidationReport& report) {
  if ((mlayer_map & ~0xFFU) != 0U) {
    std::ostringstream out;
    out << what << ' ' << xlayer_name(x)
        << ": mlayer_map has bits outside f(8)";
    report.add(out.str());
    return;
  }
  if (popcount32(mlayer_map) > sequence.seq_max_mlayer_cnt) {
    std::ostringstream out;
    out << what << ' ' << xlayer_name(x) << ": mlayer_map selects "
        << popcount32(mlayer_map) << " embedded layers but seq_max_mlayer_cnt "
        << "is " << static_cast<int>(sequence.seq_max_mlayer_cnt);
    report.add(out.str());
  }
  validate_layer_map_closure(static_cast<std::uint8_t>(x), mlayer_map,
                             tlayer_map, dependencies, report);
}

}  // namespace

bool validate_layer_map_closure(std::uint8_t xlayer_id,
                                std::uint32_t mlayer_map,
                                std::span<const std::uint8_t> tlayer_map,
                                const layers::LayerDependencyMaps& dependencies,
                                ValidationReport& report) {
  bool ok = true;
  if ((mlayer_map & ~0xFFU) != 0U) {
    std::ostringstream out;
    out << xlayer_name(xlayer_id) << ": mlayer_map has bits outside f(8)";
    report.add(out.str());
    ok = false;
  }
  for (int c = 0; c < kMaxEmbeddedLayers; ++c) {
    if ((mlayer_map & (1U << static_cast<unsigned>(c))) == 0U) continue;
    for (int r = 0; r < c; ++r) {
      if (!dependencies.mlayer[static_cast<std::size_t>(c)]
                             [static_cast<std::size_t>(r)]) {
        continue;
      }
      if ((mlayer_map & (1U << static_cast<unsigned>(r))) == 0U) {
        std::ostringstream out;
        out << xlayer_name(xlayer_id) << ": m" << c << " depends on m" << r
            << " but mlayer_map omits it (§6.8.9/§6.10.7)";
        report.add(out.str());
        ok = false;
      }
    }
    if (tlayer_map.empty()) continue;
    const std::uint8_t map = tlayer_map[static_cast<std::size_t>(c)];
    if ((map & ~0x0FU) != 0U) {
      std::ostringstream out;
      out << xlayer_name(xlayer_id) << ": tlayer_map[m" << c
          << "] has bits outside f(4)";
      report.add(out.str());
      ok = false;
      continue;
    }
    if (map == 0U) {
      std::ostringstream out;
      out << xlayer_name(xlayer_id) << ": tlayer_map[m" << c
          << "] is empty although m" << c << " is present";
      report.add(out.str());
      ok = false;
      continue;
    }
    for (int ct = 0; ct < kMaxTemporalLayers; ++ct) {
      if ((map & (1U << static_cast<unsigned>(ct))) == 0U) continue;
      for (int rt = 0; rt < ct; ++rt) {
        if (!dependencies.tlayer[static_cast<std::size_t>(c)]
                                [static_cast<std::size_t>(ct)]
                                [static_cast<std::size_t>(rt)]) {
          continue;
        }
        if ((map & (1U << static_cast<unsigned>(rt))) == 0U) {
          std::ostringstream out;
          out << xlayer_name(xlayer_id) << ": m" << c << "/t" << ct
              << " depends on t" << rt
              << " but tlayer_map omits it (§6.8.9/§6.10.7)";
          report.add(out.str());
          ok = false;
        }
      }
    }
  }
  return ok;
}

bool validate_lcr(const LayerConfigRecord& record,
                  const StreamContext& context, ValidationReport& report) {
  if (record.is_global) {
    const GlobalLcrInfo& info = record.global;
    if (info.config_record_id < 1U || info.config_record_id > 7U) {
      report.add("lcr_global_config_record_id must be 1..7 (§6.8.2)");
    }
    if (info.xlayer_map == 0U) {
      report.add("lcr_xlayer_map must be non-zero (§6.8.2)");
    }
    if (info.dependent_xlayers_flag) {
      report.add("lcr_dependent_xlayers_flag must be 0 (§6.8.2)");
    }
    if (context.lcr_active &&
        info.config_record_id != context.active_global_lcr_id) {
      report.add(
          "all extended layers must reference the same activated global LCR "
          "(§6.8.2)");
    }
    if (info.global_atlas_id_present && info.global_atlas_id > 7U) {
      report.add("lcr_global_atlas_id is outside f(3)");
    }

    std::vector<std::uint8_t> described;
    for (std::size_t i = 0; i < info.payloads.size(); ++i) {
      const GlobalPayload& payload = info.payloads[i];
      if ((info.xlayer_map &
           (1U << static_cast<unsigned>(payload.xlayer_id))) == 0U) {
        std::ostringstream out;
        out << "LCR payload " << xlayer_name(payload.xlayer_id)
            << " is not selected by lcr_xlayer_map";
        report.add(out.str());
      }
      if (i > 0 && info.payloads[i - 1].xlayer_id >= payload.xlayer_id) {
        report.add("LCR payloads must be unique and in ascending xlayer order");
      }
      described.push_back(payload.xlayer_id);
    }
    for (int x = 0; x <= kMaxExtendedLayerId; ++x) {
      if ((info.xlayer_map & (1U << static_cast<unsigned>(x))) == 0U) continue;
      if (std::find(described.begin(), described.end(),
                    static_cast<std::uint8_t>(x)) == described.end()) {
        std::ostringstream out;
        out << "lcr_xlayer_map selects " << xlayer_name(x)
            << " but the global LCR describes no payload for it";
        report.add(out.str());
      }
    }

    std::uint8_t max_level = 0;
    bool any_high_tier = false;
    bool have_ptl = false;
    if (info.profile_tier_level_info_present) {
      for (int x = 0; x <= kMaxExtendedLayerId; ++x) {
        if ((info.xlayer_map & (1U << static_cast<unsigned>(x))) == 0U) {
          continue;
        }
        const ProfileTierLevelInfo& ptl =
            info.ptl[static_cast<std::size_t>(x)];
        have_ptl = true;
        max_level = std::max(max_level, ptl.max_level_idx);
        any_high_tier = any_high_tier || ptl.tier_flag;
      }
    }
    if (info.aggregate_info_present && have_ptl) {
      if (info.aggregate.level_idx < max_level) {
        std::ostringstream out;
        out << "lcr_aggregate_level_idx " << static_cast<int>(
                   info.aggregate.level_idx)
            << " is below the per-layer maximum " << static_cast<int>(max_level);
        report.add(out.str());
      }
      if (!info.aggregate.max_tier_flag && any_high_tier) {
        report.add(
            "lcr_max_tier_flag is main tier but a per-layer PTL is high tier");
      }
    }

    bool any_non_monotonic = false;
    for (const GlobalPayload& payload : info.payloads) {
      const int x = payload.xlayer_id;
      const XLayerSequenceInfo* sequence = find_sequence(context, x);
      if (sequence == nullptr) {
        std::ostringstream out;
        out << "LCR payload " << xlayer_name(x)
            << " has no activated sequence header";
        report.add(out.str());
        continue;
      }
      any_non_monotonic =
          any_non_monotonic || !sequence->monotonic_output_order;

      const XLayerInfo& layer = payload.info;
      if (info.profile_tier_level_info_present) {
        const ProfileTierLevelInfo& ptl =
            info.ptl[static_cast<std::size_t>(x)];
        check_ptl_bounds("lcr_seq_profile_tier_level_info", x, *sequence,
                         ptl.seq_profile_idc, ptl.max_level_idx, ptl.tier_flag,
                         ptl.max_mlayer_count, report);
      }
      if (layer.rep_info_present) {
        const RepInfo& rep = layer.rep;
        if (rep.max_pic_width != sequence->max_frame_width ||
            rep.max_pic_height != sequence->max_frame_height) {
          std::ostringstream out;
          out << "lcr_rep_info " << xlayer_name(x) << " is "
              << rep.max_pic_width << 'x' << rep.max_pic_height
              << " but max_frame_width/height is " << sequence->max_frame_width
              << 'x' << sequence->max_frame_height << " (§6.8.8)";
          report.add(out.str());
        }
        if (rep.format_info_present &&
            (rep.bit_depth_idc != sequence->bit_depth_idc ||
             rep.chroma_format_idc != sequence->chroma_format_idc)) {
          std::ostringstream out;
          out << "lcr_rep_info " << xlayer_name(x)
              << ": bit_depth/chroma_format disagree with the sequence header "
                 "(§6.8.8)";
          report.add(out.str());
        }
        if (rep.cropping_window_present !=
            sequence->cropping_window_present) {
          std::ostringstream out;
          out << "lcr_rep_info " << xlayer_name(x)
              << ": lcr_cropping_window_present_flag disagrees with "
                 "seq_cropping_window_present_flag (§6.8.8)";
          report.add(out.str());
        } else if (rep.cropping_window_present &&
                   (rep.crop_left != sequence->crop_left ||
                    rep.crop_right != sequence->crop_right ||
                    rep.crop_top != sequence->crop_top ||
                    rep.crop_bottom != sequence->crop_bottom)) {
          std::ostringstream out;
          out << "lcr_rep_info " << xlayer_name(x)
              << ": cropping offsets disagree with the sequence header (§6.8.8)";
          report.add(out.str());
        }
      }
      if (layer.embedded_layer_info_present) {
        std::vector<std::uint8_t> tlayer_maps;
        tlayer_maps.resize(static_cast<std::size_t>(kMaxEmbeddedLayers), 0U);
        for (const EmbeddedLayerInfo& embedded : layer.embedded) {
          tlayer_maps[static_cast<std::size_t>(embedded.mlayer_id)] =
              embedded.tlayer_map;
          if ((embedded.dependent_layer_mask >>
               embedded.mlayer_id) != 0U) {
            report.add("lcr_dependent_layer_map references a layer >= j");
          }
          if (!embedded.same_sh_max_resolution) {
            if (embedded.max_expected_width > sequence->max_frame_width ||
                embedded.max_expected_height > sequence->max_frame_height) {
              std::ostringstream out;
              out << "lcr_max_expected_width/height for " << xlayer_name(x)
                  << " exceeds the sequence header maximum (§6.8.9)";
              report.add(out.str());
            }
          }
        }
        check_embedded_maps("LCR", x, layer.mlayer_map, tlayer_maps, *sequence,
                            context.dependencies, report);
      }
    }
    if (any_non_monotonic && !info.doh_constraint_flag) {
      report.add(
          "lcr_doh_constraint_flag must be 1 when a sequence header has "
          "monotonic_output_order_flag = 0 (§6.8.2)");
    }
    if (info.profile_tier_level_info_present && !have_ptl) {
      report.add(
          "lcr_seq_profile_tier_level_info_present_flag is 1 but no layer PTL "
          "was supplied");
    }
  } else {
    const LocalLcrInfo& info = record.local;
    if (info.local_id < 1U || info.local_id > 7U) {
      report.add("lcr_local_id must be 1..7 (§6.8.3)");
    }
    if (info.xlayer_id > kMaxExtendedLayerId) {
      report.add("local LCR obu_xlayer_id must be 0..30");
    }
    if (info.global_id != 0U && context.lcr_active &&
        info.global_id != context.active_global_lcr_id) {
      std::ostringstream out;
      out << "lcr_global_id " << static_cast<int>(info.global_id)
          << " does not match the activated global LCR "
          << static_cast<int>(context.active_global_lcr_id);
      report.add(out.str());
    }
    if (info.local_atlas_id_present && info.local_atlas_id > 7U) {
      report.add("lcr_local_atlas_id is outside f(3)");
    }
    const XLayerSequenceInfo* sequence =
        find_sequence(context, info.xlayer_id);
    if (sequence == nullptr) {
      std::ostringstream out;
      out << "local LCR " << xlayer_name(info.xlayer_id)
          << " has no activated sequence header";
      report.add(out.str());
      return report.ok();
    }
    if (info.profile_tier_level_info_present) {
      check_ptl_bounds("lcr_seq_profile_tier_level_info", info.xlayer_id,
                       *sequence, info.ptl.seq_profile_idc,
                       info.ptl.max_level_idx, info.ptl.tier_flag,
                       info.ptl.max_mlayer_count, report);
    }
    const XLayerInfo& layer = info.xlayer;
    if (layer.embedded_layer_info_present) {
      std::vector<std::uint8_t> tlayer_maps(
          static_cast<std::size_t>(kMaxEmbeddedLayers), 0U);
      for (const EmbeddedLayerInfo& embedded : layer.embedded) {
        tlayer_maps[static_cast<std::size_t>(embedded.mlayer_id)] =
            embedded.tlayer_map;
      }
      check_embedded_maps("local LCR", info.xlayer_id, layer.mlayer_map,
                          tlayer_maps, *sequence, context.dependencies,
                          report);
    }
  }
  return report.ok();
}

bool validate_ops(const OperatingPointSet& set, const StreamContext& context,
                  ValidationReport& report) {
  if (set.ops_id > kMaxOpsId) {
    report.add("ops_id is outside f(4)");
  }
  if (set.xlayer_id > kGlobalXLayerId) {
    report.add("OPS obu_xlayer_id is outside 0..31");
  }
  if (set.points.size() > static_cast<std::size_t>(kMaxOpsCnt)) {
    report.add("ops_cnt must be 0..7");
  }
  const bool is_global = global_xlayer(set);
  if (is_global) {
    if (set.mlayer_info_idc == 3U) {
      report.add("ops_mlayer_info_idc must not be 3 (§6.10.2)");
    }
  } else if (set.reserved_2bits != 0U) {
    report.add("ops_reserved_2bits must be 0 (§6.10.2)");
  }
  if (set.points.empty()) return report.ok();

  if (set.priority > 15U) {
    report.add("ops_priority is outside f(4)");
  }
  if (set.intent > 127U) {
    report.add("ops_intent is outside f(7)");
  }
  for (const OpsPoint& point : set.points) {
    if (point.initial_display_delay_present &&
        point.initial_display_delay_minus_1 > 15U) {
      report.add("ops_initial_display_delay_minus_1 is outside f(4)");
      break;
    }
  }

  if (set.color_info_present) {
    for (const OpsPoint& point : set.points) {
      if (point.color.description_idc > 3U) {
        report.add("ops_color_description_idc is outside rg(2)");
        break;
      }
    }
  }

  for (std::size_t i = 0; i < set.points.size(); ++i) {
    const OpsPoint& point = set.points[i];
    std::uint8_t max_level = 0;
    bool any_high_tier = false;
    bool have_ptl = false;

    auto check_layer = [&](int x) {
      const XLayerSequenceInfo* sequence = find_sequence(context, x);
      if (sequence == nullptr) {
        std::ostringstream out;
        out << "operating point " << i << " selects " << xlayer_name(x)
            << " but there is no activated sequence header";
        report.add(out.str());
        return;
      }
      if (set.ptl_present) {
        const OpsProfileTierLevelInfo& ptl =
            point.ptl[static_cast<std::size_t>(x)];
        have_ptl = true;
        max_level = std::max(max_level, ptl.level_idx);
        any_high_tier = any_high_tier || ptl.tier_flag;
        check_ptl_bounds("ops_seq_profile_tier_level_info", x, *sequence,
                         ptl.seq_profile_idc, ptl.level_idx, ptl.tier_flag,
                         ptl.mlayer_count, report);
      }
      if (set.mlayer_info_idc == 1U ||
          (set.mlayer_info_idc == 2U &&
           point.layer_info[static_cast<std::size_t>(x)].explicit_info)) {
        const OpsLayerMlayerInfo& layer =
            point.layer_info[static_cast<std::size_t>(x)];
        std::vector<std::uint8_t> tlayer_maps(
            static_cast<std::size_t>(kMaxEmbeddedLayers), 0U);
        for (int m = 0; m < kMaxEmbeddedLayers; ++m) {
          tlayer_maps[static_cast<std::size_t>(m)] =
              layer.tlayer_map[static_cast<std::size_t>(m)];
        }
        check_embedded_maps("OPS", x, layer.mlayer_map, tlayer_maps, *sequence,
                            context.dependencies, report);
      } else if (set.mlayer_info_idc == 2U) {
        const OpsLayerMlayerInfo& layer =
            point.layer_info[static_cast<std::size_t>(x)];
        const OpsSummary* referenced = nullptr;
        for (const OpsSummary& summary : context.known_ops) {
          if (summary.ops_id != layer.embedded_ops_id) continue;
          if (referenced == nullptr ||
              summary.xlayer_id == layers::kGlobalXLayerId) {
            referenced = &summary;
          }
        }
        if (referenced == nullptr) {
          std::ostringstream out;
          out << "operating point " << i << ' ' << xlayer_name(x)
              << ": ops_embedded_ops_id " << static_cast<int>(
                     layer.embedded_ops_id)
              << " names no OPS present in the bitstream (§6.10.2)";
          report.add(out.str());
        } else if (layer.embedded_op_index >= referenced->ops_cnt) {
          std::ostringstream out;
          out << "operating point " << i << ' ' << xlayer_name(x)
              << ": ops_embedded_op_index "
              << static_cast<int>(layer.embedded_op_index)
              << " is not below ops_cnt " << static_cast<int>(
                     referenced->ops_cnt)
              << " (§6.10.2)";
          report.add(out.str());
        }
        if (layer.embedded_ops_id == set.ops_id &&
            layer.embedded_op_index >= i) {
          std::ostringstream out;
          out << "operating point " << i << ' ' << xlayer_name(x)
              << ": self-referential inheritance must point at an earlier "
                 "operating point (§6.10.2)";
          report.add(out.str());
        }
      }
    };

    if (is_global) {
      for (int x = 0; x <= kMaxExtendedLayerId; ++x) {
        if ((point.xlayer_map & (1U << static_cast<unsigned>(x))) == 0U) {
          continue;
        }
        check_layer(x);
      }
      if (set.ptl_present && have_ptl) {
        if (point.aggregate.level_idx < max_level) {
          std::ostringstream out;
          out << "operating point " << i << ": ops_aggregate_level_idx "
              << static_cast<int>(point.aggregate.level_idx)
              << " is below the per-layer maximum "
              << static_cast<int>(max_level);
          report.add(out.str());
        }
        if (!point.aggregate.max_tier_flag && any_high_tier) {
          std::ostringstream out;
          out << "operating point " << i
              << ": ops_max_tier_flag is main tier but a layer PTL is high "
                 "tier";
          report.add(out.str());
        }
      }
    } else {
      check_layer(set.xlayer_id);
    }
  }
  return report.ok();
}

bool validate_stream(const LayerConfigRecord& record,
                     const OperatingPointSet& set,
                     const StreamContext& context, ValidationReport& report) {
  validate_lcr(record, context, report);
  validate_ops(set, context, report);
  if (record.is_global && global_xlayer(set)) {
    for (std::size_t i = 0; i < set.points.size(); ++i) {
      const std::uint32_t map = set.points[i].xlayer_map;
      if ((map & ~record.global.xlayer_map) != 0U) {
        std::ostringstream out;
        out << "operating point " << i
            << " selects an extended layer that lcr_xlayer_map does not "
               "describe";
        report.add(out.str());
      }
    }
  }
  if (!record.is_global && !global_xlayer(set) &&
      record.local.xlayer_id != set.xlayer_id) {
    report.add(
        "local LCR and local OPS in the same temporal unit must share an "
        "obu_xlayer_id");
  }
  return report.ok();
}

bool validate_local_lcr_matches_global(const LayerConfigRecord& global,
                                       const LayerConfigRecord& local,
                                       ValidationReport& report) {
  if (global.is_global != true || local.is_global != false) {
    report.add(
        "local/global LCR comparison needs a global and a local record");
    return report.ok();
  }
  const int x = local.local.xlayer_id;
  if (local.local.global_id != global.global.config_record_id) {
    std::ostringstream out;
    out << "local LCR " << xlayer_name(x) << ": lcr_global_id "
        << static_cast<int>(local.local.global_id)
        << " does not name the global LCR "
        << static_cast<int>(global.global.config_record_id);
    report.add(out.str());
  }
  const GlobalPayload* payload = nullptr;
  for (const GlobalPayload& candidate : global.global.payloads) {
    if (static_cast<int>(candidate.xlayer_id) == x) payload = &candidate;
  }
  if (payload == nullptr) {
    std::ostringstream out;
    out << "local LCR " << xlayer_name(x)
        << " has no matching payload in the global LCR";
    report.add(out.str());
    return report.ok();
  }
  if (global.global.profile_tier_level_info_present &&
      local.local.profile_tier_level_info_present) {
    const ProfileTierLevelInfo& a =
        global.global.ptl[static_cast<std::size_t>(x)];
    const ProfileTierLevelInfo& b = local.local.ptl;
    if (a.seq_profile_idc != b.seq_profile_idc ||
        a.max_level_idx != b.max_level_idx || a.tier_flag != b.tier_flag ||
        a.max_mlayer_count != b.max_mlayer_count) {
      std::ostringstream out;
      out << "local LCR " << xlayer_name(x)
          << ": lcr_seq_profile_tier_level_info differs from the global LCR";
      report.add(out.str());
    }
  }
  if (global.global.global_atlas_id_present !=
      local.local.local_atlas_id_present) {
    std::ostringstream out;
    out << "local LCR " << xlayer_name(x)
        << ": atlas-id presence differs from the global LCR, so the "
           "embedded-layer syntax cannot agree";
    report.add(out.str());
  }
  const XLayerInfo& a = payload->info;
  const XLayerInfo& b = local.local.xlayer;
  if (a.rep_info_present != b.rep_info_present ||
      a.purpose_present != b.purpose_present ||
      a.color_info_present != b.color_info_present ||
      a.embedded_layer_info_present != b.embedded_layer_info_present) {
    std::ostringstream out;
    out << "local LCR " << xlayer_name(x)
        << ": lcr_xlayer_info presence flags differ from the global LCR";
    report.add(out.str());
  }
  if (a.rep_info_present && b.rep_info_present) {
    if (a.rep.max_pic_width != b.rep.max_pic_width ||
        a.rep.max_pic_height != b.rep.max_pic_height ||
        a.rep.format_info_present != b.rep.format_info_present ||
        a.rep.bit_depth_idc != b.rep.bit_depth_idc ||
        a.rep.chroma_format_idc != b.rep.chroma_format_idc ||
        a.rep.cropping_window_present != b.rep.cropping_window_present ||
        a.rep.crop_left != b.rep.crop_left ||
        a.rep.crop_right != b.rep.crop_right ||
        a.rep.crop_top != b.rep.crop_top ||
        a.rep.crop_bottom != b.rep.crop_bottom) {
      std::ostringstream out;
      out << "local LCR " << xlayer_name(x)
          << ": lcr_rep_info differs from the global LCR";
      report.add(out.str());
    }
  }
  if (a.purpose_present && b.purpose_present &&
      a.purpose_id != b.purpose_id) {
    std::ostringstream out;
    out << "local LCR " << xlayer_name(x)
        << ": lcr_xlayer_purpose_id differs from the global LCR";
    report.add(out.str());
  }
  if (a.color_info_present && b.color_info_present) {
    if (a.color.description_idc != b.color.description_idc ||
        a.color.primaries != b.color.primaries ||
        a.color.transfer != b.color.transfer ||
        a.color.matrix != b.color.matrix ||
        a.color.full_range != b.color.full_range) {
      std::ostringstream out;
      out << "local LCR " << xlayer_name(x)
          << ": lcr_xlayer_color_info differs from the global LCR";
      report.add(out.str());
    }
  }
  if (a.embedded_layer_info_present != b.embedded_layer_info_present) {
    return report.ok();
  }
  if (a.embedded_layer_info_present) {
    if (a.mlayer_map != b.mlayer_map) {
      std::ostringstream out;
      out << "local LCR " << xlayer_name(x) << ": lcr_mlayer_map 0x"
          << std::hex << static_cast<int>(b.mlayer_map)
          << " differs from the global LCR 0x"
          << static_cast<int>(a.mlayer_map) << std::dec;
      report.add(out.str());
      return report.ok();
    }
    for (const EmbeddedLayerInfo& local_layer : b.embedded) {
      const EmbeddedLayerInfo* global_layer = nullptr;
      for (const EmbeddedLayerInfo& candidate : a.embedded) {
        if (candidate.mlayer_id == local_layer.mlayer_id) {
          global_layer = &candidate;
        }
      }
      if (global_layer == nullptr) {
        report.add("local LCR describes an embedded layer absent from the "
                   "global LCR");
        continue;
      }
      if (local_layer.tlayer_map != global_layer->tlayer_map ||
          local_layer.layer_type != global_layer->layer_type ||
          local_layer.auxiliary_type != global_layer->auxiliary_type ||
          local_layer.view_type != global_layer->view_type ||
          local_layer.view_id != global_layer->view_id ||
          local_layer.dependent_layer_mask !=
              global_layer->dependent_layer_mask ||
          local_layer.same_sh_max_resolution !=
              global_layer->same_sh_max_resolution ||
          local_layer.max_expected_width !=
              global_layer->max_expected_width ||
          local_layer.max_expected_height !=
              global_layer->max_expected_height ||
          local_layer.atlas_segment_id != global_layer->atlas_segment_id ||
          local_layer.priority_order != global_layer->priority_order ||
          local_layer.rendering_method != global_layer->rendering_method) {
        std::ostringstream out;
        out << "local LCR " << xlayer_name(x) << ": lcr_embedded_layer_info m"
            << static_cast<int>(local_layer.mlayer_id)
            << " differs from the global LCR";
        report.add(out.str());
      }
    }
  }
  return report.ok();
}

bool validate_brt_against_ops(const BrtCorrespondence& brt,
                              std::span<const OpsSummary> known_ops,
                              ValidationReport& report) {
  if (brt.br_ops_dependent_flag) {
    if (brt.br_ops_id > kMaxOpsId) {
      report.add("br_ops_id is outside f(4)");
      return report.ok();
    }
    if (brt.br_ops_cnt > static_cast<std::uint8_t>(kMaxOpsCnt)) {
      report.add("br_ops_cnt is outside f(3)");
      return report.ok();
    }
    if (brt.br_time_op.size() != brt.br_ops_cnt) {
      std::ostringstream out;
      out << "BRT carries " << brt.br_time_op.size()
          << " br_time_op entries but br_ops_cnt is "
          << static_cast<int>(brt.br_ops_cnt);
      report.add(out.str());
    }
    for (std::size_t i = 0; i < brt.br_time_op.size(); ++i) {
      if (brt.br_time_op[i] > kMaxBrTime) {
        std::ostringstream out;
        out << "br_time_op[" << i << "] is outside rg(4)";
        report.add(out.str());
      }
    }
    const OpsSummary* match = nullptr;
    for (const OpsSummary& summary : known_ops) {
      if (summary.ops_id != brt.br_ops_id) continue;
      if (match == nullptr) {
        match = &summary;
      } else if (match->ops_cnt != summary.ops_cnt) {
        std::ostringstream out;
        out << "br_ops_id " << static_cast<int>(brt.br_ops_id)
            << " is ambiguous: ops_cnt differs between xlayer "
            << static_cast<int>(match->xlayer_id) << " ("
            << static_cast<int>(match->ops_cnt) << ") and xlayer "
            << static_cast<int>(summary.xlayer_id) << " ("
            << static_cast<int>(summary.ops_cnt)
            << "); §6.11 cannot be satisfied unambiguously";
        report.add(out.str());
        return report.ok();
      }
    }
    if (match == nullptr) {
      std::ostringstream out;
      out << "br_ops_id " << static_cast<int>(brt.br_ops_id)
          << " is not an ops_id present in the bitstream (§6.11)";
      report.add(out.str());
    } else if (match->ops_cnt != brt.br_ops_cnt) {
      std::ostringstream out;
      out << "br_ops_cnt " << static_cast<int>(brt.br_ops_cnt)
          << " does not equal ops_cnt " << static_cast<int>(match->ops_cnt)
          << " for ops_id " << static_cast<int>(brt.br_ops_id) << " (§6.11)";
      report.add(out.str());
    }
  } else if (brt.br_time > kMaxBrTime) {
    report.add("br_time is outside rg(4)");
  }
  return report.ok();
}

bool resolve_ops_inheritance(
    const OpsLayerMlayerInfo& inherited,
    std::span<const OpsInheritanceSource> sources,
    OpsLayerMlayerInfo& resolved, std::string& error) {
  if (inherited.explicit_info) {
    resolved = inherited;
    return true;
  }
  for (const OpsInheritanceSource& source : sources) {
    if (source.ops_id != inherited.embedded_ops_id ||
        source.op_index != inherited.embedded_op_index) {
      continue;
    }
    resolved = inherited;
    resolved.explicit_info = true;
    resolved.mlayer_map = source.mlayer_map;
    resolved.tlayer_map = source.tlayer_map;
    return true;
  }
  std::ostringstream out;
  out << "no operating point (ops_id " << static_cast<int>(
             inherited.embedded_ops_id)
      << ", op_index " << static_cast<int>(inherited.embedded_op_index)
      << ") to inherit ops_mlayer_info from";
  return fail(error, out.str());
}

}  // namespace av2enc::svc::ops_full
