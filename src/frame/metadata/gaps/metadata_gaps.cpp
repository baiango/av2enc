#include "frame/metadata/gaps/metadata_gaps.h"

#include "core/bitstream/basic_headers.h"

#include <algorithm>
#include <array>
#include <limits>
#include <utility>

namespace av2enc::metadata::gaps {
namespace {

constexpr std::uint8_t kTrailingBitsByte = 0x80;
constexpr std::uint32_t kMaxLeb128 = std::numeric_limits<std::uint32_t>::max();

std::string size_text(std::size_t value) { return std::to_string(value); }

bool read_uleb128(std::span<const std::uint8_t> data, std::size_t& pos,
                  std::uint32_t& value, std::string& error) {
  std::uint64_t result = 0;
  unsigned shift = 0;
  for (unsigned index = 0; index < 5U; ++index) {
    if (pos >= data.size()) {
      error = "truncated leb128";
      return false;
    }
    const std::uint8_t byte = data[pos++];
    result |= static_cast<std::uint64_t>(byte & 0x7FU) << shift;
    if ((byte & 0x80U) == 0U) {
      if (index == 4U && (byte & 0x7FU) > 0x0FU) {
        error = "leb128 value exceeds 32 bits";
        return false;
      }
      value = static_cast<std::uint32_t>(result);
      return true;
    }
    shift += 7U;
  }
  error = "leb128 longer than 5 bytes";
  return false;
}

bool check_trailing_bits(std::span<const std::uint8_t> data, std::size_t pos,
                         std::string& error) {
  if (pos >= data.size()) {
    error = "missing trailing_bits byte";
    return false;
  }
  if (data[pos] != kTrailingBitsByte) {
    error = "trailing_bits must start with a single 1 bit followed by zeros";
    return false;
  }
  for (std::size_t i = pos + 1U; i < data.size(); ++i) {
    if (data[i] != 0U) {
      error = "trailing_bits contains non-zero padding";
      return false;
    }
  }
  return true;
}

std::uint32_t popcount31(std::uint32_t value) {
  value &= 0x7FFFFFFFU;
  std::uint32_t count = 0;
  while (value != 0U) {
    value &= value - 1U;
    ++count;
  }
  return count;
}

// §6.16.3: bit m of every muh_mlayer_map shall be zero for m < obu_mlayer_id.
bool check_mlayer_map_bits(std::span<const std::uint8_t> maps,
                           std::uint8_t mlayer_id, std::string& error) {
  if (mlayer_id == 0U) {
    return true;
  }
  if (mlayer_id > 7U) {
    error = "obu_mlayer_id must fit in 3 bits (§5.2.2)";
    return false;
  }
  const unsigned mask = (1U << mlayer_id) - 1U;
  for (const std::uint8_t map : maps) {
    if ((static_cast<unsigned>(map) & mask) != 0U) {
      error =
          "muh_mlayer_map sets a bit below obu_mlayer_id, which §6.16.3 "
          "requires to be zero";
      return false;
    }
  }
  return true;
}

// §7.3.3 / §5.17.1: a non-cancel unit must be a known type at its canonical
// payload size. Cancel units carry no body.
bool check_unit_payload(const MetadataGapUnit& unit, std::string& error) {
  if (!is_known_type(unit.header.type)) {
    error = "reserved metadata_type " +
            size_text(static_cast<std::uint32_t>(unit.header.type)) +
            " is not emitted";
    return false;
  }
  if (unit.header.cancel) {
    if (!unit.payload.empty()) {
      error = "a cancel metadata unit cannot carry a payload (muh_cancel_flag "
              "== 1, §5.17.3)";
      return false;
    }
    return true;
  }
  Value value;
  if (!decode_unit(unit.header.type, unit.payload, value, error)) {
    error = std::string("invalid ") + type_name(unit.header.type) +
            " metadata_unit: " + error;
    return false;
  }
  const std::size_t canonical = canonical_unit_size(value);
  if (unit.payload.size() != canonical) {
    error = std::string(type_name(unit.header.type)) + " metadata_unit is " +
            size_text(unit.payload.size()) + " bytes; canonical size is " +
            size_text(canonical);
    return false;
  }
  return true;
}

// Returns the number of layer-map bytes, or fails when the LAYER_VALUES
// description does not match the OBU's xlayer/mlayer configuration.
bool layer_map_bytes(const MetadataUnitHeader& header,
                     const GapObuConfig& config, std::size_t& out_bytes,
                     std::string& error) {
  out_bytes = 0;
  if (header.layer != hdr::MetadataLayer::kValues) {
    if (header.layer_values.xlayer_map != 0U ||
        !header.layer_values.mlayer_map.empty()) {
      error =
          "muh_xlayer_map/muh_mlayer_map are only coded when muh_layer_idc == "
          "LAYER_VALUES (§5.17.3)";
      return false;
    }
    return true;
  }

  const auto& maps = header.layer_values;
  if (config.xlayer_id == kGlobalXlayerId) {
    if ((maps.xlayer_map & 0x80000000U) != 0U) {
      error = "bit 31 of muh_xlayer_map must be zero (§6.16.3)";
      return false;
    }
    const std::size_t expected = popcount31(maps.xlayer_map);
    if (maps.mlayer_map.size() != expected) {
      error = "muh_mlayer_map has " +
              size_text(maps.mlayer_map.size()) + " bytes; GLOBAL_XLAYER_ID " +
              "requires one per set bit of muh_xlayer_map (" +
              size_text(expected) + ")";
      return false;
    }
    out_bytes = 4U + expected;
  } else {
    if (maps.xlayer_map != 0U) {
      error =
          "muh_xlayer_map is only coded when obu_xlayer_id == "
          "GLOBAL_XLAYER_ID; a non-global xlayer uses a single muh_mlayer_map";
      return false;
    }
    if (maps.mlayer_map.size() != 1U) {
      error = "LAYER_VALUES with obu_xlayer_id < GLOBAL_XLAYER_ID requires "
              "exactly one muh_mlayer_map byte";
      return false;
    }
    out_bytes = 1U;
  }
  return check_mlayer_map_bits(maps.mlayer_map, config.mlayer_id, error);
}

bool validate_config(const GapObuConfig& config, bool group, std::string& error) {
  if (config.tlayer_id > 3U) {
    error = "obu_tlayer_id must fit in 2 bits (§5.2.2)";
    return false;
  }
  if (config.mlayer_id > 7U) {
    error = "obu_mlayer_id must fit in 3 bits (§5.2.2)";
    return false;
  }
  if (config.xlayer_id > kGlobalXlayerId) {
    error = "obu_xlayer_id must fit in 5 bits (§5.2.2)";
    return false;
  }
  if (!group) {
    return true;
  }
  if (static_cast<unsigned>(config.necessity) > 3U) {
    error = "metadata_necessity_idc must fit in 2 bits (§5.17.3)";
    return false;
  }
  const auto application = static_cast<unsigned>(config.application);
  if (application > 31U) {
    error = "metadata_application_id must fit in 5 bits (§5.17.3)";
    return false;
  }
  if (application >= 6U && application <= 15U) {
    error = "metadata_application_id 6..15 is reserved for AOMedia (§6.16.3)";
    return false;
  }
  return true;
}

bool validate_common_unit(const MetadataGapUnit& unit, const GapObuConfig& config,
                          bool group, std::size_t& map_bytes,
                          std::size_t& header_size, std::string& error) {
  const auto layer = static_cast<unsigned>(unit.header.layer);
  if (group ? layer > 3U : layer >= 3U) {
    error = group ? "reserved muh_layer_idc"
                  : "OBU_METADATA_SHORT requires muh_layer_idc < 3 (§6.16.2)";
    return false;
  }
  if (static_cast<unsigned>(unit.header.persistence) > 3U) {
    error = "muh_persistence_idc 4..7 is reserved (§6.16.3)";
    return false;
  }
  if (!check_unit_payload(unit, error)) {
    return false;
  }

  if (!group) {
    if (!unit.header.layer_values.mlayer_map.empty() ||
        unit.header.layer_values.xlayer_map != 0U) {
      error =
          "OBU_METADATA_SHORT has no muh_xlayer_map/muh_mlayer_map or "
          "muh_header_extension_byte (§5.17.2); use OBU_METADATA_GROUP";
      return false;
    }
    if (!unit.header.header_extension.empty()) {
      error =
          "OBU_METADATA_SHORT has no muh_header_extension_byte (§5.17.2); "
          "use OBU_METADATA_GROUP";
      return false;
    }
    map_bytes = 0;
    header_size = 0;
    return true;
  }

  if (unit.header.cancel) {
    // §5.17.3: muh_layer_idc/muh_persistence_idc/muh_priority and the layer
    // maps are all inside if (!muh_cancel_flag), so a cancel unit codes only
    // the type, the cancel bit and any muh_header_extension_byte. Those fields
    // are not coded here; the reader reports their defaults. Callers that need
    // to preserve them must not use the cancel form.
    map_bytes = 0;
    header_size = unit.header.header_extension.size();
  } else {
    if (!layer_map_bytes(unit.header, config, map_bytes, error)) {
      return false;
    }
    if (unit.payload.size() > kMaxLeb128) {
      error = "metadata unit payload exceeds the leb128 size field";
      return false;
    }
    header_size = av2enc::basic::leb128_size(
                      static_cast<std::uint32_t>(unit.payload.size())) +
                  2U + map_bytes + unit.header.header_extension.size();
  }
  if (header_size > kMaxMetadataHeaderBytes) {
    error = "muh_header_size is " + size_text(header_size) +
            " bytes; the f(7) field caps it at 127 (§5.17.3)";
    return false;
  }
  return true;
}

std::vector<std::uint8_t> wrap_obu(std::uint8_t type,
                                   const GapObuConfig& config,
                                   std::span<const std::uint8_t> payload,
                                   std::string& error) {
  if (payload.size() > static_cast<std::size_t>(kMaxLeb128) - 2U) {
    error = "metadata OBU payload exceeds the Annex B.2 size field";
    return {};
  }
  av2enc::basic::ObuHeader header;
  header.type = static_cast<av2enc::basic::ObuType>(type);
  header.tlayer_id = config.tlayer_id;
  header.extension_flag = config.mlayer_id != 0U || config.xlayer_id != 0U;
  header.mlayer_id = config.mlayer_id;
  header.xlayer_id = config.xlayer_id;
  return av2enc::basic::make_obu(header, payload);
}

void append_u32_be(std::vector<std::uint8_t>& out, std::uint32_t value) {
  out.push_back(static_cast<std::uint8_t>((value >> 24U) & 0xFFU));
  out.push_back(static_cast<std::uint8_t>((value >> 16U) & 0xFFU));
  out.push_back(static_cast<std::uint8_t>((value >> 8U) & 0xFFU));
  out.push_back(static_cast<std::uint8_t>(value & 0xFFU));
}

std::uint32_t read_u32_be(std::span<const std::uint8_t> data, std::size_t pos) {
  return (static_cast<std::uint32_t>(data[pos]) << 24U) |
         (static_cast<std::uint32_t>(data[pos + 1U]) << 16U) |
         (static_cast<std::uint32_t>(data[pos + 2U]) << 8U) |
         static_cast<std::uint32_t>(data[pos + 3U]);
}

void append_layer_maps(std::vector<std::uint8_t>& out,
                       const MetadataUnitHeader& header,
                       const GapObuConfig& config) {
  if (header.layer != hdr::MetadataLayer::kValues) {
    return;
  }
  if (config.xlayer_id == kGlobalXlayerId) {
    append_u32_be(out, header.layer_values.xlayer_map);
  }
  out.insert(out.end(), header.layer_values.mlayer_map.begin(),
             header.layer_values.mlayer_map.end());
}

// Parsed OBU header fields needed by the group reader. Defined locally instead
// of using GapObuConfig so the reader does not depend on writer-only knobs.
struct GapObuConfigView {
  std::uint8_t mlayer_id = 0;
  std::uint8_t xlayer_id = 0;
};

bool parse_layer_maps(std::span<const std::uint8_t> payload, std::size_t& pos,
                      std::size_t& header_remaining,
                      const GapObuConfigView& view, MetadataUnitHeader& header,
                      std::string& error);

}  // namespace

// ---------------------------------------------------------------------------
// Writers
// ---------------------------------------------------------------------------

std::vector<std::uint8_t> make_short_obu(const MetadataGapUnit& unit,
                                         const GapObuConfig& config,
                                         std::string& error) {
  error.clear();
  if (!validate_config(config, /*group=*/false, error)) {
    return {};
  }
  std::size_t map_bytes = 0;
  std::size_t header_size = 0;
  if (!validate_common_unit(unit, config, /*group=*/false, map_bytes,
                            header_size, error)) {
    return {};
  }

  std::vector<std::uint8_t> payload;
  payload.reserve(2U + unit.payload.size() + 1U);
  const auto options = static_cast<std::uint8_t>(
      (config.suffix ? 0x80U : 0U) |
      (static_cast<unsigned>(unit.header.layer) << 4U) |
      (unit.header.cancel ? 0x08U : 0U) |
      static_cast<unsigned>(unit.header.persistence));
  payload.push_back(options);
  av2enc::basic::append_leb128(payload,
                    static_cast<std::uint32_t>(unit.header.type));
  if (!unit.header.cancel) {
    payload.insert(payload.end(), unit.payload.begin(), unit.payload.end());
  }
  payload.push_back(kTrailingBitsByte);
  return wrap_obu(kObuTypeMetadataShort, config, payload, error);
}

std::vector<std::uint8_t> make_short_cancel_obu(Type type,
                                                const GapObuConfig& config,
                                                std::string& error) {
  MetadataGapUnit unit;
  unit.header.type = type;
  unit.header.cancel = true;
  return make_short_obu(unit, config, error);
}

std::vector<std::uint8_t> make_group_obu(std::span<const MetadataGapUnit> units,
                                         const GapObuConfig& config,
                                         std::string& error) {
  error.clear();
  if (!validate_config(config, /*group=*/true, error)) {
    return {};
  }
  if (units.empty()) {
    error = "OBU_METADATA_GROUP needs at least one metadata unit";
    return {};
  }
  if (units.size() > 16383U) {
    error = "OBU_METADATA_GROUP unit count exceeds 16383 (§6.16.3)";
    return {};
  }

  // Validate everything before writing a single byte.
  std::vector<std::size_t> header_sizes(units.size(), 0);
  for (std::size_t i = 0; i < units.size(); ++i) {
    std::size_t map_bytes = 0;
    if (!validate_common_unit(units[i], config, /*group=*/true, map_bytes,
                              header_sizes[i], error)) {
      return {};
    }
  }

  std::vector<std::uint8_t> payload;
  const auto options = static_cast<std::uint8_t>(
      (config.suffix ? 0x80U : 0U) |
      (static_cast<unsigned>(config.necessity) << 5U) |
      static_cast<unsigned>(config.application));
  payload.push_back(options);
  av2enc::basic::append_leb128(payload,
                    static_cast<std::uint32_t>(units.size() - 1U));

  for (std::size_t i = 0; i < units.size(); ++i) {
    const MetadataGapUnit& unit = units[i];
    av2enc::basic::append_leb128(payload,
                      static_cast<std::uint32_t>(unit.header.type));
    payload.push_back(static_cast<std::uint8_t>(
        (header_sizes[i] << 1U) | (unit.header.cancel ? 1U : 0U)));
    if (unit.header.cancel) {
      // §5.17.3: muh_cancel_flag == 1 still carries the header (and thus any
      // extension bytes) but no metadata_unit body.
      payload.insert(payload.end(), unit.header.header_extension.begin(),
                     unit.header.header_extension.end());
      continue;
    }
    const auto priority = static_cast<unsigned>(unit.header.priority);
    av2enc::basic::append_leb128(payload,
                      static_cast<std::uint32_t>(unit.payload.size()));
    payload.push_back(static_cast<std::uint8_t>(
        (static_cast<unsigned>(unit.header.layer) << 5U) |
        (static_cast<unsigned>(unit.header.persistence) << 2U) |
        (priority >> 6U)));
    payload.push_back(static_cast<std::uint8_t>((priority & 0x3FU) << 2U));
    append_layer_maps(payload, unit.header, config);
    payload.insert(payload.end(), unit.header.header_extension.begin(),
                   unit.header.header_extension.end());
    payload.insert(payload.end(), unit.payload.begin(), unit.payload.end());
  }

  payload.push_back(kTrailingBitsByte);
  return wrap_obu(kObuTypeMetadataGroup, config, payload, error);
}

// ---------------------------------------------------------------------------
// Readers
// ---------------------------------------------------------------------------

namespace {

bool parse_short_from_payload(std::span<const std::uint8_t> payload,
                              ParsedShortObu& out, std::string& error) {
  out = ParsedShortObu{};
  if (payload.empty()) {
    error = "OBU_METADATA_SHORT payload is empty";
    return false;
  }
  const std::uint8_t options = payload[0];
  out.is_suffix = (options & 0x80U) != 0U;
  out.layer = static_cast<hdr::MetadataLayer>((options >> 4U) & 0x07U);
  out.cancel = (options & 0x08U) != 0U;
  out.persistence = static_cast<hdr::MetadataPersistence>(options & 0x07U);
  if (static_cast<unsigned>(out.layer) >= 3U) {
    error = "OBU_METADATA_SHORT requires muh_layer_idc < 3 (§6.16.2)";
    return false;
  }
  if (static_cast<unsigned>(out.persistence) > 3U) {
    error = "muh_persistence_idc 4..7 is reserved (§6.16.3)";
    return false;
  }

  std::size_t pos = 1;
  if (!read_uleb128(payload, pos, out.type, error)) {
    return false;
  }
  if (out.cancel) {
    if (!check_trailing_bits(payload, pos, error)) {
      return false;
    }
    return true;
  }
  if (pos >= payload.size()) {
    error = "OBU_METADATA_SHORT is missing trailing_bits";
    return false;
  }
  if (payload.back() != kTrailingBitsByte) {
    error = "OBU_METADATA_SHORT trailing_bits must be 0x80";
    return false;
  }
  out.payload = payload.subspan(pos, payload.size() - 1U - pos);
  return true;
}

bool parse_layer_maps(std::span<const std::uint8_t> payload, std::size_t& pos,
                      std::size_t& header_remaining,
                      const GapObuConfigView& view, MetadataUnitHeader& header,
                      std::string& error) {
  if (header.layer != hdr::MetadataLayer::kValues) {
    return true;
  }
  if (view.xlayer_id == kGlobalXlayerId) {
    if (header_remaining < 4U || pos + 4U > payload.size()) {
      error = "truncated muh_xlayer_map";
      return false;
    }
    header.layer_values.xlayer_map = read_u32_be(payload, pos);
    pos += 4U;
    header_remaining -= 4U;
    if ((header.layer_values.xlayer_map & 0x80000000U) != 0U) {
      error = "bit 31 of muh_xlayer_map must be zero (§6.16.3)";
      return false;
    }
    const std::size_t count = popcount31(header.layer_values.xlayer_map);
    header.layer_values.mlayer_map.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      if (header_remaining < 1U || pos >= payload.size()) {
        error = "truncated muh_mlayer_map";
        return false;
      }
      header.layer_values.mlayer_map.push_back(payload[pos++]);
      --header_remaining;
    }
  } else {
    if (header_remaining < 1U || pos >= payload.size()) {
      error = "truncated muh_mlayer_map";
      return false;
    }
    header.layer_values.mlayer_map.push_back(payload[pos++]);
    --header_remaining;
  }
  return check_mlayer_map_bits(header.layer_values.mlayer_map,
                               view.mlayer_id, error);
}

bool parse_group_from_payload(std::span<const std::uint8_t> payload,
                              const GapObuConfigView& view,
                              ParsedGroupObu& out, std::string& error) {
  out = ParsedGroupObu{};
  out.mlayer_id = view.mlayer_id;
  out.xlayer_id = view.xlayer_id;
  if (payload.size() < 2U) {
    error = "OBU_METADATA_GROUP payload is too short";
    return false;
  }

  const std::uint8_t options = payload[0];
  out.is_suffix = (options & 0x80U) != 0U;
  out.necessity = static_cast<hdr::MetadataNecessity>((options >> 5U) & 0x03U);
  out.application = static_cast<hdr::MetadataApplicationId>(options & 0x1FU);
  const auto application = static_cast<unsigned>(out.application);
  if (application >= 6U && application <= 15U) {
    error = "metadata_application_id 6..15 is reserved for AOMedia (§6.16.3)";
    return false;
  }

  std::size_t pos = 1;
  std::uint32_t count_minus_1 = 0;
  if (!read_uleb128(payload, pos, count_minus_1, error)) {
    return false;
  }
  if (count_minus_1 >= 16383U) {
    error = "metadata_unit_cnt_minus_1 must be < 16383 (§6.16.3)";
    return false;
  }
  const std::uint32_t count = count_minus_1 + 1U;
  out.units.reserve(
      std::min(static_cast<std::size_t>(count), payload.size() / 2U));

  for (std::uint32_t index = 0; index < count; ++index) {
    ParsedUnit unit;
    std::uint32_t raw_type = 0;
    if (!read_uleb128(payload, pos, raw_type, error)) {
      return false;
    }
    unit.header.type = static_cast<Type>(raw_type);
    if (pos >= payload.size()) {
      error = "truncated metadata unit header";
      return false;
    }
    const std::uint8_t header_byte = payload[pos++];
    std::size_t header_remaining = header_byte >> 1U;
    unit.header.cancel = (header_byte & 0x01U) != 0U;

    if (!unit.header.cancel) {
      std::uint32_t unit_size = 0;
      const std::size_t leb_start = pos;
      if (!read_uleb128(payload, pos, unit_size, error)) {
        return false;
      }
      const std::size_t leb_bytes = pos - leb_start;
      if (header_remaining < leb_bytes + 2U) {
        error = "muh_header_size is smaller than its fixed fields";
        return false;
      }
      header_remaining -= leb_bytes;
      if (pos + 2U > payload.size()) {
        error = "truncated metadata unit layer fields";
        return false;
      }
      const std::uint8_t layer_byte = payload[pos++];
      const std::uint8_t priority_byte = payload[pos++];
      header_remaining -= 2U;

      unit.header.layer =
          static_cast<hdr::MetadataLayer>((layer_byte >> 5U) & 0x07U);
      unit.header.persistence =
          static_cast<hdr::MetadataPersistence>((layer_byte >> 2U) & 0x07U);
      unit.header.priority = static_cast<std::uint8_t>(
          ((static_cast<unsigned>(layer_byte) & 0x03U) << 6U) |
          (static_cast<unsigned>(priority_byte) >> 2U));
      if (static_cast<unsigned>(unit.header.layer) > 3U) {
        error = "reserved muh_layer_idc";
        return false;
      }
      if (static_cast<unsigned>(unit.header.persistence) > 3U) {
        error = "muh_persistence_idc 4..7 is reserved (§6.16.3)";
        return false;
      }
      if ((priority_byte & 0x03U) != 0U) {
        error = "muh_reserved_zero_2bits must be zero (§6.16.3)";
        return false;
      }
      if (!parse_layer_maps(payload, pos, header_remaining, view,
                            unit.header, error)) {
        return false;
      }
      if (header_remaining > payload.size() - pos) {
        error = "metadata unit header exceeds the OBU payload";
        return false;
      }
      unit.header.header_extension.assign(
          payload.begin() + static_cast<std::ptrdiff_t>(pos),
          payload.begin() + static_cast<std::ptrdiff_t>(pos + header_remaining));
      pos += header_remaining;
      if (unit_size > payload.size() - pos) {
        error = "metadata unit payload exceeds the OBU payload";
        return false;
      }
      unit.payload = payload.subspan(pos, unit_size);
      pos += unit_size;
    } else {
      // Cancel unit: no payload size/layer fields; only extension bytes can
      // follow (§5.17.3, headerRemainingBytes loop is outside !cancel). The
      // non-coded header fields are reported at their defaults.
      unit.header.layer = hdr::MetadataLayer::kUnspecified;
      unit.header.persistence = hdr::MetadataPersistence::kGlobal;
      unit.header.priority = 0;
      unit.header.layer_values = LayerValueMaps{};
      if (header_remaining > payload.size() - pos) {
        error = "metadata unit header exceeds the OBU payload";
        return false;
      }
      unit.header.header_extension.assign(
          payload.begin() + static_cast<std::ptrdiff_t>(pos),
          payload.begin() + static_cast<std::ptrdiff_t>(pos + header_remaining));
      pos += header_remaining;
    }
    out.units.push_back(std::move(unit));
  }

  if (!check_trailing_bits(payload, pos, error)) {
    return false;
  }
  return true;
}

bool take_obu(std::span<const std::uint8_t> obu, hdr::ObuView& view,
              std::string& error) {
  std::size_t offset = 0;
  if (!hdr::parse_obu(obu, offset, view, error)) {
    return false;
  }
  if (offset != obu.size()) {
    error = "OBU buffer contains trailing bytes after the OBU";
    return false;
  }
  return true;
}

}  // namespace

bool parse_short_obu(std::span<const std::uint8_t> obu, ParsedShortObu& out,
                     std::string& error) {
  error.clear();
  hdr::ObuView view;
  if (!take_obu(obu, view, error)) {
    return false;
  }
  if (view.type != kObuTypeMetadataShort) {
    error = "expected OBU_METADATA_SHORT (type 8)";
    return false;
  }
  if (!parse_short_from_payload(view.payload, out, error)) {
    return false;
  }
  out.obu_bytes = view.size;
  return true;
}

bool parse_group_obu(std::span<const std::uint8_t> obu, ParsedGroupObu& out,
                     std::string& error) {
  error.clear();
  hdr::ObuView view;
  if (!take_obu(obu, view, error)) {
    return false;
  }
  if (view.type != kObuTypeMetadataGroup) {
    error = "expected OBU_METADATA_GROUP (type 9)";
    return false;
  }
  GapObuConfigView config;
  config.mlayer_id = view.mlayer_id;
  config.xlayer_id = view.xlayer_id;
  if (!parse_group_from_payload(view.payload, config, out, error)) {
    return false;
  }
  out.tlayer_id = view.tlayer_id;
  out.obu_bytes = view.size;
  return true;
}

// ---------------------------------------------------------------------------
// Temporal-unit scanning
// ---------------------------------------------------------------------------

bool scan_gap_metadata(std::span<const std::uint8_t> temporal_unit,
                       GapScanResult& out, std::string& error) {
  error.clear();
  out = GapScanResult{};

  const auto push_record = [&out](ParsedUnit* unit, bool in_group,
                                  bool is_suffix, std::size_t obu_offset,
                                  bool before_frame, hdr::MetadataLayer layer,
                                  hdr::MetadataPersistence persistence,
                                  bool cancel) {
    GapRecord record;
    record.obu_offset = obu_offset;
    record.in_group = in_group;
    record.is_suffix = is_suffix;
    record.before_first_frame = before_frame;
    if (unit != nullptr) {
      record.type = static_cast<std::uint32_t>(unit->header.type);
      record.layer = unit->header.layer;
      record.persistence = unit->header.persistence;
      record.priority = unit->header.priority;
      record.xlayer_map = unit->header.layer_values.xlayer_map;
      record.mlayer_map = unit->header.layer_values.mlayer_map;
      record.header_extension = unit->header.header_extension;
    } else {
      record.type = 0;
      record.layer = layer;
      record.persistence = persistence;
    }
    record.cancel = cancel;
    out.records.push_back(std::move(record));
  };

  std::size_t offset = 0;
  bool seen_frame = false;
  while (offset < temporal_unit.size()) {
    hdr::ObuView obu;
    if (!hdr::parse_obu(temporal_unit, offset, obu, error)) {
      return false;
    }
    if (hdr::is_frame_obu_type(obu.type)) {
      seen_frame = true;
    }
    if (obu.type == kObuTypeMetadataShort) {
      ParsedShortObu view;
      if (!parse_short_from_payload(obu.payload, view, error)) {
        return false;
      }
      ++out.metadata_obus;
      ++out.metadata_units;
      GapRecord record;
      record.obu_offset = obu.offset;
      record.in_group = false;
      record.is_suffix = view.is_suffix;
      record.before_first_frame = !seen_frame;
      record.type = view.type;
      record.layer = view.layer;
      record.persistence = view.persistence;
      record.cancel = view.cancel;
      if (view.cancel) {
        ++out.cancelled_units;
        out.records.push_back(std::move(record));
        continue;
      }
      if (!is_known_type(view.type)) {
        ++out.unsupported_units;
        continue;
      }
      Value value;
      if (!decode_unit(static_cast<Type>(view.type), view.payload, value,
                       error)) {
        ++out.invalid_units;
        error.clear();
        continue;
      }
      ++out.supported_units;
      record.has_value = true;
      record.value = std::move(value);
      out.records.push_back(std::move(record));
      continue;
    }

    if (obu.type != kObuTypeMetadataGroup) {
      continue;
    }
    ParsedGroupObu view;
    GapObuConfigView config;
    config.mlayer_id = obu.mlayer_id;
    config.xlayer_id = obu.xlayer_id;
    if (!parse_group_from_payload(obu.payload, config, view, error)) {
      return false;
    }
    ++out.metadata_obus;
    for (auto& unit : view.units) {
      ++out.metadata_units;
      if (unit.header.cancel) {
        ++out.cancelled_units;
        push_record(&unit, true, view.is_suffix, obu.offset, !seen_frame,
                    unit.header.layer, unit.header.persistence, true);
        continue;
      }
      if (static_cast<std::uint32_t>(unit.header.type) ==
          static_cast<std::uint32_t>(Type::kTemporalPointInfo)) {
        // §6.16.11: only allowed in OBU_METADATA_SHORT.
        ++out.invalid_units;
        continue;
      }
      if (!is_known_type(unit.header.type)) {
        ++out.unsupported_units;
        continue;
      }
      Value value;
      if (!decode_unit(static_cast<Type>(unit.header.type), unit.payload, value,
                       error)) {
        ++out.invalid_units;
        error.clear();
        continue;
      }
      ++out.supported_units;
      push_record(&unit, true, view.is_suffix, obu.offset, !seen_frame,
                  unit.header.layer, unit.header.persistence, false);
      out.records.back().has_value = true;
      out.records.back().value = std::move(value);
    }
  }
  return true;
}

}  // namespace av2enc::metadata::gaps
