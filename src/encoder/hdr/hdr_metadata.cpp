#include "encoder/hdr/hdr_metadata.h"

#include "bitstream/m0_headers.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <sstream>
#include <utility>

namespace av2enc::hdr {
namespace {

constexpr std::uint8_t kTrailingBitsByte = 0x80;
constexpr std::uint32_t kMaxLeb128 = std::numeric_limits<std::uint32_t>::max();

std::vector<std::uint8_t> wrap_obu(std::uint8_t type,
                                   std::span<const std::uint8_t> payload,
                                   std::string& error) {
  // `make_obu` writes 1 header byte, so the size value is 1 + payload.size().
  if (payload.size() > kMaxLeb128 - 1U) {
    error = "metadata OBU payload exceeds the Annex B.2 size field";
    return {};
  }
  m0::ObuHeader header;
  header.type = static_cast<m0::ObuType>(type);
  header.tlayer_id = 0;
  header.extension_flag = false;
  return m0::make_obu(header, payload);
}

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

void append_u16(std::vector<std::uint8_t>& out, std::uint16_t value) {
  out.push_back(static_cast<std::uint8_t>(value >> 8U));
  out.push_back(static_cast<std::uint8_t>(value & 0xFFU));
}

void append_u32(std::vector<std::uint8_t>& out, std::uint32_t value) {
  out.push_back(static_cast<std::uint8_t>(value >> 24U));
  out.push_back(static_cast<std::uint8_t>((value >> 16U) & 0xFFU));
  out.push_back(static_cast<std::uint8_t>((value >> 8U) & 0xFFU));
  out.push_back(static_cast<std::uint8_t>(value & 0xFFU));
}

std::uint16_t read_u16(std::span<const std::uint8_t> data, std::size_t pos) {
  return static_cast<std::uint16_t>(
      (static_cast<std::uint16_t>(data[pos]) << 8U) |
      static_cast<std::uint16_t>(data[pos + 1U]));
}

std::uint32_t read_u32(std::span<const std::uint8_t> data, std::size_t pos) {
  return (static_cast<std::uint32_t>(data[pos]) << 24U) |
         (static_cast<std::uint32_t>(data[pos + 1U]) << 16U) |
         (static_cast<std::uint32_t>(data[pos + 2U]) << 8U) |
         static_cast<std::uint32_t>(data[pos + 3U]);
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

bool check_short_config(const MetadataObuConfig& config, std::string& error) {
  const auto layer = static_cast<unsigned>(config.layer);
  if (layer >= 3U) {
    // §6.16.2: muh_layer_idc < 3 in OBU_METADATA_SHORT.
    error = "OBU_METADATA_SHORT requires muh_layer_idc < 3";
    return false;
  }
  return true;
}

bool check_group_config(const MetadataObuConfig& config, std::string& error) {
  const auto layer = static_cast<unsigned>(config.layer);
  if (layer > 3U) {
    error = "reserved muh_layer_idc";
    return false;
  }
  if (config.layer == MetadataLayer::kValues) {
    error =
        "writer does not emit explicit layer maps (muh_layer_idc == "
        "LAYER_VALUES)";
    return false;
  }
  return true;
}

std::string allowed_mdcv_summary(const ValidationResult& result) {
  std::string summary = result.summary();
  if (!summary.empty() && summary.back() == '\n') {
    summary.pop_back();
  }
  return summary;
}

}  // namespace

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

void ValidationResult::add(Severity severity, std::string message) {
  issues_.push_back(ValidationIssue{severity, std::move(message)});
}

bool ValidationResult::has_errors() const {
  return std::any_of(issues_.begin(), issues_.end(), [](const auto& issue) {
    return issue.severity == Severity::kError;
  });
}

bool ValidationResult::has_warnings() const {
  return std::any_of(issues_.begin(), issues_.end(), [](const auto& issue) {
    return issue.severity == Severity::kWarning;
  });
}

std::string ValidationResult::summary() const {
  std::string out;
  for (const auto& issue : issues_) {
    out += issue.severity == Severity::kError ? "error: " : "warning: ";
    out += issue.message;
    out += '\n';
  }
  return out;
}

ValidationResult validate_cll(const ContentLightLevel&, bool) {
  // §6.16.5 defines no reserved or out-of-range values for max_cll/max_fall.
  // Zero means "no upper bound is signalled"; every other 16-bit value is a
  // valid cd/m^2 bound.
  return {};
}

ValidationResult validate_mdcv(const MasteringDisplayColorVolume& mdcv,
                               bool strict) {
  ValidationResult result;
  const Severity range_severity = strict ? Severity::kError : Severity::kWarning;

  const auto check_x = [&](std::uint16_t value, const std::string& name) {
    if (value < kChromaticityXMin || value > kChromaticityXMax) {
      std::ostringstream message;
      message << name << " = " << value << " is outside [" << kChromaticityXMin
              << ", " << kChromaticityXMax
              << "]; §6.16.6 treats this as unknown/unspecified";
      result.add(range_severity, message.str());
    }
  };
  const auto check_y = [&](std::uint16_t value, const std::string& name) {
    if (value < kChromaticityYMin || value > kChromaticityYMax) {
      std::ostringstream message;
      message << name << " = " << value << " is outside [" << kChromaticityYMin
              << ", " << kChromaticityYMax
              << "]; §6.16.6 treats this as unknown/unspecified";
      result.add(range_severity, message.str());
    }
  };

  for (std::size_t i = 0; i < 3U; ++i) {
    const std::string suffix = " (primary " + std::to_string(i) + ")";
    check_x(mdcv.primary_chromaticity_x[i],
            "primary_chromaticity_x" + suffix);
    check_y(mdcv.primary_chromaticity_y[i],
            "primary_chromaticity_y" + suffix);
  }
  check_x(mdcv.white_point_chromaticity_x, "white_point_chromaticity_x");
  check_y(mdcv.white_point_chromaticity_y, "white_point_chromaticity_y");

  if (mdcv.luminance_max < kLuminanceMaxMin ||
      mdcv.luminance_max > kLuminanceMaxMax) {
    std::ostringstream message;
    message << "luminance_max = " << mdcv.luminance_max << " is outside ["
            << kLuminanceMaxMin << ", " << kLuminanceMaxMax
            << "]; §6.16.6 treats this as unknown/unspecified";
    result.add(range_severity, message.str());
  }
  if (mdcv.luminance_min < kLuminanceMinMin ||
      mdcv.luminance_min > kLuminanceMinMax) {
    std::ostringstream message;
    message << "luminance_min = " << mdcv.luminance_min << " is outside ["
            << kLuminanceMinMin << ", " << kLuminanceMinMax
            << "]; §6.16.6 treats this as unknown/unspecified";
    result.add(range_severity, message.str());
  }

  // §6.16.6: "when luminance_max is equal to 50000, luminance_min shall not be
  // equal to 50000" - a bitstream conformance requirement, always an error.
  if (mdcv.luminance_max == kLuminanceMaxMin &&
      mdcv.luminance_min == kLuminanceMinMax) {
    result.add(Severity::kError,
               "luminance_max == 50000 requires luminance_min != 50000 "
               "(§6.16.6)");
  }
  return result;
}

// ---------------------------------------------------------------------------
// Writers
// ---------------------------------------------------------------------------

std::vector<std::uint8_t> serialize_cll_unit(const ContentLightLevel& cll) {
  std::vector<std::uint8_t> out;
  out.reserve(kCllUnitBytes);
  append_u16(out, cll.max_cll);
  append_u16(out, cll.max_fall);
  return out;
}

std::vector<std::uint8_t> serialize_mdcv_unit(
    const MasteringDisplayColorVolume& mdcv) {
  std::vector<std::uint8_t> out;
  out.reserve(kMdcvUnitBytes);
  for (std::size_t i = 0; i < 3U; ++i) {
    append_u16(out, mdcv.primary_chromaticity_x[i]);
    append_u16(out, mdcv.primary_chromaticity_y[i]);
  }
  append_u16(out, mdcv.white_point_chromaticity_x);
  append_u16(out, mdcv.white_point_chromaticity_y);
  append_u32(out, mdcv.luminance_max);
  append_u32(out, mdcv.luminance_min);
  return out;
}

std::vector<std::uint8_t> make_metadata_short_obu(
    MetadataType type, std::span<const std::uint8_t> metadata_unit,
    const MetadataObuConfig& config, std::string& error) {
  error.clear();
  if (!check_short_config(config, error)) {
    return {};
  }
  if (config.cancel && !metadata_unit.empty()) {
    error = "a cancel OBU cannot carry a metadata_unit payload";
    return {};
  }

  std::vector<std::uint8_t> payload;
  payload.reserve(2U + metadata_unit.size() + 1U);
  const auto options = static_cast<std::uint8_t>(
      (config.suffix ? 0x80U : 0U) |
      (static_cast<unsigned>(config.layer) << 4U) |
      (config.cancel ? 0x08U : 0U) |
      static_cast<unsigned>(config.persistence));
  payload.push_back(options);                        // §5.17.2 header byte
  m0::append_leb128(payload, static_cast<std::uint32_t>(type));
  if (!config.cancel) {
    payload.insert(payload.end(), metadata_unit.begin(), metadata_unit.end());
  }
  payload.push_back(kTrailingBitsByte);
  return wrap_obu(kObuTypeMetadataShort, payload, error);
}

std::vector<std::uint8_t> make_cll_short_obu(const ContentLightLevel& cll,
                                             const MetadataObuConfig& config,
                                             std::string& error) {
  const std::vector<std::uint8_t> unit = serialize_cll_unit(cll);
  return make_metadata_short_obu(MetadataType::kHdrCll, unit, config, error);
}

std::vector<std::uint8_t> make_mdcv_short_obu(
    const MasteringDisplayColorVolume& mdcv, const MetadataObuConfig& config,
    std::string& error) {
  const ValidationResult validation =
      validate_mdcv(mdcv, config.strict_ranges);
  if (validation.has_errors()) {
    error = allowed_mdcv_summary(validation);
    return {};
  }
  const std::vector<std::uint8_t> unit = serialize_mdcv_unit(mdcv);
  return make_metadata_short_obu(MetadataType::kHdrMdcv, unit, config, error);
}

std::vector<std::uint8_t> make_cancel_short_obu(MetadataType type,
                                                const MetadataObuConfig& config,
                                                std::string& error) {
  MetadataObuConfig cancel = config;
  cancel.cancel = true;
  return make_metadata_short_obu(type, {}, cancel, error);
}

std::vector<std::uint8_t> make_metadata_group_obu(
    std::span<const MetadataGroupEntry> entries,
    const MetadataObuConfig& config, std::string& error) {
  error.clear();
  if (!check_group_config(config, error)) {
    return {};
  }
  if (entries.empty()) {
    error = "OBU_METADATA_GROUP needs at least one metadata unit";
    return {};
  }
  if (entries.size() > 16383U) {
    // §6.16.3: metadata_unit_cnt_minus_1 < 16383.
    error = "OBU_METADATA_GROUP unit count exceeds 16383";
    return {};
  }

  std::vector<std::uint8_t> payload;
  const auto options = static_cast<std::uint8_t>(
      (config.suffix ? 0x80U : 0U) |
      (static_cast<unsigned>(config.necessity) << 5U) |
      static_cast<unsigned>(config.application));
  payload.push_back(options);  // §5.17.3 header byte
  m0::append_leb128(payload,
                    static_cast<std::uint32_t>(entries.size() - 1U));

  for (const auto& entry : entries) {
    const auto layer = static_cast<unsigned>(entry.layer);
    if (layer > 3U) {
      error = "reserved muh_layer_idc";
      return {};
    }
    if (entry.layer == MetadataLayer::kValues) {
      error = "writer does not emit explicit layer maps";
      return {};
    }
    if (entry.cancel && !entry.unit.empty()) {
      error = "a cancel metadata unit cannot carry a payload";
      return {};
    }
    if (entry.unit.size() > kMaxLeb128) {
      error = "metadata unit payload exceeds the leb128 size field";
      return {};
    }

    m0::append_leb128(payload, static_cast<std::uint32_t>(entry.type));

    // muh_header_size counts the bytes after the 1-byte header_size/cancel
    // field: leb128(muh_payload_size) + 2 bytes of layer/persistence/priority.
    // For a cancel unit no header bytes follow, so the writer emits 0.
    std::size_t header_size = 0;
    if (!entry.cancel) {
      header_size = 2U + m0::leb128_size(
                             static_cast<std::uint32_t>(entry.unit.size()));
    }
    if (header_size > 127U) {
      error = "metadata unit header exceeds 127 bytes";
      return {};
    }
    payload.push_back(static_cast<std::uint8_t>((header_size << 1U) |
                                                (entry.cancel ? 1U : 0U)));
    if (entry.cancel) {
      continue;
    }

    m0::append_leb128(payload,
                      static_cast<std::uint32_t>(entry.unit.size()));
    payload.push_back(static_cast<std::uint8_t>(
        (layer << 5U) |
        (static_cast<unsigned>(entry.persistence) << 2U) |
        (static_cast<unsigned>(entry.priority) >> 6U)));
    payload.push_back(static_cast<std::uint8_t>(
        (static_cast<unsigned>(entry.priority) & 0x3FU) << 2U));
    payload.insert(payload.end(), entry.unit.begin(), entry.unit.end());
  }

  payload.push_back(kTrailingBitsByte);
  return wrap_obu(kObuTypeMetadataGroup, payload, error);
}

std::vector<std::uint8_t> make_hdr_metadata_obus(
    const HdrMetadata& metadata, HdrObuPacking packing,
    const MetadataObuConfig& config, std::string& error) {
  error.clear();
  if (!metadata.has_cll && !metadata.has_mdcv) {
    error = "no HDR metadata units present";
    return {};
  }
  if (config.cancel) {
    error = "make_hdr_metadata_obus does not emit cancel OBUs";
    return {};
  }

  if (metadata.has_mdcv) {
    const ValidationResult validation =
        validate_mdcv(metadata.mdcv, config.strict_ranges);
    if (validation.has_errors()) {
      error = allowed_mdcv_summary(validation);
      return {};
    }
  }

  std::vector<std::uint8_t> out;
  if (packing == HdrObuPacking::kShortObus) {
    if (metadata.has_cll) {
      const std::vector<std::uint8_t> obu =
          make_cll_short_obu(metadata.cll, config, error);
      if (obu.empty()) {
        return {};
      }
      out.insert(out.end(), obu.begin(), obu.end());
    }
    if (metadata.has_mdcv) {
      const std::vector<std::uint8_t> obu =
          make_mdcv_short_obu(metadata.mdcv, config, error);
      if (obu.empty()) {
        return {};
      }
      out.insert(out.end(), obu.begin(), obu.end());
    }
    return out;
  }

  std::vector<MetadataGroupEntry> entries;
  if (metadata.has_cll) {
    entries.push_back(MetadataGroupEntry{MetadataType::kHdrCll,
                                         serialize_cll_unit(metadata.cll),
                                         config.layer, config.persistence, 0,
                                         false});
  }
  if (metadata.has_mdcv) {
    entries.push_back(MetadataGroupEntry{MetadataType::kHdrMdcv,
                                         serialize_mdcv_unit(metadata.mdcv),
                                         config.layer, config.persistence, 0,
                                         false});
  }
  return make_metadata_group_obu(entries, config, error);
}

std::vector<std::uint8_t> make_legacy_av1_metadata_obu(
    MetadataType type, std::span<const std::uint8_t> unit,
    std::string& error) {
  error.clear();
  std::vector<std::uint8_t> payload;
  payload.reserve(unit.size() + 2U);
  m0::append_leb128(payload, static_cast<std::uint32_t>(type));
  payload.insert(payload.end(), unit.begin(), unit.end());
  payload.push_back(kTrailingBitsByte);
  return wrap_obu(kObuTypeMetadataShort, payload, error);
}

std::vector<std::uint8_t> make_legacy_av1_hdr_metadata_obus(
    const HdrMetadata& metadata, std::string& error) {
  error.clear();
  if (!metadata.has_cll && !metadata.has_mdcv) {
    error = "no HDR metadata units present";
    return {};
  }
  std::vector<std::uint8_t> out;
  if (metadata.has_cll) {
    const std::vector<std::uint8_t> unit = serialize_cll_unit(metadata.cll);
    const std::vector<std::uint8_t> obu =
        make_legacy_av1_metadata_obu(MetadataType::kHdrCll, unit, error);
    if (obu.empty()) {
      return {};
    }
    out.insert(out.end(), obu.begin(), obu.end());
  }
  if (metadata.has_mdcv) {
    const std::vector<std::uint8_t> unit = serialize_mdcv_unit(metadata.mdcv);
    const std::vector<std::uint8_t> obu =
        make_legacy_av1_metadata_obu(MetadataType::kHdrMdcv, unit, error);
    if (obu.empty()) {
      return {};
    }
    out.insert(out.end(), obu.begin(), obu.end());
  }
  return out;
}

// ---------------------------------------------------------------------------
// Readers
// ---------------------------------------------------------------------------

bool parse_obu(std::span<const std::uint8_t> data, std::size_t& offset,
               ObuView& out, std::string& error) {
  error.clear();
  const std::size_t start = offset;
  std::uint32_t size = 0;
  if (!read_uleb128(data, offset, size, error)) {
    return false;
  }
  if (size == 0U) {
    error = "OBU size is zero";
    return false;
  }

  const std::size_t header_start = offset;
  if (offset >= data.size()) {
    error = "truncated OBU header";
    return false;
  }
  const std::uint8_t header_byte = data[offset++];
  const bool extension_flag = (header_byte & 0x80U) != 0U;
  const std::uint8_t type = static_cast<std::uint8_t>((header_byte >> 2U) & 0x1FU);
  const std::uint8_t tlayer_id =
      static_cast<std::uint8_t>(header_byte & 0x03U);

  std::uint8_t mlayer_id = 0;
  std::uint8_t xlayer_id = 0;
  if (extension_flag) {
    if (offset >= data.size()) {
      error = "truncated OBU header extension";
      return false;
    }
    const std::uint8_t extension = data[offset++];
    mlayer_id = static_cast<std::uint8_t>((extension >> 5U) & 0x07U);
    xlayer_id = static_cast<std::uint8_t>(extension & 0x1FU);
  }

  const std::size_t header_bytes = offset - header_start;
  if (size < header_bytes) {
    error = "OBU size is smaller than its header";
    return false;
  }
  // `size` counts the OBU header plus payload from `header_start`.
  if (static_cast<std::size_t>(size) > data.size() - header_start) {
    error = "OBU size exceeds the available bytes";
    return false;
  }

  out = ObuView{};
  out.type = type;
  out.tlayer_id = tlayer_id;
  out.extension_flag = extension_flag;
  out.mlayer_id = mlayer_id;
  out.xlayer_id = xlayer_id;
  out.offset = start;
  out.header_bytes = offset - start;
  out.size = static_cast<std::size_t>(size) + (header_start - start);
  out.payload = data.subspan(offset, static_cast<std::size_t>(size) -
                                         header_bytes);
  offset = start + out.size;
  return true;
}

bool parse_obu_sequence(std::span<const std::uint8_t> data,
                        std::vector<ObuView>& out, std::string& error) {
  error.clear();
  out.clear();
  std::size_t offset = 0;
  while (offset < data.size()) {
    ObuView obu;
    if (!parse_obu(data, offset, obu, error)) {
      return false;
    }
    out.push_back(obu);
  }
  return true;
}

bool parse_metadata_short(std::span<const std::uint8_t> payload,
                          MetadataShortView& out, std::string& error) {
  error.clear();
  out = MetadataShortView{};
  if (payload.empty()) {
    error = "OBU_METADATA_SHORT payload is empty";
    return false;
  }

  const std::uint8_t options = payload[0];
  out.is_suffix = (options & 0x80U) != 0U;
  out.layer = static_cast<MetadataLayer>((options >> 4U) & 0x07U);
  out.cancel = (options & 0x08U) != 0U;
  out.persistence = static_cast<MetadataPersistence>(options & 0x07U);

  if (static_cast<unsigned>(out.layer) >= 3U) {
    error = "OBU_METADATA_SHORT requires muh_layer_idc < 3 (§6.16.2)";
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

  // metadataPayloadSize = obuPayloadSize - 2 - Leb128Bytes (§5.17.2), i.e. the
  // final byte is always trailing_bits.
  if (pos >= payload.size()) {
    error = "OBU_METADATA_SHORT is missing trailing_bits";
    return false;
  }
  if (payload.back() != kTrailingBitsByte) {
    error = "OBU_METADATA_SHORT trailing_bits must be 0x80";
    return false;
  }
  out.unit = payload.subspan(pos, payload.size() - 1U - pos);
  return true;
}

bool parse_metadata_group(std::span<const std::uint8_t> payload,
                          std::uint8_t xlayer_id, MetadataGroupView& out,
                          std::string& error) {
  error.clear();
  out = MetadataGroupView{};
  if (payload.size() < 2U) {
    error = "OBU_METADATA_GROUP payload is too short";
    return false;
  }

  const std::uint8_t options = payload[0];
  out.is_suffix = (options & 0x80U) != 0U;
  out.necessity = static_cast<MetadataNecessity>((options >> 5U) & 0x03U);
  out.application = static_cast<MetadataApplicationId>(options & 0x1FU);

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

  // Each unit needs at least a metadata_type leb128 byte and a header byte, so
  // never reserve more entries than the payload can possibly hold.
  out.units.reserve(
      std::min(static_cast<std::size_t>(count), payload.size() / 2U));
  for (std::uint32_t index = 0; index < count; ++index) {
    MetadataUnitView unit;
    if (!read_uleb128(payload, pos, unit.type, error)) {
      return false;
    }
    if (pos >= payload.size()) {
      error = "truncated metadata unit header";
      return false;
    }
    const std::uint8_t header_byte = payload[pos++];
    std::size_t header_remaining = header_byte >> 1U;
    unit.cancel = (header_byte & 0x01U) != 0U;

    if (unit.cancel) {
      if (header_remaining > payload.size() - pos) {
        error = "metadata unit header exceeds the OBU payload";
        return false;
      }
      pos += header_remaining;  // muh_header_extension_byte
      out.units.push_back(unit);
      continue;
    }

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

    unit.layer = static_cast<MetadataLayer>((layer_byte >> 5U) & 0x07U);
    unit.persistence =
        static_cast<MetadataPersistence>((layer_byte >> 2U) & 0x07U);
    unit.priority = static_cast<std::uint8_t>(
        ((static_cast<unsigned>(layer_byte) & 0x03U) << 6U) |
        (static_cast<unsigned>(priority_byte) >> 2U));
    if (static_cast<unsigned>(unit.layer) > 3U) {
      error = "reserved muh_layer_idc";
      return false;
    }
    if ((priority_byte & 0x03U) != 0U) {
      error = "muh_reserved_zero_2bits must be zero (§6.16.3)";
      return false;
    }

    if (unit.layer == MetadataLayer::kValues) {
      if (xlayer_id == kGlobalXlayerId) {
        if (header_remaining < 4U || pos + 4U > payload.size()) {
          error = "truncated muh_xlayer_map";
          return false;
        }
        const std::uint32_t xlayer_map = read_u32(payload, pos);
        pos += 4U;
        header_remaining -= 4U;
        if ((xlayer_map & 0x80000000U) != 0U) {
          error = "bit 31 of muh_xlayer_map must be zero (§6.16.3)";
          return false;
        }
        for (unsigned bit = 0; bit < 31U; ++bit) {
          if ((xlayer_map & (1U << bit)) == 0U) {
            continue;
          }
          if (header_remaining < 1U || pos >= payload.size()) {
            error = "truncated muh_mlayer_map";
            return false;
          }
          ++pos;
          --header_remaining;
        }
      } else {
        if (header_remaining < 1U || pos >= payload.size()) {
          error = "truncated muh_mlayer_map";
          return false;
        }
        ++pos;
        --header_remaining;
      }
    }

    if (header_remaining > payload.size() - pos) {
      error = "metadata unit header exceeds the OBU payload";
      return false;
    }
    pos += header_remaining;
    if (unit_size > payload.size() - pos) {
      error = "metadata unit payload exceeds the OBU payload";
      return false;
    }
    unit.unit = payload.subspan(pos, unit_size);
    pos += unit_size;
    out.units.push_back(unit);
  }

  if (!check_trailing_bits(payload, pos, error)) {
    return false;
  }
  return true;
}

bool decode_cll(std::span<const std::uint8_t> unit, ContentLightLevel& out,
                std::string& error) {
  error.clear();
  if (unit.size() < kCllUnitBytes) {
    error = "metadata_hdr_cll payload is shorter than 4 bytes";
    return false;
  }
  out.max_cll = read_u16(unit, 0);
  out.max_fall = read_u16(unit, 2);
  return true;
}

bool decode_mdcv(std::span<const std::uint8_t> unit,
                 MasteringDisplayColorVolume& out, std::string& error) {
  error.clear();
  if (unit.size() < kMdcvUnitBytes) {
    error = "metadata_hdr_mdcv payload is shorter than 24 bytes";
    return false;
  }
  std::size_t pos = 0;
  for (std::size_t i = 0; i < 3U; ++i) {
    out.primary_chromaticity_x[i] = read_u16(unit, pos);
    pos += 2U;
    out.primary_chromaticity_y[i] = read_u16(unit, pos);
    pos += 2U;
  }
  out.white_point_chromaticity_x = read_u16(unit, pos);
  pos += 2U;
  out.white_point_chromaticity_y = read_u16(unit, pos);
  pos += 2U;
  out.luminance_max = read_u32(unit, pos);
  pos += 4U;
  out.luminance_min = read_u32(unit, pos);
  return true;
}

bool is_frame_obu_type(std::uint8_t type) {
  switch (type) {
    case 4:   // OBU_CLOSED_LOOP_KEY
    case 5:   // OBU_OPEN_LOOP_KEY
    case 6:   // OBU_LEADING_TILE_GROUP
    case 7:   // OBU_REGULAR_TILE_GROUP
    case 10:  // OBU_SWITCH
    case 11:  // OBU_LEADING_SEF
    case 12:  // OBU_REGULAR_SEF
    case 13:  // OBU_LEADING_TIP
    case 14:  // OBU_REGULAR_TIP
    case 19:  // OBU_BRIDGE_FRAME
    case 21:  // OBU_RAS_FRAME
      return true;
    default:
      return false;
  }
}

bool scan_hdr_metadata(std::span<const std::uint8_t> temporal_unit,
                       HdrScanResult& out, std::string& error) {
  error.clear();
  out = HdrScanResult{};

  const auto record_unit = [&](std::uint32_t type, bool cancel,
                               std::span<const std::uint8_t> unit,
                               bool in_group, bool is_suffix,
                               std::size_t obu_offset, bool before_frame) {
    if (cancel) {
      return;
    }
    if (type == static_cast<std::uint32_t>(MetadataType::kHdrCll)) {
      ContentLightLevel cll;
      if (!decode_cll(unit, cll, error)) {
        ++out.invalid_units;
        error.clear();
        return;
      }
      HdrMetadataRecord record;
      record.obu_offset = obu_offset;
      record.in_group = in_group;
      record.is_suffix = is_suffix;
      record.before_first_frame = before_frame;
      record.type = MetadataType::kHdrCll;
      record.cll = cll;
      out.records.push_back(record);
      return;
    }
    if (type == static_cast<std::uint32_t>(MetadataType::kHdrMdcv)) {
      MasteringDisplayColorVolume mdcv;
      if (!decode_mdcv(unit, mdcv, error)) {
        ++out.invalid_units;
        error.clear();
        return;
      }
      HdrMetadataRecord record;
      record.obu_offset = obu_offset;
      record.in_group = in_group;
      record.is_suffix = is_suffix;
      record.before_first_frame = before_frame;
      record.type = MetadataType::kHdrMdcv;
      record.mdcv = mdcv;
      out.records.push_back(record);
      return;
    }
    ++out.unknown_units;
  };

  std::size_t offset = 0;
  bool seen_frame = false;
  while (offset < temporal_unit.size()) {
    ObuView obu;
    if (!parse_obu(temporal_unit, offset, obu, error)) {
      return false;
    }
    if (is_frame_obu_type(obu.type)) {
      seen_frame = true;
    }

    if (obu.type == kObuTypeMetadataShort) {
      MetadataShortView view;
      if (!parse_metadata_short(obu.payload, view, error)) {
        return false;
      }
      ++out.metadata_obus;
      ++out.metadata_units;
      record_unit(view.type, view.cancel, view.unit, false, view.is_suffix,
                  obu.offset, !seen_frame);
    } else if (obu.type == kObuTypeMetadataGroup) {
      MetadataGroupView view;
      if (!parse_metadata_group(obu.payload, obu.xlayer_id, view, error)) {
        return false;
      }
      ++out.metadata_obus;
      for (const auto& unit : view.units) {
        ++out.metadata_units;
        record_unit(unit.type, unit.cancel, unit.unit, true, view.is_suffix,
                    obu.offset, !seen_frame);
      }
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Ordering
// ---------------------------------------------------------------------------

bool insert_metadata_obus(std::span<const std::uint8_t> temporal_unit,
                          std::span<const std::uint8_t> metadata_obus,
                          MetadataPlacement placement,
                          std::vector<std::uint8_t>& out, std::string& error,
                          std::size_t* inserted_at) {
  error.clear();
  out.clear();
  if (metadata_obus.empty()) {
    error = "no metadata OBUs to insert";
    return false;
  }

  std::vector<ObuView> metadata;
  if (!parse_obu_sequence(metadata_obus, metadata, error)) {
    return false;
  }
  const bool want_suffix = placement == MetadataPlacement::kEndOfTemporalUnit;
  for (const auto& obu : metadata) {
    if (!obu.is_metadata()) {
      error = "only OBU_METADATA_SHORT/GROUP may be inserted";
      return false;
    }
    bool is_suffix = false;
    if (obu.type == kObuTypeMetadataShort) {
      MetadataShortView view;
      if (!parse_metadata_short(obu.payload, view, error)) {
        return false;
      }
      is_suffix = view.is_suffix;
    } else {
      MetadataGroupView view;
      if (!parse_metadata_group(obu.payload, obu.xlayer_id, view, error)) {
        return false;
      }
      is_suffix = view.is_suffix;
    }
    if (is_suffix != want_suffix) {
      error = want_suffix
                  ? "suffix placement requires metadata_is_suffix == 1"
                  : "prefix placement requires metadata_is_suffix == 0";
      return false;
    }
  }

  std::vector<ObuView> obus;
  if (!parse_obu_sequence(temporal_unit, obus, error)) {
    return false;
  }

  std::size_t insert_at = temporal_unit.size();
  if (placement == MetadataPlacement::kBeforeFirstFrame) {
    const auto frame = std::find_if(
        obus.begin(), obus.end(),
        [](const ObuView& obu) { return is_frame_obu_type(obu.type); });
    if (frame == obus.end()) {
      error = "prefix metadata requires a frame OBU";
      return false;
    }
    const bool sequence_before =
        std::any_of(obus.begin(), frame, [](const ObuView& obu) {
          return obu.type == 1U;  // OBU_SEQUENCE_HEADER
        });
    if (!sequence_before) {
      error = "prefix metadata requires a sequence header before the frame";
      return false;
    }
    insert_at = frame->offset;
  } else {
    // §7.3.3: suffix metadata follows the coded frame, so a frame must exist.
    const bool has_frame =
        std::any_of(obus.begin(), obus.end(), [](const ObuView& obu) {
          return is_frame_obu_type(obu.type);
        });
    if (!has_frame) {
      error = "suffix metadata requires a frame OBU";
      return false;
    }
  }

  out.reserve(temporal_unit.size() + metadata_obus.size());
  out.insert(out.end(), temporal_unit.begin(),
             temporal_unit.begin() + static_cast<std::ptrdiff_t>(insert_at));
  out.insert(out.end(), metadata_obus.begin(), metadata_obus.end());
  out.insert(out.end(),
             temporal_unit.begin() + static_cast<std::ptrdiff_t>(insert_at),
             temporal_unit.end());
  if (inserted_at != nullptr) {
    *inserted_at = insert_at;
  }
  return true;
}

}  // namespace av2enc::hdr
