#include "frame/metadata/types/metadata_types.h"

#include "core/bitstream/bitwriter.h"
#include "core/bitstream/basic_headers.h"

#include <algorithm>
#include <sstream>
#include <type_traits>
#include <utility>

namespace av2enc::metadata {
namespace {

std::string strip_trailing_newline(std::string text) {
  if (!text.empty() && text.back() == '\n') {
    text.pop_back();
  }
  return text;
}

std::string summary_of(const hdr::ValidationResult& result) {
  return strip_trailing_newline(result.summary());
}

std::string size_text(std::size_t value) { return std::to_string(value); }

// MSB-first bit reader for the non-byte-aligned metadata_unit() payloads.
class BitReader {
 public:
  explicit BitReader(std::span<const std::uint8_t> data) : data_(data) {}

  bool get(unsigned num_bits, std::uint32_t& value) {
    if (num_bits > 32U) {
      return false;
    }
    if (bit_pos_ + num_bits > data_.size() * 8U) {
      return false;
    }
    std::uint32_t result = 0;
    for (unsigned i = 0; i < num_bits; ++i) {
      const std::size_t bit = bit_pos_ + i;
      const auto byte = data_[bit / 8U];
      const unsigned shift = 7U - static_cast<unsigned>(bit & 7U);
      result = (result << 1U) |
               (static_cast<std::uint32_t>((byte >> shift) & 0x01U));
    }
    bit_pos_ += num_bits;
    value = result;
    return true;
  }

  std::size_t bits_remaining() const {
    return data_.size() * 8U - bit_pos_;
  }

 private:
  std::span<const std::uint8_t> data_;
  std::size_t bit_pos_ = 0;
};

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

std::size_t cer_bytes(std::size_t bits) { return (bits + 7U) / 8U; }

void add_range_issue(hdr::ValidationResult& result, bool strict,
                     std::string message) {
  result.add(strict ? hdr::Severity::kError : hdr::Severity::kWarning,
             std::move(message));
}

bool validate_short_config(const hdr::MetadataObuConfig& config,
                           std::string& error) {
  const auto layer = static_cast<unsigned>(config.layer);
  if (layer >= 3U) {
    error = "OBU_METADATA_SHORT requires muh_layer_idc < 3 (§6.16.2)";
    return false;
  }
  if (static_cast<unsigned>(config.persistence) > 3U) {
    error = "muh_persistence_idc 4..7 is reserved (§6.16.3)";
    return false;
  }
  if (config.cancel) {
    error = "use make_short_cancel_obu() for muh_cancel_flag == 1";
    return false;
  }
  return true;
}

bool validate_group_config(const hdr::MetadataObuConfig& config,
                           std::string& error) {
  if (config.cancel) {
    error =
        "OBU_METADATA_GROUP carries per-unit cancel flags; config.cancel is "
        "not used";
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

bool check_unit_payload(const MetadataUnitPayload& unit, bool strict,
                        std::string& error) {
  if (!is_known_type(unit.type)) {
    error = "reserved metadata_type " + size_text(unit.type == Type::kReserved
                                                     ? 0U
                                                     : static_cast<unsigned>(
                                                           unit.type));
    return false;
  }
  Value value;
  if (!decode_unit(unit.type, unit.bytes, value, error)) {
    error = std::string("invalid ") + type_name(unit.type) +
            " metadata_unit: " + error;
    return false;
  }
  const std::size_t canonical = canonical_unit_size(value);
  if (unit.bytes.size() != canonical) {
    error = std::string(type_name(unit.type)) + " metadata_unit is " +
            size_text(unit.bytes.size()) + " bytes; canonical size is " +
            size_text(canonical);
    return false;
  }
  hdr::ValidationResult validation = std::visit(
      [strict](const auto& item) -> hdr::ValidationResult {
        using T = std::decay_t<decltype(item)>;
        if constexpr (std::is_same_v<T, hdr::ContentLightLevel>) {
          return hdr::validate_cll(item, strict);
        } else if constexpr (std::is_same_v<T,
                                            hdr::MasteringDisplayColorVolume>) {
          return hdr::validate_mdcv(item, strict);
        } else if constexpr (std::is_same_v<T, ItutT35>) {
          return validate_itut_t35(item, strict);
        } else if constexpr (std::is_same_v<T, Timecode>) {
          return validate_timecode(item, strict);
        } else if constexpr (std::is_same_v<T, BandingHints>) {
          return validate_banding_hints(item, strict);
        } else if constexpr (std::is_same_v<T, IccProfile>) {
          return validate_icc_profile(item, strict);
        } else if constexpr (std::is_same_v<T, ScanType>) {
          return validate_scan_type(item, strict);
        } else if constexpr (std::is_same_v<T, TemporalPointInfo>) {
          return validate_temporal_point_info(item, strict);
        } else if constexpr (std::is_same_v<T, DecodedFrameHash>) {
          return validate_decoded_frame_hash(item, strict);
        } else {
          static_assert(std::is_same_v<T, UserDataUnregistered>);
          return validate_user_data_unregistered(item, strict);
        }
      },
      value);
  if (validation.has_errors()) {
    error = summary_of(validation);
    return false;
  }
  return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

bool is_known_type(std::uint32_t type) {
  return type >= static_cast<std::uint32_t>(Type::kHdrCll) &&
         type <= static_cast<std::uint32_t>(Type::kUserDataUnregistered);
}

bool is_known_type(Type type) {
  return is_known_type(static_cast<std::uint32_t>(type));
}

bool is_group_allowed(Type type) {
  return type != Type::kTemporalPointInfo;
}

const char* type_name(std::uint32_t type) {
  switch (type) {
    case static_cast<std::uint32_t>(Type::kHdrCll):
      return "hdr_cll";
    case static_cast<std::uint32_t>(Type::kHdrMdcv):
      return "hdr_mdcv";
    case static_cast<std::uint32_t>(Type::kItutT35):
      return "itut_t35";
    case static_cast<std::uint32_t>(Type::kTimecode):
      return "timecode";
    case static_cast<std::uint32_t>(Type::kDecodedFrameHash):
      return "decoded_frame_hash";
    case static_cast<std::uint32_t>(Type::kBandingHints):
      return "banding_hints";
    case static_cast<std::uint32_t>(Type::kIccProfile):
      return "icc_profile";
    case static_cast<std::uint32_t>(Type::kScanType):
      return "scan_type";
    case static_cast<std::uint32_t>(Type::kTemporalPointInfo):
      return "temporal_point_info";
    case static_cast<std::uint32_t>(Type::kUserDataUnregistered):
      return "user_data_unregistered";
    default:
      return "reserved";
  }
}

const char* type_name(Type type) {
  return type_name(static_cast<std::uint32_t>(type));
}

std::optional<IccProfileVersion> icc_profile_version(
    const IccProfile& profile) {
  if (profile.data.size() < 10U) {
    return std::nullopt;
  }
  IccProfileVersion version;
  version.major = profile.data[8];
  version.minor = static_cast<std::uint8_t>(profile.data[9] >> 4U);
  return version;
}

std::optional<std::uint8_t> scan_type_required_ci_scan_type_idc(
    std::uint8_t mps_pic_struct_type) {
  switch (mps_pic_struct_type) {
    case 0:
    case 7:
    case 8:
      return 1;
    case 1:
    case 2:
    case 9:
    case 10:
    case 11:
    case 12:
      return 2;
    case 3:
    case 4:
    case 5:
    case 6:
      return 3;
    default:
      return std::nullopt;
  }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

hdr::ValidationResult validate_itut_t35(const ItutT35& value, bool) {
  hdr::ValidationResult result;
  if (value.payload.empty()) {
    // §6.16.4: the terminal provider code and terminal provider oriented code
    // "shall be contained in the first one or more bytes".
    result.add(hdr::Severity::kError,
               "itu_t_t35_payload_bytes is empty; §6.16.4 requires at least "
               "the terminal provider code");
  }
  if (value.country_code != kItutT35ExtensionCountryCode &&
      value.country_code_extension_byte != 0U) {
    result.add(hdr::Severity::kError,
               "itu_t_t35_country_code_extension_byte is only coded when "
               "itu_t_t35_country_code == 0xFF (§5.17.4)");
  }
  return result;
}

hdr::ValidationResult validate_timecode(const Timecode& value, bool strict) {
  hdr::ValidationResult result;
  if (value.counting_type > 6U) {
    add_range_issue(result, strict,
                    "counting_type = " +
                        size_text(value.counting_type) +
                        " is reserved (7..31, §6.16.7)");
  }
  if (value.n_frames > 511U) {
    result.add(hdr::Severity::kError,
               "n_frames = " + size_text(value.n_frames) +
                   " does not fit in 9 bits (§5.17.7)");
  }
  if (value.time_offset_length > 31U) {
    result.add(hdr::Severity::kError,
               "time_offset_length = " + size_text(value.time_offset_length) +
                   " does not fit in 5 bits (§5.17.7)");
  } else if (value.time_offset_length == 0U) {
    if (value.time_offset_value != 0U) {
      result.add(hdr::Severity::kError,
                 "time_offset_value is not coded when time_offset_length == 0");
    }
  } else {
    const std::uint32_t limit =
        value.time_offset_length == 31U
            ? 0x7FFFFFFFU
            : ((1U << value.time_offset_length) - 1U);
    if (value.time_offset_value > limit) {
      result.add(hdr::Severity::kError,
                 "time_offset_value = " +
                     size_text(value.time_offset_value) + " does not fit in " +
                     size_text(value.time_offset_length) + " bits");
    }
  }

  const auto check_range = [&result](std::uint8_t value_, std::uint8_t max,
                                     const char* name) {
    if (value_ > max) {
      result.add(hdr::Severity::kError,
                 std::string(name) + " = " + size_text(value_) +
                     " is outside [0, " + size_text(max) + "] (§6.16.7)");
    }
  };

  if (value.full_timestamp_flag) {
    check_range(value.seconds_value, 59U, "seconds_value");
    check_range(value.minutes_value, 59U, "minutes_value");
    check_range(value.hours_value, 23U, "hours_value");
    return result;
  }

  if (value.minutes_flag && !value.seconds_flag) {
    result.add(hdr::Severity::kError,
               "minutes_flag requires seconds_flag (§5.17.7)");
  }
  if (value.hours_flag && !value.minutes_flag) {
    result.add(hdr::Severity::kError,
               "hours_flag requires minutes_flag (§5.17.7)");
  }
  if (value.seconds_flag) {
    check_range(value.seconds_value, 59U, "seconds_value");
  } else if (value.seconds_value != 0U) {
    result.add(hdr::Severity::kError,
               "seconds_value is not coded when seconds_flag == 0");
  }
  if (value.minutes_flag) {
    check_range(value.minutes_value, 59U, "minutes_value");
  } else if (value.minutes_value != 0U) {
    result.add(hdr::Severity::kError,
               "minutes_value is not coded when minutes_flag == 0");
  }
  if (value.hours_flag) {
    check_range(value.hours_value, 23U, "hours_value");
  } else if (value.hours_value != 0U) {
    result.add(hdr::Severity::kError,
               "hours_value is not coded when hours_flag == 0");
  }
  return result;
}

hdr::ValidationResult validate_banding_hints(const BandingHints& value,
                                             bool) {
  hdr::ValidationResult result;
  const auto components_default = [&value](std::size_t begin) {
    for (std::size_t i = begin; i < 3U; ++i) {
      if (value.components[i] != BandingComponentHint{}) {
        return false;
      }
    }
    return true;
  };

  if (!value.coding_banding_present_flag) {
    if (value.banding_hints_flag || value.three_color_components_flag ||
        !components_default(0) ||
        value.band_units_information_present_flag ||
        value.varying_size_band_units_flag ||
        value.band_block_in_luma_samples != 0U ||
        !value.vert_size_in_band_blocks_minus_1.empty() ||
        !value.horz_size_in_band_blocks_minus_1.empty() ||
        !value.banding_in_band_unit_present.empty()) {
      result.add(hdr::Severity::kError,
                 "banding hint fields are not coded when "
                 "coding_banding_present_flag == 0 (§5.17.8)");
    }
    return result;
  }

  if (!value.banding_hints_flag) {
    if (value.three_color_components_flag || !components_default(0) ||
        value.band_units_information_present_flag ||
        value.varying_size_band_units_flag ||
        value.band_block_in_luma_samples != 0U ||
        !value.vert_size_in_band_blocks_minus_1.empty() ||
        !value.horz_size_in_band_blocks_minus_1.empty() ||
        !value.banding_in_band_unit_present.empty()) {
      result.add(hdr::Severity::kError,
                 "banding component hints are not coded when "
                 "banding_hints_flag == 0 (§5.17.8)");
    }
    return result;
  }

  const std::size_t component_count =
      value.three_color_components_flag ? 3U : 1U;
  for (std::size_t i = 0; i < 3U; ++i) {
    const auto& component = value.components[i];
    if (i >= component_count) {
      if (component != BandingComponentHint{}) {
        result.add(hdr::Severity::kError,
                   "component " + size_text(i) +
                       " hint is not coded when "
                       "three_color_components_flag == 0 (§5.17.8)");
      }
      continue;
    }
    if (!component.banding_in_component_present_flag) {
      if (component.max_band_width_minus_4 != 0U ||
          component.max_band_step_minus_1 != 0U) {
        result.add(hdr::Severity::kError,
                   "component " + size_text(i) +
                       " band width/step are not coded when "
                       "banding_in_component_present_flag == 0");
      }
      continue;
    }
    if (component.max_band_width_minus_4 > 63U) {
      result.add(hdr::Severity::kError,
                 "max_band_width_minus_4 does not fit in 6 bits");
    }
    if (component.max_band_step_minus_1 > 15U) {
      result.add(hdr::Severity::kError,
                 "max_band_step_minus_1 does not fit in 4 bits");
    }
  }

  if (!value.band_units_information_present_flag) {
    if (value.varying_size_band_units_flag ||
        value.band_block_in_luma_samples != 0U ||
        !value.vert_size_in_band_blocks_minus_1.empty() ||
        !value.horz_size_in_band_blocks_minus_1.empty() ||
        !value.banding_in_band_unit_present.empty()) {
      result.add(hdr::Severity::kError,
                 "band-unit information is not coded when "
                 "band_units_information_present_flag == 0 (§5.17.8)");
    }
    return result;
  }

  if (value.num_band_units_rows_minus_1 > 31U) {
    result.add(hdr::Severity::kError,
               "num_band_units_rows_minus_1 does not fit in 5 bits");
  }
  if (value.num_band_units_cols_minus_1 > 31U) {
    result.add(hdr::Severity::kError,
               "num_band_units_cols_minus_1 does not fit in 5 bits");
  }
  if (value.vert_size_in_band_blocks_minus_1.size() > 32U ||
      value.horz_size_in_band_blocks_minus_1.size() > 32U) {
    result.add(hdr::Severity::kError,
               "band-unit per-row/column array exceeds 32 entries");
  }

  const std::size_t rows =
      static_cast<std::size_t>(value.num_band_units_rows_minus_1) + 1U;
  const std::size_t cols =
      static_cast<std::size_t>(value.num_band_units_cols_minus_1) + 1U;

  if (value.varying_size_band_units_flag) {
    if (value.band_block_in_luma_samples > 7U) {
      result.add(hdr::Severity::kError,
                 "band_block_in_luma_samples does not fit in 3 bits");
    }
    if (value.vert_size_in_band_blocks_minus_1.size() != rows) {
      result.add(hdr::Severity::kError,
                 "vert_size_in_band_blocks_minus_1 has " +
                     size_text(value.vert_size_in_band_blocks_minus_1.size()) +
                     " entries, expected " + size_text(rows));
    }
    if (value.horz_size_in_band_blocks_minus_1.size() != cols) {
      result.add(hdr::Severity::kError,
                 "horz_size_in_band_blocks_minus_1 has " +
                     size_text(value.horz_size_in_band_blocks_minus_1.size()) +
                     " entries, expected " + size_text(cols));
    }
    for (const std::uint8_t entry :
         value.vert_size_in_band_blocks_minus_1) {
      if (entry > 31U) {
        result.add(hdr::Severity::kError,
                   "vert_size_in_band_blocks_minus_1 does not fit in 5 bits");
        break;
      }
    }
    for (const std::uint8_t entry :
         value.horz_size_in_band_blocks_minus_1) {
      if (entry > 31U) {
        result.add(hdr::Severity::kError,
                   "horz_size_in_band_blocks_minus_1 does not fit in 5 bits");
        break;
      }
    }
  } else {
    if (value.band_block_in_luma_samples != 0U) {
      result.add(hdr::Severity::kError,
                 "band_block_in_luma_samples is not coded when "
                 "varying_size_band_units_flag == 0");
    }
    if (!value.vert_size_in_band_blocks_minus_1.empty() ||
        !value.horz_size_in_band_blocks_minus_1.empty()) {
      result.add(hdr::Severity::kError,
                 "band-unit size arrays are not coded when "
                 "varying_size_band_units_flag == 0");
    }
  }

  if (value.banding_in_band_unit_present.size() != rows * cols) {
    result.add(hdr::Severity::kError,
               "banding_in_band_unit_present has " +
                   size_text(value.banding_in_band_unit_present.size()) +
                   " entries, expected " + size_text(rows * cols));
  }
  for (const std::uint8_t present : value.banding_in_band_unit_present) {
    if (present > 1U) {
      result.add(hdr::Severity::kError,
                 "banding_in_band_unit_present entries must be 0 or 1");
      break;
    }
  }
  return result;
}

hdr::ValidationResult validate_icc_profile(const IccProfile& value,
                                           bool strict) {
  hdr::ValidationResult result;
  if (value.data.size() < 10U) {
    add_range_issue(
        result, strict,
        "ICC profile is " + size_text(value.data.size()) +
            " bytes; ICCmajorVer/ICCminorVer need at least 10 (§6.16.9)");
  }
  return result;
}

hdr::ValidationResult validate_scan_type(const ScanType& value, bool) {
  hdr::ValidationResult result;
  if (value.mps_pic_struct_type > 12U) {
    result.add(hdr::Severity::kError,
               "mps_pic_struct_type = " +
                   size_text(value.mps_pic_struct_type) +
                   " is reserved and shall not be present (§6.16.10)");
  }
  if (value.mps_source_scan_type_idc > 3U) {
    result.add(hdr::Severity::kError,
               "mps_source_scan_type_idc does not fit in 2 bits");
  }
  return result;
}

hdr::ValidationResult validate_temporal_point_info(const TemporalPointInfo&,
                                                   bool) {
  return {};
}

hdr::ValidationResult validate_decoded_frame_hash(const DecodedFrameHash& value,
                                                  bool strict) {
  hdr::ValidationResult result;
  if (value.hash_type > 15U) {
    result.add(hdr::Severity::kError, "hash_type does not fit in 4 bits");
    return result;
  }
  if (value.hash_type != 0U) {
    add_range_issue(result, strict,
                    "hash_type = " + size_text(value.hash_type) +
                        " is reserved for AOMedia use (§6.16.13)");
  }

  const auto hash_nonzero = [](const FrameHash& hash) {
    return std::any_of(hash.begin(), hash.end(),
                       [](std::uint8_t byte) { return byte != 0U; });
  };

  if (value.per_plane) {
    if (hash_nonzero(value.frame_hash)) {
      result.add(hdr::Severity::kError,
                 "frame_hash is not coded when per_plane == 1");
    }
    const std::size_t plane_count = value.is_monochrome ? 1U : 3U;
    for (std::size_t i = plane_count; i < 3U; ++i) {
      if (hash_nonzero(value.plane_hash[i])) {
        result.add(hdr::Severity::kError,
                   "plane_hash[" + size_text(i) +
                       "] is not coded for a monochrome frame");
      }
    }
  } else {
    for (std::size_t i = 0; i < 3U; ++i) {
      if (hash_nonzero(value.plane_hash[i])) {
        result.add(hdr::Severity::kError,
                   "plane_hash[" + size_text(i) +
                       "] is not coded when per_plane == 0");
        break;
      }
    }
  }
  return result;
}

hdr::ValidationResult validate_user_data_unregistered(
    const UserDataUnregistered&, bool) {
  return {};
}

// ---------------------------------------------------------------------------
// Canonical sizes and serialization
// ---------------------------------------------------------------------------

std::size_t itut_t35_unit_size(const ItutT35& value) {
  return 1U +
         (value.country_code == kItutT35ExtensionCountryCode ? 1U : 0U) +
         value.payload.size();
}

std::size_t timecode_syntax_bits(const Timecode& value) {
  std::size_t bits = 5U + 1U + 1U + 1U + 9U;  // fixed prefix
  if (value.full_timestamp_flag) {
    bits += 6U + 6U + 5U;
  } else {
    bits += 1U;  // seconds_flag
    if (value.seconds_flag) {
      bits += 6U + 1U;  // seconds_value + minutes_flag
      if (value.minutes_flag) {
        bits += 6U + 1U;  // minutes_value + hours_flag
        if (value.hours_flag) {
          bits += 5U;
        }
      }
    }
  }
  bits += 5U + value.time_offset_length;
  return bits;
}

std::size_t timecode_unit_size(const Timecode& value) {
  return cer_bytes(timecode_syntax_bits(value));
}

std::size_t banding_hints_syntax_bits(const BandingHints& value) {
  std::size_t bits = 2U;  // coding + source
  if (!value.coding_banding_present_flag) {
    return bits;
  }
  bits += 1U;  // banding_hints_flag
  if (!value.banding_hints_flag) {
    return bits;
  }
  bits += 1U;  // three_color_components_flag
  const std::size_t component_count =
      value.three_color_components_flag ? 3U : 1U;
  bits += component_count;
  for (std::size_t i = 0; i < component_count; ++i) {
    if (value.components[i].banding_in_component_present_flag) {
      bits += 6U + 4U;
    }
  }
  bits += 1U;  // band_units_information_present_flag
  if (!value.band_units_information_present_flag) {
    return bits;
  }
  bits += 5U + 5U + 1U;  // rows, cols, varying
  const std::size_t rows =
      static_cast<std::size_t>(value.num_band_units_rows_minus_1) + 1U;
  const std::size_t cols =
      static_cast<std::size_t>(value.num_band_units_cols_minus_1) + 1U;
  if (value.varying_size_band_units_flag) {
    bits += 3U;  // band_block_in_luma_samples
    bits += 5U * value.vert_size_in_band_blocks_minus_1.size();
    bits += 5U * value.horz_size_in_band_blocks_minus_1.size();
  }
  bits += rows * cols;
  return bits;
}

std::size_t banding_hints_unit_size(const BandingHints& value) {
  return cer_bytes(banding_hints_syntax_bits(value));
}

std::size_t icc_profile_unit_size(const IccProfile& value) {
  return value.data.size();
}

std::size_t temporal_point_info_unit_size(const TemporalPointInfo& value) {
  return av2enc::basic::leb128_size(value.frame_presentation_time);
}

std::size_t decoded_frame_hash_unit_size(const DecodedFrameHash& value) {
  if (!value.per_plane) {
    return kDecodedFrameHashWholeUnitBytes;
  }
  return value.is_monochrome ? kDecodedFrameHashPerPlaneMonoUnitBytes
                             : kDecodedFrameHashPerPlaneColorUnitBytes;
}

std::size_t user_data_unregistered_unit_size(
    const UserDataUnregistered& value) {
  return kUserDataUnregisteredUuidBytes + value.payload.size();
}

std::vector<std::uint8_t> serialize_itut_t35_unit(const ItutT35& value) {
  std::vector<std::uint8_t> out;
  out.reserve(itut_t35_unit_size(value));
  out.push_back(value.country_code);
  if (value.country_code == kItutT35ExtensionCountryCode) {
    out.push_back(value.country_code_extension_byte);
  }
  out.insert(out.end(), value.payload.begin(), value.payload.end());
  return out;
}

std::vector<std::uint8_t> serialize_timecode_unit(const Timecode& value) {
  BitWriter writer;
  writer.put_bits(value.counting_type, 5U);
  writer.put_bits(value.full_timestamp_flag ? 1U : 0U, 1U);
  writer.put_bits(value.discontinuity_flag ? 1U : 0U, 1U);
  writer.put_bits(value.cnt_dropped_flag ? 1U : 0U, 1U);
  writer.put_bits(value.n_frames, 9U);
  if (value.full_timestamp_flag) {
    writer.put_bits(value.seconds_value, 6U);
    writer.put_bits(value.minutes_value, 6U);
    writer.put_bits(value.hours_value, 5U);
  } else {
    writer.put_bits(value.seconds_flag ? 1U : 0U, 1U);
    if (value.seconds_flag) {
      writer.put_bits(value.seconds_value, 6U);
      writer.put_bits(value.minutes_flag ? 1U : 0U, 1U);
      if (value.minutes_flag) {
        writer.put_bits(value.minutes_value, 6U);
        writer.put_bits(value.hours_flag ? 1U : 0U, 1U);
        if (value.hours_flag) {
          writer.put_bits(value.hours_value, 5U);
        }
      }
    }
  }
  writer.put_bits(value.time_offset_length, 5U);
  if (value.time_offset_length > 0U) {
    writer.put_bits(value.time_offset_value, value.time_offset_length);
  }
  writer.align_to_byte();
  return writer.take_bytes();
}

std::vector<std::uint8_t> serialize_banding_hints_unit(
    const BandingHints& value) {
  BitWriter writer;
  writer.put_bits(value.coding_banding_present_flag ? 1U : 0U, 1U);
  writer.put_bits(value.source_banding_present_flag ? 1U : 0U, 1U);
  if (value.coding_banding_present_flag) {
    writer.put_bits(value.banding_hints_flag ? 1U : 0U, 1U);
    if (value.banding_hints_flag) {
      writer.put_bits(value.three_color_components_flag ? 1U : 0U, 1U);
      const std::size_t component_count =
          value.three_color_components_flag ? 3U : 1U;
      for (std::size_t i = 0; i < component_count; ++i) {
        const auto& component = value.components[i];
        writer.put_bits(
            component.banding_in_component_present_flag ? 1U : 0U, 1U);
        if (component.banding_in_component_present_flag) {
          writer.put_bits(component.max_band_width_minus_4, 6U);
          writer.put_bits(component.max_band_step_minus_1, 4U);
        }
      }
      writer.put_bits(value.band_units_information_present_flag ? 1U : 0U,
                      1U);
      if (value.band_units_information_present_flag) {
        writer.put_bits(value.num_band_units_rows_minus_1, 5U);
        writer.put_bits(value.num_band_units_cols_minus_1, 5U);
        writer.put_bits(value.varying_size_band_units_flag ? 1U : 0U, 1U);
        const std::size_t rows =
            static_cast<std::size_t>(value.num_band_units_rows_minus_1) + 1U;
        const std::size_t cols =
            static_cast<std::size_t>(value.num_band_units_cols_minus_1) + 1U;
        if (value.varying_size_band_units_flag) {
          writer.put_bits(value.band_block_in_luma_samples, 3U);
          for (std::size_t r = 0; r < rows; ++r) {
            const std::uint8_t entry =
                r < value.vert_size_in_band_blocks_minus_1.size()
                    ? value.vert_size_in_band_blocks_minus_1[r]
                    : 0U;
            writer.put_bits(entry, 5U);
          }
          for (std::size_t c = 0; c < cols; ++c) {
            const std::uint8_t entry =
                c < value.horz_size_in_band_blocks_minus_1.size()
                    ? value.horz_size_in_band_blocks_minus_1[c]
                    : 0U;
            writer.put_bits(entry, 5U);
          }
        }
        for (std::size_t r = 0; r < rows; ++r) {
          for (std::size_t c = 0; c < cols; ++c) {
            const std::size_t index = r * cols + c;
            const std::uint8_t present =
                index < value.banding_in_band_unit_present.size()
                    ? value.banding_in_band_unit_present[index]
                    : 0U;
            writer.put_bits(present, 1U);
          }
        }
      }
    }
  }
  writer.align_to_byte();
  return writer.take_bytes();
}

std::vector<std::uint8_t> serialize_icc_profile_unit(const IccProfile& value) {
  return value.data;
}

std::vector<std::uint8_t> serialize_scan_type_unit(const ScanType& value) {
  const auto byte = static_cast<std::uint8_t>(
      (static_cast<unsigned>(value.mps_pic_struct_type) << 3U) |
      (static_cast<unsigned>(value.mps_source_scan_type_idc) << 1U) |
      (value.mps_duplicate_flag ? 1U : 0U));
  return {byte};
}

std::vector<std::uint8_t> serialize_temporal_point_info_unit(
    const TemporalPointInfo& value) {
  std::vector<std::uint8_t> out;
  out.reserve(temporal_point_info_unit_size(value));
  av2enc::basic::append_leb128(out, value.frame_presentation_time);
  return out;
}

std::vector<std::uint8_t> serialize_decoded_frame_hash_unit(
    const DecodedFrameHash& value) {
  std::vector<std::uint8_t> out;
  out.reserve(decoded_frame_hash_unit_size(value));
  out.push_back(static_cast<std::uint8_t>(
      (static_cast<unsigned>(value.hash_type) << 4U) |
      (value.per_plane ? 0x08U : 0U) | (value.has_grain ? 0x04U : 0U) |
      (value.is_monochrome ? 0x02U : 0U)));
  if (value.per_plane) {
    const std::size_t plane_count = value.is_monochrome ? 1U : 3U;
    for (std::size_t i = 0; i < plane_count; ++i) {
      out.insert(out.end(), value.plane_hash[i].begin(),
                 value.plane_hash[i].end());
    }
  } else {
    out.insert(out.end(), value.frame_hash.begin(), value.frame_hash.end());
  }
  return out;
}

std::vector<std::uint8_t> serialize_user_data_unregistered_unit(
    const UserDataUnregistered& value) {
  std::vector<std::uint8_t> out;
  out.reserve(user_data_unregistered_unit_size(value));
  out.insert(out.end(), value.uuid.begin(), value.uuid.end());
  out.insert(out.end(), value.payload.begin(), value.payload.end());
  return out;
}

namespace {

template <class T>
std::optional<MetadataUnitPayload> finish_unit(
    Type type, const hdr::ValidationResult& validation, const T& value,
    std::string& error) {
  if (validation.has_errors()) {
    error = summary_of(validation);
    return std::nullopt;
  }
  MetadataUnitPayload unit;
  unit.type = type;
  if constexpr (std::is_same_v<T, ItutT35>) {
    unit.bytes = serialize_itut_t35_unit(value);
  } else if constexpr (std::is_same_v<T, Timecode>) {
    unit.bytes = serialize_timecode_unit(value);
  } else if constexpr (std::is_same_v<T, BandingHints>) {
    unit.bytes = serialize_banding_hints_unit(value);
  } else if constexpr (std::is_same_v<T, IccProfile>) {
    unit.bytes = serialize_icc_profile_unit(value);
  } else if constexpr (std::is_same_v<T, ScanType>) {
    unit.bytes = serialize_scan_type_unit(value);
  } else if constexpr (std::is_same_v<T, TemporalPointInfo>) {
    unit.bytes = serialize_temporal_point_info_unit(value);
  } else if constexpr (std::is_same_v<T, DecodedFrameHash>) {
    unit.bytes = serialize_decoded_frame_hash_unit(value);
  } else {
    static_assert(std::is_same_v<T, UserDataUnregistered>);
    unit.bytes = serialize_user_data_unregistered_unit(value);
  }
  return unit;
}

}  // namespace

std::optional<MetadataUnitPayload> make_itut_t35_unit(const ItutT35& value,
                                                      std::string& error,
                                                      bool strict) {
  error.clear();
  return finish_unit(Type::kItutT35, validate_itut_t35(value, strict), value,
                     error);
}

std::optional<MetadataUnitPayload> make_timecode_unit(const Timecode& value,
                                                      std::string& error,
                                                      bool strict) {
  error.clear();
  return finish_unit(Type::kTimecode, validate_timecode(value, strict), value,
                     error);
}

std::optional<MetadataUnitPayload> make_banding_hints_unit(
    const BandingHints& value, std::string& error, bool strict) {
  error.clear();
  return finish_unit(Type::kBandingHints,
                     validate_banding_hints(value, strict), value, error);
}

std::optional<MetadataUnitPayload> make_icc_profile_unit(
    const IccProfile& value, std::string& error, bool strict) {
  error.clear();
  return finish_unit(Type::kIccProfile, validate_icc_profile(value, strict),
                     value, error);
}

std::optional<MetadataUnitPayload> make_scan_type_unit(const ScanType& value,
                                                       std::string& error,
                                                       bool strict) {
  error.clear();
  return finish_unit(Type::kScanType, validate_scan_type(value, strict), value,
                     error);
}

std::optional<MetadataUnitPayload> make_temporal_point_info_unit(
    const TemporalPointInfo& value, std::string& error, bool strict) {
  error.clear();
  return finish_unit(Type::kTemporalPointInfo,
                     validate_temporal_point_info(value, strict), value,
                     error);
}

std::optional<MetadataUnitPayload> make_decoded_frame_hash_unit(
    const DecodedFrameHash& value, std::string& error, bool strict) {
  error.clear();
  return finish_unit(Type::kDecodedFrameHash,
                     validate_decoded_frame_hash(value, strict), value, error);
}

std::optional<MetadataUnitPayload> make_user_data_unregistered_unit(
    const UserDataUnregistered& value, std::string& error, bool strict) {
  error.clear();
  return finish_unit(Type::kUserDataUnregistered,
                     validate_user_data_unregistered(value, strict), value,
                     error);
}

std::size_t canonical_unit_size(const Value& value) {
  return std::visit(
      [](const auto& item) -> std::size_t {
        using T = std::decay_t<decltype(item)>;
        if constexpr (std::is_same_v<T, hdr::ContentLightLevel>) {
          return hdr::kCllUnitBytes;
        } else if constexpr (std::is_same_v<T,
                                            hdr::MasteringDisplayColorVolume>) {
          return hdr::kMdcvUnitBytes;
        } else if constexpr (std::is_same_v<T, ItutT35>) {
          return itut_t35_unit_size(item);
        } else if constexpr (std::is_same_v<T, Timecode>) {
          return timecode_unit_size(item);
        } else if constexpr (std::is_same_v<T, BandingHints>) {
          return banding_hints_unit_size(item);
        } else if constexpr (std::is_same_v<T, IccProfile>) {
          return icc_profile_unit_size(item);
        } else if constexpr (std::is_same_v<T, ScanType>) {
          return kScanTypeUnitBytes;
        } else if constexpr (std::is_same_v<T, TemporalPointInfo>) {
          return temporal_point_info_unit_size(item);
        } else if constexpr (std::is_same_v<T, DecodedFrameHash>) {
          return decoded_frame_hash_unit_size(item);
        } else {
          static_assert(std::is_same_v<T, UserDataUnregistered>);
          return user_data_unregistered_unit_size(item);
        }
      },
      value);
}

// ---------------------------------------------------------------------------
// OBU writers
// ---------------------------------------------------------------------------

std::vector<std::uint8_t> make_short_obu(const MetadataUnitPayload& unit,
                                         const hdr::MetadataObuConfig& config,
                                         std::string& error) {
  error.clear();
  if (!validate_short_config(config, error)) {
    return {};
  }
  if (!check_unit_payload(unit, config.strict_ranges, error)) {
    return {};
  }
  return hdr::make_metadata_short_obu(unit.type, unit.bytes, config, error);
}

std::vector<std::uint8_t> make_short_cancel_obu(
    Type type, const hdr::MetadataObuConfig& config, std::string& error) {
  error.clear();
  if (!validate_short_config(config, error)) {
    return {};
  }
  if (!is_known_type(type)) {
    error = "reserved metadata_type is not emitted";
    return {};
  }
  return hdr::make_cancel_short_obu(type, config, error);
}

std::vector<std::uint8_t> make_group_obu(std::span<const GroupUnit> units,
                                         const hdr::MetadataObuConfig& config,
                                         std::string& error) {
  error.clear();
  if (!validate_group_config(config, error)) {
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

  std::vector<hdr::MetadataGroupEntry> entries;
  entries.reserve(units.size());
  for (const auto& unit : units) {
    if (!is_known_type(unit.payload.type)) {
      error = "reserved metadata_type is not emitted";
      return {};
    }
    if (unit.payload.type == Type::kTemporalPointInfo) {
      error =
          "METADATA_TYPE_TEMPORAL_POINT_INFO shall only appear in an "
          "OBU_METADATA_SHORT (§6.16.11)";
      return {};
    }
    if (static_cast<unsigned>(unit.layer) > 3U) {
      error = "reserved muh_layer_idc";
      return {};
    }
    if (unit.layer == hdr::MetadataLayer::kValues) {
      error =
          "writer does not emit explicit layer maps (muh_layer_idc == "
          "LAYER_VALUES)";
      return {};
    }
    if (static_cast<unsigned>(unit.persistence) > 3U) {
      error = "muh_persistence_idc 4..7 is reserved (§6.16.3)";
      return {};
    }
    if (unit.cancel) {
      if (!unit.payload.bytes.empty()) {
        error = "a cancel metadata unit cannot carry a payload";
        return {};
      }
    } else if (!check_unit_payload(unit.payload, config.strict_ranges,
                                   error)) {
      return {};
    }
    hdr::MetadataGroupEntry entry;
    entry.type = unit.payload.type;
    entry.unit = unit.payload.bytes;
    entry.layer = unit.layer;
    entry.persistence = unit.persistence;
    entry.priority = unit.priority;
    entry.cancel = unit.cancel;
    entries.push_back(std::move(entry));
  }
  hdr::MetadataObuConfig framing = config;
  // The group writer takes muh_layer_idc/persistence/priority per entry.
  // HDR's group writer still range-checks config.layer, so pin it to a valid
  // value to keep config.layer meaning "unused" for group framing.
  framing.layer = hdr::MetadataLayer::kCurrent;
  return hdr::make_metadata_group_obu(entries, framing, error);
}

// ---------------------------------------------------------------------------
// Readers
// ---------------------------------------------------------------------------

bool decode_itut_t35(std::span<const std::uint8_t> unit, ItutT35& out,
                     std::string& error) {
  error.clear();
  out = ItutT35{};
  if (unit.empty()) {
    error = "metadata_itut_t35 payload is empty (no country code)";
    return false;
  }
  out.country_code = unit[0];
  std::size_t pos = 1;
  out.country_code_extension_byte = 0;
  if (out.country_code == kItutT35ExtensionCountryCode) {
    if (pos >= unit.size()) {
      error = "metadata_itut_t35 is missing the country code extension byte";
      return false;
    }
    out.country_code_extension_byte = unit[pos++];
  }
  out.payload.assign(unit.begin() + static_cast<std::ptrdiff_t>(pos),
                     unit.end());
  return true;
}

bool decode_timecode(std::span<const std::uint8_t> unit, Timecode& out,
                     std::string& error) {
  error.clear();
  out = Timecode{};
  BitReader reader(unit);
  std::uint32_t field = 0;
  const auto fail = [&error]() {
    error = "truncated metadata_timecode payload";
    return false;
  };

  if (!reader.get(5U, field)) return fail();
  out.counting_type = static_cast<std::uint8_t>(field);
  if (!reader.get(1U, field)) return fail();
  out.full_timestamp_flag = field != 0U;
  if (!reader.get(1U, field)) return fail();
  out.discontinuity_flag = field != 0U;
  if (!reader.get(1U, field)) return fail();
  out.cnt_dropped_flag = field != 0U;
  if (!reader.get(9U, field)) return fail();
  out.n_frames = static_cast<std::uint16_t>(field);

  if (out.full_timestamp_flag) {
    if (!reader.get(6U, field)) return fail();
    out.seconds_value = static_cast<std::uint8_t>(field);
    if (!reader.get(6U, field)) return fail();
    out.minutes_value = static_cast<std::uint8_t>(field);
    if (!reader.get(5U, field)) return fail();
    out.hours_value = static_cast<std::uint8_t>(field);
    out.seconds_flag = true;
    out.minutes_flag = true;
    out.hours_flag = true;
  } else {
    if (!reader.get(1U, field)) return fail();
    out.seconds_flag = field != 0U;
    if (out.seconds_flag) {
      if (!reader.get(6U, field)) return fail();
      out.seconds_value = static_cast<std::uint8_t>(field);
      if (!reader.get(1U, field)) return fail();
      out.minutes_flag = field != 0U;
      if (out.minutes_flag) {
        if (!reader.get(6U, field)) return fail();
        out.minutes_value = static_cast<std::uint8_t>(field);
        if (!reader.get(1U, field)) return fail();
        out.hours_flag = field != 0U;
        if (out.hours_flag) {
          if (!reader.get(5U, field)) return fail();
          out.hours_value = static_cast<std::uint8_t>(field);
        }
      }
    }
  }

  if (!reader.get(5U, field)) return fail();
  out.time_offset_length = static_cast<std::uint8_t>(field);
  if (out.time_offset_length > 0U) {
    if (out.time_offset_length > 31U) return fail();
    if (!reader.get(out.time_offset_length, field)) return fail();
    out.time_offset_value = field;
  }
  return true;
}

bool decode_banding_hints(std::span<const std::uint8_t> unit,
                          BandingHints& out, std::string& error) {
  error.clear();
  out = BandingHints{};
  BitReader reader(unit);
  std::uint32_t field = 0;
  const auto fail = [&error]() {
    error = "truncated metadata_banding_hints payload";
    return false;
  };

  if (!reader.get(1U, field)) return fail();
  out.coding_banding_present_flag = field != 0U;
  if (!reader.get(1U, field)) return fail();
  out.source_banding_present_flag = field != 0U;
  if (!out.coding_banding_present_flag) {
    return true;
  }

  if (!reader.get(1U, field)) return fail();
  out.banding_hints_flag = field != 0U;
  if (!out.banding_hints_flag) {
    return true;
  }

  if (!reader.get(1U, field)) return fail();
  out.three_color_components_flag = field != 0U;
  const std::size_t component_count =
      out.three_color_components_flag ? 3U : 1U;
  for (std::size_t i = 0; i < component_count; ++i) {
    if (!reader.get(1U, field)) return fail();
    auto& component = out.components[i];
    component.banding_in_component_present_flag = field != 0U;
    if (component.banding_in_component_present_flag) {
      if (!reader.get(6U, field)) return fail();
      component.max_band_width_minus_4 = static_cast<std::uint8_t>(field);
      if (!reader.get(4U, field)) return fail();
      component.max_band_step_minus_1 = static_cast<std::uint8_t>(field);
    }
  }

  if (!reader.get(1U, field)) return fail();
  out.band_units_information_present_flag = field != 0U;
  if (!out.band_units_information_present_flag) {
    return true;
  }

  if (!reader.get(5U, field)) return fail();
  out.num_band_units_rows_minus_1 = static_cast<std::uint8_t>(field);
  if (!reader.get(5U, field)) return fail();
  out.num_band_units_cols_minus_1 = static_cast<std::uint8_t>(field);
  if (!reader.get(1U, field)) return fail();
  out.varying_size_band_units_flag = field != 0U;

  const std::size_t rows =
      static_cast<std::size_t>(out.num_band_units_rows_minus_1) + 1U;
  const std::size_t cols =
      static_cast<std::size_t>(out.num_band_units_cols_minus_1) + 1U;

  if (out.varying_size_band_units_flag) {
    if (!reader.get(3U, field)) return fail();
    out.band_block_in_luma_samples = static_cast<std::uint8_t>(field);
    out.vert_size_in_band_blocks_minus_1.resize(rows);
    out.horz_size_in_band_blocks_minus_1.resize(cols);
    for (std::size_t r = 0; r < rows; ++r) {
      if (!reader.get(5U, field)) return fail();
      out.vert_size_in_band_blocks_minus_1[r] =
          static_cast<std::uint8_t>(field);
    }
    for (std::size_t c = 0; c < cols; ++c) {
      if (!reader.get(5U, field)) return fail();
      out.horz_size_in_band_blocks_minus_1[c] =
          static_cast<std::uint8_t>(field);
    }
  }

  out.banding_in_band_unit_present.resize(rows * cols);
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t c = 0; c < cols; ++c) {
      if (!reader.get(1U, field)) return fail();
      out.banding_in_band_unit_present[r * cols + c] =
          static_cast<std::uint8_t>(field);
    }
  }
  return true;
}

bool decode_icc_profile(std::span<const std::uint8_t> unit, IccProfile& out,
                        std::string& error) {
  error.clear();
  out.data.assign(unit.begin(), unit.end());
  return true;
}

bool decode_scan_type(std::span<const std::uint8_t> unit, ScanType& out,
                      std::string& error) {
  error.clear();
  out = ScanType{};
  if (unit.empty()) {
    error = "metadata_scan_type payload is empty";
    return false;
  }
  const std::uint8_t byte = unit[0];
  out.mps_pic_struct_type = static_cast<std::uint8_t>((byte >> 3U) & 0x1FU);
  out.mps_source_scan_type_idc =
      static_cast<std::uint8_t>((byte >> 1U) & 0x03U);
  out.mps_duplicate_flag = (byte & 0x01U) != 0U;
  return true;
}

bool decode_temporal_point_info(std::span<const std::uint8_t> unit,
                                TemporalPointInfo& out, std::string& error) {
  error.clear();
  out = TemporalPointInfo{};
  if (unit.empty()) {
    error = "metadata_temporal_point_info payload is empty";
    return false;
  }
  std::size_t pos = 0;
  if (!read_uleb128(unit, pos, out.frame_presentation_time, error)) {
    return false;
  }
  return true;
}

bool decode_decoded_frame_hash(std::span<const std::uint8_t> unit,
                               DecodedFrameHash& out, std::string& error) {
  error.clear();
  out = DecodedFrameHash{};
  BitReader reader(unit);
  std::uint32_t field = 0;
  const auto fail = [&error]() {
    error = "truncated metadata_decoded_frame_hash payload";
    return false;
  };
  if (!reader.get(4U, field)) return fail();
  out.hash_type = static_cast<std::uint8_t>(field);
  if (!reader.get(1U, field)) return fail();
  out.per_plane = field != 0U;
  if (!reader.get(1U, field)) return fail();
  out.has_grain = field != 0U;
  if (!reader.get(1U, field)) return fail();
  out.is_monochrome = field != 0U;
  if (!reader.get(1U, field)) return fail();
  if (field != 0U) {
    error = "metadata_decoded_frame_hash reserved bit must be zero (§6.16.13)";
    return false;
  }

  const std::size_t plane_count =
      out.per_plane ? (out.is_monochrome ? 1U : 3U) : 1U;
  for (std::size_t plane = 0; plane < plane_count; ++plane) {
    FrameHash hash{};
    for (std::size_t byte_index = 0; byte_index < kFrameHashBytes;
         ++byte_index) {
      if (!reader.get(8U, field)) return fail();
      hash[byte_index] = static_cast<std::uint8_t>(field);
    }
    if (out.per_plane) {
      out.plane_hash[plane] = hash;
    } else {
      out.frame_hash = hash;
    }
  }
  return true;
}

bool decode_user_data_unregistered(std::span<const std::uint8_t> unit,
                                   UserDataUnregistered& out,
                                   std::string& error) {
  error.clear();
  out = UserDataUnregistered{};
  if (unit.size() < kUserDataUnregisteredUuidBytes) {
    error = "metadata_user_data_unregistered payload is shorter than the "
            "16-byte uuid_iso_iec_11578";
    return false;
  }
  std::copy_n(unit.begin(), kUserDataUnregisteredUuidBytes, out.uuid.begin());
  out.payload.assign(
      unit.begin() + static_cast<std::ptrdiff_t>(kUserDataUnregisteredUuidBytes),
      unit.end());
  return true;
}

bool decode_unit(Type type, std::span<const std::uint8_t> unit, Value& out,
                 std::string& error) {
  error.clear();
  switch (type) {
    case Type::kHdrCll: {
      hdr::ContentLightLevel value;
      if (!hdr::decode_cll(unit, value, error)) return false;
      out = value;
      return true;
    }
    case Type::kHdrMdcv: {
      hdr::MasteringDisplayColorVolume value;
      if (!hdr::decode_mdcv(unit, value, error)) return false;
      out = value;
      return true;
    }
    case Type::kItutT35: {
      ItutT35 value;
      if (!decode_itut_t35(unit, value, error)) return false;
      out = std::move(value);
      return true;
    }
    case Type::kTimecode: {
      Timecode value;
      if (!decode_timecode(unit, value, error)) return false;
      out = value;
      return true;
    }
    case Type::kDecodedFrameHash: {
      DecodedFrameHash value;
      if (!decode_decoded_frame_hash(unit, value, error)) return false;
      out = value;
      return true;
    }
    case Type::kBandingHints: {
      BandingHints value;
      if (!decode_banding_hints(unit, value, error)) return false;
      out = std::move(value);
      return true;
    }
    case Type::kIccProfile: {
      IccProfile value;
      if (!decode_icc_profile(unit, value, error)) return false;
      out = std::move(value);
      return true;
    }
    case Type::kScanType: {
      ScanType value;
      if (!decode_scan_type(unit, value, error)) return false;
      out = value;
      return true;
    }
    case Type::kTemporalPointInfo: {
      TemporalPointInfo value;
      if (!decode_temporal_point_info(unit, value, error)) return false;
      out = value;
      return true;
    }
    case Type::kUserDataUnregistered: {
      UserDataUnregistered value;
      if (!decode_user_data_unregistered(unit, value, error)) return false;
      out = std::move(value);
      return true;
    }
    default:
      error = "unsupported metadata_type " +
              size_text(static_cast<std::uint32_t>(type));
      return false;
  }
}

// ---------------------------------------------------------------------------
// Stream scanning
// ---------------------------------------------------------------------------

bool scan_metadata(std::span<const std::uint8_t> temporal_unit,
                   ScanResult& out, std::string& error) {
  error.clear();
  out = ScanResult{};

  const auto push_record = [&out](std::uint32_t metadata_type,
                                  const hdr::MetadataUnitView* group_unit,
                                  bool in_group, bool is_suffix,
                                  std::size_t obu_offset, bool before_frame,
                                  hdr::MetadataLayer short_layer,
                                  hdr::MetadataPersistence short_persistence,
                                  bool cancel) {
    MetadataRecord record;
    record.obu_offset = obu_offset;
    record.in_group = in_group;
    record.is_suffix = is_suffix;
    record.before_first_frame = before_frame;
    record.metadata_type = metadata_type;
    if (group_unit != nullptr) {
      record.layer = group_unit->layer;
      record.persistence = group_unit->persistence;
      record.priority = group_unit->priority;
    } else {
      record.layer = short_layer;
      record.persistence = short_persistence;
      record.priority = 0;
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

    if (obu.type == hdr::kObuTypeMetadataShort) {
      hdr::MetadataShortView view;
      if (!hdr::parse_metadata_short(obu.payload, view, error)) {
        return false;
      }
      ++out.metadata_obus;
      ++out.metadata_units;
      if (view.cancel) {
        ++out.cancelled_units;
        push_record(view.type, nullptr, false, view.is_suffix, obu.offset,
                    !seen_frame, view.layer, view.persistence, true);
        continue;
      }
      if (!is_known_type(view.type)) {
        ++out.unsupported_units;
        continue;
      }
      Value value;
      if (!decode_unit(static_cast<Type>(view.type), view.unit, value, error)) {
        ++out.invalid_units;
        error.clear();
        continue;
      }
      ++out.supported_units;
      push_record(view.type, nullptr, false, view.is_suffix, obu.offset,
                  !seen_frame, view.layer, view.persistence, false);
      out.records.back().has_value = true;
      out.records.back().value = std::move(value);
      continue;
    }

    if (obu.type == hdr::kObuTypeMetadataGroup) {
      hdr::MetadataGroupView view;
      if (!hdr::parse_metadata_group(obu.payload, obu.xlayer_id, view,
                                     error)) {
        return false;
      }
      ++out.metadata_obus;
      for (const auto& unit : view.units) {
        ++out.metadata_units;
        if (unit.cancel) {
          ++out.cancelled_units;
          push_record(unit.type, &unit, true, view.is_suffix, obu.offset,
                      !seen_frame, hdr::MetadataLayer::kUnspecified,
                      hdr::MetadataPersistence::kGlobal, true);
          continue;
        }
        if (unit.type ==
            static_cast<std::uint32_t>(Type::kTemporalPointInfo)) {
          // §6.16.11: only allowed in OBU_METADATA_SHORT.
          ++out.invalid_units;
          continue;
        }
        if (!is_known_type(unit.type)) {
          ++out.unsupported_units;
          continue;
        }
        Value value;
        if (!decode_unit(static_cast<Type>(unit.type), unit.unit, value,
                         error)) {
          ++out.invalid_units;
          error.clear();
          continue;
        }
        ++out.supported_units;
        push_record(unit.type, &unit, true, view.is_suffix, obu.offset,
                    !seen_frame, hdr::MetadataLayer::kUnspecified,
                    hdr::MetadataPersistence::kGlobal, false);
        out.records.back().has_value = true;
        out.records.back().value = std::move(value);
      }
    }
  }
  return true;
}

}  // namespace av2enc::metadata
