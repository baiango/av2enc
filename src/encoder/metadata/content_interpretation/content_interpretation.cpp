#include "encoder/metadata/content_interpretation/content_interpretation.h"

#include "bitstream/bitwriter.h"
#include "bitstream/m0_headers.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <sstream>
#include <utility>

namespace av2enc::ci {
namespace {

constexpr std::uint32_t kMaxUvlc = std::numeric_limits<std::uint32_t>::max();
constexpr std::uint32_t kMaxTicksMinus1 = kMaxUvlc - 1U;

bool is_known_primaries(std::uint8_t value) {
  switch (value) {
    case 1:   // CP_BT_709
    case 2:   // CP_UNSPECIFIED
    case 4:   // CP_BT_470_M
    case 5:   // CP_BT_470_B_G
    case 6:   // CP_BT_601
    case 7:   // CP_SMPTE_240
    case 8:   // CP_GENERIC_FILM
    case 9:   // CP_BT_2020
    case 10:  // CP_XYZ
    case 11:  // CP_SMPTE_431
    case 12:  // CP_SMPTE_432
    case 22:  // CP_EBU_3213
      return true;
    default:
      return false;
  }
}

bool is_known_transfer(std::uint8_t value) {
  switch (value) {
    case 1:   // TC_BT_709
    case 2:   // TC_UNSPECIFIED
    case 4:   // TC_BT_470_M
    case 5:   // TC_BT_470_B_G
    case 6:   // TC_BT_601
    case 7:   // TC_SMPTE_240
    case 8:   // TC_LINEAR
    case 9:   // TC_LOG_100
    case 10:  // TC_LOG_100_SQRT10
    case 11:  // TC_IEC_61966
    case 12:  // TC_BT_1361
    case 13:  // TC_SRGB
    case 14:  // TC_BT_2020_10_BIT
    case 16:  // TC_SMPTE_2084
    case 17:  // TC_SMPTE_428
    case 18:  // TC_HLG
      return true;
    default:
      return false;
  }
}

bool is_known_matrix(std::uint8_t value) {
  return value == 0 || value == 1 || value == 2 ||
         (value >= 4 && value <= 17);
}

// §5.2.1: OBU_CONTENT_INTERPRETATION is an extensible OBU, so the payload is
// [content_interpretation_obu()] obu_extension_flag f(1)
//   if (obu_extension_flag) obu_extension_data() else trailing_bits().
// The writer always emits extension flag 0.
class BitReader {
 public:
  explicit BitReader(std::span<const std::uint8_t> bytes) : bytes_(bytes) {}

  bool read_bit(bool& out, std::string& error) {
    if (bit_pos_ >= bytes_.size() * 8U) {
      error = "truncated content interpretation payload";
      return false;
    }
    const std::size_t byte_index = bit_pos_ >> 3U;
    const unsigned shift = 7U - static_cast<unsigned>(bit_pos_ & 7U);
    out = ((bytes_[byte_index] >> shift) & 1U) != 0U;
    ++bit_pos_;
    return true;
  }

  bool read_bits(unsigned count, std::uint32_t& out, std::string& error) {
    std::uint64_t value = 0;
    for (unsigned index = 0; index < count; ++index) {
      bool bit = false;
      if (!read_bit(bit, error)) {
        return false;
      }
      value = (value << 1U) | (bit ? 1U : 0U);
    }
    out = static_cast<std::uint32_t>(value);
    return true;
  }

  // §4.11.3 uvlc().
  bool read_uvlc(std::uint32_t& out, std::string& error) {
    std::uint32_t leading_zeros = 0;
    while (true) {
      bool bit = false;
      if (!read_bit(bit, error)) {
        return false;
      }
      if (bit) {
        break;
      }
      ++leading_zeros;
    }
    if (leading_zeros >= 32U) {
      out = kMaxUvlc;
      return true;
    }
    std::uint32_t payload = 0;
    if (!read_bits(leading_zeros, payload, error)) {
      return false;
    }
    const std::uint64_t value =
        static_cast<std::uint64_t>(payload) + ((1ULL << leading_zeros) - 1ULL);
    out = static_cast<std::uint32_t>(value);
    return true;
  }

  // §4.11.10 rg(n): q one bits, a zero bit, then n fixed bits. After 32 one
  // bits the descriptor returns the -1 sentinel (§4.11.10), which is a
  // conformance violation and must not be accepted.
  bool read_rg(unsigned n, std::uint32_t& out, std::string& error) {
    for (unsigned q = 0; q < 32U; ++q) {
      bool bit = false;
      if (!read_bit(bit, error)) {
        return false;
      }
      if (!bit) {
        std::uint32_t remainder = 0;
        if (!read_bits(n, remainder, error)) {
          return false;
        }
        out = (q << n) | remainder;
        return true;
      }
    }
    error = "rg() returned the -1 sentinel (32 one bits)";
    return false;
  }

  bool read_trailing_bits(std::string& error) {
    // §5.2.3: one 1 bit then zeros to the end of the OBU payload.
    bool bit = false;
    if (!read_bit(bit, error)) {
      error = "missing trailing_bits";
      return false;
    }
    if (!bit) {
      error = "trailing_bits must start with a single 1 bit";
      return false;
    }
    while (bit_pos_ < bytes_.size() * 8U) {
      if (!read_bit(bit, error)) {
        return false;
      }
      if (bit) {
        error = "trailing_bits contains a non-zero padding bit";
        return false;
      }
    }
    return true;
  }

  std::size_t bits_read() const { return bit_pos_; }

 private:
  std::span<const std::uint8_t> bytes_;
  std::size_t bit_pos_ = 0;
};

bool put_uvlc(BitWriter& writer, std::uint32_t value) {
  if (value >= kMaxUvlc) {
    return false;  // 2^32-1 is only reachable through the >=32-zero path.
  }
  std::uint32_t leading_zeros = 0;
  while (static_cast<std::uint64_t>(value) >=
         ((1ULL << (leading_zeros + 1U)) - 1ULL)) {
    ++leading_zeros;
    if (leading_zeros >= 32U) {
      return false;
    }
  }
  writer.put_bits(0U, leading_zeros);
  writer.put_bit(true);
  writer.put_bits(static_cast<std::uint64_t>(value) -
                      ((1ULL << leading_zeros) - 1ULL),
                  leading_zeros);
  return true;
}

// §4.11.10 rg(2) writer.
void put_rg(BitWriter& writer, std::uint32_t value, unsigned n) {
  const std::uint32_t q = value >> n;
  const std::uint32_t remainder = value & ((1UL << n) - 1UL);
  for (std::uint32_t index = 0; index < q; ++index) {
    writer.put_bit(true);
  }
  writer.put_bit(false);
  writer.put_bits(remainder, n);
}

std::string first_error(const ValidationResult& result) {
  for (const ValidationIssue& issue : result.issues()) {
    if (issue.severity == Severity::kError) {
      return issue.message;
    }
  }
  return "content interpretation parameters are invalid";
}

ContentInterpretation make_defaults() {
  ContentInterpretation params;
  params.scan_type = ScanType::kUnspecified;
  params.has_color_description = false;
  params.color = ColorDescription{};
  params.has_chroma_sample_position = false;
  params.chroma_top = ChromaSamplePosition::kUnspecified;
  params.chroma_bottom = ChromaSamplePosition::kUnspecified;
  params.has_aspect_ratio = false;
  params.aspect = AspectRatio{};
  params.has_timing_info = false;
  params.timing = TimingInfo{};
  return params;
}

}  // namespace

ChromaSampleOffset chroma_sample_offset(ChromaSamplePosition position,
                                        ChromaFormat format) {
  ChromaSampleOffset offset;
  switch (position) {
    case ChromaSamplePosition::kLeft:
      offset.applicable = true;
      offset.horizontal = 0.0;
      offset.vertical = format == ChromaFormat::k420 ? 0.5 : 0.0;
      return offset;
    case ChromaSamplePosition::kCenter:
      offset.applicable = true;
      offset.horizontal = 0.5;
      offset.vertical = format == ChromaFormat::k420 ? 0.5 : 0.0;
      return offset;
    case ChromaSamplePosition::kTopLeft:
      offset.applicable = format == ChromaFormat::k420;
      offset.horizontal = 0.0;
      offset.vertical = 0.0;
      return offset;
    case ChromaSamplePosition::kTop:
      offset.applicable = format == ChromaFormat::k420;
      offset.horizontal = 0.5;
      offset.vertical = 0.0;
      return offset;
    case ChromaSamplePosition::kBottomLeft:
      offset.applicable = format == ChromaFormat::k420;
      offset.horizontal = 0.0;
      offset.vertical = 1.0;
      return offset;
    case ChromaSamplePosition::kBottom:
      offset.applicable = format == ChromaFormat::k420;
      offset.horizontal = 0.5;
      offset.vertical = 1.0;
      return offset;
    case ChromaSamplePosition::kUnspecified:
      break;
  }
  return offset;
}

const char* chroma_sample_position_name(ChromaSamplePosition position) {
  switch (position) {
    case ChromaSamplePosition::kLeft:
      return "CSP_LEFT";
    case ChromaSamplePosition::kCenter:
      return "CSP_CENTER";
    case ChromaSamplePosition::kTopLeft:
      return "CSP_TOPLEFT";
    case ChromaSamplePosition::kTop:
      return "CSP_TOP";
    case ChromaSamplePosition::kBottomLeft:
      return "CSP_BOTTOMLEFT";
    case ChromaSamplePosition::kBottom:
      return "CSP_BOTTOM";
    case ChromaSamplePosition::kUnspecified:
      return "CSP_UNSPECIFIED";
  }
  return "unknown";
}

const char* scan_type_name(ScanType scan_type) {
  switch (scan_type) {
    case ScanType::kUnspecified:
      return "unspecified";
    case ScanType::kProgressive:
      return "progressive";
    case ScanType::kInterlaceField:
      return "interlace-field";
    case ScanType::kInterlaceComplementaryPair:
      return "interlace-complementary-pair";
  }
  return "unknown";
}

ColorDescription derive_color_description(std::uint8_t idc) {
  ColorDescription description;
  description.idc = idc;
  switch (idc) {
    case 1:  // BT.709 SDR
      description.primaries = 1;
      description.transfer = 1;
      description.matrix = 1;
      break;
    case 2:  // BT.2100 PQ
      description.primaries = 9;
      description.transfer = 16;
      description.matrix = 9;
      break;
    case 3:  // BT.2100 HLG
      description.primaries = 9;
      description.transfer = 18;
      description.matrix = 9;
      break;
    case 4:  // sRGB
      description.primaries = 1;
      description.transfer = 13;
      description.matrix = 0;
      break;
    case 5:  // sYCC
      description.primaries = 1;
      description.transfer = 13;
      description.matrix = 5;
      break;
    default:
      // 0 is explicit; 6..127 are reserved for AOMedia.
      break;
  }
  return description;
}

std::uint32_t AspectRatio::resolved_width() const {
  if (idc == 255U) {
    return sar_width;
  }
  if (idc < kAspectRatioWidth.size()) {
    return kAspectRatioWidth[idc];
  }
  return 0;
}

std::uint32_t AspectRatio::resolved_height() const {
  if (idc == 255U) {
    return sar_height;
  }
  if (idc < kAspectRatioHeight.size()) {
    return kAspectRatioHeight[idc];
  }
  return 0;
}

bool AspectRatio::resolved() const {
  return resolved_width() != 0U && resolved_height() != 0U;
}

ContentInterpretation default_content_interpretation() { return make_defaults(); }

void ValidationResult::add(Severity severity, std::string message) {
  issues_.push_back(ValidationIssue{severity, std::move(message)});
}

bool ValidationResult::has_errors() const {
  return std::any_of(issues_.begin(), issues_.end(),
                     [](const ValidationIssue& issue) {
                       return issue.severity == Severity::kError;
                     });
}

bool ValidationResult::has_warnings() const {
  return std::any_of(issues_.begin(), issues_.end(),
                     [](const ValidationIssue& issue) {
                       return issue.severity == Severity::kWarning;
                     });
}

std::string ValidationResult::summary() const {
  std::ostringstream out;
  for (const ValidationIssue& issue : issues_) {
    out << (issue.severity == Severity::kError ? "error: " : "warning: ")
        << issue.message;
    if (&issue != &issues_.back()) {
      out << '\n';
    }
  }
  return out.str();
}

ValidationResult validate_content_interpretation(
    const ContentInterpretation& params, ChromaFormat format, bool strict) {
  ValidationResult result;
  const auto warn_or_error = [&](bool hard, std::string message) {
    result.add(hard || strict ? Severity::kError : Severity::kWarning,
               std::move(message));
  };

  switch (params.scan_type) {
    case ScanType::kUnspecified:
    case ScanType::kProgressive:
    case ScanType::kInterlaceField:
    case ScanType::kInterlaceComplementaryPair:
      break;
    default:
      result.add(Severity::kError, "ci_scan_type_idc is outside 0..3");
      break;
  }

  // --- colour description ---------------------------------------------------
  if (params.has_color_description) {
    if (params.color.idc > 127U) {
      result.add(Severity::kError,
                 "ci_color_description_idc is outside 0..127 (rg(2))");
    } else if (params.color.idc >= 6U) {
      warn_or_error(false,
                    "ci_color_description_idc " +
                        std::to_string(params.color.idc) +
                        " is reserved for AOMedia (Table 6.13)");
    }

    if (params.color.idc == 0U) {
      if (!is_known_primaries(params.color.primaries)) {
        warn_or_error(false,
                      "ci_color_primaries " +
                          std::to_string(params.color.primaries) +
                          " is reserved/unknown (Table 6.14)");
      }
      if (!is_known_transfer(params.color.transfer)) {
        warn_or_error(false,
                      "ci_transfer_characteristics " +
                          std::to_string(params.color.transfer) +
                          " is reserved/unknown");
      }
      if (!is_known_matrix(params.color.matrix)) {
        warn_or_error(false,
                      "ci_matrix_coefficients " +
                          std::to_string(params.color.matrix) +
                          " is reserved/unknown (Table 6.15)");
      }
    } else {
      // Non-zero idc makes the explicit fields redundant.
      const ColorDescription derived =
          derive_color_description(params.color.idc);
      if (params.color.idc <= 5U &&
          (params.color.primaries != derived.primaries ||
           params.color.transfer != derived.transfer ||
           params.color.matrix != derived.matrix)) {
        result.add(Severity::kWarning,
                   "explicit primaries/transfer/matrix are ignored because "
                   "ci_color_description_idc is non-zero");
      }
    }
  } else {
    if (params.color != ColorDescription{}) {
      result.add(Severity::kWarning,
                 "colour description fields are ignored because "
                 "ci_color_description_present_flag is 0");
    }
  }

  // --- chroma sample position ----------------------------------------------
  if (params.has_chroma_sample_position) {
    const auto check_position = [&](ChromaSamplePosition position,
                                    const char* name) {
      const auto raw = static_cast<unsigned>(position);
      if (raw > 5U) {
        result.add(Severity::kError,
                   std::string(name) +
                       " must be <= 5 when present (§6.14); got " +
                       std::to_string(raw));
      } else if (!chroma_sample_offset(position, format).applicable) {
        warn_or_error(false,
                      std::string(name) + " " +
                          chroma_sample_position_name(position) +
                          " is N/A for the selected chroma format");
      }
    };

    if (params.scan_type == ScanType::kProgressive) {
      // Only ci_chroma_sample_position_top is coded; bottom := top.
      if (params.chroma_bottom != ChromaSamplePosition::kUnspecified &&
          params.chroma_bottom != params.chroma_top) {
        result.add(Severity::kError,
                   "progressive scan codes only "
                   "ci_chroma_sample_position_top; bottom must equal top");
      }
      check_position(params.chroma_top, "ci_chroma_sample_position_top");
    } else {
      check_position(params.chroma_top, "ci_chroma_sample_position_top");
      check_position(params.chroma_bottom, "ci_chroma_sample_position_bottom");
    }
  } else {
    if (params.chroma_top != ChromaSamplePosition::kUnspecified ||
        params.chroma_bottom != ChromaSamplePosition::kUnspecified) {
      result.add(Severity::kWarning,
                 "chroma sample positions are ignored because "
                 "ci_chroma_sample_position_present_flag is 0");
    }
  }

  // --- aspect ratio ---------------------------------------------------------
  if (params.has_aspect_ratio) {
    const std::uint8_t idc = params.aspect.idc;
    if (idc != 255U && idc > 16U) {
      result.add(Severity::kError,
                 "ci_aspect_ratio_idc must be <= 16 or 255 (§6.14); got " +
                     std::to_string(idc));
    }
    if (idc == 255U) {
      if (params.aspect.sar_width == 0U && params.aspect.sar_height == 0U) {
        result.add(Severity::kWarning,
                   "explicit SAR is 0:0, which means unspecified");
      } else if (params.aspect.sar_width == 0U ||
                 params.aspect.sar_height == 0U) {
        result.add(Severity::kWarning,
                   "one explicit SAR dimension is 0; the sample aspect ratio "
                   "is unspecified (§6.14)");
      }
    } else if (idc <= 16U && (params.aspect.sar_width != 0U ||
                              params.aspect.sar_height != 0U)) {
      if (params.aspect.sar_width !=
              kAspectRatioWidth[params.aspect.idc] ||
          params.aspect.sar_height !=
              kAspectRatioHeight[params.aspect.idc]) {
        result.add(Severity::kWarning,
                   "explicit SAR is ignored when ci_aspect_ratio_idc != 255; "
                   "the Aspect_Ratio_* table value is used");
      }
    }
  } else {
    if (params.aspect != AspectRatio{}) {
      result.add(Severity::kWarning,
                 "aspect ratio fields are ignored because "
                 "ci_aspect_ratio_info_present_flag is 0");
    }
  }

  // --- timing ---------------------------------------------------------------
  if (params.has_timing_info) {
    if (params.timing.num_units_in_display_tick == 0U) {
      result.add(Severity::kError,
                 "num_units_in_display_tick must be greater than 0 (§6.4.12)");
    }
    if (params.timing.time_scale == 0U) {
      result.add(Severity::kError, "time_scale must be greater than 0 (§6.4.12)");
    }
    if (params.timing.equal_picture_interval) {
      if (params.timing.num_ticks_per_picture_minus_1 > kMaxTicksMinus1) {
        result.add(Severity::kError,
                   "num_ticks_per_picture_minus_1 must be <= 2^32 - 2");
      }
    } else if (params.timing.num_ticks_per_picture_minus_1 != 0U) {
      result.add(Severity::kWarning,
                 "num_ticks_per_picture_minus_1 is not coded when "
                 "equal_picture_interval is 0");
    }
  } else {
    if (params.timing != TimingInfo{}) {
      result.add(Severity::kWarning,
                 "timing fields are ignored because "
                 "ci_timing_info_present_flag is 0");
    }
  }

  return result;
}

std::vector<std::uint8_t> serialize_content_interpretation_payload(
    const ContentInterpretation& params, std::string& error,
    ChromaFormat format, bool strict) {
  error.clear();
  const ValidationResult validation =
      validate_content_interpretation(params, format, strict);
  if (validation.has_errors()) {
    error = first_error(validation);
    return {};
  }

  BitWriter writer;
  writer.put_bits(static_cast<unsigned>(params.scan_type), 2U);
  writer.put_bit(params.has_color_description);
  writer.put_bit(params.has_chroma_sample_position);
  writer.put_bit(params.has_aspect_ratio);
  writer.put_bit(params.has_timing_info);
  writer.put_bits(0U, 2U);  // ci_reserved_2bit must be 0

  if (params.has_color_description) {
    put_rg(writer, params.color.idc, 2U);
    if (params.color.idc == 0U) {
      writer.put_bits(params.color.primaries, 8U);
      writer.put_bits(params.color.transfer, 8U);
      writer.put_bits(params.color.matrix, 8U);
    }
    writer.put_bit(params.color.full_range);
  }

  if (params.has_chroma_sample_position) {
    if (!put_uvlc(writer, static_cast<std::uint32_t>(params.chroma_top))) {
      error = "ci_chroma_sample_position_top is not uvlc()-encodable";
      return {};
    }
    if (params.scan_type != ScanType::kProgressive) {
      if (!put_uvlc(writer,
                    static_cast<std::uint32_t>(params.chroma_bottom))) {
        error = "ci_chroma_sample_position_bottom is not uvlc()-encodable";
        return {};
      }
    }
  }

  if (params.has_aspect_ratio) {
    writer.put_bits(params.aspect.idc, 8U);
    if (params.aspect.idc == 255U) {
      if (!put_uvlc(writer, params.aspect.sar_width) ||
          !put_uvlc(writer, params.aspect.sar_height)) {
        error = "explicit SAR is not uvlc()-encodable";
        return {};
      }
    }
  }

  if (params.has_timing_info) {
    writer.put_bits(params.timing.num_units_in_display_tick, 32U);
    writer.put_bits(params.timing.time_scale, 32U);
    writer.put_bit(params.timing.equal_picture_interval);
    if (params.timing.equal_picture_interval &&
        !put_uvlc(writer, params.timing.num_ticks_per_picture_minus_1)) {
      error = "num_ticks_per_picture_minus_1 is not uvlc()-encodable";
      return {};
    }
  }

  // §5.2.1 extensible OBU: obu_extension_flag = 0 then trailing_bits().
  writer.put_bit(false);
  writer.put_bit(true);
  writer.align_to_byte();
  return writer.take_bytes();
}

std::vector<std::uint8_t> make_content_interpretation_obu(
    const ContentInterpretation& params, std::string& error,
    std::size_t* payload_bits, ChromaFormat format, bool strict) {
  error.clear();
  const std::vector<std::uint8_t> payload =
      serialize_content_interpretation_payload(params, error, format, strict);
  if (payload.empty() && !error.empty()) {
    return {};
  }
  if (payload.size() >
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) - 1U) {
    error = "content interpretation OBU payload exceeds the Annex B.2 size field";
    return {};
  }
  if (payload_bits != nullptr) {
    *payload_bits = payload.size() * 8U;
  }
  m0::ObuHeader header;
  header.type = static_cast<m0::ObuType>(kObuTypeContentInterpretation);
  header.tlayer_id = 0;
  header.extension_flag = false;
  return m0::make_obu(header, payload);
}

bool parse_obu(std::span<const std::uint8_t> data, std::size_t& offset,
               ObuView& out, std::string& error) {
  error.clear();
  const std::size_t start = offset;
  if (start >= data.size()) {
    error = "no OBU at offset " + std::to_string(start);
    return false;
  }

  std::uint64_t size = 0;
  unsigned shift = 0;
  std::size_t prefix_bytes = 0;
  while (true) {
    if (start + prefix_bytes >= data.size()) {
      error = "truncated leb128 OBU size";
      return false;
    }
    if (prefix_bytes == 5U) {
      error = "leb128 OBU size longer than 5 bytes";
      return false;
    }
    const std::uint8_t byte = data[start + prefix_bytes];
    ++prefix_bytes;
    size |= static_cast<std::uint64_t>(byte & 0x7FU) << shift;
    if ((byte & 0x80U) == 0U) {
      if (prefix_bytes == 5U && (byte & 0x7FU) > 0x0FU) {
        error = "leb128 OBU size exceeds 32 bits";
        return false;
      }
      break;
    }
    shift += 7U;
  }

  if (size == 0U) {
    error = "OBU size must be at least the header byte";
    return false;
  }
  if (size > data.size() - start - prefix_bytes) {
    error = "OBU size extends past the end of the temporal unit";
    return false;
  }

  const std::size_t header_pos = start + prefix_bytes;
  const std::uint8_t header_byte = data[header_pos];
  const bool extension_flag = (header_byte & 0x80U) != 0U;
  const std::uint8_t type = static_cast<std::uint8_t>((header_byte >> 2U) & 0x1FU);
  const std::uint8_t tlayer_id = static_cast<std::uint8_t>(header_byte & 0x03U);
  std::size_t header_bytes = prefix_bytes + 1U;
  std::uint8_t mlayer_id = 0;
  std::uint8_t xlayer_id = 0;
  if (extension_flag) {
    if (size < 2U) {
      error = "OBU header extension flag set but the OBU is too short";
      return false;
    }
    const std::uint8_t extension = data[header_pos + 1U];
    mlayer_id = static_cast<std::uint8_t>(extension >> 5U);
    xlayer_id = static_cast<std::uint8_t>(extension & 0x1FU);
    header_bytes += 1U;
  }

  const std::size_t payload_bytes =
      static_cast<std::size_t>(size) - (header_bytes - prefix_bytes);

  out.type = type;
  out.tlayer_id = tlayer_id;
  out.extension_flag = extension_flag;
  out.mlayer_id = mlayer_id;
  out.xlayer_id = xlayer_id;
  out.offset = start;
  out.header_bytes = header_bytes;
  out.size = prefix_bytes + static_cast<std::size_t>(size);
  out.payload = data.subspan(start + header_bytes, payload_bytes);
  offset = start + out.size;
  return true;
}

bool parse_obu_sequence(std::span<const std::uint8_t> data,
                        std::vector<ObuView>& out, std::string& error) {
  out.clear();
  std::size_t offset = 0;
  while (offset < data.size()) {
    ObuView view;
    if (!parse_obu(data, offset, view, error)) {
      return false;
    }
    out.push_back(view);
  }
  return true;
}

bool parse_content_interpretation(std::span<const std::uint8_t> payload,
                                  ContentInterpretationView& out,
                                  std::string& error, bool strict,
                                  ChromaFormat format) {
  error.clear();
  out = ContentInterpretationView{};
  BitReader reader(payload);

  std::uint32_t scan_type = 0;
  bool color_present = false;
  bool chroma_present = false;
  bool aspect_present = false;
  bool timing_present = false;
  std::uint32_t reserved = 0;
  if (!reader.read_bits(2U, scan_type, error) ||
      !reader.read_bit(color_present, error) ||
      !reader.read_bit(chroma_present, error) ||
      !reader.read_bit(aspect_present, error) ||
      !reader.read_bit(timing_present, error) ||
      !reader.read_bits(2U, reserved, error)) {
    return false;
  }

  ContentInterpretation params = make_defaults();
  params.scan_type = static_cast<ScanType>(scan_type);
  params.has_color_description = color_present;
  params.has_chroma_sample_position = chroma_present;
  params.has_aspect_ratio = aspect_present;
  params.has_timing_info = timing_present;
  out.reserved_bits = static_cast<std::uint8_t>(reserved);

  if (color_present) {
    std::uint32_t idc = 0;
    if (!reader.read_rg(2U, idc, error)) {
      return false;
    }
    params.color.idc = static_cast<std::uint8_t>(idc);
    if (idc == 0U) {
      std::uint32_t primaries = 0;
      std::uint32_t transfer = 0;
      std::uint32_t matrix = 0;
      if (!reader.read_bits(8U, primaries, error) ||
          !reader.read_bits(8U, transfer, error) ||
          !reader.read_bits(8U, matrix, error)) {
        return false;
      }
      params.color.primaries = static_cast<std::uint8_t>(primaries);
      params.color.transfer = static_cast<std::uint8_t>(transfer);
      params.color.matrix = static_cast<std::uint8_t>(matrix);
    } else {
      const ColorDescription derived = derive_color_description(
          static_cast<std::uint8_t>(idc));
      params.color.primaries = derived.primaries;
      params.color.transfer = derived.transfer;
      params.color.matrix = derived.matrix;
    }
    bool full_range = false;
    if (!reader.read_bit(full_range, error)) {
      return false;
    }
    params.color.full_range = full_range;
  }

  if (chroma_present) {
    std::uint32_t top = 0;
    if (!reader.read_uvlc(top, error)) {
      return false;
    }
    if (top > static_cast<std::uint32_t>(ChromaSamplePosition::kUnspecified)) {
      error = "ci_chroma_sample_position_top is outside 0..6";
      return false;
    }
    params.chroma_top = static_cast<ChromaSamplePosition>(top);
    if (params.scan_type != ScanType::kProgressive) {
      std::uint32_t bottom = 0;
      if (!reader.read_uvlc(bottom, error)) {
        return false;
      }
      if (bottom >
          static_cast<std::uint32_t>(ChromaSamplePosition::kUnspecified)) {
        error = "ci_chroma_sample_position_bottom is outside 0..6";
        return false;
      }
      params.chroma_bottom = static_cast<ChromaSamplePosition>(bottom);
      out.chroma_bottom_signalled = true;
    } else {
      params.chroma_bottom = params.chroma_top;
    }
  }

  if (aspect_present) {
    std::uint32_t idc = 0;
    if (!reader.read_bits(8U, idc, error)) {
      return false;
    }
    params.aspect.idc = static_cast<std::uint8_t>(idc);
    if (idc == 255U) {
      std::uint32_t width = 0;
      std::uint32_t height = 0;
      if (!reader.read_uvlc(width, error) ||
          !reader.read_uvlc(height, error)) {
        return false;
      }
      params.aspect.sar_width = width;
      params.aspect.sar_height = height;
    } else if (idc <= 16U) {
      params.aspect.sar_width = kAspectRatioWidth[idc];
      params.aspect.sar_height = kAspectRatioHeight[idc];
    }
  }

  if (timing_present) {
    std::uint32_t num_units = 0;
    std::uint32_t time_scale = 0;
    bool equal_picture_interval = false;
    if (!reader.read_bits(32U, num_units, error) ||
        !reader.read_bits(32U, time_scale, error) ||
        !reader.read_bit(equal_picture_interval, error)) {
      return false;
    }
    params.timing.num_units_in_display_tick = num_units;
    params.timing.time_scale = time_scale;
    params.timing.equal_picture_interval = equal_picture_interval;
    if (equal_picture_interval) {
      std::uint32_t ticks = 0;
      if (!reader.read_uvlc(ticks, error)) {
        return false;
      }
      params.timing.num_ticks_per_picture_minus_1 = ticks;
    }
  }

  out.syntax_bits = reader.bits_read();

  // §5.2.1 extensible-OBU payload.
  bool extension_flag = false;
  if (!reader.read_bit(extension_flag, error)) {
    return false;
  }
  if (extension_flag) {
    error = "content interpretation OBU uses obu_extension_data(), which this "
            "module does not parse";
    return false;
  }
  if (!reader.read_trailing_bits(error)) {
    return false;
  }

  if (strict) {
    // Enforce the hard §6.14 / §6.4.12 conformance requirements. Reserved
    // codepoints stay warnings: §6.14 requires decoders to ignore them, so a
    // reader must not reject them.
    const ValidationResult validation =
        validate_content_interpretation(params, format, false);
    if (validation.has_errors()) {
      error = first_error(validation);
      return false;
    }
  }
  out.params = params;
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

bool is_tu_prefix_obu_type(std::uint8_t type) {
  switch (type) {
    case 1:   // OBU_SEQUENCE_HEADER
    case 2:   // OBU_TEMPORAL_DELIMITER
    case 16:  // OBU_LAYER_CONFIGURATION_RECORD
    case 17:  // OBU_ATLAS_SEGMENT
    case 18:  // OBU_OPERATING_POINT_SET
    case 20:  // OBU_MSDO
      return true;
    default:
      return false;
  }
}

bool scan_content_interpretation(std::span<const std::uint8_t> temporal_unit,
                                 ContentInterpretationScanResult& out,
                                 std::string& error, bool strict,
                                 ChromaFormat format) {
  out = ContentInterpretationScanResult{};
  std::vector<ObuView> obus;
  if (!parse_obu_sequence(temporal_unit, obus, error)) {
    return false;
  }

  for (const ObuView& obu : obus) {
    if (obu.type == 1U) {
      out.has_sequence_header = true;
    }
    if (is_frame_obu_type(obu.type)) {
      ++out.frame_obus;
      if (!out.has_first_frame) {
        out.has_first_frame = true;
        out.first_frame_offset = obu.offset;
      }
    }
    if (obu.type != kObuTypeContentInterpretation) {
      continue;
    }
    ++out.ci_obus;
    ContentInterpretationRecord record;
    record.obu_offset = obu.offset;
    record.obu_size = obu.size;
    record.before_first_frame = !out.has_first_frame;

    // §7.3.3: at most one CI OBU per coded output frame unit, as its first
    // OBU.
    bool bad = out.ci_obus > 1U || !record.before_first_frame;
    if (!parse_content_interpretation(obu.payload, record.view, error, strict,
                                      format)) {
      bad = true;
      error.clear();
    } else {
      record.valid = true;
    }
    if (bad) {
      ++out.invalid_ci_obus;
    }
    out.records.push_back(record);
  }
  return true;
}

bool insert_content_interpretation(std::span<const std::uint8_t> temporal_unit,
                                   std::span<const std::uint8_t> ci_obu,
                                   std::vector<std::uint8_t>& out,
                                   std::string& error,
                                   std::size_t* inserted_at) {
  error.clear();
  out.assign(temporal_unit.begin(), temporal_unit.end());

  std::vector<ObuView> obus;
  if (!parse_obu_sequence(temporal_unit, obus, error)) {
    return false;
  }
  if (obus.empty()) {
    error = "temporal unit contains no OBUs";
    return false;
  }
  for (const ObuView& obu : obus) {
    if (obu.type == kObuTypeContentInterpretation) {
      error =
          "temporal unit already contains a content interpretation OBU; "
          "§7.3.3 allows at most one per coded output frame unit";
      return false;
    }
  }

  std::size_t first_frame = obus.size();
  for (std::size_t index = 0; index < obus.size(); ++index) {
    if (is_frame_obu_type(obus[index].type)) {
      first_frame = index;
      break;
    }
  }
  if (first_frame == obus.size()) {
    error = "temporal unit has no frame OBU; CI OBU placement is undefined";
    return false;
  }

  std::size_t last_prefix = obus.size();
  bool has_sequence_header = false;
  bool has_sequence_xlayer = false;
  std::uint8_t sequence_xlayer = 0;
  for (std::size_t index = 0; index < first_frame; ++index) {
    const ObuView& obu = obus[index];
    if (is_tu_prefix_obu_type(obu.type)) {
      last_prefix = index;
    }
    if (obu.type == 1U) {
      has_sequence_header = true;
      if (!has_sequence_xlayer) {
        has_sequence_xlayer = true;
        sequence_xlayer = obu.xlayer_id;
      } else if (obu.xlayer_id != sequence_xlayer) {
        error =
            "multiple sequence headers with different obu_xlayer_id; the CI "
            "insertion point is ambiguous";
        return false;
      }
    }
    // Global metadata is part of the temporal unit, not of a coded extended
    // layer unit, and must stay ahead of every sequence header.
    if ((obu.type == 8U || obu.type == 9U) && obu.extension_flag &&
        obu.xlayer_id == 31U) {
      error =
          "global metadata OBU precedes the coded extended layer unit; the CI "
          "insertion point is ambiguous in this single-layer helper";
      return false;
    }
  }
  if (!has_sequence_header) {
    error = "temporal unit has no sequence header before the frame OBU";
    return false;
  }
  if (last_prefix == obus.size()) {
    error = "temporal unit has no prefix OBU before the frame unit";
    return false;
  }

  const std::size_t insert_offset = obus[last_prefix + 1U].offset;
  std::vector<std::uint8_t> result;
  result.reserve(temporal_unit.size() + ci_obu.size());
  result.insert(result.end(), temporal_unit.begin(),
                temporal_unit.begin() + static_cast<std::ptrdiff_t>(insert_offset));
  result.insert(result.end(), ci_obu.begin(), ci_obu.end());
  result.insert(result.end(),
                temporal_unit.begin() + static_cast<std::ptrdiff_t>(insert_offset),
                temporal_unit.end());
  out = std::move(result);
  if (inserted_at != nullptr) {
    *inserted_at = insert_offset;
  }
  return true;
}

std::vector<std::uint8_t> make_nonconforming_chroma_probe_obu(
    std::string& error) {
  error.clear();
  BitWriter writer;
  writer.put_bits(static_cast<unsigned>(ScanType::kProgressive), 2U);
  writer.put_bit(false);  // ci_color_description_present_flag
  writer.put_bit(true);   // ci_chroma_sample_position_present_flag
  writer.put_bit(false);  // ci_aspect_ratio_info_present_flag
  writer.put_bit(false);  // ci_timing_info_present_flag
  writer.put_bits(0U, 2U);
  // uvlc(6): 00 1 11. §6.14 requires <= 5 when present.
  writer.put_bits(0U, 2U);
  writer.put_bit(true);
  writer.put_bits(3U, 2U);
  writer.put_bit(false);  // obu_extension_flag
  writer.put_bit(true);   // trailing_bits
  writer.align_to_byte();

  m0::ObuHeader header;
  header.type = static_cast<m0::ObuType>(kObuTypeContentInterpretation);
  header.tlayer_id = 0;
  header.extension_flag = false;
  return m0::make_obu(header, writer.take_bytes());
}

void CvsConstancyTracker::reset() {
  layers_.clear();
  timing_established_ = false;
  timing_ = TimingInfo{};
}

bool CvsConstancyTracker::established(std::uint8_t mlayer) const {
  const auto it = layers_.find(mlayer);
  return it != layers_.end() && it->second.established;
}

ContentInterpretation CvsConstancyTracker::params(std::uint8_t mlayer) const {
  const auto it = layers_.find(mlayer);
  if (it == layers_.end()) {
    return make_defaults();
  }
  return it->second.params;
}

bool CvsConstancyTracker::observe(const CiObservation& observation,
                                  std::string& error) {
  error.clear();
  LayerState& state = layers_[observation.mlayer];

  if (observation.is_first_tu_of_cvs) {
    state = LayerState{};
    state.first_tu_seen = true;
    state.first_tu_had_ci = observation.params != nullptr;
    if (observation.params != nullptr) {
      state.established = true;
      state.params = *observation.params;
    }
  } else {
    if (!state.first_tu_seen) {
      error =
          "CI constancy tracker: the first observation for embedded layer " +
          std::to_string(observation.mlayer) +
          " must be the first temporal unit of the coded video sequence";
      return false;
    }
    if (observation.params != nullptr) {
      if (!state.first_tu_had_ci) {
        error =
            "CI OBU present in a later temporal unit but absent from the first "
            "temporal unit of the coded video sequence (§7.3.8.10)";
        return false;
      }
      if (!state.established) {
        state.established = true;
        state.params = *observation.params;
      } else if (observation.has_random_access) {
        // A new random access point re-initialises the parameters.
        state.params = *observation.params;
      } else if (!(*observation.params == state.params)) {
        error =
            "content interpretation parameters changed within the coded video "
            "sequence (§6.14); all instances must be identical";
        return false;
      }
    } else if (observation.has_random_access) {
      // §7.3.8.11: with no CI OBU at the random access point the parameters
      // fall back to the defaults (single embedded layer).
      state.established = true;
      state.params = make_defaults();
    }
  }

  if (observation.params != nullptr && observation.params->has_timing_info) {
    if (!timing_established_) {
      timing_established_ = true;
      timing_ = observation.params->timing;
    } else if (!(observation.params->timing == timing_)) {
      error =
          "timing_info() differs across embedded layers within the coded "
          "video sequence (§6.14)";
      return false;
    }
  }
  return true;
}

ValidationResult check_cvs_constancy(
    std::span<const CiObservation> observations) {
  ValidationResult result;
  CvsConstancyTracker tracker;
  for (const CiObservation& observation : observations) {
    std::string error;
    if (!tracker.observe(observation, error)) {
      result.add(Severity::kError, std::move(error));
    }
  }
  return result;
}

}  // namespace av2enc::ci
