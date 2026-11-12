#include "codec/rate/rc/hrd/hrd.h"

#include "core/bitstream/basic_headers.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace av2enc::hrd {
namespace {

void set_error(std::string* error, const std::string& message) {
  if (error != nullptr) *error = message;
}

// uvlc() (§4.11.3). Maximum representable value is (1 << 32) - 2.
bool put_uvlc(BitWriter& writer, std::uint64_t value, std::string* error) {
  if (value > kMaxUvlc) {
    set_error(error, "uvlc() value exceeds (1 << 32) - 2");
    return false;
  }
  unsigned leading_zeros = 0;
  while (value >= ((1ULL << (leading_zeros + 1U)) - 1ULL)) {
    ++leading_zeros;
  }
  writer.put_bits(0U, leading_zeros);
  writer.put_bit(true);
  const std::uint64_t payload =
      value - ((1ULL << leading_zeros) - 1ULL);
  writer.put_bits(payload, leading_zeros);
  return true;
}

// rg(n) (§4.11.10). q is limited to 31, so values >= (32 << n) are not
// representable. Only used with n = 4 for BRT here.
bool put_rg(BitWriter& writer, std::uint32_t value, unsigned n,
            std::string* error) {
  if (n == 0U || n > 24U) {
    set_error(error, "rg(n) parameter out of range");
    return false;
  }
  const std::uint32_t q = value >> n;
  if (q >= 32U) {
    set_error(error, "rg() value does not fit in 31 unary bits");
    return false;
  }
  for (std::uint32_t i = 0; i < q; ++i) writer.put_bit(true);
  writer.put_bit(false);
  writer.put_bits(value & ((1U << n) - 1U), n);
  return true;
}

std::uint32_t ceil_log2(std::uint32_t value) {
  std::uint32_t bits = 0;
  std::uint32_t v = value - 1U;
  while (v != 0U) {
    v >>= 1U;
    ++bits;
  }
  return bits;
}

bool consume_trailing_bits(BitReader& reader, std::string* error) {
  bool bit = false;
  if (!reader.read_bit(bit)) {
    set_error(error, "truncated trailing_bits()");
    return false;
  }
  if (!bit) {
    set_error(error, "trailing_bits() does not start with a one bit");
    return false;
  }
  while (reader.bits_remaining() > 0U) {
    if (!reader.read_bit(bit)) {
      set_error(error, "truncated trailing_bits() padding");
      return false;
    }
    if (bit) {
      set_error(error, "trailing_bits() padding contains a non-zero bit");
      return false;
    }
  }
  return true;
}

// Reads the §5.4.1 prefix through seq_cropping_window_present_flag and its
// optional offsets. `timing_position` is the bit position of the timing block
// (the position right after the cropping fields).
struct SequencePrefix {
  bool single_picture_header_flag = false;
  std::size_t timing_position = 0;
};

bool read_sequence_prefix(BitReader& reader, SequencePrefix& prefix,
                          std::string* error) {
  std::uint32_t value = 0;
  if (!reader.read_uvlc(value, error)) return false;  // seq_header_id
  if (!reader.read_bits(5, value)) {                  // seq_profile_idc
    set_error(error, "truncated seq_profile_idc");
    return false;
  }
  bool single = false;
  if (!reader.read_bit(single)) {
    set_error(error, "truncated single_picture_header_flag");
    return false;
  }
  std::uint32_t level = 0;
  if (!reader.read_bits(5, level)) {  // seq_level_idx
    set_error(error, "truncated seq_level_idx");
    return false;
  }
  if (level > 3U && !single) {
    bool tier = false;
    if (!reader.read_bit(tier)) {  // seq_tier
      set_error(error, "truncated seq_tier");
      return false;
    }
  }
  if (!reader.read_uvlc(value, error)) return false;  // chroma_format_idc
  if (!reader.read_uvlc(value, error)) return false;  // bit_depth_idc
  prefix.single_picture_header_flag = single;
  if (single) {
    // Still picture: no seq_lcr_id/still_picture/max_* fields and no timing
    // block. Frame size follows.
    prefix.timing_position = reader.bit_position();
  } else {
    if (!reader.read_bits(3, value)) {  // seq_lcr_id
      set_error(error, "truncated seq_lcr_id");
      return false;
    }
    bool still = false;
    if (!reader.read_bit(still)) {  // still_picture
      set_error(error, "truncated still_picture");
      return false;
    }
    std::uint32_t max_tlayer = 0;
    std::uint32_t max_mlayer = 0;
    if (!reader.read_bits(2, max_tlayer) || !reader.read_bits(3, max_mlayer)) {
      set_error(error, "truncated max_tlayer_id/max_mlayer_id");
      return false;
    }
    if (max_mlayer > 0U) {
      const std::uint32_t n = ceil_log2(max_mlayer + 1U);
      if (!reader.read_bits(n, value)) {  // seq_max_mlayer_cnt_minus_1
        set_error(error, "truncated seq_max_mlayer_cnt_minus_1");
        return false;
      }
    }
    bool monotonic = false;
    if (!reader.read_bit(monotonic)) {  // monotonic_output_order_flag
      set_error(error, "truncated monotonic_output_order_flag");
      return false;
    }
  }

  std::uint32_t width_bits_minus_1 = 0;
  std::uint32_t height_bits_minus_1 = 0;
  if (!reader.read_bits(4, width_bits_minus_1) ||
      !reader.read_bits(4, height_bits_minus_1)) {
    set_error(error, "truncated frame_width_bits_minus_1/frame_height_bits_minus_1");
    return false;
  }
  if (!reader.read_bits(width_bits_minus_1 + 1U, value) ||
      !reader.read_bits(height_bits_minus_1 + 1U, value)) {
    set_error(error, "truncated max_frame_width/height");
    return false;
  }
  bool cropping = false;
  if (!reader.read_bit(cropping)) {  // seq_cropping_window_present_flag
    set_error(error, "truncated seq_cropping_window_present_flag");
    return false;
  }
  if (cropping) {
    for (int i = 0; i < 4; ++i) {
      if (!reader.read_uvlc(value, error)) return false;
    }
  }
  prefix.timing_position = reader.bit_position();
  return true;
}

bool append_payload_bits(BitWriter& writer, std::span<const std::uint8_t> payload,
                         std::size_t begin, std::size_t bit_count) {
  if (begin > payload.size() * 8U || bit_count > payload.size() * 8U - begin) {
    return false;
  }
  for (std::size_t i = 0; i < bit_count; ++i) {
    const std::size_t pos = begin + i;
    const std::size_t byte = pos >> 3U;
    const unsigned shift = 7U - static_cast<unsigned>(pos & 7U);
    writer.put_bit(((payload[byte] >> shift) & 1U) != 0U);
  }
  return true;
}

// Locates the end of the syntax bits in an extensible OBU payload. The tail is
//   [syntax bits] obu_extension_flag(0) trailing_one_bit(1) zero padding
// so the last set bit in the payload is the trailing one and the bit before it
// is the (zero) extension flag.
bool extensible_syntax_bits(std::span<const std::uint8_t> payload,
                            std::size_t& syntax_bits, std::string* error) {
  std::size_t last_set = 0;
  bool found = false;
  for (std::size_t bit = 0; bit < payload.size() * 8U; ++bit) {
    const std::size_t byte = bit >> 3U;
    const unsigned shift = 7U - static_cast<unsigned>(bit & 7U);
    if (((payload[byte] >> shift) & 1U) != 0U) {
      last_set = bit;
      found = true;
    }
  }
  if (!found || last_set == 0U) {
    set_error(error, "OBU payload has no trailing one bit");
    return false;
  }
  const std::size_t extension_pos = last_set - 1U;
  const std::size_t byte = extension_pos >> 3U;
  const unsigned shift = 7U - static_cast<unsigned>(extension_pos & 7U);
  if (((payload[byte] >> shift) & 1U) != 0U) {
    set_error(error, "OBU carries obu_extension_data, which is not supported here");
    return false;
  }
  syntax_bits = extension_pos;
  return true;
}

std::vector<std::uint8_t> make_obu_bytes(ObuType type, std::uint8_t tlayer_id,
                                         std::span<const std::uint8_t> payload) {
  av2enc::basic::ObuHeader header;
  header.type = static_cast<av2enc::basic::ObuType>(type);
  header.tlayer_id = tlayer_id;
  header.extension_flag = false;
  return av2enc::basic::make_obu(header, payload);
}

bool single_obu_of_type(std::span<const std::uint8_t> bytes, ObuType expected,
                        ParsedObu& obu, std::string* error) {
  std::vector<ParsedObu> obus;
  if (!parse_obus(bytes, obus, error)) return false;
  if (obus.size() != 1U) {
    set_error(error, "expected exactly one OBU");
    return false;
  }
  if (obus[0].type != expected) {
    set_error(error, std::string("unexpected OBU type: expected ") +
                         obu_type_name(expected) + ", got " +
                         obu_type_name(obus[0].type));
    return false;
  }
  obu = obus[0];
  return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Limits / validation
// ---------------------------------------------------------------------------

bool timing_info_valid(const TimingInfo& timing, std::string* error) {
  if (timing.num_units_in_display_tick == 0U) {
    set_error(error, "num_units_in_display_tick shall be greater than 0");
    return false;
  }
  if (timing.time_scale == 0U) {
    set_error(error, "time_scale shall be greater than 0");
    return false;
  }
  if (timing.equal_picture_interval &&
      timing.num_ticks_per_picture_minus_1 > kMaxUvlc) {
    set_error(error, "num_ticks_per_picture_minus_1 exceeds uvlc() range");
    return false;
  }
  return true;
}

double display_tick_seconds(const TimingInfo& timing,
                            std::string* error) noexcept {
  if (!timing_info_valid(timing, error)) return 0.0;
  return static_cast<double>(timing.num_units_in_display_tick) /
         static_cast<double>(timing.time_scale);
}

bool seq_decoder_model_info_valid(const SeqDecoderModelInfo& info,
                                  std::string* error) {
  if (info.decoder_buffer_delay > kMaxUvlc) {
    set_error(error, "decoder_buffer_delay exceeds uvlc() range");
    return false;
  }
  if (info.encoder_buffer_delay > kMaxUvlc) {
    set_error(error, "encoder_buffer_delay exceeds uvlc() range");
    return false;
  }
  return true;
}

bool sequence_timing_valid(const SequenceTimingConfig& config,
                           std::string* error) {
  if (config.seq_initial_display_delay_minus_1 > 15U) {
    set_error(error, "seq_initial_display_delay_minus_1 is f(4)");
    return false;
  }
  if (config.seq_decoder_model_info_present_flag &&
      !config.decoder_model_info_present_flag) {
    set_error(error,
              "seq_decoder_model_info_present_flag requires "
              "decoder_model_info_present_flag");
    return false;
  }
  if (config.decoder_model_info_present_flag) {
    if (config.num_units_in_decoding_tick == 0U) {
      set_error(error, "num_units_in_decoding_tick shall be greater than 0");
      return false;
    }
    if (config.seq_decoder_model_info_present_flag &&
        !seq_decoder_model_info_valid(config.decoder_model_info, error)) {
      return false;
    }
  }
  return true;
}

bool content_interpretation_valid(const ContentInterpretation& ci,
                                  std::string* error) {
  if (ci.ci_scan_type_idc > 3U) {
    set_error(error, "ci_scan_type_idc is f(2)");
    return false;
  }
  if (ci.ci_color_description_idc > 127U) {
    set_error(error, "ci_color_description_idc is rg(2) (0..127)");
    return false;
  }
  if (ci.ci_chroma_sample_position_present_flag) {
    if (ci.ci_chroma_sample_position_top > 5U ||
        ci.ci_chroma_sample_position_bottom > 5U) {
      set_error(error, "chroma sample position shall be <= 5");
      return false;
    }
  }
  if (ci.ci_aspect_ratio_info_present_flag) {
    if (ci.ci_aspect_ratio_idc > 16U && ci.ci_aspect_ratio_idc != 255U) {
      set_error(error, "ci_aspect_ratio_idc shall be <= 16 or 255");
      return false;
    }
  }
  if (ci.ci_timing_info_present_flag &&
      !timing_info_valid(ci.timing, error)) {
    return false;
  }
  return true;
}

bool buffer_removal_timing_valid(const BufferRemovalTiming& brt,
                                 std::string* error) {
  if (!brt.br_ops_dependent_flag) {
    if (brt.br_time > kMaxBrTime) {
      set_error(error, "br_time does not fit in rg(4) (max 511)");
      return false;
    }
    return true;
  }
  if (brt.br_ops_id > kMaxOpsId) {
    set_error(error, "br_ops_id is f(4)");
    return false;
  }
  if (brt.op_times.empty() || brt.op_times.size() > kMaxOpsCnt) {
    set_error(error, "br_ops_cnt shall be in 1..7 for an active OPS");
    return false;
  }
  for (std::size_t i = 0; i < brt.op_times.size(); ++i) {
    const BrOpTime& entry = brt.op_times[i];
    if (!entry.br_decoder_model_present_op_flag) continue;
    if (entry.br_time > kMaxBrTime) {
      set_error(error, "br_time_op[" + std::to_string(i) +
                           "] does not fit in rg(4) (max 511)");
      return false;
    }
  }
  return true;
}

bool validate_brt_tick_ranges(const BufferRemovalTiming& brt,
                              std::string* error) {
  return buffer_removal_timing_valid(brt, error);
}

bool validate_brt_against_ops(
    const BufferRemovalTiming& brt, std::uint8_t brt_xlayer_id,
    std::span<const OperatingPointSetSummary> ops, std::string* error) {
  if (!validate_brt_tick_ranges(brt, error)) return false;
  if (!brt.br_ops_dependent_flag) {
    // The correspondence requirement only applies to br_time_op values.
    return true;
  }
  if (brt_xlayer_id > 31U) {
    set_error(error, "OBU extended layer id is f(5)");
    return false;
  }
  for (const OperatingPointSetSummary& summary : ops) {
    if (summary.ops_id > kMaxOpsId) continue;
    if (summary.ops_cnt == 0U || summary.ops_cnt > kMaxOpsCnt) continue;
    if (summary.xlayer_id != brt_xlayer_id) continue;
    if (summary.ops_id != brt.br_ops_id) continue;
    if (summary.ops_cnt != brt.op_times.size()) {
      set_error(error,
                "br_ops_cnt[" + std::to_string(brt.br_ops_id) +
                    "] = " + std::to_string(brt.op_times.size()) +
                    " does not match ops_cnt = " +
                    std::to_string(summary.ops_cnt));
      return false;
    }
    return true;
  }
  set_error(error,
            "br_ops_id " + std::to_string(brt.br_ops_id) +
                " does not match an active OPS for xlayer " +
                std::to_string(brt_xlayer_id));
  return false;
}

// ---------------------------------------------------------------------------
// Syntax writers
// ---------------------------------------------------------------------------

bool write_timing_info(BitWriter& writer, const TimingInfo& timing,
                       std::string* error) {
  if (!timing_info_valid(timing, error)) return false;
  writer.put_bits(timing.num_units_in_display_tick, 32U);
  writer.put_bits(timing.time_scale, 32U);
  writer.put_bit(timing.equal_picture_interval);
  if (timing.equal_picture_interval) {
    if (!put_uvlc(writer, timing.num_ticks_per_picture_minus_1, error)) {
      return false;
    }
  }
  return true;
}

bool write_seq_decoder_model_info(BitWriter& writer,
                                  const SeqDecoderModelInfo& info,
                                  std::string* error) {
  if (!seq_decoder_model_info_valid(info, error)) return false;
  if (!put_uvlc(writer, info.decoder_buffer_delay, error)) return false;
  if (!put_uvlc(writer, info.encoder_buffer_delay, error)) return false;
  writer.put_bit(info.low_delay_mode_flag);
  return true;
}

bool write_sequence_timing(BitWriter& writer,
                           const SequenceTimingConfig& config,
                           std::string* error) {
  if (!sequence_timing_valid(config, error)) return false;
  writer.put_bit(config.seq_initial_display_delay_present_flag);
  if (config.seq_initial_display_delay_present_flag) {
    writer.put_bits(config.seq_initial_display_delay_minus_1, 4U);
  }
  writer.put_bit(config.decoder_model_info_present_flag);
  if (config.decoder_model_info_present_flag) {
    writer.put_bits(config.num_units_in_decoding_tick, 32U);
    writer.put_bit(config.seq_decoder_model_info_present_flag);
    if (config.seq_decoder_model_info_present_flag) {
      if (!write_seq_decoder_model_info(writer, config.decoder_model_info,
                                        error)) {
        return false;
      }
    }
  }
  return true;
}

bool write_buffer_removal_timing(BitWriter& writer,
                                 const BufferRemovalTiming& brt,
                                 std::string* error) {
  if (!buffer_removal_timing_valid(brt, error)) return false;
  writer.put_bit(brt.br_ops_dependent_flag);
  if (!brt.br_ops_dependent_flag) {
    return put_rg(writer, brt.br_time, 4U, error);
  }
  writer.put_bits(brt.br_ops_id, 4U);
  writer.put_bits(static_cast<std::uint32_t>(brt.op_times.size()), 3U);
  for (const BrOpTime& entry : brt.op_times) {
    writer.put_bit(entry.br_decoder_model_present_op_flag);
    if (!entry.br_decoder_model_present_op_flag) continue;
    if (!put_rg(writer, entry.br_time, 4U, error)) return false;
  }
  return true;
}

bool write_content_interpretation(BitWriter& writer,
                                  const ContentInterpretation& ci,
                                  std::string* error) {
  if (!content_interpretation_valid(ci, error)) return false;
  writer.put_bits(ci.ci_scan_type_idc, 2U);
  writer.put_bit(ci.ci_color_description_present_flag);
  writer.put_bit(ci.ci_chroma_sample_position_present_flag);
  writer.put_bit(ci.ci_aspect_ratio_info_present_flag);
  writer.put_bit(ci.ci_timing_info_present_flag);
  writer.put_bits(0U, 2U);  // ci_reserved_2bit

  if (ci.ci_color_description_present_flag) {
    if (!put_rg(writer, ci.ci_color_description_idc, 2U, error)) return false;
    if (ci.ci_color_description_idc == 0U) {
      writer.put_bits(ci.ci_color_primaries, 8U);
      writer.put_bits(ci.ci_transfer_characteristics, 8U);
      writer.put_bits(ci.ci_matrix_coefficients, 8U);
    }
    writer.put_bit(ci.ci_full_range_flag);
  }
  if (ci.ci_chroma_sample_position_present_flag) {
    if (!put_uvlc(writer, ci.ci_chroma_sample_position_top, error)) return false;
    if (ci.ci_scan_type_idc != 1U) {
      if (!put_uvlc(writer, ci.ci_chroma_sample_position_bottom, error)) {
        return false;
      }
    }
  }
  if (ci.ci_aspect_ratio_info_present_flag) {
    writer.put_bits(ci.ci_aspect_ratio_idc, 8U);
    if (ci.ci_aspect_ratio_idc == 255U) {
      if (!put_uvlc(writer, ci.ci_sar_width, error)) return false;
      if (!put_uvlc(writer, ci.ci_sar_height, error)) return false;
    }
  }
  if (ci.ci_timing_info_present_flag) {
    return write_timing_info(writer, ci.timing, error);
  }
  return true;
}

// ---------------------------------------------------------------------------
// Syntax readers
// ---------------------------------------------------------------------------

bool read_timing_info(BitReader& reader, TimingInfo& timing,
                      std::string* error) {
  std::uint32_t value = 0;
  if (!reader.read_bits(32, timing.num_units_in_display_tick)) {
    set_error(error, "truncated num_units_in_display_tick");
    return false;
  }
  if (!reader.read_bits(32, timing.time_scale)) {
    set_error(error, "truncated time_scale");
    return false;
  }
  bool equal = false;
  if (!reader.read_bit(equal)) {
    set_error(error, "truncated equal_picture_interval");
    return false;
  }
  timing.equal_picture_interval = equal;
  timing.num_ticks_per_picture_minus_1 = 0;
  if (equal) {
    if (!reader.read_uvlc(value, error)) return false;
    timing.num_ticks_per_picture_minus_1 = value;
  }
  if (!timing_info_valid(timing, error)) return false;
  return true;
}

bool read_seq_decoder_model_info(BitReader& reader, SeqDecoderModelInfo& info,
                                 std::string* error) {
  std::uint32_t value = 0;
  if (!reader.read_uvlc(value, error)) return false;
  info.decoder_buffer_delay = value;
  if (!reader.read_uvlc(value, error)) return false;
  info.encoder_buffer_delay = value;
  bool low_delay = false;
  if (!reader.read_bit(low_delay)) {
    set_error(error, "truncated low_delay_mode_flag");
    return false;
  }
  info.low_delay_mode_flag = low_delay;
  return seq_decoder_model_info_valid(info, error);
}

bool read_sequence_timing(BitReader& reader, SequenceTimingConfig& config,
                          std::string* error) {
  config = SequenceTimingConfig{};
  bool flag = false;
  if (!reader.read_bit(flag)) {
    set_error(error, "truncated seq_initial_display_delay_present_flag");
    return false;
  }
  config.seq_initial_display_delay_present_flag = flag;
  if (flag) {
    std::uint32_t delay = 0;
    if (!reader.read_bits(4, delay)) {
      set_error(error, "truncated seq_initial_display_delay_minus_1");
      return false;
    }
    config.seq_initial_display_delay_minus_1 =
        static_cast<std::uint8_t>(delay);
  }
  if (!reader.read_bit(flag)) {
    set_error(error, "truncated decoder_model_info_present_flag");
    return false;
  }
  config.decoder_model_info_present_flag = flag;
  config.seq_decoder_model_info_present_flag = false;
  config.num_units_in_decoding_tick = 0;
  if (!flag) {
    if (!sequence_timing_valid(config, error)) return false;
    return true;
  }
  if (!reader.read_bits(32, config.num_units_in_decoding_tick)) {
    set_error(error, "truncated num_units_in_decoding_tick");
    return false;
  }
  bool sdmi = false;
  if (!reader.read_bit(sdmi)) {
    set_error(error, "truncated seq_decoder_model_info_present_flag");
    return false;
  }
  config.seq_decoder_model_info_present_flag = sdmi;
  if (sdmi && !read_seq_decoder_model_info(reader, config.decoder_model_info,
                                           error)) {
    return false;
  }
  return sequence_timing_valid(config, error);
}

bool read_buffer_removal_timing(BitReader& reader, BufferRemovalTiming& brt,
                                std::string* error) {
  bool dependent = false;
  if (!reader.read_bit(dependent)) {
    set_error(error, "truncated br_ops_dependent_flag");
    return false;
  }
  brt = BufferRemovalTiming{};
  brt.br_ops_dependent_flag = dependent;
  if (!dependent) {
    if (!reader.read_rg(4U, brt.br_time, error)) return false;
    return buffer_removal_timing_valid(brt, error);
  }
  std::uint32_t ops_id = 0;
  std::uint32_t count = 0;
  if (!reader.read_bits(4, ops_id) || !reader.read_bits(3, count)) {
    set_error(error, "truncated br_ops_id/br_ops_cnt");
    return false;
  }
  brt.br_ops_id = static_cast<std::uint8_t>(ops_id);
  brt.op_times.resize(count);
  for (std::uint32_t i = 0; i < count; ++i) {
    bool present = false;
    if (!reader.read_bit(present)) {
      set_error(error, "truncated br_decoder_model_present_op_flag");
      return false;
    }
    brt.op_times[i].br_decoder_model_present_op_flag = present;
    if (present && !reader.read_rg(4U, brt.op_times[i].br_time, error)) {
      return false;
    }
  }
  return buffer_removal_timing_valid(brt, error);
}

bool read_content_interpretation(BitReader& reader, ContentInterpretation& ci,
                                 std::string* error) {
  ci = ContentInterpretation{};
  std::uint32_t scan = 0;
  if (!reader.read_bits(2, scan)) {
    set_error(error, "truncated ci_scan_type_idc");
    return false;
  }
  ci.ci_scan_type_idc = static_cast<std::uint8_t>(scan);
  bool flag = false;
  if (!reader.read_bit(flag)) {
    set_error(error, "truncated ci_color_description_present_flag");
    return false;
  }
  ci.ci_color_description_present_flag = flag;
  if (!reader.read_bit(flag)) {
    set_error(error, "truncated ci_chroma_sample_position_present_flag");
    return false;
  }
  ci.ci_chroma_sample_position_present_flag = flag;
  if (!reader.read_bit(flag)) {
    set_error(error, "truncated ci_aspect_ratio_info_present_flag");
    return false;
  }
  ci.ci_aspect_ratio_info_present_flag = flag;
  if (!reader.read_bit(flag)) {
    set_error(error, "truncated ci_timing_info_present_flag");
    return false;
  }
  ci.ci_timing_info_present_flag = flag;
  std::uint32_t reserved = 0;
  if (!reader.read_bits(2, reserved)) {
    set_error(error, "truncated ci_reserved_2bit");
    return false;
  }

  if (ci.ci_color_description_present_flag) {
    std::uint32_t idc = 0;
    if (!reader.read_rg(2U, idc, error)) return false;
    ci.ci_color_description_idc = static_cast<std::uint8_t>(idc);
    if (idc == 0U) {
      std::uint32_t primaries = 0;
      std::uint32_t transfer = 0;
      std::uint32_t matrix = 0;
      if (!reader.read_bits(8, primaries) ||
          !reader.read_bits(8, transfer) ||
          !reader.read_bits(8, matrix)) {
        set_error(error, "truncated CI colour description");
        return false;
      }
      ci.ci_color_primaries = static_cast<std::uint8_t>(primaries);
      ci.ci_transfer_characteristics = static_cast<std::uint8_t>(transfer);
      ci.ci_matrix_coefficients = static_cast<std::uint8_t>(matrix);
    }
    bool full_range = false;
    if (!reader.read_bit(full_range)) {
      set_error(error, "truncated ci_full_range_flag");
      return false;
    }
    ci.ci_full_range_flag = full_range;
  }
  if (ci.ci_chroma_sample_position_present_flag) {
    if (!reader.read_uvlc(ci.ci_chroma_sample_position_top, error)) return false;
    if (ci.ci_scan_type_idc != 1U) {
      if (!reader.read_uvlc(ci.ci_chroma_sample_position_bottom, error)) {
        return false;
      }
    } else {
      ci.ci_chroma_sample_position_bottom = ci.ci_chroma_sample_position_top;
    }
  }
  if (ci.ci_aspect_ratio_info_present_flag) {
    std::uint32_t idc = 0;
    if (!reader.read_bits(8, idc)) {
      set_error(error, "truncated ci_aspect_ratio_idc");
      return false;
    }
    ci.ci_aspect_ratio_idc = static_cast<std::uint8_t>(idc);
    if (idc == 255U) {
      if (!reader.read_uvlc(ci.ci_sar_width, error)) return false;
      if (!reader.read_uvlc(ci.ci_sar_height, error)) return false;
    }
  }
  if (ci.ci_timing_info_present_flag &&
      !read_timing_info(reader, ci.timing, error)) {
    return false;
  }
  return content_interpretation_valid(ci, error);
}

// ---------------------------------------------------------------------------
// OBU assembly / parsing
// ---------------------------------------------------------------------------

const char* obu_type_name(ObuType type) noexcept {
  switch (type) {
    case ObuType::kSequenceHeader: return "OBU_SEQUENCE_HEADER";
    case ObuType::kTemporalDelimiter: return "OBU_TEMPORAL_DELIMITER";
    case ObuType::kClosedLoopKey: return "OBU_CLOSED_LOOP_KEY";
    case ObuType::kRegularTileGroup: return "OBU_REGULAR_TILE_GROUP";
    case ObuType::kBufferRemovalTiming: return "OBU_BUFFER_REMOVAL_TIMING";
    case ObuType::kLayerConfigurationRecord:
      return "OBU_LAYER_CONFIGURATION_RECORD";
    case ObuType::kAtlasSegment: return "OBU_ATLAS_SEGMENT";
    case ObuType::kOperatingPointSet: return "OBU_OPERATING_POINT_SET";
    case ObuType::kContentInterpretation: return "OBU_CONTENT_INTERPRETATION";
  }
  return "OBU_RESERVED";
}

bool parse_obus(std::span<const std::uint8_t> bytes,
                std::vector<ParsedObu>& obus, std::string* error) {
  obus.clear();
  std::size_t offset = 0;
  while (offset < bytes.size()) {
    const std::size_t start = offset;
    std::uint64_t obu_size_64 = 0;
    unsigned shift = 0;
    bool complete = false;
    for (int i = 0; i < 8; ++i) {
      if (offset >= bytes.size()) {
        set_error(error, "truncated leb128 OBU size");
        return false;
      }
      const std::uint8_t byte = bytes[offset++];
      obu_size_64 |= static_cast<std::uint64_t>(byte & 0x7FU) << shift;
      shift += 7U;
      if ((byte & 0x80U) == 0U) {
        complete = true;
        break;
      }
    }
    if (!complete) {
      set_error(error, "leb128 OBU size uses more than 8 bytes");
      return false;
    }
    if (obu_size_64 > std::numeric_limits<std::uint32_t>::max()) {
      set_error(error, "leb128 OBU size exceeds 32 bits");
      return false;
    }
    const std::uint32_t obu_size =
        static_cast<std::uint32_t>(obu_size_64);
    if (obu_size == 0U) {
      set_error(error, "zero-length OBU");
      return false;
    }
    if (obu_size > bytes.size() - offset) {
      set_error(error, "OBU size exceeds the remaining input");
      return false;
    }
    const std::size_t header_start = offset;
    const std::uint8_t first = bytes[offset++];
    ParsedObu obu;
    obu.extension_flag = (first & 0x80U) != 0U;
    obu.raw_type = static_cast<std::uint8_t>((first >> 2U) & 0x1FU);
    obu.tlayer_id = static_cast<std::uint8_t>(first & 0x03U);
    if (obu.extension_flag) {
      if (offset >= header_start + obu_size || offset >= bytes.size()) {
        set_error(error, "truncated obu_header extension");
        return false;
      }
      const std::uint8_t second = bytes[offset++];
      obu.mlayer_id = static_cast<std::uint8_t>((second >> 5U) & 0x07U);
      obu.xlayer_id = static_cast<std::uint8_t>(second & 0x1FU);
    } else {
      obu.mlayer_id = 0;
      obu.xlayer_id =
          (obu.raw_type == 2U || obu.raw_type == 20U) ? 31U : 0U;
    }
    const std::size_t header_bytes = offset - header_start;
    if (header_bytes > obu_size) {
      set_error(error, "OBU header exceeds its declared size");
      return false;
    }
    obu.payload = bytes.subspan(offset, obu_size - header_bytes);
    obu.offset = start;
    obu.size = (offset - start) + obu.payload.size();
    obu.type = static_cast<ObuType>(obu.raw_type);
    obus.push_back(obu);
    offset = start + obu.size;
  }
  return true;
}

bool validate_temporal_unit_order(std::span<const ParsedObu> obus,
                                  std::string* error) {
  if (obus.empty()) {
    set_error(error, "empty temporal unit");
    return false;
  }
  if (obus[0].type != ObuType::kTemporalDelimiter) {
    set_error(error, "temporal unit does not start with OBU_TEMPORAL_DELIMITER");
    return false;
  }
  bool seen_sequence = false;
  bool seen_ci = false;
  bool seen_brt = false;
  bool seen_frame = false;
  std::size_t frame_count = 0;
  for (std::size_t i = 0; i < obus.size(); ++i) {
    const ParsedObu& obu = obus[i];
    switch (obu.type) {
      case ObuType::kTemporalDelimiter:
        if (i != 0U) {
          set_error(error, "multiple OBU_TEMPORAL_DELIMITER in one unit");
          return false;
        }
        break;
      case ObuType::kLayerConfigurationRecord:
      case ObuType::kAtlasSegment:
      case ObuType::kOperatingPointSet:
        if (seen_frame) {
          set_error(error, std::string(obu_type_name(obu.type)) +
                               " appears after the frame OBUs");
          return false;
        }
        break;
      case ObuType::kSequenceHeader:
        if (seen_frame) {
          set_error(error, "OBU_SEQUENCE_HEADER appears after the frame OBUs");
          return false;
        }
        if (seen_sequence) {
          set_error(error, "multiple OBU_SEQUENCE_HEADER in one unit");
          return false;
        }
        seen_sequence = true;
        break;
      case ObuType::kContentInterpretation:
        if (!seen_sequence) {
          set_error(error,
                     "OBU_CONTENT_INTERPRETATION precedes the sequence header");
          return false;
        }
        if (seen_frame) {
          set_error(error,
                     "OBU_CONTENT_INTERPRETATION appears after the frame OBUs");
          return false;
        }
        if (seen_ci) {
          set_error(error,
                     "multiple OBU_CONTENT_INTERPRETATION in one frame unit");
          return false;
        }
        seen_ci = true;
        break;
      case ObuType::kBufferRemovalTiming:
        if (seen_frame) {
          set_error(error,
                     "OBU_BUFFER_REMOVAL_TIMING appears after the frame OBUs");
          return false;
        }
        if (seen_brt) {
          set_error(error,
                     "multiple OBU_BUFFER_REMOVAL_TIMING in one frame unit");
          return false;
        }
        seen_brt = true;
        break;
      case ObuType::kClosedLoopKey:
      case ObuType::kRegularTileGroup:
        seen_frame = true;
        ++frame_count;
        break;
      default:
        set_error(error, std::string("unexpected OBU type in the M2 unit: ") +
                             obu_type_name(obu.type));
        return false;
    }
  }
  if (!seen_frame) {
    set_error(error, "temporal unit contains no coded frame OBU");
    return false;
  }
  if (frame_count > 1U) {
    set_error(error, "the pinned M2 gate expects one frame OBU per unit");
    return false;
  }
  return true;
}

bool build_buffer_removal_timing_obu(const BufferRemovalTiming& brt,
                                     std::vector<std::uint8_t>& out,
                                     std::string* error) {
  BitWriter writer;
  if (!write_buffer_removal_timing(writer, brt, error)) return false;
  writer.put_bit(true);       // trailing_one_bit (§5.2.3)
  writer.align_to_byte();
  out = make_obu_bytes(ObuType::kBufferRemovalTiming, 0U, writer.bytes());
  return true;
}

bool build_content_interpretation_obu(const ContentInterpretation& ci,
                                      std::vector<std::uint8_t>& out,
                                      std::string* error) {
  BitWriter writer;
  if (!write_content_interpretation(writer, ci, error)) return false;
  writer.put_bit(false);  // obu_extension_flag (extensible OBU, §5.2.1)
  writer.put_bit(true);   // trailing_one_bit
  writer.align_to_byte();
  out = make_obu_bytes(ObuType::kContentInterpretation, 0U, writer.bytes());
  return true;
}

bool read_buffer_removal_timing_obu(std::span<const std::uint8_t> obu,
                                    BufferRemovalTiming& brt,
                                    std::string* error) {
  ParsedObu parsed;
  if (!single_obu_of_type(obu, ObuType::kBufferRemovalTiming, parsed, error)) {
    return false;
  }
  BitReader reader(parsed.payload);
  if (!read_buffer_removal_timing(reader, brt, error)) return false;
  return consume_trailing_bits(reader, error);
}

bool read_content_interpretation_obu(std::span<const std::uint8_t> obu,
                                     ContentInterpretation& ci,
                                     std::string* error) {
  ParsedObu parsed;
  if (!single_obu_of_type(obu, ObuType::kContentInterpretation, parsed,
                          error)) {
    return false;
  }
  BitReader reader(parsed.payload);
  if (!read_content_interpretation(reader, ci, error)) return false;
  bool extension = false;
  if (!reader.read_bit(extension)) {
    set_error(error, "truncated obu_extension_flag");
    return false;
  }
  if (extension) {
    set_error(error, "obu_extension_data is not supported here");
    return false;
  }
  return consume_trailing_bits(reader, error);
}

// ---------------------------------------------------------------------------
// Sequence-header timing splice
// ---------------------------------------------------------------------------

bool splice_sequence_timing(std::span<const std::uint8_t> sequence_header_obu,
                            const SequenceTimingConfig& config,
                            std::vector<std::uint8_t>& out,
                            std::string* error) {
  ParsedObu parsed;
  if (!single_obu_of_type(sequence_header_obu, ObuType::kSequenceHeader,
                          parsed, error)) {
    return false;
  }
  if (parsed.extension_flag) {
    set_error(error, "sequence header extension data is not supported");
    return false;
  }
  if (!sequence_timing_valid(config, error)) return false;
  std::size_t syntax_bits = 0;
  if (!extensible_syntax_bits(parsed.payload, syntax_bits, error)) return false;

  BitReader reader(parsed.payload);
  SequencePrefix prefix;
  if (!read_sequence_prefix(reader, prefix, error)) return false;
  if (prefix.single_picture_header_flag) {
    set_error(error,
              "single_picture_header_flag = 1 has no timing block to splice");
    return false;
  }
  bool flag = false;
  if (!reader.read_bit(flag)) {  // existing seq_initial_display_delay_present_flag
    set_error(error, "truncated baseline timing block");
    return false;
  }
  if (flag) {
    set_error(error, "baseline sequence header already signals initial display delay");
    return false;
  }
  if (!reader.read_bit(flag)) {  // existing decoder_model_info_present_flag
    set_error(error, "truncated baseline timing block");
    return false;
  }
  if (flag) {
    set_error(error, "baseline sequence header already signals decoder model info");
    return false;
  }
  const std::size_t rest_position = reader.bit_position();

  BitWriter writer;
  if (!append_payload_bits(writer, parsed.payload, 0U, prefix.timing_position)) {
    set_error(error, "failed to copy sequence-header prefix");
    return false;
  }
  if (!write_sequence_timing(writer, config, error)) return false;
  if (rest_position > syntax_bits) {
    set_error(error, "baseline sequence header ends inside its timing block");
    return false;
  }
  if (rest_position < syntax_bits) {
    if (!append_payload_bits(writer, parsed.payload, rest_position,
                             syntax_bits - rest_position)) {
      set_error(error, "failed to copy sequence-header suffix");
      return false;
    }
  }
  writer.put_bit(false);  // obu_extension_flag
  writer.put_bit(true);   // trailing_one_bit
  writer.align_to_byte();
  out = make_obu_bytes(ObuType::kSequenceHeader, parsed.tlayer_id,
                       writer.bytes());
  return true;
}

bool read_sequence_header_timing(
    std::span<const std::uint8_t> sequence_header_obu,
    SequenceTimingConfig& config, std::string* error) {
  ParsedObu parsed;
  if (!single_obu_of_type(sequence_header_obu, ObuType::kSequenceHeader,
                          parsed, error)) {
    return false;
  }
  BitReader reader(parsed.payload);
  SequencePrefix prefix;
  if (!read_sequence_prefix(reader, prefix, error)) return false;
  config = SequenceTimingConfig{};
  if (prefix.single_picture_header_flag) return true;
  return read_sequence_timing(reader, config, error);
}

}  // namespace av2enc::hrd
