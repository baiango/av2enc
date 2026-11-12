#pragma once

// HRD / timing syntax for AV2 v1.0.0 (spec-only).
//
// This component owns the byte-level syntax around the decoder model:
//
//   * §5.4.12  timing_info()                  (carried by the Content
//                                              Interpretation OBU, §5.15)
//   * §5.4.13  seq_decoder_model_info()
//   * §5.4.1   seq_initial_display_delay_present_flag /
//              decoder_model_info_present_flag / num_units_in_decoding_tick
//   * §5.12    buffer_removal_timing_obu()    (both br_ops_dependent_flag
//                                              forms, per-operating-point
//                                              br_time_op values)
//   * §6.4.12 / §6.4.13 / §6.11 semantics
//   * §5.2.2   OBU header parsing/ordering for the pieces above
//
// It deliberately does NOT modify the M1/M2 core, the root CMakeLists.txt or
// src/main.cpp. `hrd_buffer.h` adds the rate-control -> br_time planner.
//
// Normative constants / derivations used below:
//   DispCT = num_units_in_display_tick / time_scale              (§6.4.12)
//   DecCT  = num_units_in_decoding_tick / time_scale             (§6.4.13)
//   rg(4)  is Rice-Golomb with n = 4, so the representable br_time range is
//          0 .. (32 << 4) - 1 == 511                              (§4.11.10)
//   br_ops_cnt is f(3): 0..7, and a non-reset OPS has 1..7 points (§6.10.2)
//
// The PDF prints timing_info() and buffer_removal_timing_obu() on pages where
// the surrounding §-markers are detached; the syntax bodies above are from the
// same v1.0.0 PDF text (sections 5.4.12/5.4.13, 5.12) and are pinned by the
// bit goldens in hrd_test.cpp.

#include "core/bitstream/bitwriter.h"
#include "codec/rate/rc/hrd/bit_reader.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace av2enc::hrd {

// ---------------------------------------------------------------------------
// Limits from the syntax descriptors
// ---------------------------------------------------------------------------

inline constexpr std::uint32_t kMaxBrTime = 511;  // (32 << 4) - 1 for rg(4)
inline constexpr std::uint8_t kMaxOpsId = 15;     // br_ops_id f(4)
inline constexpr std::uint8_t kMaxOpsCnt = 7;     // br_ops_cnt f(3)
inline constexpr std::uint32_t kMaxUvlc = 0xFFFFFFFEU;  // uvlc() maximum

// ---------------------------------------------------------------------------
// §5.4.12 timing_info()
// ---------------------------------------------------------------------------

struct TimingInfo {
  std::uint32_t num_units_in_display_tick = 1;
  std::uint32_t time_scale = 1;
  bool equal_picture_interval = false;
  // num_ticks_per_picture_minus_1 (uvlc), present only when
  // equal_picture_interval is true. The display interval is
  // (value + 1) * DispCT.
  std::uint32_t num_ticks_per_picture_minus_1 = 0;

  // (num_ticks_per_picture_minus_1 + 1), or 1 when the interval is not
  // signalled. This is the multiplier on DispCT, not a DecCT count.
  std::uint64_t ticks_per_picture() const noexcept {
    return equal_picture_interval
               ? static_cast<std::uint64_t>(num_ticks_per_picture_minus_1) + 1U
               : 1U;
  }
};

bool timing_info_valid(const TimingInfo& timing, std::string* error = nullptr);

// §6.4.12 DispCT in seconds. Returns 0 (and sets *error) for an invalid
// timing structure.
double display_tick_seconds(const TimingInfo& timing,
                            std::string* error = nullptr) noexcept;

// ---------------------------------------------------------------------------
// §5.4.13 seq_decoder_model_info()
// ---------------------------------------------------------------------------

struct SeqDecoderModelInfo {
  // Both delays are in units of 1/90000 seconds. uvlc(), so the maximum is
  // 2^32 - 2. For a sequence with random access points the sum
  // decoder_buffer_delay + encoder_buffer_delay is constant.
  std::uint32_t decoder_buffer_delay = 70000;
  std::uint32_t encoder_buffer_delay = 20000;
  bool low_delay_mode_flag = false;
};

bool seq_decoder_model_info_valid(const SeqDecoderModelInfo& info,
                                  std::string* error = nullptr);

// ---------------------------------------------------------------------------
// §5.4.1 sequence-header timing block
// ---------------------------------------------------------------------------

// The sequence-header fields that follow seq_cropping_window_present_flag and
// its optional offsets. When single_picture_header_flag == 1 the whole block
// is absent and decoder_model_info_present_flag is inferred 0.
struct SequenceTimingConfig {
  bool seq_initial_display_delay_present_flag = false;
  // f(4). Present only when the flag above is true.
  std::uint8_t seq_initial_display_delay_minus_1 = 0;
  bool decoder_model_info_present_flag = false;
  // f(32), present only when decoder_model_info_present_flag is true.
  std::uint32_t num_units_in_decoding_tick = 0;
  bool seq_decoder_model_info_present_flag = false;
  SeqDecoderModelInfo decoder_model_info{};
};

bool sequence_timing_valid(const SequenceTimingConfig& config,
                           std::string* error = nullptr);

// ---------------------------------------------------------------------------
// §5.15 content_interpretation_obu() and §5.4.12 timing_info()
// ---------------------------------------------------------------------------

// Complete content_interpretation_obu() syntax. The timing information is the
// only reason this component emits a CI OBU, but the full syntax is written
// and read so the OBU round-trips byte-exactly.
struct ContentInterpretation {
  std::uint8_t ci_scan_type_idc = 0;  // f(2)
  bool ci_color_description_present_flag = false;
  // rg(2), valid range 0..127. When 0, the three colour fields are explicit.
  std::uint8_t ci_color_description_idc = 0;
  std::uint8_t ci_color_primaries = 0;          // f(8), when idc == 0
  std::uint8_t ci_transfer_characteristics = 0; // f(8), when idc == 0
  std::uint8_t ci_matrix_coefficients = 0;      // f(8), when idc == 0
  bool ci_full_range_flag = false;              // when colour description present

  bool ci_chroma_sample_position_present_flag = false;
  std::uint32_t ci_chroma_sample_position_top = 0;    // uvlc(), <= 5
  std::uint32_t ci_chroma_sample_position_bottom = 0; // uvlc(), <= 5

  bool ci_aspect_ratio_info_present_flag = false;
  std::uint8_t ci_aspect_ratio_idc = 0;  // f(8), <= 16 or 255
  std::uint32_t ci_sar_width = 1;        // uvlc(), when idc == 255
  std::uint32_t ci_sar_height = 1;       // uvlc(), when idc == 255

  bool ci_timing_info_present_flag = false;
  TimingInfo timing{};
};

bool content_interpretation_valid(const ContentInterpretation& ci,
                                  std::string* error = nullptr);

// ---------------------------------------------------------------------------
// §5.12 buffer_removal_timing_obu()
// ---------------------------------------------------------------------------

struct BrOpTime {
  bool br_decoder_model_present_op_flag = true;
  // br_time_op[br_ops_id][i], rg(4). Valid only when the flag above is true.
  std::uint32_t br_time = 0;
};

struct BufferRemovalTiming {
  bool br_ops_dependent_flag = false;
  // br_ops_id f(4), present only for the operating-point-dependent form.
  std::uint8_t br_ops_id = 0;
  // br_ops_cnt[br_ops_id] entries, in operating-point index order.
  std::vector<BrOpTime> op_times;
  // br_time rg(4), present only for the non-dependent form.
  std::uint32_t br_time = 0;
};

bool buffer_removal_timing_valid(const BufferRemovalTiming& brt,
                                 std::string* error = nullptr);

// A parsed OPS OBU summary, used for the §6.11 correspondence requirement:
//   br_ops_id == an ops_id present in the bitstream
//   br_ops_cnt[br_ops_id] == ops_cnt[obu_xlayer_id][br_ops_id]
struct OperatingPointSetSummary {
  std::uint8_t xlayer_id = 0;
  std::uint8_t ops_id = 0;
  std::uint8_t ops_cnt = 0;  // 1..7 for an active OPS, 0 for a reset
};

bool validate_brt_against_ops(
    const BufferRemovalTiming& brt, std::uint8_t brt_xlayer_id,
    std::span<const OperatingPointSetSummary> ops,
    std::string* error = nullptr);

// Tick-range validation: br_time and every present br_time_op must be <= 511,
// br_ops_id <= 15 and br_ops_cnt <= 7.
bool validate_brt_tick_ranges(const BufferRemovalTiming& brt,
                              std::string* error = nullptr);

// ---------------------------------------------------------------------------
// Syntax writers (payload bits only; no OBU header/trailing bits)
// ---------------------------------------------------------------------------

bool write_timing_info(BitWriter& writer, const TimingInfo& timing,
                       std::string* error = nullptr);
bool write_seq_decoder_model_info(BitWriter& writer,
                                  const SeqDecoderModelInfo& info,
                                  std::string* error = nullptr);
bool write_sequence_timing(BitWriter& writer,
                           const SequenceTimingConfig& config,
                           std::string* error = nullptr);
bool write_buffer_removal_timing(BitWriter& writer,
                                 const BufferRemovalTiming& brt,
                                 std::string* error = nullptr);
bool write_content_interpretation(BitWriter& writer,
                                  const ContentInterpretation& ci,
                                  std::string* error = nullptr);

// ---------------------------------------------------------------------------
// Syntax readers (payload bits only)
// ---------------------------------------------------------------------------

bool read_timing_info(BitReader& reader, TimingInfo& timing,
                      std::string* error = nullptr);
bool read_seq_decoder_model_info(BitReader& reader, SeqDecoderModelInfo& info,
                                 std::string* error = nullptr);
bool read_sequence_timing(BitReader& reader, SequenceTimingConfig& config,
                          std::string* error = nullptr);
bool read_buffer_removal_timing(BitReader& reader, BufferRemovalTiming& brt,
                                std::string* error = nullptr);
bool read_content_interpretation(BitReader& reader, ContentInterpretation& ci,
                                 std::string* error = nullptr);

// ---------------------------------------------------------------------------
// OBU assembly and parsing
// ---------------------------------------------------------------------------

// OBU types from §6.2.2 Table 6.1 (values reused across the repo).
enum class ObuType : std::uint8_t {
  kSequenceHeader = 1,
  kTemporalDelimiter = 2,
  kClosedLoopKey = 4,
  kRegularTileGroup = 7,
  kBufferRemovalTiming = 15,
  kLayerConfigurationRecord = 16,
  kAtlasSegment = 17,
  kOperatingPointSet = 18,
  kContentInterpretation = 24,
};

const char* obu_type_name(ObuType type) noexcept;

struct ParsedObu {
  ObuType type = ObuType::kTemporalDelimiter;
  std::uint8_t raw_type = 0;
  std::uint8_t tlayer_id = 0;
  std::uint8_t mlayer_id = 0;
  std::uint8_t xlayer_id = 0;
  bool extension_flag = false;
  std::span<const std::uint8_t> payload;  // excludes the OBU header
  std::size_t offset = 0;                 // offset of the leb128 prefix
  std::size_t size = 0;                   // prefix + header + payload
};

// Annex B.2 leb128 prefix, §5.2.2 obu_header() and OBU payload boundaries.
// The returned spans alias `bytes`; keep `bytes` alive while using them.
bool parse_obus(std::span<const std::uint8_t> bytes,
                std::vector<ParsedObu>& obus, std::string* error = nullptr);

// Checks the §7.3.3/§7.3.6/§7.3.7 ordering rules this component can affect:
// at most one BRT per coded frame unit, BRT before the frame OBUs, at most one
// CI, CI before the frame OBUs and after any sequence header, and a
// temporal-delimiter/frame structure the M2 stream uses.
bool validate_temporal_unit_order(std::span<const ParsedObu> obus,
                                  std::string* error = nullptr);

// Builds a complete Annex B.2 length-delimited OBU with obu_extension_flag = 0.
// BRT is not an extensible OBU (§5.2.2 is_extensible_obu), so its payload is
// syntax bits + trailing_bits(); CI is extensible and gets
// obu_extension_flag = 0 before its trailing bits.
bool build_buffer_removal_timing_obu(const BufferRemovalTiming& brt,
                                     std::vector<std::uint8_t>& out,
                                     std::string* error = nullptr);
bool build_content_interpretation_obu(const ContentInterpretation& ci,
                                      std::vector<std::uint8_t>& out,
                                      std::string* error = nullptr);

// Parses a complete OBU (leb128 + header + payload) of the expected type.
bool read_buffer_removal_timing_obu(std::span<const std::uint8_t> obu,
                                    BufferRemovalTiming& brt,
                                    std::string* error = nullptr);
bool read_content_interpretation_obu(std::span<const std::uint8_t> obu,
                                     ContentInterpretation& ci,
                                     std::string* error = nullptr);

// ---------------------------------------------------------------------------
// Sequence-header timing splice
// ---------------------------------------------------------------------------

// Rebuilds a `single_picture_header_flag == 0` sequence_header_obu() so that
// the timing block (initial display delay, decoder model info) is present.
//
// Preconditions, all checked: the input is one complete Annex B.2 OBU of type
// OBU_SEQUENCE_HEADER with obu_extension_flag = 0 and no extension data; its
// existing timing block is the M0/M2 default
// (seq_initial_display_delay_present_flag = 0,
//  decoder_model_info_present_flag = 0).
//
// This is the integration path while M1/M2 own their own header writers: the
// caller keeps the baseline header and only this block changes. A future
// integration may instead call write_sequence_timing() directly from the
// sequence-header writer.
bool splice_sequence_timing(std::span<const std::uint8_t> sequence_header_obu,
                            const SequenceTimingConfig& config,
                            std::vector<std::uint8_t>& out,
                            std::string* error = nullptr);

// Reads the timing block from a complete sequence_header_obu(). The returned
// config has decoder_model_info_present_flag = false when the block is absent.
bool read_sequence_header_timing(std::span<const std::uint8_t> sequence_header_obu,
                                 SequenceTimingConfig& config,
                                 std::string* error = nullptr);

}  // namespace av2enc::hrd
