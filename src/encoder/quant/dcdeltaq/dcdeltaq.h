// AV2 v1.0.0 DC/AC delta-Q: sequence/frame syntax, per-plane quantizer and
// dequantizer, QP-0 lossless interaction, QM/TCQ interplay, and an encoder-side
// RD search for the frame delta values.
//
// Normative references (AV2 Bitstream & Decoding Process Specification
// v1.0.0):
//   §5.4.8    sequence_transform_quant_entropy_config()
//   §5.18.6.1 quantization_params()
//   §5.18.6.2 setup_qm_params() (consumed through encoder/qm)
//   §5.18.6.3 read_delta_q()
//   §5.18.2   LosslessArray / CodedLossless
//   §7.14.2   qlookup(), get_q(), get_qindex(), get_dc_quant(), get_ac_quant()
//   §7.14.4   dequantization, including the QM and TCQ shifts
//
// Design notes
// ------------
// * The sequence header stores *base* deltas (BaseYDcDeltaQ / BaseUVDcDeltaQ /
//   BaseUVAcDeltaQ), each in [DELTA_DCQUANT_MIN, DELTA_DCQUANT_MAX] = [-23, 8].
// * The frame header stores *frame* deltas (DeltaQYDc / DeltaQUDc / DeltaQUAc /
//   DeltaQVDc / DeltaQVAc), each `su(7)` = [-64, 63], and only when the
//   sequence-level `*_delta_q_enabled` flag is set.
// * `equal_ac_dc_q = 1` makes the sequence header write *no* Y-DC or UV-DC
//   base field at all; Y-DC therefore has no delta syntax and is always
//   base_q_idx.  A non-zero Y-DC delta requires `equal_ac_dc_q = 0`.
// * `separate_uv_delta_q = 0` makes the frame header code only one U delta
//   pair and copy it to V (`DeltaQV* = DeltaQU*`).
// * The coefficient CDF slice (not this module) is always selected from the
//   *raw* `base_q_idx` (§6.17.2), never from an effective delta qindex.
#pragma once

#include "bitstream/bitwriter.h"
#include "encoder/m1/deltaq/bitreader.h"
#include "encoder/qm/qm.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace av2enc::dcdeltaq {

// ---------------------------------------------------------------------------
// Constants (§9)
// ---------------------------------------------------------------------------

inline constexpr int kMaxQ = 255;               // MAXQ_8_BITS
inline constexpr int kDeltaDcQuantBits = 5;     // DELTA_DCQUANT_BITS
inline constexpr int kDeltaDcQuantMin = -23;    // DELTA_DCQUANT_MIN
inline constexpr int kDeltaDcQuantMax = 8;      // DELTA_DCQUANT_MAX
inline constexpr int kBaseDeltaCount = 32;      // 1 << DELTA_DCQUANT_BITS
inline constexpr int kFrameDeltaMin = -64;      // su(7)
inline constexpr int kFrameDeltaMax = 63;
inline constexpr int kNumPlanes = 3;
inline constexpr int kMaxSegments = 16;         // MAX_SEGMENTS
inline constexpr int kBaseSegments = 8;         // enable_ext_seg = 0
inline constexpr int kQuantTableBits = 3;       // QUANT_TABLE_BITS

// ---------------------------------------------------------------------------
// Sequence-level delta-Q state (§5.4.8)
// ---------------------------------------------------------------------------

struct SequenceDeltaQ {
  bool monochrome = false;
  bool separate_uv_delta_q = false;
  bool equal_ac_dc_q = true;
  bool y_dc_delta_q_enabled = false;
  bool uv_dc_delta_q_enabled = false;
  bool uv_ac_delta_q_enabled = false;
  // Resolved values, i.e. DELTA_DCQUANT_MIN + raw_bits.
  int base_y_dc_delta_q = 0;
  int base_uv_dc_delta_q = 0;
  int base_uv_ac_delta_q = 0;
};

// Checks the internal consistency that §5.4.8 derives.  In particular:
//   * monochrome forces separate_uv_delta_q = 0 and all UV fields off;
//   * equal_ac_dc_q = 1 forces BaseYDcDeltaQ = BaseUVDcDeltaQ = 0 and both DC
//     enables off (the fields are not coded);
//   * BaseUVDcDeltaQ is BaseUVAcDeltaQ when equal_ac_dc_q = 1.
bool validate_sequence_delta_q(const SequenceDeltaQ& seq, std::string* error);

// Normalized view: with equal_ac_dc_q = 1, BaseUVDcDeltaQ := BaseUVAcDeltaQ.
SequenceDeltaQ normalized_sequence_delta_q(const SequenceDeltaQ& seq);

// Full §5.4.8 inputs.  `monochrome` and `single_picture_header_flag` are
// sequence-level facts from §5.4.1; the reader also fills the tool flags it
// parses.  The delta-Q fields come from `SequenceDeltaQ`.
struct SeqTransformQuantEntropyConfig {
  bool monochrome = false;
  bool single_picture_header_flag = true;
  bool enable_fsc = false;
  bool enable_idtx_intra = false;
  bool enable_intra_ist = false;
  bool enable_inter_ist = false;
  bool enable_chroma_dctonly = false;
  bool enable_inter_ddt = false;
  bool reduced_tx_part_set = false;
  bool enable_cctx = false;
  bool enable_tcq = false;
  bool choose_tcq_per_frame = false;
  bool enable_parity_hiding = false;
  bool enable_avg_cdf = false;
  bool avg_cdf_type = false;
};

// Writes/reads the complete §5.4.8 group, from `enable_fsc` through
// `uv_ac_delta_q_enabled`.  Returns the number of bits written (0 on error).
std::size_t write_sequence_transform_quant_entropy_config(
    BitWriter& writer, const SeqTransformQuantEntropyConfig& config,
    const SequenceDeltaQ& seq, std::string* error = nullptr);
bool read_sequence_transform_quant_entropy_config(
    av2enc::deltaq::BitReader& reader, SeqTransformQuantEntropyConfig& config,
    SequenceDeltaQ& seq, std::string* error = nullptr);

// Delta-only form: starts at `separate_uv_delta_q` and stops after
// `uv_ac_delta_q_enabled`.  This is the exact splice for headers that already
// wrote enable_fsc .. enable_parity_hiding.
std::size_t write_sequence_delta_q_fields(BitWriter& writer,
                                          const SequenceDeltaQ& seq,
                                          std::string* error = nullptr);
bool read_sequence_delta_q_fields(av2enc::deltaq::BitReader& reader, bool monochrome,
                                  SequenceDeltaQ& seq,
                                  std::string* error = nullptr);

// ---------------------------------------------------------------------------
// Frame-level delta-Q state (§5.18.6.1 / §5.18.6.3)
// ---------------------------------------------------------------------------

struct FrameDeltaQ {
  // Values exactly as §5.18.6.1 names them: frame deltas, excluding the
  // sequence-level base deltas.
  int delta_q_y_dc = 0;
  int delta_q_u_dc = 0;
  int delta_q_v_dc = 0;
  int delta_q_u_ac = 0;
  int delta_q_v_ac = 0;
  // `diff_uv_delta` is coded only when separate_uv_delta_q = 1.  On write it
  // is derived from the U/V values; on read it is the decoded bit.
  bool diff_uv_delta = false;
  // Presence bits, filled by the reader.  They say whether read_delta_q()
  // returned a non-zero value, not whether the field was enabled.
  bool y_dc_coded = false;
  bool u_dc_coded = false;
  bool v_dc_coded = false;
  bool u_ac_coded = false;
  bool v_ac_coded = false;
};

struct QuantizationParams {
  int base_q_idx = 0;
  FrameDeltaQ delta;
};

bool validate_frame_delta_q(const SequenceDeltaQ& seq,
                            const FrameDeltaQ& frame, std::string* error);

// Complete §5.18.6.1 `quantization_params()` for a non-TIP frame:
//   base_q_idx; DeltaQYDc; [diff_uv_delta]; DeltaQUDc; DeltaQUAc; ...
// `bit_depth` must be 8 (the module's qindex domain).  Returns bits written.
std::size_t write_quantization_params(BitWriter& writer, int base_q_idx,
                                      const SequenceDeltaQ& seq,
                                      const FrameDeltaQ& frame,
                                      std::string* error = nullptr);
bool read_quantization_params(av2enc::deltaq::BitReader& reader,
                              const SequenceDeltaQ& seq, int bit_depth,
                              QuantizationParams& out,
                              std::string* error = nullptr);

// Convenience for callers that only need base_q_idx (all deltas zero).
std::size_t write_disabled_quantization_params(BitWriter& writer,
                                               int base_q_idx,
                                               const SequenceDeltaQ& seq,
                                               std::string* error = nullptr);

// `read_delta_q()` itself: f(1) delta_coded; if set, su(7) delta_q.
std::size_t write_delta_q(BitWriter& writer, int value,
                          std::string* error = nullptr);
bool read_delta_q(av2enc::deltaq::BitReader& reader, int& value,
                  bool* coded = nullptr, std::string* error = nullptr);

// ---------------------------------------------------------------------------
// Derived quantizers (§7.14.2)
// ---------------------------------------------------------------------------

// DeltaQYDc + BaseYDcDeltaQ, DeltaQUDc + BaseUVDcDeltaQ, ...
struct ResolvedDeltaQ {
  int y_dc = 0;
  int u_dc = 0;
  int v_dc = 0;
  int u_ac = 0;
  int v_ac = 0;
};

ResolvedDeltaQ resolve_delta_q(const SequenceDeltaQ& seq,
                               const FrameDeltaQ& frame);

struct QuantFrameInputs {
  int base_q_idx = 0;
  bool delta_q_present = false;
  // CurrentQIndex from §5.20.1/.2; used only when delta_q_present.
  int current_q_index = 0;
  SequenceDeltaQ seq;
  FrameDeltaQ frame;
  // SEG_LVL_ALT_Q data.  `num_segments` is MaxSegments (8 or 16).
  int num_segments = kBaseSegments;
  bool alt_q_enabled[kMaxSegments] = {};
  int alt_q[kMaxSegments] = {};
};

// §7.14.2 qlookup() and get_q() for the 8-bit qindex domain.
int qlookup(int q);
int get_q(int qindex, int delta, int max_q = kMaxQ);

// §7.14.2 get_qindex(ignoreDeltaQ, segmentId).
int get_qindex(const QuantFrameInputs& in, bool ignore_delta_q,
               int segment_id);

struct PlaneQuant {
  int dc = 0;
  int ac = 0;
};

struct QuantFrameState {
  int qindex = 0;             // get_qindex(0, segment_id)
  ResolvedDeltaQ delta;       // DeltaQ* + Base* totals
  PlaneQuant plane[kNumPlanes];
  bool lossless = false;      // Lossless for this segment
};

QuantFrameState derive_quant(const QuantFrameInputs& in, int segment_id = 0);

// §5.18.2 LosslessArray[] and CodedLossless.
struct LosslessFlags {
  bool coded_lossless = false;
  bool has_lossless_segment = false;
  bool lossless[kMaxSegments] = {};
  int num_segments = kBaseSegments;
};
LosslessFlags derive_lossless_flags(const QuantFrameInputs& in);

// ---------------------------------------------------------------------------
// Quantizer / dequantizer (§7.14.4)
// ---------------------------------------------------------------------------

inline constexpr int kMaxAdjustedDim = 32;
inline constexpr int kMaxQuantElems = kMaxAdjustedDim * kMaxAdjustedDim;

// Everything one transform block needs.  `coefficient_scale` is the forward
// transform's coefficient multiplier at q2 = 528 (the value tx_variants
// bakes into TxParams::quant_factor); `quantize_block` rescales it per
// coefficient from the §7.14.2 q and §7.14.4 q2.
struct QuantBlock {
  int plane = 0;
  int coded_w = 4;
  int coded_h = 4;
  int tx_size = 0;          // TxSize index, needed for QM tables
  int tx_type = 0;          // TxType; IDTX/V_*/H_* affect QM/TCQ
  int segment_id = 0;
  bool lossless = false;    // §5.18.2 Lossless for this block
  bool allow_tcq = false;   // §5.20.7.27 allow_tcq
  bool use_fsc = false;     // Luma IDTX FSC path
  bool using_qmatrix = false;
  const qm::QmFrameState* qm_state = nullptr;
  PlaneQuant q;             // §7.14.2 get_dc_quant / get_ac_quant
  double coefficient_scale = 1.0;
};

// §7.14.4 dqDenom shift: (pels > 256) + (pels > 1024) [+ TCQ extra].
int dequant_shift(const QuantBlock& block);
int adjusted_width(int coded_w);
int adjusted_height(int coded_h);

// Fills `q2[tw * th]` with the §7.14.4 q2 matrix (DC at index 0, AC elsewhere,
// QM applied when active).  Returns false on bad parameters.
bool build_q2_matrix(const QuantBlock& block, std::int32_t* q2,
                     std::string* error = nullptr);

// `coeff[tw * th]` -> levels.  In lossless mode the coefficients are passed
// through with Round2(x, 0) (the reversible WHT/IDTX path does not use this
// quantizer; the pass-through keeps the API total and is what a lossless
// encoder calls).
bool quantize_block(const double* coeff, const QuantBlock& block,
                    std::int32_t* levels, std::string* error = nullptr);

// levels -> dequant.  In lossless mode this copies the levels (the decoder's
// lossless path does not invoke §7.14.4 dequantization).
bool dequantize_block(const std::int32_t* levels, const QuantBlock& block,
                      std::int32_t* dequant, std::string* error = nullptr);

// §4.8 Round2 for the non-negative domain used by dequantization.
std::int32_t round2(std::int32_t value, int n);

// TCQ's normative level mapping (§5.20.7.27), exposed for the interplay tests.
int tcq_state_q0(int state);
int tcq_syntax_level(int full_level, int pre_max);
int tcq_full_level(int syntax_level, int state);

// ---------------------------------------------------------------------------
// Encoder-side RD search for the frame delta values
// ---------------------------------------------------------------------------

struct FramePlanesView {
  int width = 0;
  int height = 0;
  std::span<const std::uint8_t> y;
  std::span<const std::uint8_t> u;
  std::span<const std::uint8_t> v;
};

struct DeltaQSearchConfig {
  int base_q_idx = 90;
  // `equal_ac_dc_q = 1` makes Y-DC non-signallable, so the search only moves
  // the UV quantizers.  Set false to search Y-DC.
  bool equal_ac_dc_q = false;
  bool separate_uv_delta_q = false;
  // QP 0: by default the search keeps every total delta <= 0 so that
  // CodedLossless stays 1 (the §5.18.2 interaction).
  bool allow_lossy_at_qp0 = false;
  // Candidate magnitudes are bounded by this (<= 63).
  int max_delta_magnitude = 24;
  // Coordinate-descent passes; the search is deterministic and bounded.
  int max_passes = 3;
  // RD lambda scale.  The 0.002 default is calibrated for this module's
  // deterministic rate proxy (search.cpp); it is not an AV1 lambda.
  double lambda_scale = 0.002;
  // Cap on the number of 8x8 luma blocks visited (deterministic subsample
  // when the frame is larger).
  int max_luma_blocks = 2048;
  bool use_qm = false;
  bool tcq = false;
};

struct DeltaQSearchResult {
  SequenceDeltaQ seq;
  FrameDeltaQ frame;
  QuantFrameState quant;
  double rd = 0.0;
  double sse = 0.0;
  double bits = 0.0;
  int evaluations = 0;
  bool lossless = false;
};

// Chooses the frame deltas that minimise SSE + lambda * rate for one frame.
// The result is the legal sequence/frame header pair plus its derived
// quantizers.  `error` is set on invalid input.
bool search_frame_delta_q(const FramePlanesView& planes,
                          const DeltaQSearchConfig& config,
                          DeltaQSearchResult* result,
                          std::string* error = nullptr);

}  // namespace av2enc::dcdeltaq
