// AV2 v1.0.0 lossless still-picture coding (Q=0, DPCM, FSC).
//
// This module is deliberately separate from the M1/M2 syntax writers.  A
// lossless frame changes the frame-level tool set (CodedLossless = 1,
// TxMode = ONLY_4X4, no deblock/CDEF/GDF/CCSO/restoration), the block mode
// syntax (use_dpcm_y/uv, dpcm_mode_y/uv, optional fsc_mode), the residual
// transform (reversible WHT for DCT_DCT, identity for IDTX), the DPCM residual
// cumulative sum, and the lossless_tx_size syntax.  Reusing a lossy writer
// would require editing it; that is out of scope for this module.
//
// Normative references (AV2 Bitstream & Decoding Process Specification
// v1.0.0, 2026-05-28, commit 38f28e7):
//   §5.18.2     CodedLossless / LosslessArray
//   §5.18.6.1   quantization_params(), base_q_idx = 0
//   §5.18.5.2   deblocking_filter_params() early return for CodedLossless
//   §5.18.7.9/.10/.11/.12  gdf/cdef/lr/ccso early returns
//   §5.18.8.1   read_tx_mode(): CodedLossless -> ONLY_4X4
//   §5.20.5.5   read_intra_y_mode(): use_dpcm_y, dpcm_mode_y, fsc_mode
//   §5.20.5.6   read_intra_uv_mode(): use_dpcm_uv, dpcm_mode_uv
//   §5.20.6.1   read_tx_size() lossless branch
//   §5.20.7.27  coeffs(): FSC coefficient paths (bob/idtx)
//   §5.20.7.29  compute_tx_type(): lossless DCT_DCT / IDTX
//   §7.14.4     dequantization at q=0 (q2 = 64)
//   §7.15.2.2   inverse WHT (reversible 4x4 transform)
//   §7.15.4     lossless 2D inverse transform and DPCM cumulative sum
//   §8.3.2      CDF selection for every symbol written here
//
// Scope of the CPU reference encoder:
//   * 8-bit 4:2:0 still pictures, single tile, one frame per temporal unit.
//   * width/height multiples of 8 (MiCols/MiRows are the rounded values).
//   * 8x8 luma coding blocks.  Non-FSC blocks use four 4x4 DCT_DCT transforms
//     per luma block; chroma uses one 4x4 transform per plane.  This keeps
//     every lossless WHT a 4x4 reversible transform.  FSC blocks use one 8x8
//     IDTX luma transform (lossless_tx_size = 1) and one 4x4 IDTX chroma
//     transform per plane.
//   * partition tree: PARTITION_NONE at 8x8, HORZ/VERT above it.  The tree
//     follows §5.20.3.2's implied-partition rule at frame boundaries.
//   * one segment, base_q_idx = 0, no delta-Q, no palette/IntraBC/CfL/IST/
//     TCQ/parity hiding/CCSO/CDEF/GDF/restoration/film grain.
//   * CDF adaptation is disabled by default (disable_cdf_update = 1), matching
//     the reference "correctness first" policy.  The symbol stream is still
//     the normative §8.2 coder with the §9.3 default CDFs.
//
// The reader is a true syntax reader: it parses the OBUs, frame header, tile
// partition tree, mode info and coefficients, and reconstructs the frame.  It
// is used by the tests to prove the writer's choices round-trip, and by the
// decoder gate to compare against dav2d.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace av2enc::lossless {

// ---------------------------------------------------------------------------
// Lossless frame-level flags (§5.18.2)
// ---------------------------------------------------------------------------

inline constexpr int kMaxSegments = 16;
inline constexpr int kBaseSegments = 8;  // enable_ext_seg = 0

struct LosslessInputs {
  // §5.18.6.1 quantization_params().  `base_q_idx` is the raw 8-bit field.
  int base_q_idx = 0;
  bool delta_q_present = false;

  // Base and per-plane deltas from the sequence/frame header.  The comparison
  // is against zero, exactly as §5.18.2 writes it.
  int delta_q_y_dc = 0;
  int base_y_dc_delta_q = 0;
  int delta_q_u_dc = 0;
  int delta_q_v_dc = 0;
  int delta_q_u_ac = 0;
  int delta_q_v_ac = 0;
  int base_uv_dc_delta_q = 0;
  int base_uv_ac_delta_q = 0;

  // SEG_LVL_ALT_Q per segment.  `alt_q_enabled[s] == false` means
  // get_qindex(1, s) returns base_q_idx for that segment.
  bool alt_q_enabled[kMaxSegments] = {};
  int alt_q[kMaxSegments] = {};
  // Number of segments actually present in the frame (MaxSegments).
  int num_segments = kBaseSegments;
};

struct LosslessFlags {
  bool coded_lossless = false;       // every segment in LosslessArray is 1
  bool has_lossless_segment = false; // at least one segment is lossless
  bool lossless[kMaxSegments] = {};
};

// §5.18.2 computation of LosslessArray[] and CodedLossless.  This is pure
// frame-level derivation; it does not read or write a bitstream.
LosslessFlags compute_lossless_flags(const LosslessInputs& inputs);

// get_qindex(ignoreDeltaQ = 1, segmentId) with the §5.18.2 inputs, clipped to
// [0, MaxQ] with MaxQ = 255 for 8-bit.  Exposed for tests and rate control.
int qindex_for_segment(const LosslessInputs& inputs, int segment_id);

// ---------------------------------------------------------------------------
// Reversible transforms and DPCM helpers (exposed for tests)
// ---------------------------------------------------------------------------

// §7.15.2.2 inverse WHT, in place on 4 samples.  `shift` is the pre-scale
// shift (3 for the row pass of a lossless 4x4 transform, 0 otherwise).
void inverse_wht4(const int in[4], int shift, int out[4]);

// The WHT is an involution with shift 0; this is the encoder-side forward
// transform that exactly matches the decoder's two inverse passes.
void forward_wht4(const int in[4], int out[4]);

// Forward/inverse lossless 4x4 DCT_DCT transform.  `forward_lossless_4x4`
// produces the coefficient levels that the decoder's WHT pair reconstructs bit
// exactly.  `inverse_lossless_4x4` is the decoder path.
void forward_lossless_4x4(const int residual[16], int levels[16]);
void inverse_lossless_4x4(const int levels[16], int residual[16]);

// DPCM residual domain conversion.  The decoder applies a cumulative sum after
// the inverse transform, so the encoder must store the first difference.
// vertical == true is V_PRED (sum down columns); false is H_PRED (sum across
// rows).
void dpcm_difference_4x4(const int residual[16], bool vertical, int out[16]);
void dpcm_cumsum_4x4(const int in[16], bool vertical, int out[16]);
void dpcm_difference_8x8(const int residual[64], bool vertical, int out[64]);
void dpcm_cumsum_8x8(const int in[64], bool vertical, int out[64]);

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

enum class FscMode : std::uint8_t {
  kOff = 0,   // enable_fsc = 0, no fsc_mode syntax
  kAuto = 1,  // per-block rate estimate chooses FSC
  kForce = 2, // every luma block uses FSC (for path/gate testing)
};

struct StillPicture {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::span<const std::uint8_t> y;
  std::span<const std::uint8_t> u;
  std::span<const std::uint8_t> v;
};

struct Options {
  // 64 only: this reference encoder emits one 64x64 superblock grid.
  int sb_size = 64;
  // 31 = "maximum parameters", which keeps a single tile legal at any size.
  int seq_level_idx = 31;
  // When false (default), the frame header sets disable_cdf_update = 1 and
  // every CDF stays at its default value.  When true, both sides adapt.
  bool adaptive_cdf = false;
  // DPCM intra search: V/H DPCM vs DC per block.  When false, every block uses
  // DC prediction and the DPCM symbols are inferred/skipped.
  bool enable_dpcm = true;
  // FSC (forward skip coding / identity-only lossless coefficient coding).
  FscMode fsc = FscMode::kOff;
  // Disable the intra edge filter; the lossless reference does not need it.
  bool enable_intra_edge_filter = false;
  // Verify the reconstruction is bit exact before returning success.
  bool verify_exact = true;
};

struct Stats {
  int luma_blocks = 0;
  int fsc_blocks = 0;
  int dpcm_y_blocks = 0;
  int dpcm_uv_blocks = 0;
  std::uint32_t tile_bytes = 0;
  std::uint32_t frame_bytes = 0;
  std::uint64_t max_abs_recon_error = 0;
  bool exact = false;
};

struct EncodeResult {
  // Annex B length-delimited temporal unit (TD + sequence header + CLK).
  std::vector<std::uint8_t> temporal_unit;
  std::vector<std::uint8_t> recon_y;
  std::vector<std::uint8_t> recon_u;
  std::vector<std::uint8_t> recon_v;
  Stats stats;
};

// Encodes one lossless still picture.  Returns false with `error` set on a
// geometry/syntax failure.  The recon planes are the decoder's exact output;
// `stats.exact` is true only when every sample matches the source.
bool encode_still(const StillPicture& picture, const Options& options,
                  EncodeResult& result, std::string& error);

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

struct DecodeResult {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
  Stats stats;
};

// Parses one Annex B temporal unit produced by encode_still() and reconstructs
// the frame.  This is a full lossless syntax reader, not a shortcut through the
// encoder's decisions: every partition, mode and coefficient symbol is read
// from the bitstream with the §8.2 decoder state machine.
bool decode_temporal_unit(std::span<const std::uint8_t> temporal_unit,
                          DecodeResult& result, std::string& error);

}  // namespace av2enc::lossless
