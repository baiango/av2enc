// AV2 intra prediction — CPU reference implementation.
//
// Normative source: AV2 Bitstream & Decoding Process Specification v1.0.0
// (2026-05-28, commit 38f28e7), §7.13.2 Intra prediction process.
//
// Scope of this component (M1, workstream 2):
//   - §7.13.2.1  edge/neighbour sample preparation (MRlIndex = 0)
//   - §7.13.2.2  basic (Paeth) intra prediction
//   - §7.13.2.7  directional prediction, angle deltas + wide-angle mapping
//   - §7.13.2.8  single directional prediction (including luma IDIF)
//   - §7.13.2.10 DC prediction
//   - §7.13.2.13 smooth / smooth-v / smooth-h prediction
//   - §7.13.2.14 filter corner
//   - §7.13.2.17 intra edge filter strength selection
//   - §7.13.2.18 intra edge filter
//
// Deliberately not implemented here (later milestones / other workstreams):
//   DIP (§7.13.2.3), MRL (§7.13.2.7 MrlIndex > 0), IBP (§7.13.2.9/.12),
//   CfL (§7.13.5), IntraBC, palette, MHCCP.
//
// The module is bit-exact CPU reference code: no SIMD, no threads, no
// allocations in the prediction path. The CUDA port in src/cuda/intra_modes/
// must reproduce these integer operations exactly.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace av2enc::intra {

// ---------------------------------------------------------------------------
// Mode identifiers (AV2 YMode / UVMode; UV_CFL_PRED is handled by CfL)
// ---------------------------------------------------------------------------

enum class Mode : std::uint8_t {
  kDc = 0,
  kV = 1,
  kH = 2,
  kD45 = 3,
  kD135 = 4,
  kD113 = 5,
  kD157 = 6,
  kD203 = 7,
  kD67 = 8,
  kSmooth = 9,
  kSmoothV = 10,
  kSmoothH = 11,
  kPaeth = 12,
};

inline constexpr int kNumModes = 13;
inline constexpr int kNumDirectionalModes = 8;

// ANGLE_STEP (§3) and the signalled angle-delta range: -3..+3, i.e. +/- 9 deg.
inline constexpr int kAngleStep = 3;
inline constexpr int kMaxAngleDelta = 3;

// AV2 intra transform blocks are 4..64 samples per axis and power-of-two.
inline constexpr int kMinBlockDim = 4;
inline constexpr int kMaxBlockDim = 64;

// 13 base modes + 8 directional modes * 6 non-zero angle deltas.
inline constexpr int kMaxModeCandidates =
    kNumModes + kNumDirectionalModes * 2 * kMaxAngleDelta;

enum class IntraStatus : std::uint8_t {
  kOk = 0,
  kInvalidArgument = 1,
};

const char* status_string(IntraStatus status) noexcept;

bool is_directional(Mode mode) noexcept;

// Mode_To_Angle[] (§ 5.20.7.29 constants table).
int mode_to_angle(Mode mode) noexcept;

struct MappedAngle {
  Mode mode;
  int p_angle;
};

// Applies AngleDeltaY * ANGLE_STEP and the wide-angle intra prediction
// mapping (wide_angle_mapping, §5.20.7.29) for an intra block of w x h.
// Returns {mode, mode_to_angle(mode)} for non-directional modes.
MappedAngle map_angle(Mode mode, int angle_delta, int w, int h) noexcept;

// ---------------------------------------------------------------------------
// Edge sample preparation (§7.13.2.1, MrlIndex = 0)
// ---------------------------------------------------------------------------

// Reconstructed samples of one plane, row-major, tightly or padded stride.
struct PlaneView {
  const std::uint8_t* data = nullptr;
  int stride = 0;
};

// Inputs describing the transform block whose neighbours are needed.
//
// x/y are the top-left sample coordinates in the reconstructed plane.
// max_x/max_y are the largest *visible* coordinates (inclusive), not the
// allocation size. have_left/have_above are the block-context flags (§7.13.2.1
// haveLeft/haveAbove), which may be false at frame/tile boundaries.
//
// num4_above_right / num4_below_left are the §7.13.2.1 counts in 4x4 units.
// For chroma with w>32 or h>32 the spec forces the corresponding count to 0;
// prepare_edges() applies that rule from color_plane.
struct NeighborRequest {
  PlaneView recon;
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
  int max_x = 0;
  int max_y = 0;
  bool have_left = false;
  bool have_above = false;
  int num4_above_right = 0;
  int num4_below_left = 0;
  int color_plane = 0;  // 0 = luma, >0 = chroma
};

// One prepared edge array (AboveRow or LeftCol). Logical indices run from -2
// to 2*kMaxBlockDim+1; the spec uses -1 (top-left corner / virtual sample),
// 0..w+h-1 (samples), and up to w+h+1 after directional padding.
class EdgeRow {
 public:
  static constexpr int kMinIndex = -2;
  static constexpr int kMaxIndex = 2 * kMaxBlockDim + 1;  // 129
  static constexpr int kStorage = kMaxIndex - kMinIndex + 1;

  std::int16_t& operator[](int index) noexcept {
    return samples_[static_cast<std::size_t>(index - kMinIndex)];
  }
  std::int16_t operator[](int index) const noexcept {
    return samples_[static_cast<std::size_t>(index - kMinIndex)];
  }

  void clear() noexcept { samples_.fill(0); }

 private:
  std::array<std::int16_t, kStorage> samples_{};
};

struct IntraEdges {
  EdgeRow above;  // AboveRow
  EdgeRow left;   // LeftCol
  int w = 0;
  int h = 0;
  int bit_depth = 8;
  bool have_left = false;
  bool have_above = false;
  bool is_luma = true;
  // min(w, max_x - x + 1) and min(h, max_y - y + 1), used by the edge filter
  // extent (§7.13.2.7 numPx). Stored so prediction does not need geometry.
  int valid_above_px = 0;
  int valid_left_px = 0;
};

// Fills AboveRow/LeftCol exactly as §7.13.2.1 for MrlIndex = 0, including the
// unavailable-neighbour fillers (127/129/128 for 8-bit) and the -1 corner.
// `bit_depth` is 8..12; the current PlaneView always carries 8-bit samples.
IntraStatus prepare_edges(const NeighborRequest& request, int bit_depth,
                          IntraEdges& edges) noexcept;

// ---------------------------------------------------------------------------
// Prediction
// ---------------------------------------------------------------------------

struct PredictParams {
  Mode mode = Mode::kDc;
  int angle_delta = 0;  // -3..3, directional modes only
  // Sequence-level enable_intra_edge_filter. Ignored for the exact 90/180
  // degree predictions, as in §7.13.2.7.
  bool enable_edge_filter = true;
  // filterTypeAbove / filterTypeLeft (§7.13.2.15/.16): true when the
  // neighbouring block above/left uses SMOOTH, SMOOTH_V or SMOOTH_H.
  bool above_neighbor_is_smooth = false;
  bool left_neighbor_is_smooth = false;
};

// Writes an h x w prediction block. The input edges are never modified, so the
// same IntraEdges can be reused for every RDO candidate.
IntraStatus predict_intra(const IntraEdges& edges, const PredictParams& params,
                          std::uint8_t* dst, std::size_t dst_stride) noexcept;

// Convenience wrapper: prepare_edges() + predict_intra().
IntraStatus predict_block(const NeighborRequest& request, int bit_depth,
                          const PredictParams& params, std::uint8_t* dst,
                          std::size_t dst_stride) noexcept;

// ---------------------------------------------------------------------------
// RDO / mode-decision hooks
// ---------------------------------------------------------------------------

struct PredictionCost {
  std::uint64_t sad = 0;  // sum of absolute differences
  std::uint64_t sse = 0;  // sum of squared errors
};

PredictionCost prediction_cost(const std::uint8_t* src, std::size_t src_stride,
                               const std::uint8_t* pred,
                               std::size_t pred_stride, int w,
                               int h) noexcept;

// Predicts into `pred_scratch` (row-major, stride w; must hold w*h samples)
// and returns the distortion against `src`. Used by a greedy/SATD-style mode
// decision now and by the later RDO pass.
IntraStatus predict_and_cost(const IntraEdges& edges,
                             const PredictParams& params,
                             std::span<std::uint8_t> pred_scratch,
                             const std::uint8_t* src, std::size_t src_stride,
                             PredictionCost& cost) noexcept;

struct ModeCandidate {
  Mode mode = Mode::kDc;
  std::int8_t angle_delta = 0;
};

// Fills `out` with the candidate list: all 13 base modes first, then the six
// non-zero angle deltas for each directional mode when `include_angle_deltas`
// is true. Returns the number written, or -1 if `out` is too small.
int build_mode_candidates(std::span<ModeCandidate> out,
                          bool include_angle_deltas) noexcept;

// ---------------------------------------------------------------------------
// Exposed normative helpers (tests and CPU/CUDA parity)
// ---------------------------------------------------------------------------

// §7.13.2.17. `delta` is the angle difference in degrees; its absolute value
// is used. filter_type is 1 when either neighbour uses a smooth mode.
int edge_filter_strength(int w, int h, bool smooth_neighbors,
                         int delta) noexcept;

// §7.13.2.18. Five-tap filter of one edge row. `size` is the number of pixels
// to consider (1..129). Reads come from a snapshot of the original samples
// (the spec's derived `edge[]` array); writes go to logical indices
// -1..size-2, so filtered samples are never fed back into later taps.
void intra_edge_filter(EdgeRow& row, int size, int strength) noexcept;

}  // namespace av2enc::intra
