// AV2 v1.0.0 data-driven intra prediction (DIP) — CPU reference.
//
// Normative source: AV2 Bitstream & Decoding Process Specification v1.0.0
// (2026-05-28, commit 38f28e7):
//   §5.4.5      sequence_intra_config(): enable_dip
//   §5.20.7.9   dip_mode_info(): use_dip, dip_transpose, dip_mode
//   §6.19.7.8   DIP mode info semantics
//   §7.13.2.3   data driven intra prediction process
//   §7.13.2.4   DIP features process (11 features, 5-tap dip_avg)
//   §7.13.2.5   DIP transform process (Dip_Weights[6][64][11])
//   §7.13.2.6   DIP resample process (8x8 -> w x h)
//   §7.13.2.1   AboveRow / LeftCol preparation with useDip
//   §8.3.1/.2   TileUseDipCdf / TileDipModeCdf symbol selection
//   §9.3        Default_Use_Dip_Cdf, Default_Dip_Mode_Cdf, Dip_Weights
//
// Scope and constraints:
//   * DIP is a luma-only *DC_PRED* tool.  The syntax condition is
//         enable_dip && YMode == DC_PRED && PaletteSizeY == 0 &&
//         Block_Width > 4 && Block_Height > 4 &&
//         Block_Width * Block_Height >= 128
//     and the decode process applies it only when plane == 0.  Chroma keeps
//     ordinary DC prediction even if use_dip was signalled for the block.
//   * Valid DIP extents are the AV2 transform-block powers of two in 4..64
//     satisfying the availability rule: 8x16, 8x32, 8x64, 16x8, 16x16, ...,
//     64x64.  8x8 and anything with a 4-sample axis is rejected.
//   * 8..12-bit samples are supported by the CPU reference.  The public
//     `predict8()` wrapper is the integration point for the existing 8-bit
//     M1 pipeline.  The CUDA kernel in src/cuda/dip/ is 8-bit, matching M1.
//   * All arithmetic is integer.  Round2(c, 10) is the spec's
//     (c + 512) >> 10 with an arithmetic right shift, including negative c.
//
// The prediction is fully determined by the prepared AboveRow/LeftCol,
// dip_mode and dip_transpose.  No encoder-side tuning is allowed: this module
// is a bit-exact decoder reference.  The decision API only chooses among the
// normative candidates.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace av2enc::dip {

// ---------------------------------------------------------------------------
// Normative constants
// ---------------------------------------------------------------------------

inline constexpr int kNumModes = 6;       // Dip_Weights first dimension
inline constexpr int kNumFeatures = 11;   // f[0..10]
inline constexpr int kTransformSize = 8;  // 8x8 linear predictor
inline constexpr int kTransformSamples = kTransformSize * kTransformSize;
inline constexpr int kMaxBlockDim = 64;
inline constexpr int kMaxEdgeSamples = kMaxBlockDim + kMaxBlockDim / 4;  // 80
inline constexpr int kUseDipContexts = 3;  // DIP_CTXS

// AV2 transform-block extents accepted by this module.
bool valid_block_dim(int v) noexcept;

// §5.20.7.9 availability: Block_Width > 4 && Block_Height > 4 &&
// Block_Width * Block_Height >= 128.  This is independent of enable_dip,
// YMode and palette; see dip_mode_info_present() for the full syntax gate.
bool dip_block_available(int w, int h) noexcept;

// ---------------------------------------------------------------------------
// Source reconstruction and prepared edges
// ---------------------------------------------------------------------------

// Reconstructed samples for one plane.  `stride` is in samples, not bytes.
// `bytes_per_sample` is 1 for 8-bit and 2 for 9..12-bit planes.  The data
// pointer is never null for a valid block.
struct PlaneView {
  const void* data = nullptr;
  int stride = 0;
  int bytes_per_sample = 1;

  static PlaneView from_u8(const std::uint8_t* data, int stride) noexcept {
    return PlaneView{data, stride, 1};
  }
  static PlaneView from_u16(const std::uint16_t* data, int stride) noexcept {
    return PlaneView{data, stride, 2};
  }
};

struct Block {
  PlaneView recon;
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
  int max_x = 0;  // largest visible column, inclusive
  int max_y = 0;  // largest visible row, inclusive
  bool have_left = false;
  bool have_above = false;
  // §7.13.2.1 counts in 4x4 units.  They affect the AboveRow/LeftCol clamp,
  // not the number of samples DIP reads.
  int num4_above_right = 0;
  int num4_below_left = 0;
};

// One prepared edge line.  Logical indices are -1 (corner) and 0..79.
class EdgeArray {
 public:
  static constexpr int kMinIndex = -1;
  static constexpr int kMaxIndex = kMaxEdgeSamples;  // 80
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

struct Edges {
  EdgeArray above;  // AboveRow
  EdgeArray left;   // LeftCol
  int w = 0;
  int h = 0;
  int bit_depth = 8;
  bool have_left = false;
  bool have_above = false;
};

enum class Status : std::uint8_t {
  kOk = 0,
  kInvalidArgument = 1,
  kUnavailable = 2,
};

const char* status_string(Status status) noexcept;

// §7.13.2.1 with useDip = 1 (MrlIndex is necessarily 0 because DIP requires
// YMode == DC_PRED and MRL requires a directional mode).  Fills AboveRow
// 0..w+w/4-1 and LeftCol 0..h+h/4-1 plus index -1.  The extra w/4, h/4
// samples are required by the 5-tap dip_avg downsampler, not by resampling.
Status prepare_edges(const Block& block, int bit_depth, Edges& edges) noexcept;

// ---------------------------------------------------------------------------
// §7.13.2.4 features / §7.13.2.5 transform / §7.13.2.6 resample
// ---------------------------------------------------------------------------

// `above_avg` and `left_avg` are the five outputs of dip_avg(0, w) and
// dip_avg(1, h).  Exposed separately so tests and the CUDA device header can
// share the exact downsampling rule.
Status dip_avg(const EdgeArray& row, int n, std::array<int, 5>& out) noexcept;

// Fills f[0..10] exactly as §7.13.2.4.  Edges must have been prepared for the
// same w/h and at least the required extent.
Status extract_features(const Edges& edges, bool transpose,
                        std::array<int, kNumFeatures>& f) noexcept;

// §7.13.2.5.  `pred` receives an 8x8 row-major array after the optional
// transpose.  `f`, `mode` and `bit_depth` (8..12) must be valid.
void transform(const std::array<int, kNumFeatures>& f, int mode, bool transpose,
               int bit_depth,
               std::array<std::uint16_t, kTransformSamples>& pred) noexcept;

// §7.13.2.6.  Resamples the 8x8 `dip_pred` to w x h using AboveRow/LeftCol.
void resample(const Edges& edges,
              const std::array<std::uint16_t, kTransformSamples>& dip_pred,
              std::uint16_t* dst, std::size_t dst_stride) noexcept;

struct Params {
  int mode = 0;          // dip_mode 0..5
  bool transpose = false;
};

Status validate_params(const Edges& edges, const Params& params) noexcept;

// Full prediction into 16-bit output (8..12-bit samples).
Status predict(const Edges& edges, const Params& params, std::uint16_t* dst,
               std::size_t dst_stride) noexcept;

// 8-bit integration wrapper.  Requires edges.bit_depth == 8 and clips into
// [0,255].  This is the entry point for the M1 8-bit pipeline.
Status predict8(const Edges& edges, const Params& params, std::uint8_t* dst,
                std::size_t dst_stride) noexcept;

Status predict_block(const Block& block, int bit_depth, const Params& params,
                     std::uint16_t* dst, std::size_t dst_stride) noexcept;

Status predict_block8(const Block& block, int bit_depth, const Params& params,
                      std::uint8_t* dst, std::size_t dst_stride) noexcept;

// ---------------------------------------------------------------------------
// Decision API (encoder RDO / mode search)
// ---------------------------------------------------------------------------

struct PredictionCost {
  std::uint64_t sad = 0;
  std::uint64_t sse = 0;
};

PredictionCost prediction_cost(const std::uint16_t* src, std::size_t src_stride,
                               const std::uint16_t* pred,
                               std::size_t pred_stride, int w,
                               int h) noexcept;

PredictionCost prediction_cost8(const std::uint8_t* src,
                                std::size_t src_stride,
                                const std::uint8_t* pred,
                                std::size_t pred_stride, int w,
                                int h) noexcept;

enum class CostMetric : std::uint8_t {
  kSad = 0,
  kSse = 1,
};

struct DecisionParams {
  bool enable_dip = false;
  int y_mode = 0;             // AV2 YMode; DIP requires DC_PRED (0)
  int palette_size_y = 0;     // DIP requires 0
  bool tree_is_chroma_part = false;
  CostMetric metric = CostMetric::kSse;
};

struct Decision {
  bool available = false;
  int mode = 0;
  bool transpose = false;
  PredictionCost cost{};
};

// Enumerates the normative candidates (mode 0..5, transpose 0/1), predicts
// each with the supplied edges, and returns the lowest cost.  Ties are broken
// by (mode, transpose) so the result is deterministic: lower mode first, then
// transpose == false.  Returns kUnavailable when the syntax conditions are not
// met; no candidate is evaluated in that case.
Status decide(const Edges& edges, const std::uint8_t* src,
              std::size_t src_stride, const DecisionParams& params,
              Decision& decision) noexcept;

// ---------------------------------------------------------------------------
// §5.20.7.9 / §6.19.7.8 syntax requirements
// ---------------------------------------------------------------------------

struct SyntaxContext {
  bool enable_dip = false;
  int y_mode = 0;
  int palette_size_y = 0;
  bool tree_is_chroma_part = false;
  int w = 0;
  int h = 0;
  // TileUseDipCdf context: number of above/left neighbours with UseDip == 1.
  // The caller derives it from the neighbour grid; dip_mode_info_present()
  // does not depend on it.
  int use_dip_ctx = 0;
};

// True exactly when dip_mode_info() is present in the bitstream:
//   enable_dip && YMode == DC_PRED && PaletteSizeY == 0 &&
//   Block_Width > 4 && Block_Height > 4 && W*H >= 128 &&
//   TreeType != CHROMA_PART.
bool dip_mode_info_present(const SyntaxContext& context) noexcept;

struct DipModeInfo {
  bool use_dip = false;
  bool dip_transpose = false;  // only present when use_dip
  int dip_mode = 0;            // only present when use_dip; 0..5
};

// Validates a parsed/signalled DipModeInfo against the syntax context.
bool validate_dip_mode_info(const SyntaxContext& context,
                            const DipModeInfo& info,
                            std::string* error = nullptr) noexcept;

// §8.3.1: TileUseDipCdf[ctx], ctx = number of above/left neighbours whose
// UseDip flag is 1.  DIP_CTXS = 3.
int use_dip_context(bool above_use_dip, bool left_use_dip) noexcept;

// Default CDFs from §9.3.  Layout is the repository convention:
//   N symbols -> N-1 cumulative values, base adaptation rate, counter.
inline constexpr std::uint16_t kDefaultUseDipCdf[kUseDipContexts][3] = {
    {11914, 31, 0},
    {5128, 31, 0},
    {3555, 31, 0},
};

inline constexpr std::uint16_t kDefaultDipModeCdf[kNumModes + 1] = {
    5753, 15408, 19070, 22631, 27578, 0, 0,
};

// ---------------------------------------------------------------------------
// Exposed normative table access (tests, CUDA device header)
// ---------------------------------------------------------------------------

std::int16_t dip_weight(int mode, int position, int feature) noexcept;

}  // namespace av2enc::dip
