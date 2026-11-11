// AV2 inter motion estimation — shared host/device core.
//
// This header is the single implementation of the ME cost model, the search
// state machines, the normative subpel interpolation, and the MV candidate
// derivation.  The CPU reference (me.cpp) and the CUDA kernel
// (src/cuda/inter/me/me.cu) instantiate the *same* inline functions, so their
// costs and selected MVs are bit-exact by construction.
//
// Scope
// -----
//   * 8-bit 4:2:0 luma motion estimation.  Chroma planes are part of the data
//     model and the cost/interp helpers accept plane indices 1/2, but the
//     search itself runs on luma only (the MC component owns chroma
//     prediction and chroma MV derivation).
//   * 8-bit samples, 1/16-sample MV precision (SUBPEL_BITS = 4), matching AV2
//     §7.13.3.  The search refinement stages are integer (16/16), half (8/16),
//     quarter (4/16), eighth (2/16) and sixteenth (1/16) sample.
//   * Blocks from 1x1 to 128x128 luma samples, including partial blocks at the
//     right/bottom frame edge.
//   * Multiple references: the caller supplies ref indices per job.
//
// Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
// (2026-05-28, commit 38f28e7), §7.13.3.14 (subpel prediction) and
// §7.13.3.16 (rounding variables).  Subpel_Filters below is transcribed from
// the spec text (6 filters x 16 phases x 8 taps, all phases sum to 128).
//
// This is encoder-side decision data only.  Nothing here writes bitstream
// syntax, and the emitted MV still goes through MC/entropy in other components.

#ifndef AV2ENC_ENCODER_INTER_ME_ME_CORE_H_
#define AV2ENC_ENCODER_INTER_ME_ME_CORE_H_

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define AV2ENC_ME_HD __host__ __device__ __forceinline__
#if defined(__CUDA_ARCH__)
#define AV2ENC_ME_TABLE __device__ constexpr
#else
#define AV2ENC_ME_TABLE inline constexpr
#endif
#else
#define AV2ENC_ME_HD inline
#define AV2ENC_ME_TABLE inline constexpr
#endif

namespace av2enc::me {

// --- AV2 constants ---------------------------------------------------------
constexpr int kSubpelBits = 4;
constexpr int kSubpelOne = 1 << kSubpelBits;   // 16 MV units per luma sample
constexpr int kSubpelMask = kSubpelOne - 1;
constexpr int kInterRound0 = 3;                // §7.13.3.16
constexpr int kInterRound1 = 11;               // single-reference prediction
constexpr int kFilterBits = 7;

constexpr int kMeMaxRefs = 8;                  // active refs per job
constexpr int kMeMaxPredictors = 16;           // explicit MVPs per job
constexpr int kMeMaxMerge = 8;                 // spatial merge candidates kept
constexpr int kMeMaxBlockDim = 128;
constexpr int kMeMaxInterpFilter = 5;

// --- enums / option values -------------------------------------------------
enum MeMetric { kMeSad = 0, kMeSatd = 1 };
enum MeSearchMethod { kMeFull = 0, kMeDiamond = 1, kMeTestZone = 2 };
enum MeSubpelStages {
  kMeSubpelNone = 0,
  kMeSubpelHalf = 1,        // 8/16 sample
  kMeSubpelQuarter = 2,     // 4/16 sample
  kMeSubpelEighth = 4,      // 2/16 sample
  kMeSubpelSixteenth = 8,   // 1/16 sample
};
// Values map to the Subpel_Filters row.  kMeInterpBilinear is the normative
// 8-tap bilinear filter (row 3); it reduces to two-tap linear interpolation.
enum MeInterp {
  kMeInterpRegular = 0,
  kMeInterpSmooth = 1,
  kMeInterpSharp = 2,
  kMeInterpBilinear = 3,
  kMeInterpRegular4 = 4,
  kMeInterpSmooth4 = 5,
};
enum MeSearchStage {
  kMeStageNone = -1,
  kMeStagePredictor = 0,
  kMeStageInteger = 1,
  kMeStageHalf = 2,
  kMeStageQuarter = 3,
  kMeStageEighth = 4,
  kMeStageSixteenth = 5,
};

// --- data model ------------------------------------------------------------
// One plane.  For refs used by the search, plane[0] is required.  Chroma
// planes (1, 2) are half width/height and are used only by the cost/interp
// helpers; the caller supplies plane coordinates and chroma-precision MVs.
struct MeCorePlane {
  const std::uint8_t* data = nullptr;
  int stride = 0;   // samples per row; must be >= width
  int width = 0;
  int height = 0;
};

struct MeCoreFrame {
  MeCorePlane plane[3];
  int width = 0;    // luma width
  int height = 0;   // luma height
};

struct MeCoreRef {
  MeCorePlane plane[3];
};

// Motion-vector field on the 4x4 luma MI grid.  `stride` <= 0 means `cols`.
// `valid` may be null (all cells valid).  `ref_index` may be null (all cells
// use the caller-provided default reference).  All pointers are host pointers
// for the CPU API and device pointers for the CUDA API.
struct MeCoreMvField {
  const std::int32_t* mv_x = nullptr;
  const std::int32_t* mv_y = nullptr;
  const std::int16_t* ref_index = nullptr;
  const std::uint8_t* valid = nullptr;
  int cols = 0;
  int rows = 0;
  int stride = 0;
};

struct MeMvCandidate {
  std::int32_t mv_x = 0;
  std::int32_t mv_y = 0;
  std::int16_t ref_index = 0;
};

// One ME job.  `x`,`y` are the luma origin in the current frame; `w`,`h` are
// the nominal block size.  The visible part is clipped to the current frame,
// so a partial block simply has a smaller cost area.  `pred_mv` is an optional
// explicit predictor list (e.g. NEWMV/NEARMV candidates); the spatial merge
// candidates come from `MeCoreMvField` supplied to the batch call.
struct MeCoreJob {
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
  int num_refs = 0;
  std::int8_t ref_indices[kMeMaxRefs] = {};
  int num_pred = 0;
  std::int32_t pred_mv[kMeMaxPredictors][2] = {};
};

struct MeCoreOptions {
  int method = kMeFull;
  int metric = kMeSad;
  int range = 16;              // integer search radius in luma samples
  int integer_step = 1;        // full-search stride, samples
  int max_candidates = 0;      // full-search cap; 0 = unlimited
  int max_iterations = 4;      // per-step cap for diamond/TZ
  int diamond_step = 0;        // 0 = auto (max(1, range/4))
  int tz_start_step = 0;       // 0 = auto (range)
  int subpel_stages = kMeSubpelHalf | kMeSubpelQuarter;
  int subpel_interp = kMeInterpRegular;
  int subpel_iterations = 2;
  int use_predictors = 1;
  int use_merge_candidates = 1;
  int max_merge_candidates = 4;
};

struct MeCoreResult {
  std::int32_t mv_x = 0;       // 1/16 luma samples
  std::int32_t mv_y = 0;
  int ref_index = -1;
  std::int32_t cost = 0;
  std::int32_t evals = 0;      // cost evaluations, identical on CPU and GPU
  std::int32_t stage = kMeStageNone;
  int visible_w = 0;
  int visible_h = 0;
};

struct MeCostJob {
  int plane = 0;
  int ref_index = 0;
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
  std::int32_t mv_x = 0;
  std::int32_t mv_y = 0;
  int metric = kMeSad;
  int interp = kMeInterpRegular;
};

struct MeInterpJob {
  int plane = 0;
  int ref_index = 0;
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
  std::int32_t mv_x = 0;
  std::int32_t mv_y = 0;
  int interp = kMeInterpRegular;
  std::int32_t out_offset = 0;
};

// --- small integer helpers -------------------------------------------------
AV2ENC_ME_HD int me_imin(int a, int b) { return a < b ? a : b; }
AV2ENC_ME_HD int me_imax(int a, int b) { return a > b ? a : b; }
AV2ENC_ME_HD int me_iclamp(int lo, int v, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}
AV2ENC_ME_HD std::int32_t me_round2(std::int32_t x, int n) {
  if (n == 0) {
    return x;
  }
  return (x + (static_cast<std::int32_t>(1) << (n - 1))) >> n;
}
AV2ENC_ME_HD bool me_is_integer_mv(std::int32_t mv_x, std::int32_t mv_y) {
  return ((mv_x | mv_y) & kSubpelMask) == 0;
}

// --- normative Subpel_Filters[6][16][8] (AV2 §7.13.3.14) -------------------
AV2ENC_ME_TABLE std::int16_t kMeSubpelFilters[6][16][8] = {
    // 0: REGULAR (8-tap)
    {
        {0, 0, 0, 128, 0, 0, 0, 0},
        {0, 2, -6, 126, 8, -2, 0, 0},
        {0, 2, -10, 122, 18, -4, 0, 0},
        {0, 2, -12, 116, 28, -8, 2, 0},
        {0, 2, -14, 110, 38, -10, 2, 0},
        {0, 2, -14, 102, 48, -12, 2, 0},
        {0, 2, -16, 94, 58, -12, 2, 0},
        {0, 2, -14, 84, 66, -12, 2, 0},
        {0, 2, -14, 76, 76, -14, 2, 0},
        {0, 2, -12, 66, 84, -14, 2, 0},
        {0, 2, -12, 58, 94, -16, 2, 0},
        {0, 2, -12, 48, 102, -14, 2, 0},
        {0, 2, -10, 38, 110, -14, 2, 0},
        {0, 2, -8, 28, 116, -12, 2, 0},
        {0, 0, -4, 18, 122, -10, 2, 0},
        {0, 0, -2, 8, 126, -6, 2, 0},
    },
    // 1: SMOOTH (8-tap)
    {
        {0, 0, 0, 128, 0, 0, 0, 0},
        {0, 2, 28, 62, 34, 2, 0, 0},
        {0, 0, 26, 62, 36, 4, 0, 0},
        {0, 0, 22, 62, 40, 4, 0, 0},
        {0, 0, 20, 60, 42, 6, 0, 0},
        {0, 0, 18, 58, 44, 8, 0, 0},
        {0, 0, 16, 56, 46, 10, 0, 0},
        {0, -2, 16, 54, 48, 12, 0, 0},
        {0, -2, 14, 52, 52, 14, -2, 0},
        {0, 0, 12, 48, 54, 16, -2, 0},
        {0, 0, 10, 46, 56, 16, 0, 0},
        {0, 0, 8, 44, 58, 18, 0, 0},
        {0, 0, 6, 42, 60, 20, 0, 0},
        {0, 0, 4, 40, 62, 22, 0, 0},
        {0, 0, 4, 36, 62, 26, 0, 0},
        {0, 0, 2, 34, 62, 28, 2, 0},
    },
    // 2: SHARP (8-tap)
    {
        {0, 0, 0, 128, 0, 0, 0, 0},
        {-2, 2, -6, 126, 8, -2, 2, 0},
        {-2, 6, -12, 124, 16, -6, 4, -2},
        {-2, 8, -18, 120, 26, -10, 6, -2},
        {-4, 10, -22, 116, 38, -14, 6, -2},
        {-4, 10, -22, 108, 48, -18, 8, -2},
        {-4, 10, -24, 100, 60, -20, 8, -2},
        {-4, 10, -24, 90, 70, -22, 10, -2},
        {-4, 12, -24, 80, 80, -24, 12, -4},
        {-2, 10, -22, 70, 90, -24, 10, -4},
        {-2, 8, -20, 60, 100, -24, 10, -4},
        {-2, 8, -18, 48, 108, -22, 10, -4},
        {-2, 6, -14, 38, 116, -22, 10, -4},
        {-2, 6, -10, 26, 120, -18, 8, -2},
        {-2, 4, -6, 16, 124, -12, 6, -2},
        {0, 2, -2, 8, 126, -6, 2, -2},
    },
    // 3: BILINEAR (8-tap representation, two non-zero taps)
    {
        {0, 0, 0, 128, 0, 0, 0, 0},
        {0, 0, 0, 120, 8, 0, 0, 0},
        {0, 0, 0, 112, 16, 0, 0, 0},
        {0, 0, 0, 104, 24, 0, 0, 0},
        {0, 0, 0, 96, 32, 0, 0, 0},
        {0, 0, 0, 88, 40, 0, 0, 0},
        {0, 0, 0, 80, 48, 0, 0, 0},
        {0, 0, 0, 72, 56, 0, 0, 0},
        {0, 0, 0, 64, 64, 0, 0, 0},
        {0, 0, 0, 56, 72, 0, 0, 0},
        {0, 0, 0, 48, 80, 0, 0, 0},
        {0, 0, 0, 40, 88, 0, 0, 0},
        {0, 0, 0, 32, 96, 0, 0, 0},
        {0, 0, 0, 24, 104, 0, 0, 0},
        {0, 0, 0, 16, 112, 0, 0, 0},
        {0, 0, 0, 8, 120, 0, 0, 0},
    },
    // 4: REGULAR 4-tap (used for blocks with a dimension <= 4)
    {
        {0, 0, 0, 128, 0, 0, 0, 0},
        {0, 0, -4, 126, 8, -2, 0, 0},
        {0, 0, -8, 122, 18, -4, 0, 0},
        {0, 0, -10, 116, 28, -6, 0, 0},
        {0, 0, -12, 110, 38, -8, 0, 0},
        {0, 0, -12, 102, 48, -10, 0, 0},
        {0, 0, -14, 94, 58, -10, 0, 0},
        {0, 0, -12, 84, 66, -10, 0, 0},
        {0, 0, -12, 76, 76, -12, 0, 0},
        {0, 0, -10, 66, 84, -12, 0, 0},
        {0, 0, -10, 58, 94, -14, 0, 0},
        {0, 0, -10, 48, 102, -12, 0, 0},
        {0, 0, -8, 38, 110, -12, 0, 0},
        {0, 0, -6, 28, 116, -10, 0, 0},
        {0, 0, -4, 18, 122, -8, 0, 0},
        {0, 0, -2, 8, 126, -4, 0, 0},
    },
    // 5: SMOOTH 4-tap
    {
        {0, 0, 0, 128, 0, 0, 0, 0},
        {0, 0, 30, 62, 34, 2, 0, 0},
        {0, 0, 26, 62, 36, 4, 0, 0},
        {0, 0, 22, 62, 40, 4, 0, 0},
        {0, 0, 20, 60, 42, 6, 0, 0},
        {0, 0, 18, 58, 44, 8, 0, 0},
        {0, 0, 16, 56, 46, 10, 0, 0},
        {0, 0, 14, 54, 48, 12, 0, 0},
        {0, 0, 12, 52, 52, 12, 0, 0},
        {0, 0, 12, 48, 54, 14, 0, 0},
        {0, 0, 10, 46, 56, 16, 0, 0},
        {0, 0, 8, 44, 58, 18, 0, 0},
        {0, 0, 6, 42, 60, 20, 0, 0},
        {0, 0, 4, 40, 62, 22, 0, 0},
        {0, 0, 4, 36, 62, 26, 0, 0},
        {0, 0, 2, 34, 62, 30, 0, 0},
    },
};

// --- subpel interpolation --------------------------------------------------
// One sample of the normative two-pass 8-tap interpolation.  `xq`/`yq` are
// absolute reference coordinates in 1/16 luma samples.  Every tap is clipped
// to the visible plane, matching the spec's Clip3(firstX,lastX,...).
AV2ENC_ME_HD int me_interp_sample(const MeCorePlane& ref, int xq, int yq,
                                  int filter) {
  const int ix = xq >> kSubpelBits;
  const int iy = yq >> kSubpelBits;
  const int fx = xq & kSubpelMask;
  const int fy = yq & kSubpelMask;
  if ((fx | fy) == 0) {
    const int cx = me_iclamp(0, ix, ref.width - 1);
    const int cy = me_iclamp(0, iy, ref.height - 1);
    return static_cast<int>(
        ref.data[static_cast<std::size_t>(cy) * static_cast<std::size_t>(ref.stride) +
                 static_cast<std::size_t>(cx)]);
  }
  const std::int16_t(*coeff)[8] =
      kMeSubpelFilters[me_iclamp(0, filter, kMeMaxInterpFilter)];
  std::int32_t h[8];
  for (int t = 0; t < 8; ++t) {
    const int ry = me_iclamp(0, iy + t - 3, ref.height - 1);
    const std::uint8_t* row =
        ref.data + static_cast<std::size_t>(ry) * static_cast<std::size_t>(ref.stride);
    std::int32_t s = 0;
    for (int u = 0; u < 8; ++u) {
      const int rx = me_iclamp(0, ix + u - 3, ref.width - 1);
      s += static_cast<std::int32_t>(coeff[fx][u]) *
           static_cast<std::int32_t>(row[rx]);
    }
    h[t] = me_round2(s, kInterRound0);
  }
  std::int32_t s = 0;
  for (int t = 0; t < 8; ++t) {
    s += static_cast<std::int32_t>(coeff[fy][t]) * h[t];
  }
  return me_iclamp(0, static_cast<int>(me_round2(s, kInterRound1)), 255);
}

// Predicted sample for current-frame position (px,py) with MV (mv_x,mv_y) in
// 1/16 luma samples.  Integer positions bypass the filter.
AV2ENC_ME_HD int me_sample_at(const MeCorePlane& ref, int px, int py,
                              std::int32_t mv_x, std::int32_t mv_y, int filter) {
  const int xq = (px << kSubpelBits) + mv_x;
  const int yq = (py << kSubpelBits) + mv_y;
  if (((xq | yq) & kSubpelMask) == 0) {
    return static_cast<int>(
        ref.data[static_cast<std::size_t>(yq >> kSubpelBits) *
                     static_cast<std::size_t>(ref.stride) +
                 static_cast<std::size_t>(xq >> kSubpelBits)]);
  }
  return me_interp_sample(ref, xq, yq, filter);
}

// --- cost model ------------------------------------------------------------
// SAD over rows [r0,r1) with stride `rstep`.  The CUDA warp cost uses
// r0=lane, rstep=32 so each lane owns a subset of rows.
AV2ENC_ME_HD std::int32_t me_sad_rows(const MeCorePlane& cur,
                                      const MeCorePlane& ref, int x, int y,
                                      int w, int mv_x, std::int32_t mv_y_int,
                                      int r0, int r1, int rstep) {
  const int ix = x + (mv_x >> kSubpelBits);
  const int iy = y + mv_y_int;
  std::int32_t sum = 0;
  for (int r = r0; r < r1; r += rstep) {
    const std::uint8_t* cr =
        cur.data + static_cast<std::size_t>(y + r) *
                       static_cast<std::size_t>(cur.stride) +
        static_cast<std::size_t>(x);
    const std::uint8_t* rr =
        ref.data + static_cast<std::size_t>(iy + r) *
                       static_cast<std::size_t>(ref.stride) +
        static_cast<std::size_t>(ix);
    for (int c = 0; c < w; ++c) {
      const int d = static_cast<int>(cr[c]) - static_cast<int>(rr[c]);
      sum += d < 0 ? -d : d;
    }
  }
  return sum;
}

AV2ENC_ME_HD std::int32_t me_sad_subpel_rows(const MeCorePlane& cur,
                                             const MeCorePlane& ref, int x,
                                             int y, int w, std::int32_t mv_x,
                                             std::int32_t mv_y, int filter,
                                             int r0, int r1, int rstep) {
  std::int32_t sum = 0;
  for (int r = r0; r < r1; r += rstep) {
    const std::uint8_t* cr =
        cur.data + static_cast<std::size_t>(y + r) *
                       static_cast<std::size_t>(cur.stride) +
        static_cast<std::size_t>(x);
    for (int c = 0; c < w; ++c) {
      const int p = me_sample_at(ref, x + c, y + r, mv_x, mv_y, filter);
      const int d = static_cast<int>(cr[c]) - p;
      sum += d < 0 ? -d : d;
    }
  }
  return sum;
}

AV2ENC_ME_HD void me_hadamard8(int* v) {
  for (int len = 1; len < 8; len <<= 1) {
    for (int i = 0; i < 8; i += len << 1) {
      for (int j = 0; j < len; ++j) {
        const int a = v[i + j];
        const int b = v[i + j + len];
        v[i + j] = a + b;
        v[i + j + len] = a - b;
      }
    }
  }
}

AV2ENC_ME_HD void me_hadamard8_2d(int* v) {
  for (int r = 0; r < 8; ++r) {
    me_hadamard8(v + r * 8);
  }
  for (int len = 1; len < 8; len <<= 1) {
    for (int i = 0; i < 8; i += len << 1) {
      for (int j = 0; j < len; ++j) {
        for (int c = 0; c < 8; ++c) {
          const int a = v[(i + j) * 8 + c];
          const int b = v[(i + j + len) * 8 + c];
          v[(i + j) * 8 + c] = a + b;
          v[(i + j + len) * 8 + c] = a - b;
        }
      }
    }
  }
}

// One 8x8 SATD tile.  Tiles are zero-padded at the right/bottom edge: pixels
// outside the visible block contribute a zero residual.  The unnormalised
// 2-D Hadamard has gain 16 for 8x8, so the sum of |coeff| is scaled by
// (sum + 8) >> 4, which keeps SATD and SAD in the same numeric range.
AV2ENC_ME_HD std::int32_t me_satd_unit(const MeCorePlane& cur,
                                       const MeCorePlane& ref, int x, int y,
                                       int w, int h, std::int32_t mv_x,
                                       std::int32_t mv_y, int filter,
                                       int tx, int ty) {
  int m[64];
  for (int r = 0; r < 8; ++r) {
    for (int c = 0; c < 8; ++c) {
      const int rr = ty + r;
      const int cc = tx + c;
      int v = 0;
      if (rr < h && cc < w) {
        const int cur_v = static_cast<int>(
            cur.data[static_cast<std::size_t>(y + rr) *
                         static_cast<std::size_t>(cur.stride) +
                     static_cast<std::size_t>(x + cc)]);
        const int ref_v =
            me_sample_at(ref, x + cc, y + rr, mv_x, mv_y, filter);
        v = cur_v - ref_v;
      }
      m[r * 8 + c] = v;
    }
  }
  me_hadamard8_2d(m);
  std::int32_t sum = 0;
  for (int i = 0; i < 64; ++i) {
    sum += m[i] < 0 ? -m[i] : m[i];
  }
  return (sum + 8) >> 4;
}

AV2ENC_ME_HD int me_satd_unit_count(int w, int h) {
  return ((w + 7) >> 3) * ((h + 7) >> 3);
}

AV2ENC_ME_HD std::int32_t me_cost_plane(const MeCorePlane& cur,
                                        const MeCorePlane& ref, int x, int y,
                                        int w, int h, std::int32_t mv_x,
                                        std::int32_t mv_y, int metric,
                                        int filter) {
  if (metric == kMeSad) {
    if (me_is_integer_mv(mv_x, mv_y)) {
      return me_sad_rows(cur, ref, x, y, w, mv_x, mv_y >> kSubpelBits, 0, h,
                         1);
    }
    return me_sad_subpel_rows(cur, ref, x, y, w, mv_x, mv_y, filter, 0, h, 1);
  }
  const int tiles_x = (w + 7) >> 3;
  const int tiles_y = (h + 7) >> 3;
  std::int32_t sum = 0;
  for (int ty = 0; ty < tiles_y; ++ty) {
    for (int tx = 0; tx < tiles_x; ++tx) {
      sum += me_satd_unit(cur, ref, x, y, w, h, mv_x, mv_y, filter, tx << 3,
                          ty << 3);
    }
  }
  return sum;
}

// Scalar cost functor used by the CPU search.
struct MeScalarCost {
  AV2ENC_ME_HD std::int32_t operator()(const MeCorePlane& cur,
                                       const MeCorePlane& ref, int x, int y,
                                       int w, int h, std::int32_t mv_x,
                                       std::int32_t mv_y, int metric,
                                       int filter) const {
    return me_cost_plane(cur, ref, x, y, w, h, mv_x, mv_y, metric, filter);
  }
};

// --- MV prediction / merge candidates --------------------------------------
AV2ENC_ME_HD int me_mv_field_stride(const MeCoreMvField& f) {
  return f.stride > 0 ? f.stride : f.cols;
}

// Reads one MI cell.  Returns false when the cell is outside the grid or
// marked invalid.  `default_ref` is used when the field has no ref array.
AV2ENC_ME_HD bool me_mv_field_read(const MeCoreMvField& f, int cx, int cy,
                                   int default_ref, MeMvCandidate* out) {
  if (cx < 0 || cy < 0 || cx >= f.cols || cy >= f.rows) {
    return false;
  }
  const int stride = me_mv_field_stride(f);
  const std::size_t idx =
      static_cast<std::size_t>(cy) * static_cast<std::size_t>(stride) +
      static_cast<std::size_t>(cx);
  if (f.valid != nullptr && f.valid[idx] == 0) {
    return false;
  }
  if (f.mv_x == nullptr || f.mv_y == nullptr) {
    return false;
  }
  out->mv_x = f.mv_x[idx];
  out->mv_y = f.mv_y[idx];
  out->ref_index = f.ref_index != nullptr ? f.ref_index[idx]
                                          : static_cast<std::int16_t>(default_ref);
  return true;
}

AV2ENC_ME_HD bool me_mv_candidate_equal(const MeMvCandidate& a,
                                        const MeMvCandidate& b) {
  return a.mv_x == b.mv_x && a.mv_y == b.mv_y && a.ref_index == b.ref_index;
}

// Spatial merge candidate derivation on the 4x4 MI grid, in AV1/AV2 scan
// order: A1 (left), B1 (above), B0 (above-right), A0 (left-below), B2
// (above-left).  Duplicates are removed by (mv, ref).  A zero-MV candidate
// with `default_ref` is appended when absent.  Returns the number written.
AV2ENC_ME_HD int me_collect_mv_candidates_core(const MeCoreMvField& f, int x,
                                               int y, int w, int h,
                                               int default_ref,
                                               MeMvCandidate* out, int max_out) {
  if (out == nullptr || max_out <= 0) {
    return 0;
  }
  int count = 0;
  const int cx_a1 = (x - 1) >> 2;
  const int cy_a1 = (y + h - 1) >> 2;
  const int cx_b1 = (x + w - 1) >> 2;
  const int cy_b1 = (y - 1) >> 2;
  const int cx_b0 = (x + w) >> 2;
  const int cy_b0 = (y - 1) >> 2;
  const int cx_a0 = (x - 1) >> 2;
  const int cy_a0 = (y + h) >> 2;
  const int cx_b2 = (x - 1) >> 2;
  const int cy_b2 = (y - 1) >> 2;
  const int pos[5][2] = {
      {cx_a1, cy_a1}, {cx_b1, cy_b1}, {cx_b0, cy_b0},
      {cx_a0, cy_a0}, {cx_b2, cy_b2},
  };
  for (int p = 0; p < 5 && count < max_out; ++p) {
    MeMvCandidate c;
    if (!me_mv_field_read(f, pos[p][0], pos[p][1], default_ref, &c)) {
      continue;
    }
    bool dup = false;
    for (int i = 0; i < count; ++i) {
      if (me_mv_candidate_equal(out[i], c)) {
        dup = true;
        break;
      }
    }
    if (!dup) {
      out[count++] = c;
    }
  }
  if (count < max_out) {
    MeMvCandidate zero;
    zero.mv_x = 0;
    zero.mv_y = 0;
    zero.ref_index = static_cast<std::int16_t>(default_ref);
    bool dup = false;
    for (int i = 0; i < count; ++i) {
      if (me_mv_candidate_equal(out[i], zero)) {
        dup = true;
        break;
      }
    }
    if (!dup) {
      out[count++] = zero;
    }
  }
  return count;
}

// --- search ----------------------------------------------------------------
AV2ENC_ME_HD int me_full_search_step(int range, int step, int max_candidates) {
  if (max_candidates <= 0 || range <= 0) {
    return step;
  }
  int side = 1;
  while ((side + 2) * (side + 2) <= max_candidates) {
    side += 2;
  }
  const int half = me_imax(1, side / 2);
  if (half >= range) {
    return step;
  }
  const int needed = (range + half - 1) / half;
  return needed > step ? needed : step;
}

AV2ENC_ME_HD std::int32_t me_clamp_mv_x(std::int32_t mv, int x, int w,
                                        int ref_w, bool allow_fraction) {
  std::int32_t lo = -static_cast<std::int32_t>(x) * kSubpelOne;
  std::int32_t hi =
      static_cast<std::int32_t>(ref_w - x - w) * kSubpelOne;
  if (allow_fraction) {
    lo -= kSubpelMask;
    hi += kSubpelMask;
  }
  return me_iclamp(lo, mv, hi);
}

AV2ENC_ME_HD std::int32_t me_clamp_mv_y(std::int32_t mv, int y, int h,
                                        int ref_h, bool allow_fraction) {
  std::int32_t lo = -static_cast<std::int32_t>(y) * kSubpelOne;
  std::int32_t hi =
      static_cast<std::int32_t>(ref_h - y - h) * kSubpelOne;
  if (allow_fraction) {
    lo -= kSubpelMask;
    hi += kSubpelMask;
  }
  return me_iclamp(lo, mv, hi);
}

// Clamp a candidate to the requested search radius around `base`, then to the
// reference plane.  Used by diamond/TZ, whose centre moves between iterations.
AV2ENC_ME_HD std::int32_t me_clamp_search_x(std::int32_t mv,
                                            std::int32_t base,
                                            std::int32_t range_units, int x,
                                            int w, int ref_w,
                                            bool allow_fraction) {
  const std::int32_t v =
      me_iclamp(base - range_units, mv, base + range_units);
  return me_clamp_mv_x(v, x, w, ref_w, allow_fraction);
}

AV2ENC_ME_HD std::int32_t me_clamp_search_y(std::int32_t mv,
                                            std::int32_t base,
                                            std::int32_t range_units, int y,
                                            int h, int ref_h,
                                            bool allow_fraction) {
  const std::int32_t v =
      me_iclamp(base - range_units, mv, base + range_units);
  return me_clamp_mv_y(v, y, h, ref_h, allow_fraction);
}

AV2ENC_ME_HD int me_add_unique_predictor(std::int32_t (*list)[2], int count,
                                         int max_count, std::int32_t mv_x,
                                         std::int32_t mv_y) {
  for (int i = 0; i < count; ++i) {
    if (list[i][0] == mv_x && list[i][1] == mv_y) {
      return count;
    }
  }
  if (count < max_count) {
    list[count][0] = mv_x;
    list[count][1] = mv_y;
    return count + 1;
  }
  return count;
}

// Search one job against all listed references.  `cost` is the scalar or warp
// cooperative cost functor described above.  All decision-making uses integer
// comparisons, and candidate iteration order is fixed, so the CPU and CUDA
// instantiations return the same MV, ref index, cost and evaluation count.
template <class CostFn>
AV2ENC_ME_HD MeCoreResult me_search_core(const MeCoreFrame& cur,
                                         const MeCoreRef* refs,
                                         const MeCoreJob& job,
                                         const MeCoreOptions& opt,
                                         const MeCoreMvField* field,
                                         const CostFn& cost) {
  MeCoreResult res;
  const int w = me_imin(job.w, cur.width - job.x);
  const int h = me_imin(job.h, cur.height - job.y);
  res.visible_w = w;
  res.visible_h = h;
  if (w <= 0 || h <= 0 || job.num_refs <= 0) {
    return res;
  }

  // Predictor list: explicit MVPs, then spatial merge MVs, then zero.
  std::int32_t pred[kMeMaxPredictors + kMeMaxMerge + 1][2];
  int num_pred = 0;
  const int max_pred = kMeMaxPredictors + kMeMaxMerge + 1;
  if (opt.use_predictors != 0) {
    for (int i = 0; i < job.num_pred && i < kMeMaxPredictors; ++i) {
      num_pred = me_add_unique_predictor(pred, num_pred, max_pred,
                                         job.pred_mv[i][0], job.pred_mv[i][1]);
    }
  }
  if (opt.use_merge_candidates != 0 && field != nullptr && field->cols > 0 &&
      field->rows > 0) {
    MeMvCandidate cand[kMeMaxMerge];
    int max_merge = opt.max_merge_candidates;
    if (max_merge < 0) {
      max_merge = 0;
    }
    if (max_merge > kMeMaxMerge) {
      max_merge = kMeMaxMerge;
    }
    const int default_ref = job.ref_indices[0];
    const int n =
        me_collect_mv_candidates_core(*field, job.x, job.y, job.w, job.h,
                                      default_ref, cand, max_merge);
    for (int i = 0; i < n; ++i) {
      num_pred = me_add_unique_predictor(pred, num_pred, max_pred, cand[i].mv_x,
                                         cand[i].mv_y);
    }
  }
  num_pred = me_add_unique_predictor(pred, num_pred, max_pred, 0, 0);
  if (num_pred <= 0) {
    pred[0][0] = 0;
    pred[0][1] = 0;
    num_pred = 1;
  }

  std::int32_t best_cost = 0;
  std::int32_t best_x = 0;
  std::int32_t best_y = 0;
  int best_ref = -1;
  int best_stage = kMeStageNone;
  bool have_best = false;
  std::int32_t evals = 0;

  for (int ri = 0; ri < job.num_refs; ++ri) {
    const int ref_index = job.ref_indices[ri];
    const MeCorePlane& ref = refs[ref_index].plane[0];
    const int ref_w = ref.width;
    const int ref_h = ref.height;

    // Predictor stage (MVPs may be fractional; they are used as-is unless the
    // block would leave the reference).
    std::int32_t cbest = 0;
    std::int32_t cx = 0;
    std::int32_t cy = 0;
    int cstage = kMeStageNone;
    bool c_valid = false;
    for (int i = 0; i < num_pred; ++i) {
      const std::int32_t mx =
          me_clamp_mv_x(pred[i][0], job.x, w, ref_w, false);
      const std::int32_t my =
          me_clamp_mv_y(pred[i][1], job.y, h, ref_h, false);
      const std::int32_t c =
          cost(cur.plane[0], ref, job.x, job.y, w, h, mx, my, opt.metric,
               opt.subpel_interp);
      ++evals;
      if (!c_valid || c < cbest) {
        cbest = c;
        cx = mx;
        cy = my;
        cstage = kMeStagePredictor;
        c_valid = true;
      }
    }

    // Integer search around the best predictor.  `base_x`/`base_y` anchor the
    // search radius for diamond/TZ, so iterative moves cannot drift outside
    // the requested range.
    const std::int32_t base_x = cx;
    const std::int32_t base_y = cy;
    const std::int32_t range_units =
        static_cast<std::int32_t>(opt.range) * kSubpelOne;
    if (opt.range > 0) {
      if (opt.method == kMeFull) {
        const int step =
            me_full_search_step(opt.range, me_imax(1, opt.integer_step),
                                opt.max_candidates);
        const std::int32_t start_x = cx;
        const std::int32_t start_y = cy;
        for (int dy = -opt.range; dy <= opt.range; dy += step) {
          for (int dx = -opt.range; dx <= opt.range; dx += step) {
            if (dx == 0 && dy == 0) {
              continue;
            }
            const std::int32_t mx = me_clamp_mv_x(
                start_x + static_cast<std::int32_t>(dx) * kSubpelOne, job.x,
                w, ref_w, false);
            const std::int32_t my = me_clamp_mv_y(
                start_y + static_cast<std::int32_t>(dy) * kSubpelOne, job.y,
                h, ref_h, false);
            const std::int32_t c =
                cost(cur.plane[0], ref, job.x, job.y, w, h, mx, my,
                     opt.metric, opt.subpel_interp);
            ++evals;
            if (c < cbest) {
              cbest = c;
              cx = mx;
              cy = my;
              cstage = kMeStageInteger;
            }
          }
        }
      } else if (opt.method == kMeDiamond) {
        int step = opt.diamond_step > 0 ? opt.diamond_step
                                        : me_imax(1, opt.range / 4);
        int iter = 0;
        while (step >= 1 && iter < opt.max_iterations) {
          bool improved = false;
          const int off[4][2] = {{0, -1}, {-1, 0}, {1, 0}, {0, 1}};
          for (int k = 0; k < 4; ++k) {
            const std::int32_t mx = me_clamp_search_x(
                cx + static_cast<std::int32_t>(off[k][0] * step) * kSubpelOne,
                base_x, range_units, job.x, w, ref_w, false);
            const std::int32_t my = me_clamp_search_y(
                cy + static_cast<std::int32_t>(off[k][1] * step) * kSubpelOne,
                base_y, range_units, job.y, h, ref_h, false);
            if (mx == cx && my == cy) {
              continue;
            }
            const std::int32_t c =
                cost(cur.plane[0], ref, job.x, job.y, w, h, mx, my,
                     opt.metric, opt.subpel_interp);
            ++evals;
            if (c < cbest) {
              cbest = c;
              cx = mx;
              cy = my;
              improved = true;
              cstage = kMeStageInteger;
            }
          }
          if (!improved) {
            step >>= 1;
          }
          ++iter;
        }
      } else {
        // TZ-style coarse-to-fine grid.  Each step level moves while it
        // improves, then halves.  The 8-neighbour ring is evaluated in a fixed
        // order for deterministic tie-breaking.
        int step = opt.tz_start_step > 0 ? opt.tz_start_step : opt.range;
        const int off[8][2] = {{-1, -1}, {0, -1}, {1, -1}, {-1, 0},
                               {1, 0},   {-1, 1}, {0, 1},  {1, 1}};
        while (step >= 1) {
          for (int it = 0; it < opt.max_iterations; ++it) {
            bool improved = false;
            for (int k = 0; k < 8; ++k) {
              const std::int32_t mx = me_clamp_search_x(
                  cx + static_cast<std::int32_t>(off[k][0] * step) * kSubpelOne,
                  base_x, range_units, job.x, w, ref_w, false);
              const std::int32_t my = me_clamp_search_y(
                  cy + static_cast<std::int32_t>(off[k][1] * step) * kSubpelOne,
                  base_y, range_units, job.y, h, ref_h, false);
              if (mx == cx && my == cy) {
                continue;
              }
              const std::int32_t c =
                  cost(cur.plane[0], ref, job.x, job.y, w, h, mx, my,
                       opt.metric, opt.subpel_interp);
              ++evals;
              if (c < cbest) {
                cbest = c;
                cx = mx;
                cy = my;
                improved = true;
                cstage = kMeStageInteger;
              }
            }
            if (!improved) {
              break;
            }
          }
          step >>= 1;
        }
      }
    }

    // Subpel refinement.  Each stage is a fixed 8-neighbour ring with the
    // stage's MV step; fractional samples use the selected interpolation
    // filter for the cost (the MC component still does the normative
    // prediction for the reconstructed frame).
    const int stage_mask[4] = {kMeSubpelHalf, kMeSubpelQuarter,
                               kMeSubpelEighth, kMeSubpelSixteenth};
    const int stage_units[4] = {8, 4, 2, 1};
    const int stage_id[4] = {kMeStageHalf, kMeStageQuarter, kMeStageEighth,
                             kMeStageSixteenth};
    const int off[8][2] = {{-1, -1}, {0, -1}, {1, -1}, {-1, 0},
                           {1, 0},   {-1, 1}, {0, 1},  {1, 1}};
    for (int s = 0; s < 4; ++s) {
      if ((opt.subpel_stages & stage_mask[s]) == 0) {
        continue;
      }
      const std::int32_t step = stage_units[s];
      for (int it = 0; it < opt.subpel_iterations; ++it) {
        bool improved = false;
        for (int k = 0; k < 8; ++k) {
          const std::int32_t mx = me_clamp_search_x(
              cx + off[k][0] * step, base_x, range_units + kSubpelMask,
              job.x, w, ref_w, true);
          const std::int32_t my = me_clamp_search_y(
              cy + off[k][1] * step, base_y, range_units + kSubpelMask,
              job.y, h, ref_h, true);
          if (mx == cx && my == cy) {
            continue;
          }
          const std::int32_t c =
              cost(cur.plane[0], ref, job.x, job.y, w, h, mx, my, opt.metric,
                   opt.subpel_interp);
          ++evals;
          if (c < cbest) {
            cbest = c;
            cx = mx;
            cy = my;
            improved = true;
            cstage = stage_id[s];
          }
        }
        if (!improved) {
          break;
        }
      }
    }

    if (!have_best || cbest < best_cost) {
      best_cost = cbest;
      best_x = cx;
      best_y = cy;
      best_ref = ref_index;
      best_stage = cstage;
      have_best = true;
    }
  }

  if (!have_best) {
    return res;
  }
  res.mv_x = best_x;
  res.mv_y = best_y;
  res.ref_index = best_ref;
  res.cost = best_cost;
  res.evals = evals;
  res.stage = best_stage;
  return res;
}

}  // namespace av2enc::me

#endif  // AV2ENC_ENCODER_INTER_ME_ME_CORE_H_
