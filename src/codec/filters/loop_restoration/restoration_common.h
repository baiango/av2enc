// AV2 in-loop restoration — shared host/device constants and helpers.
//
// Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
// (2026-05-28, commit 38f28e7).  Section numbers are cited at each use site.
//
// This header is compiled twice: as ordinary host C++ for the scalar CPU
// reference (restoration.cpp) and as CUDA device code for
// src/cuda/restoration/*.cu.  Every inline function here is bit-exact between
// the two compilers because it only uses integer arithmetic (no floating
// point, no fused multiply-add, no library calls).
//
// Scope: 8-bit 4:2:0 planar.  The arithmetic itself follows the spec's
// BitDepth parameter; the filters reject other bit depths explicitly rather
// than silently approximating them.

#ifndef AV2ENC_ENCODER_FILTERS_RESTORATION_RESTORATION_COMMON_H_
#define AV2ENC_ENCODER_FILTERS_RESTORATION_RESTORATION_COMMON_H_

#include <cstdint>

#if defined(__CUDACC__)
#define AV2ENC_LR_HD __host__ __device__ __forceinline__
// During the device compilation pass the normative tables must live in device
// memory; the host pass sees ordinary inline constexpr arrays.
#if defined(__CUDA_ARCH__)
#define AV2ENC_LR_TABLE __device__ constexpr
#else
#define AV2ENC_LR_TABLE inline constexpr
#endif
#else
#define AV2ENC_LR_HD inline
#define AV2ENC_LR_TABLE inline constexpr
#endif

namespace av2enc::restoration {

// ---------------------------------------------------------------------------
// Geometry / constants
// ---------------------------------------------------------------------------

constexpr int kMiSize = 4;
constexpr int kMiSizeLog2 = 2;

// Maximum number of segments (spec MAX_SEGMENTS).  LosslessArray is indexed by
// segment id.
constexpr int kMaxSegments = 8;

// §4.8 Round2 / Round2Signed.  Round2 is floor((x + 2^(n-1)) / 2^n) using
// mathematical (real) division, i.e. an arithmetic right shift.
AV2ENC_LR_HD constexpr int clip3(int lo, int hi, int v) {
  return v < lo ? lo : (v > hi ? hi : v);
}

AV2ENC_LR_HD constexpr int round2(int x, int n) {
  return n == 0 ? x : (x + (1 << (n - 1))) >> n;
}

AV2ENC_LR_HD constexpr int round2_signed(int x, int n) {
  return x >= 0 ? round2(x, n) : -round2(-x, n);
}

AV2ENC_LR_HD constexpr int abs_int(int x) { return x < 0 ? -x : x; }

// Loop restoration -----------------------------------------------------------------

// Spec constants (spec/tables/constants.csv).
constexpr int kRestorationTileSizeMax = 512;
constexpr int kWienerNsPrecBits = 7;
constexpr int kWienerNsTapsY = 32;
constexpr int kWienerNsTapsUv = 12;
constexpr int kWienerNsChromaCoeffs = 18;
constexpr int kWienerNsLumaCoeffs = 16;
constexpr int kWienerNsClasses = 16;
constexpr int kWienerNsPlanes = 3;
constexpr int kWienerNsShortCoeffs = 6;

// Coefficient arrays are stored with a uniform stride of 18 so one pointer
// type serves luma (16 used) and chroma (18 used).
constexpr int kWienerNsCoeffStride = 18;

// §5.18.7.11 Decode_Num_Filter_Classes[8]; num_filter_classes_idx is f(3).
AV2ENC_LR_TABLE std::int8_t kDecodeNumFilterClasses[8] = {1, 2, 3, 4,
                                                          6, 8, 12, 16};

// §5.18.7.11 get_filter_set_index(): base_q_idx -> filter set.
AV2ENC_LR_HD int lr_filter_set_index(int base_q_idx) {
  if (base_q_idx < 130) {
    return 0;
  }
  if (base_q_idx < 190) {
    return 1;
  }
  if (base_q_idx < 220) {
    return 2;
  }
  return 3;
}

// §6.17.7.7 / §7.20.3 coefficient ranges.  These are the normative
// Wiener_Ns_Taps_Min / Wiener_Ns_Taps_K tables; they bound every signaled
// coefficient, so an encoder-side fit must clip into [min, min + 2^k - 1].
AV2ENC_LR_TABLE std::int8_t kWienerNsTapsMin[2][18] = {
    {-24, -24, -14, -14, -16, -16, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8,
     -8, -8},
    {-24, -24, -14, -14, -16, -16, -16, -16, -16, -16, -8, -8, -8, -8, -8,
     -8, -8, -8},
};
AV2ENC_LR_TABLE std::int8_t kWienerNsTapsK[2][18] = {
    {6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4},
    {6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4},
};

AV2ENC_LR_HD int wiener_ns_coeff_min(bool chroma, int j) {
  return kWienerNsTapsMin[chroma ? 1 : 0][j];
}

AV2ENC_LR_HD int wiener_ns_coeff_max(bool chroma, int j) {
  return kWienerNsTapsMin[chroma ? 1 : 0][j] +
         (1 << kWienerNsTapsK[chroma ? 1 : 0][j]) - 1;
}

AV2ENC_LR_HD int wiener_ns_clip_coeff(bool chroma, int j, int v) {
  return clip3(wiener_ns_coeff_min(chroma, j),
               wiener_ns_coeff_max(chroma, j), v);
}

// §6.17.7.7 FrameRestorationType values.
enum RestorationType : int {
  kRestoreNone = 0,
  kRestorePcWiener = 1,
  kRestoreWienerNs = 2,
  kRestoreSwitchable = 3,
};

// §7.20.3 Wiener_Ns_Config_Y[32][3]: { dy, dx, coeffIndex }.
AV2ENC_LR_TABLE std::int8_t kWienerNsConfigY[kWienerNsTapsY][3] = {
    {1, 0, 0},   {-1, 0, 0},  {0, 1, 1},   {0, -1, 1},  {2, 0, 2},
    {-2, 0, 2},  {0, 2, 3},   {0, -2, 3},  {1, 1, 4},   {-1, -1, 4},
    {-1, 1, 5},  {1, -1, 5},  {2, 1, 6},   {-2, -1, 6}, {2, -1, 7},
    {-2, 1, 7},  {1, 2, 8},   {-1, -2, 8}, {1, -2, 9},  {-1, 2, 9},
    {3, 0, 10},  {-3, 0, 10}, {0, 3, 11},  {0, -3, 11}, {4, 0, 12},
    {-4, 0, 12}, {0, 4, 13},  {0, -4, 13}, {3, 3, 14},  {-3, -3, 14},
    {3, -3, 15}, {-3, 3, 15},
};

// §7.20.3 Wiener_Ns_Config_Uv[12][3].
AV2ENC_LR_TABLE std::int8_t kWienerNsConfigUv[kWienerNsTapsUv][3] = {
    {1, 0, 0},  {-1, 0, 0}, {0, 1, 1},  {0, -1, 1},
    {1, 1, 2},  {-1, -1, 2}, {-1, 1, 3}, {1, -1, 3},
    {2, 0, 4},  {-2, 0, 4}, {0, 2, 5},  {0, -2, 5},
};

// §7.20.3 Wiener_Filters_420[2][2][2] used by get_luma_sample().
AV2ENC_LR_TABLE std::int8_t kWienerFilters420[2][2][2] = {
    {{1, 1}, {1, 1}},
    {{2, 0}, {2, 0}},
};

// CCSO -----------------------------------------------------------------------

// Spec constants.
constexpr int kCcsoBandNum = 64;
constexpr int kCcsoInputInterval = 3;
constexpr int kCcsoLumaSizeLog2 = 8;
constexpr int kCcsoOffsetRange = 8;
constexpr int kCcsoMaxClasses = 3;

// §6.19.2.1 bru_mode values.
constexpr int kBruInactive = 0;
constexpr int kBruSupport = 1;
constexpr int kBruActive = 2;

// §5.18.7.11 LoopRestorationSize[plane] derivation.  The header signals one of
// lr_luma_use_{half,max,quarter}_size (and the chroma pair); these helpers map
// the resulting shift to plane samples.  RESTORATION_TILESIZE_MAX is 512.
AV2ENC_LR_HD int lr_unit_size_from_shift(int shift) {
  return kRestorationTileSizeMax >> (shift < 0 ? 0 : shift);
}

// §5.18.7.12 CcsoLumaSizeLog2 derivation.  `tile_start_or` is
// OR(MiColStarts[]) | OR(MiRowStarts[]) - only its low six bits matter.
AV2ENC_LR_HD int ccso_select_luma_size_log2(int sb_size, int tile_start_or,
                                            bool unit_matches_sb_size) {
  if (unit_matches_sb_size) {
    int log2 = 0;
    int v = sb_size;
    while (v > 4) {
      v >>= 1;
      ++log2;
    }
    return log2 + kMiSizeLog2;  // Mi_Width_Log2[SbSize] + MI_SIZE_LOG2
  }
  if ((tile_start_or & 63) == 0) {
    return 8;
  }
  if ((tile_start_or & 31) == 0) {
    return 7;
  }
  return 6;
}

// CcsoBlks[plane] grid size: ceil(plane samples / CCSO block size).  For 4:2:0
// the block is (1 << (log2 - 1)) in each chroma dimension.
AV2ENC_LR_HD void ccso_blk_grid_size(int plane_width, int plane_height,
                                     int ccso_luma_size_log2, int sub_x,
                                     int sub_y, int* cols, int* rows) {
  const int shift_x = ccso_luma_size_log2 - (sub_x ? sub_x : 0);
  const int shift_y = ccso_luma_size_log2 - (sub_y ? sub_y : 0);
  const int blk_w = 1 << (shift_x < 0 ? 0 : shift_x);
  const int blk_h = 1 << (shift_y < 0 ? 0 : shift_y);
  *cols = (plane_width + blk_w - 1) / blk_w;
  *rows = (plane_height + blk_h - 1) / blk_h;
}

// §5.18.7.12 header limits for the CCSO scalar fields.  `ccso_max_band_log2`
// is f(2 + ccso_bo_only): non-band-only allows 0..3, band-only allows 0..6
// (the conformance cap is CCSO_BAND_NUM == 64).  `ccso_edge_clf` is f(1), so a
// bitstream can only produce 0 or 1 even though the filter core supports 2.
constexpr int kCcsoMaxBandLog2Normal = 3;
constexpr int kCcsoMaxBandLog2BoOnly = 6;
constexpr int kCcsoMaxEdgeClf = 1;

// §7.19.1 CCSO_Quant_Sz[scale_idx][quant_idx].
AV2ENC_LR_TABLE std::int16_t kCcsoQuantSz[4][4] = {
    {16, 8, 32, 0},
    {56, 40, 64, 128},
    {48, 24, 96, 192},
    {80, 112, 160, 256},
};

// §7.19.1 Ccso_Pos[ext_filter][2]: { dy, dx }.
AV2ENC_LR_TABLE std::int8_t kCcsoPos[7][2] = {
    {-1, 0}, {0, -1}, {-1, -1}, {-1, 1}, {-1, -2}, {1, -2}, {0, 2},
};

// §5.18.7.12 Ccso_Offset[8].
AV2ENC_LR_TABLE std::int8_t kCcsoOffset[kCcsoOffsetRange] = {
    0, 1, -1, 3, -3, 7, -7, -10,
};

// §7.19.1 ccso_score().
AV2ENC_LR_HD int ccso_score(int diff, int quant_step, int edge_classifier) {
  if (diff > quant_step && edge_classifier == 0) {
    return 2;
  }
  if (diff < -quant_step) {
    return 0;
  }
  return 1;
}

// Offset table indexing: [band][cls0][cls1] with 64 bands and 3x3 classes.
AV2ENC_LR_HD int ccso_offset_index(int band, int cls0, int cls1) {
  return (band * kCcsoMaxClasses + cls0) * kCcsoMaxClasses + cls1;
}

constexpr int kCcsoOffsetEntries =
    kCcsoBandNum * kCcsoMaxClasses * kCcsoMaxClasses;

}  // namespace av2enc::restoration

#endif  // AV2ENC_ENCODER_FILTERS_RESTORATION_RESTORATION_COMMON_H_
