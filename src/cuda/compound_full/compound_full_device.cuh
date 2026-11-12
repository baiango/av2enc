#ifndef AV2ENC_CUDA_COMPOUND_FULL_COMPOUND_FULL_DEVICE_CUH_
#define AV2ENC_CUDA_COMPOUND_FULL_COMPOUND_FULL_DEVICE_CUH_

// Shared device body for the full compound path.
//
// Compiled twice:
//   * by nvcc, for the kernels in compound_full.cu;
//   * as host code by compound_full_emulation_test.cpp, with the CUDA keywords
//     stubbed, so the emulation parity test runs the exact same arithmetic.
//
// There is one transcription of each formula here. Where the CPU reference
// (`inter::adv::*`, `inter::temporal::*`) exposes a device-safe helper, this
// header calls it (masks, rounding, relative distance, MV scaling) instead of
// restating it.

#include "compound_full.h"

#include <cstddef>
#include <cstdint>

#include "encoder/inter/advanced/warp_common.h"
#include "encoder/inter/mc/mc_core.h"

#if defined(__CUDACC__)
#define AV2ENC_CF_HD __host__ __device__ __forceinline__
#if defined(__CUDA_ARCH__)
#define AV2ENC_CF_SYNC() __syncthreads()
#else
#define AV2ENC_CF_SYNC() ((void)0)
#endif
#else
#define AV2ENC_CF_HD inline
#define AV2ENC_CF_SYNC() ((void)0)
#endif

namespace av2enc::cuda::compound_full {
namespace detail {

using av2enc::inter::adv::abs_int;
using av2enc::inter::adv::clip3;
using av2enc::inter::adv::get_relative_dist;
using av2enc::inter::adv::round2;
using av2enc::inter::adv::round2_signed;
using av2enc::inter::adv::round2_signed64;

enum : int {
  kCompoundPostRound = 4,  // adv::kCompoundPostRound
  // Mirrors temporal_types.h constants used by OPFL (that header is not
  // device-callable, so the values are restated here).
  kOpflMaxLsBits = 26,       // kMaxLsBits
  kOpflMvDeltaLimit = 16,    // 1 << MV_REFINE_PREC_BITS
  kCfMiSize = 4,             // adv::kMiSize
  kCfMvLow = -(1 << 16),     // adv::kMvLow
  kCfMvUpp = 1 << 16,        // adv::kMvUpp
  kCfBlockSizes = 29,        // adv::tables::kBlockSizes
  kCfInterIntraModes = 4,    // adv::kInterIntraModes
  kCfWedgeTypes = 68,        // adv::kWedgeTypes
  kCfWedgeEntries = 2 * 68,  // sign x index
  kCfBlock16x16 = 6,         // adv::kBlock16x16 (last SHARP wedge size)
  kCfMaskMasterSize = 128,   // adv::kMaskMasterSize
  kCfScaleSubpelBits = 10,   // mc::kScaleSubpelBits
  kCfSubpelMask = 15,        // mc::kSubpelMask
  kCfSubpelTaps = 8,         // mc::kSubpelTaps
  kCfInterRound0 = 3,        // mc::kInterRound0
  kCfInterRound1Single = 11,    // mc::kInterRound1Single
  kCfInterRound1Compound = 7,   // mc::kInterRound1Compound
};

// ---------------------------------------------------------------------------
// Wedge-mask cache layout.
//
// One contiguous uint8_t image holds every (block size, sign, wedge index, i, j)
// value that an M2 block can select. Block sizes are the §9.2 entries with
// WEDGE_BITS > 0 and unknown/tallest dimensions <= 64 (the M2 maximum).
// ---------------------------------------------------------------------------
AV2ENC_CF_HD int wedge_cache_block_size(int slot) {
  switch (slot) {
    case 0: return 3;
    case 1: return 4;
    case 2: return 5;
    case 3: return 6;
    case 4: return 7;
    case 5: return 8;
    case 6: return 9;
    case 7: return 10;
    case 8: return 11;
    case 9: return 12;
    case 10: return 21;
    case 11: return 22;
    case 12: return 23;
    case 13: return 24;
    case 14: return 27;
    case 15: return 28;
    default: return -1;
  }
}

AV2ENC_CF_HD int wedge_cache_offset(int bsize) {
  // Byte offset of `bsize`'s cache block. -1 when the block size is not
  // cached (the caller then evaluates the mask on the fly).
  switch (bsize) {
    case 3: return 0;
    case 4: return 8704;
    case 5: return 26112;
    case 6: return 43520;
    case 7: return 78336;
    case 8: return 147968;
    case 9: return 217600;
    case 10: return 356864;
    case 11: return 635392;
    case 12: return 913920;
    case 21: return 1470976;
    case 22: return 1505792;
    case 23: return 1540608;
    case 24: return 1679872;
    case 27: return 1819136;
    case 28: return 1888768;
    default: return -1;
  }
}

// Device copies of the §5.20.7.18 / §7.13.3.27/.29 tables used by the kernel.
// Namespace-scope arrays must not be host-only when a kernel indexes them
// (nvcc: "identifier is undefined in device code"), so the device pass gets
// __device__ constexpr storage. The host pass keeps the inline form for the
// emulation test. Values are copied verbatim from
// encoder/inter/advanced/tables/wedge_tables.h.
#if defined(__CUDA_ARCH__)
#define AV2ENC_CF_DEV_TABLE __device__ constexpr
#else
#define AV2ENC_CF_DEV_TABLE inline constexpr
#endif

AV2ENC_CF_DEV_TABLE std::int8_t kCfWedgeCosLut[20] = {
       4,    4,    4,    2,    2,    0,   -2,   -2,   -4,   -4,   -4,   -4,   -4,   -2,   -2,    0,    2,    2,    4,    4,
};

AV2ENC_CF_DEV_TABLE std::int8_t kCfWedgeSinLut[20] = {
       0,   -1,   -2,   -2,   -4,   -4,   -4,   -2,   -2,   -1,    0,    1,    2,    2,    4,    4,    4,    2,    2,    1,
};

AV2ENC_CF_DEV_TABLE std::int8_t kCfPosDist2BldWeight[32] = {
       8,    8,    9,    9,   10,   10,   11,   11,   12,   12,   12,   13,   13,   13,   14,   14,
      14,   14,   14,   15,   15,   15,   15,   15,   15,   15,   15,   15,   16,   16,   16,   16,
};

AV2ENC_CF_DEV_TABLE std::int8_t kCfNegDist2BldWeight[32] = {
       8,    8,    7,    7,    6,    6,    5,    5,    4,    4,    4,    3,    3,    3,    2,    2,
       2,    2,    2,    1,    1,    1,    1,    1,    1,    1,    1,    1,    0,    0,    0,    0,
};

AV2ENC_CF_DEV_TABLE std::int8_t kCfWedgeCodebook[kCfWedgeTypes][3] = {
    { 0,  5,  4},
    { 0,  6,  4},
    { 0,  7,  4},
    { 1,  4,  4},
    { 1,  5,  4},
    { 1,  6,  4},
    { 1,  7,  4},
    { 2,  4,  4},
    { 2,  5,  4},
    { 2,  6,  4},
    { 2,  7,  4},
    { 3,  4,  4},
    { 3,  5,  4},
    { 3,  6,  4},
    { 3,  7,  4},
    { 4,  4,  4},
    { 4,  4,  3},
    { 4,  4,  2},
    { 4,  4,  1},
    { 5,  4,  3},
    { 5,  4,  2},
    { 5,  4,  1},
    { 6,  4,  4},
    { 6,  4,  3},
    { 6,  4,  2},
    { 6,  4,  1},
    { 7,  4,  4},
    { 7,  3,  4},
    { 7,  2,  4},
    { 7,  1,  4},
    { 8,  4,  4},
    { 8,  3,  4},
    { 8,  2,  4},
    { 8,  1,  4},
    { 9,  4,  4},
    { 9,  3,  4},
    { 9,  2,  4},
    { 9,  1,  4},
    {10,  3,  4},
    {10,  2,  4},
    {10,  1,  4},
    {11,  3,  4},
    {11,  2,  4},
    {11,  1,  4},
    {12,  3,  4},
    {12,  2,  4},
    {12,  1,  4},
    {13,  3,  4},
    {13,  2,  4},
    {13,  1,  4},
    {14,  4,  5},
    {14,  4,  6},
    {14,  4,  7},
    {15,  4,  5},
    {15,  4,  6},
    {15,  4,  7},
    {16,  4,  5},
    {16,  4,  6},
    {16,  4,  7},
    {17,  5,  4},
    {17,  6,  4},
    {17,  7,  4},
    {18,  5,  4},
    {18,  6,  4},
    {18,  7,  4},
    {19,  5,  4},
    {19,  6,  4},
    {19,  7,  4},
};

AV2ENC_CF_DEV_TABLE std::int16_t kCfWedgeBlockWidth[29] = {
       4,    4,    8,    8,    8,   16,   16,   16,   32,   32,
      32,   64,   64,   64,  128,  128,  128,  256,  256,    4,
      16,    8,   32,   16,   64,    4,   32,    8,   64,
};

AV2ENC_CF_DEV_TABLE std::int16_t kCfWedgeBlockHeight[29] = {
       4,    8,    4,    8,   16,    8,   16,   32,   16,   32,
      64,   32,   64,  128,   64,  128,  256,  128,  256,   16,
       4,   32,    8,   64,   16,   32,    4,   64,    8,
};

AV2ENC_CF_DEV_TABLE std::int8_t kCfWedgeBits[kCfBlockSizes] = {
       0,    0,    0,    4,    4,    4,    4,    4,    4,    4,    4,    4,
       4,    0,    0,    0,    0,    0,    0,    0,    0,    4,    4,    4,
       4,    0,    0,    4,    4,
};

AV2ENC_CF_DEV_TABLE std::int8_t kCfIiWeights1d[128] = {
      60,   58,   56,   54,   52,   50,   48,   47,   45,   44,   42,   41,
      39,   38,   37,   35,   34,   33,   32,   31,   30,   29,   28,   27,
      26,   25,   24,   23,   22,   22,   21,   20,   19,   19,   18,   18,
      17,   16,   16,   15,   15,   14,   14,   13,   13,   12,   12,   12,
      11,   11,   10,   10,   10,    9,    9,    9,    8,    8,    8,    8,
       7,    7,    7,    7,    6,    6,    6,    6,    6,    5,    5,    5,
       5,    5,    4,    4,    4,    4,    4,    4,    4,    4,    3,    3,
       3,    3,    3,    3,    3,    3,    3,    2,    2,    2,    2,    2,
       2,    2,    2,    2,    2,    2,    2,    2,    2,    2,    1,    1,
       1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
       1,    1,    1,    1,    1,    1,    1,    1,
};


// §9.2 Wedge_Bits[] test: device copy, host-only static_assert against the
// canonical host table (advanced_common.h is not owned by this module).
AV2ENC_CF_HD constexpr bool cf_bsize_has_wedge(int bsize) {
  return bsize >= 0 && bsize < kCfBlockSizes && kCfWedgeBits[bsize] > 0;
}

#if !defined(__CUDA_ARCH__)
constexpr bool cf_wedge_bits_match_adv() {
  for (int i = 0; i < kCfBlockSizes; ++i) {
    if (kCfWedgeBits[i] != av2enc::inter::adv::tables::kWedgeBits[i]) {
      return false;
    }
  }
  return true;
}
constexpr bool cf_wedge_dims_match_adv() {
  for (int i = 0; i < kCfBlockSizes; ++i) {
    if (kCfWedgeBlockWidth[i] != av2enc::inter::adv::tables::kWedgeBlockWidth[i] ||
        kCfWedgeBlockHeight[i] != av2enc::inter::adv::tables::kWedgeBlockHeight[i]) {
      return false;
    }
  }
  return true;
}
constexpr bool cf_wedge_tables_match_adv() {
  for (int i = 0; i < 20; ++i) {
    if (kCfWedgeCosLut[i] != av2enc::inter::adv::tables::kWedgeCosLut[i] ||
        kCfWedgeSinLut[i] != av2enc::inter::adv::tables::kWedgeSinLut[i]) {
      return false;
    }
  }
  for (int i = 0; i < 32; ++i) {
    if (kCfPosDist2BldWeight[i] !=
            av2enc::inter::adv::tables::kPosDist2BldWeight[i] ||
        kCfNegDist2BldWeight[i] !=
            av2enc::inter::adv::tables::kNegDist2BldWeight[i]) {
      return false;
    }
  }
  for (int i = 0; i < kCfWedgeTypes; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (kCfWedgeCodebook[i][j] !=
          av2enc::inter::adv::tables::kWedgeCodebook[i][j]) {
        return false;
      }
    }
  }
  for (int i = 0; i < 128; ++i) {
    if (kCfIiWeights1d[i] != av2enc::inter::adv::tables::kIiWeights1d[i]) {
      return false;
    }
  }
  return true;
}
static_assert(cf_wedge_bits_match_adv(),
              "kCfWedgeBits must match adv::tables::kWedgeBits");
static_assert(cf_wedge_dims_match_adv(),
              "kCfWedgeBlockWidth/Height must match the host tables");
static_assert(cf_wedge_tables_match_adv(),
              "device wedge/intra tables must match the host tables");
#endif

AV2ENC_CF_HD int cf_wedge_master_value(int boundary, int angle, int n, int m) {
  const int y = ((n << 1) - kCfMaskMasterSize + 1) * kCfWedgeSinLut[angle];
  int d = ((m << 1) - kCfMaskMasterSize + 1) * kCfWedgeCosLut[angle] + y;
  if (boundary == 0) {
    d *= 2;
  }
  const int clamp_d = clip3(-31, 31, d);
  const int w = clamp_d >= 0 ? kCfPosDist2BldWeight[clamp_d]
                             : kCfNegDist2BldWeight[-clamp_d];
  return w << 2;
}

// §7.13.3.27 wedge mask, device-side transcription of adv::wedge_mask_value.
AV2ENC_CF_HD int cf_wedge_mask_value(int bsize, int sign, int index, int i,
                                     int j) {
  const int w = kCfWedgeBlockWidth[bsize];
  const int h = kCfWedgeBlockHeight[bsize];
  const int boundary = (bsize <= kCfBlock16x16) ? 0 : 1;
  const int dir = kCfWedgeCodebook[index][0];
  const int xoff = kCfMaskMasterSize / 2 - ((kCfWedgeCodebook[index][1] * w) >> 3);
  const int yoff = kCfMaskMasterSize / 2 - ((kCfWedgeCodebook[index][2] * h) >> 3);
  const int master = cf_wedge_master_value(boundary, dir, yoff + i, xoff + j);
  return sign ? (64 - master) : master;
}

// §7.13.3.29 intra-mode variant mask (device copy of kIiWeights1d).
AV2ENC_CF_HD int cf_intra_mode_mask_value(av2enc::inter::adv::InterIntraMode mode,
                                          int w, int h, int i, int j) {
  const int size_scale = 128 / (h > w ? h : w);
  switch (mode) {
    case av2enc::inter::adv::InterIntraMode::kVertical:
      return kCfIiWeights1d[i * size_scale];
    case av2enc::inter::adv::InterIntraMode::kHorizontal:
      return kCfIiWeights1d[j * size_scale];
    case av2enc::inter::adv::InterIntraMode::kSmooth:
      return kCfIiWeights1d[(i < j ? i : j) * size_scale];
    default:  // kDc
      return 32;
  }
}

// Total size of the cache image. Kept in sync with wedge_cache_offset().

AV2ENC_CF_HD int wedge_cache_sample(const std::uint8_t* cache, int bsize,
                                           int sign, int index, int i, int j) {
  const int offset = wedge_cache_offset(bsize);
  if (cache == nullptr || offset < 0) {
    return cf_wedge_mask_value(bsize, sign, index, i, j);
  }
  const int w = kCfWedgeBlockWidth[bsize];
  const int h = kCfWedgeBlockHeight[bsize];
  const int entry = sign * kCfWedgeTypes + index;
  return static_cast<int>(cache[offset + (entry * h + i) * w + j]);
}

// Fills one block size's cache region. Threads stride over every entry.
AV2ENC_CF_HD void wedge_cache_build_slot_device(std::uint8_t* cache,
                                                       int slot, int tid,
                                                       int nthr) {
  if (slot < 0 || slot >= 16) {
    return;
  }
  const int bsize = wedge_cache_block_size(slot);
  const int w = kCfWedgeBlockWidth[bsize];
  const int h = kCfWedgeBlockHeight[bsize];
  const int offset = wedge_cache_offset(bsize);
  const int total = kWedgeCacheEntries * h * w;
  for (int index = tid; index < total; index += nthr) {
    const int j = index % w;
    const int entry_row = index / w;
    const int i = entry_row % h;
    const int entry = entry_row / h;
    const int sign = entry / kCfWedgeTypes;
    const int wedge_index = entry % kCfWedgeTypes;
    cache[offset + index] = static_cast<std::uint8_t>(
        cf_wedge_mask_value(bsize, sign, wedge_index, i, j));
  }
}

// ---------------------------------------------------------------------------
// Typed sample access (8-bit uint8_t / 10-bit LSB-aligned uint16_t).
// ---------------------------------------------------------------------------

AV2ENC_CF_HD int load_sample(const void* base, std::size_t index,
                                    int is_16bit) {
  return is_16bit
             ? static_cast<int>(static_cast<const std::uint16_t*>(base)[index])
             : static_cast<int>(static_cast<const std::uint8_t*>(base)[index]);
}

AV2ENC_CF_HD void store_sample(void* base, std::size_t index,
                                      int is_16bit, int value) {
  if (is_16bit) {
    static_cast<std::uint16_t*>(base)[index] =
        static_cast<std::uint16_t>(value);
  } else {
    static_cast<std::uint8_t*>(base)[index] = static_cast<std::uint8_t>(value);
  }
}

AV2ENC_CF_HD int clip1_bd(int value, int bit_depth) {
  return clip3(0, (1 << bit_depth) - 1, value);
}

AV2ENC_CF_HD int ref_read(const RefPlane& ref, int x, int y) {
  const int cx = clip3(0, ref.width - 1, x);
  const int cy = clip3(0, ref.height - 1, y);
  return load_sample(ref.data,
                     static_cast<std::size_t>(cy) *
                             static_cast<std::size_t>(ref.stride) +
                         static_cast<std::size_t>(cx),
                     ref.is_16bit);
}

// ---------------------------------------------------------------------------
// §7.13.3.17/§7.13.3.18 prediction primitives, generalised to 10-bit reads.
// ---------------------------------------------------------------------------

struct SampleLoc {
  int start_x = 0;
  int start_y = 0;
  int step_x = 0;
  int step_y = 0;
};

// Unscaled reference: identical to mc::mc_derive_sample_location().
AV2ENC_CF_HD SampleLoc location_unscaled(int x, int y, int mv_row,
                                                int mv_col, int sub_x,
                                                int sub_y) {
  const mc::McSampleLocation loc = mc::mc_derive_sample_location(
      x, y, mc::Mv{mv_row, mv_col}, sub_x, sub_y);
  SampleLoc out;
  out.start_x = loc.start_x;
  out.start_y = loc.start_y;
  out.step_x = loc.step_x;
  out.step_y = loc.step_y;
  return out;
}

// Scaled reference, matching refinemv.cpp's scaled_start_* helpers.
AV2ENC_CF_HD SampleLoc location_scaled(const RefPlane& ref, int x, int y,
                                               int mv_row, int mv_col,
                                               int sub_x, int sub_y,
                                               int frame_w, int frame_h) {
  const int mv_pair[2] = {mv_row, mv_col};
  const adv::MvScaleOutput loc = adv::motion_vector_scaling(
      0, sub_x, sub_y, x, y, mv_pair, false, ref.width << sub_x,
      ref.height << sub_y, frame_w, frame_h);
  SampleLoc out;
  out.start_x = loc.start_x;
  out.start_y = loc.start_y;
  out.step_x = loc.step_x;
  out.step_y = loc.step_y;
  return out;
}

// One output sample of the two-pass subpel filter. `round1` is 7 for 1/16
// sample compound predictions and 11 for integer single-reference ones.
// `first_*`/`last_*` are the inclusive reference area (§7.13.3.4).
AV2ENC_CF_HD int filter_sample(const RefPlane& ref, int first_x,
                                      int first_y, int last_x, int last_y,
                                      const SampleLoc& loc,
                                      const std::int16_t* taps_h,
                                      const std::int16_t* taps_v, int r, int c,
                                      int round1) {
  const int p_x = loc.start_x + loc.step_x * c;
  const int col = (p_x >> kCfScaleSubpelBits) - 3;
  const int phase_h = (p_x >> 6) & kCfSubpelMask;
  const std::int16_t* fh = taps_h + phase_h * kCfSubpelTaps;

  const int p_y = (loc.start_y & 1023) + loc.step_y * r;
  const int row_base = p_y >> kCfScaleSubpelBits;
  const int phase_v = (p_y >> 6) & kCfSubpelMask;
  const std::int16_t* fv = taps_v + phase_v * kCfSubpelTaps;

  int sum_v = 0;
  for (int t = 0; t < 8; ++t) {
    const int row = clip3(first_y, last_y,
                          (loc.start_y >> kCfScaleSubpelBits) + row_base + t -
                              3);
    int hsum = 0;
    for (int u = 0; u < 8; ++u) {
      const int cx = clip3(first_x, last_x, col + u);
      hsum += static_cast<int>(fh[u]) * ref_read(ref, cx, row);
    }
    const int intermediate = mc::mc_round2(hsum, kCfInterRound0);
    sum_v += static_cast<int>(fv[t]) * intermediate;
  }
  return mc::mc_round2(sum_v, round1);
}

// Convenience: unscaled single-reference integer prediction (refine-MV search).
AV2ENC_CF_HD int predict_unscaled_u8(const RefPlane& ref,
                                            const int area[4], int x, int y,
                                            int w, int h, int mv_row,
                                            int mv_col, int filter,
                                            int bit_depth, int r, int c) {
  const SampleLoc loc =
      location_unscaled(x, y, mv_row, mv_col, 0, 0);
  const mc::InterpFilter fh_enum =
      mc::mc_pass_filter(static_cast<mc::InterpFilter>(filter), w <= 4);
  const mc::InterpFilter fv_enum =
      mc::mc_pass_filter(static_cast<mc::InterpFilter>(filter), h <= 4);
  const std::int16_t* taps_h =
      &mc::kSubpelFilters[static_cast<int>(fh_enum)][0][0];
  const std::int16_t* taps_v =
      &mc::kSubpelFilters[static_cast<int>(fv_enum)][0][0];
  const int value = filter_sample(ref, area[0], area[1], area[2], area[3], loc,
                                  taps_h, taps_v, r, c, kCfInterRound1Single);
  return clip1_bd(value, bit_depth);
}

// ---------------------------------------------------------------------------
// Compound / inter-intra blend
// ---------------------------------------------------------------------------

AV2ENC_CF_HD int implicit_mask_sample(const BlendJob& job, int i, int j) {
  // §7.13.3.14 get_mask(), transcribed locally so no device construction of
  // adv::CompoundJob is needed.
  const int last_x = (job.mi_cols * kCfMiSize >> job.sub_x) - 1;
  const int last_y = (job.mi_rows * kCfMiSize >> job.sub_y) - 1;
  const int ref_x0 = job.ref_start_x[0] + j;
  const int ref_x1 = job.ref_start_x[1] + j;
  const int ref_y0 = job.ref_start_y[0] + i;
  const int ref_y1 = job.ref_start_y[1] + i;
  const bool on0 = ref_x0 >= 0 && ref_x0 <= last_x && ref_y0 >= 0 &&
                   ref_y0 <= last_y;
  const bool on1 = ref_x1 >= 0 && ref_x1 <= last_x && ref_y1 >= 0 &&
                   ref_y1 <= last_y;
  if (on0 && !on1) {
    return 2;
  }
  if (on1 && !on0) {
    return 0;
  }
  return 1;
}

// Luma mask value for sample (i, j) of the job's luma block.
AV2ENC_CF_HD int luma_mask_value(const BlendJob& job, int i, int j) {
  switch (job.type) {
    case kCompoundWedge:
      return wedge_cache_sample(job.wedge_cache, job.bsize, job.wedge_sign,
                                job.wedge_index, i, j);
    case kCompoundDiffWtd:
      return adv::diffwtd_mask_value(job.p0[i * job.p0_stride + j],
                                     job.p1[i * job.p1_stride + j],
                                     job.mask_type, kCompoundPostRound,
                                     job.bit_depth);
    case kCompoundIntra:
      return cf_intra_mode_mask_value(
          static_cast<adv::InterIntraMode>(job.interintra_mode), job.rect_w,
          job.rect_h, i, j);
    case kCompoundAverage:
    default:
      return 32;
  }
}

// Mask value at plane sample (i, j): luma evaluates directly; chroma follows
// §7.13.3.30 (subsample the luma mask, or regenerate the intra-mode mask).
AV2ENC_CF_HD int mask_sample(const BlendJob& job, int i, int j) {
  const bool mask_from_luma =
      job.type == kCompoundWedge || job.type == kCompoundDiffWtd;
  if (job.plane == 0 || !mask_from_luma) {
    return luma_mask_value(job, i, j);
  }

  // A chroma DIFFWTD job may carry the luma predictions instead of the luma
  // mask; deriving the four luma mask samples here removes any batch ordering
  // dependency. The values are identical to subsampling the plane-0 mask.
  if (job.type == kCompoundDiffWtd && job.luma_p0 != nullptr &&
      job.luma_p1 != nullptr && job.luma_mask == nullptr) {
    const int stride0 = job.luma_p0_stride;
    const int stride1 = job.luma_p1_stride;
    const int m00 = adv::diffwtd_mask_value(
        job.luma_p0[(2 * i) * stride0 + 2 * j],
        job.luma_p1[(2 * i) * stride1 + 2 * j], job.mask_type,
        kCompoundPostRound, job.bit_depth);
    const int m01 = adv::diffwtd_mask_value(
        job.luma_p0[(2 * i) * stride0 + 2 * j + 1],
        job.luma_p1[(2 * i) * stride1 + 2 * j + 1], job.mask_type,
        kCompoundPostRound, job.bit_depth);
    if (!job.sub_y) {
      return adv::mask_subsample(m00, m01, 0, 0, job.sub_x, job.sub_y);
    }
    const int m10 = adv::diffwtd_mask_value(
        job.luma_p0[(2 * i + 1) * stride0 + 2 * j],
        job.luma_p1[(2 * i + 1) * stride1 + 2 * j], job.mask_type,
        kCompoundPostRound, job.bit_depth);
    const int m11 = adv::diffwtd_mask_value(
        job.luma_p0[(2 * i + 1) * stride0 + 2 * j + 1],
        job.luma_p1[(2 * i + 1) * stride1 + 2 * j + 1], job.mask_type,
        kCompoundPostRound, job.bit_depth);
    return adv::mask_subsample(m00, m01, m10, m11, job.sub_x, job.sub_y);
  }

  if (job.luma_mask == nullptr) {
    return 32;  // matches the CPU reference's defined fallback
  }
  const int stride = job.luma_mask_stride;
  if (!job.sub_x) {
    return job.luma_mask[i * stride + j];
  }
  if (!job.sub_y) {
    return adv::mask_subsample(job.luma_mask[i * stride + 2 * j],
                               job.luma_mask[i * stride + 2 * j + 1], 0, 0,
                               job.sub_x, job.sub_y);
  }
  return adv::mask_subsample(
      job.luma_mask[(2 * i) * stride + 2 * j],
      job.luma_mask[(2 * i) * stride + 2 * j + 1],
      job.luma_mask[(2 * i + 1) * stride + 2 * j],
      job.luma_mask[(2 * i + 1) * stride + 2 * j + 1], job.sub_x, job.sub_y);
}

// Blends one sample. For 8-bit this is byte-identical to
// adv::compound_sample_at(); the only change is Clip1's bit depth.
AV2ENC_CF_HD int blend_sample_at(const BlendJob& job, int i, int j,
                                        int pred0, int pred1, int intra_pred,
                                        int inter_pred) {
  const int bd = job.bit_depth;
  if (job.is_inter_intra) {
    const int m = job.type == kCompoundWedge
                      ? mask_sample(job, i, j)
                      : cf_intra_mode_mask_value(
                            static_cast<adv::InterIntraMode>(
                                job.interintra_mode),
                            job.rect_w, job.rect_h, i, j);
    return adv::inter_intra_blend_sample(m, intra_pred, inter_pred);
  }
  if (job.type == kCompoundAverage) {
    if (job.imp_mask_eligible) {
      const int m = implicit_mask_sample(job, i, j);
      return clip1_bd(round2(m * pred0 + (2 - m) * pred1,
                             1 + kCompoundPostRound),
                      bd);
    }
    const int cw = job.cwp_weight;
    return clip1_bd(
        round2(cw * pred0 + (16 - cw) * pred1, 4 + kCompoundPostRound), bd);
  }
  const int m = mask_sample(job, i, j);
  return clip1_bd(round2(m * pred0 + (64 - m) * pred1, 6 + kCompoundPostRound),
                  bd);
}

AV2ENC_CF_HD int mask_out_at(const BlendJob& job, int i, int j) {
  return job.type == kCompoundWedge
             ? wedge_cache_sample(job.wedge_cache, job.bsize, job.wedge_sign,
                                  job.wedge_index, i, j)
             : luma_mask_value(job, i, j);
}

// Runs one blend job. `tid`/`nthr` define the sample stride.
AV2ENC_CF_HD void run_blend_job_device(const BlendJob& job, int tid,
                                              int nthr) {
  const int w = job.rect_w;
  const int h = job.rect_h;
  if (w <= 0 || h <= 0) {
    return;
  }
  const int total = w * h;
  for (int index = tid; index < total; index += nthr) {
    const int i = index / w;
    const int j = index - i * w;
    const std::int16_t* p0_row =
        job.p0 ? job.p0 + static_cast<std::ptrdiff_t>(i) * job.p0_stride
               : nullptr;
    const std::int16_t* p1_row =
        job.p1 ? job.p1 + static_cast<std::ptrdiff_t>(i) * job.p1_stride
               : nullptr;
    const int pred0 = p0_row ? p0_row[j] : 0;
    const int pred1 = p1_row ? p1_row[j] : 0;
    std::size_t dst_index =
        static_cast<std::size_t>(job.rect_y + i) *
            static_cast<std::size_t>(job.dst_stride) +
        static_cast<std::size_t>(job.rect_x + j);
    const int intra_pred =
        job.intra
            ? load_sample(job.intra,
                          static_cast<std::size_t>(i) *
                                  static_cast<std::size_t>(job.intra_stride) +
                              static_cast<std::size_t>(j),
                          job.intra_16bit)
            : 0;
    const int inter_pred = load_sample(job.dst, dst_index, job.dst_16bit);
    store_sample(job.dst, dst_index, job.dst_16bit,
                 blend_sample_at(job, i, j, pred0, pred1, intra_pred,
                                 inter_pred));
  }
  if (job.mask_out != nullptr && job.plane == 0) {
    for (int index = tid; index < total; index += nthr) {
      const int i = index / w;
      const int j = index - i * w;
      job.mask_out[static_cast<std::ptrdiff_t>(i) * job.mask_out_stride + j] =
          static_cast<std::uint8_t>(mask_out_at(job, i, j));
    }
  }
}

// ---------------------------------------------------------------------------
// Skip-mode derived prediction: both lists, same MV, compound average, no
// residual.
// ---------------------------------------------------------------------------

AV2ENC_CF_HD void run_skip_job_device(const SkipJob& job, int tid,
                                             int nthr) {
  const int mv_pair[2] = {job.mv_row, job.mv_col};
  (void)mv_pair;
  for (int plane = 0; plane < 3; ++plane) {
    const int sub_x = plane > 0 ? 1 : 0;
    const int sub_y = plane > 0 ? 1 : 0;
    const int px = job.x >> sub_x;
    const int py = job.y >> sub_y;
    const int pw = job.w >> sub_x;
    const int ph = job.h >> sub_y;
    if (pw <= 0 || ph <= 0 || job.dst[plane] == nullptr) {
      continue;
    }
    const SampleLoc loc = location_unscaled(px, py, job.mv_row, job.mv_col,
                                            sub_x, sub_y);
    const mc::InterpFilter filter_h =
        mc::mc_pass_filter(static_cast<mc::InterpFilter>(job.filter),
                           pw <= 4);
    const mc::InterpFilter filter_v =
        mc::mc_pass_filter(static_cast<mc::InterpFilter>(job.filter),
                           ph <= 4);
    const std::int16_t* taps_h =
        &mc::kSubpelFilters[static_cast<int>(filter_h)][0][0];
    const std::int16_t* taps_v =
        &mc::kSubpelFilters[static_cast<int>(filter_v)][0][0];
    const int total = pw * ph;
    for (int index = tid; index < total; index += nthr) {
      const int r = index / pw;
      const int c = index - r * pw;
      const int p0 = filter_sample(job.ref[0][plane], 0, 0,
                                   job.ref[0][plane].width - 1,
                                   job.ref[0][plane].height - 1, loc, taps_h,
                                   taps_v, r, c, kCfInterRound1Compound);
      const int p1 = filter_sample(job.ref[1][plane], 0, 0,
                                   job.ref[1][plane].width - 1,
                                   job.ref[1][plane].height - 1, loc, taps_h,
                                   taps_v, r, c, kCfInterRound1Compound);
      const int cw = job.cwp_weight;
      const int value = clip1_bd(
          round2(cw * p0 + (16 - cw) * p1, 4 + kCompoundPostRound),
          job.bit_depth);
      store_sample(job.dst[plane],
                   static_cast<std::size_t>(py + r) *
                           static_cast<std::size_t>(job.dst_stride[plane]) +
                       static_cast<std::size_t>(px + c),
                   job.dst_16bit[plane], value);
    }
  }
}

// ---------------------------------------------------------------------------
// §7.13.3.4/.5 get_ref_area()
// ---------------------------------------------------------------------------

AV2ENC_CF_HD void ref_area_compute(const RefAreaInput& in,
                                          int out[2][4]) {
  const int refs = in.num_refs < 2 ? in.num_refs : 2;
  for (int list = 0; list < refs; ++list) {
    const int last_x = ((in.ref_mi_cols[list] * kCfMiSize) >> in.sub_x) - 1;
    const int last_y = ((in.ref_mi_rows[list] * kCfMiSize) >> in.sub_y) - 1;
    const int mv_pair[2] = {in.cand_mv[list][0], in.cand_mv[list][1]};
    const adv::MvScaleOutput loc = adv::motion_vector_scaling(
        0, in.sub_x, in.sub_y, in.x, in.y, mv_pair, false,
        in.ref_luma_w[list], in.ref_luma_h[list], in.frame_w, in.frame_h);
    const int pad_lo_x = in.w == 4 ? 1 : 3;
    const int pad_hi_x = in.w == 4 ? 2 : 4;
    const int pad_lo_y = in.h == 4 ? 1 : 3;
    const int pad_hi_y = in.h == 4 ? 2 : 4;
    out[list][0] = clip3(0, last_x, (loc.start_x >> 10) - pad_lo_x);
    out[list][1] = clip3(0, last_y, (loc.start_y >> 10) - pad_lo_y);
    out[list][2] = clip3(
        0, last_x, ((loc.start_x + loc.step_x * (in.w - 1)) >> 10) + pad_hi_x);
    out[list][3] = clip3(
        0, last_y, ((loc.start_y + loc.step_y * (in.h - 1)) >> 10) + pad_hi_y);
  }
}

// ---------------------------------------------------------------------------
// OPFL helpers (§7.13.3.9-.12), mirroring temporal_math.h exactly.
// ---------------------------------------------------------------------------

AV2ENC_CF_HD int get_msb(std::uint32_t value) {
  int result = 0;
  while (value > 1) {
    value >>= 1;
    ++result;
  }
  return result;
}

AV2ENC_CF_HD bool opfl_scale_distances(int d0, int d1, int out[2]) {
  if (d0 == 0 || d1 == 0) {
    return false;
  }
  if (abs_int(d0) == abs_int(d1)) {
    out[0] = d0 < 0 ? -1 : 1;
    out[1] = d1 < 0 ? -1 : 1;
  } else if (abs_int(d0) > abs_int(d1)) {
    out[0] = d0 < 0 ? -2 : 2;
    out[1] = d1 < 0 ? -1 : 1;
  } else {
    out[0] = d0 < 0 ? -1 : 1;
    out[1] = d1 < 0 ? -2 : 2;
  }
  return true;
}

AV2ENC_CF_HD void divide_and_round_array(const int sol[2], int den,
                                                int shift, int result[2]) {
  int inv_den = 1;
  int den_shift = 0;
  if (den != 1) {
    const adv::DivScaleShift ds = adv::resolve_divisor(den);
    den_shift = ds.shift;
    inv_den = ds.factor;
  }
  const int inv_den_msb = get_msb(static_cast<std::uint32_t>(
      inv_den < 0 ? -inv_den : inv_den));
  for (int i = 0; i < 2; ++i) {
    result[i] = 0;
    if (sol[i] == 0) {
      continue;
    }
    const bool sgn = sol[i] > 0;
    int tmp = sgn ? sol[i] : -sol[i];
    const int raw_red_bits =
        get_msb(static_cast<std::uint32_t>(tmp)) + inv_den_msb + 4 -
        kOpflMaxLsBits;
    const int num_red_bits = raw_red_bits > 0 ? raw_red_bits : 0;
    if (num_red_bits > 0) {
      tmp = round2_signed(tmp, num_red_bits);
    }
    const int inc_bits = shift + num_red_bits - den_shift;
    std::int64_t value;
    if (inc_bits <= -31) {
      tmp = round2_signed(tmp, -inc_bits - 30);
      const std::int64_t mult = static_cast<std::int64_t>(tmp) * inv_den;
      value = round2_signed64(mult, 30);
    } else {
      const std::int64_t mult = static_cast<std::int64_t>(tmp) * inv_den;
      if (inc_bits >= 0) {
        value = mult << inc_bits;
      } else {
        value = round2_signed64(mult, -inc_bits);
      }
    }
    result[i] = sgn ? static_cast<int>(value) : -static_cast<int>(value);
  }
}

AV2ENC_CF_HD int opfl_x_grad(const int* tmp, int w, int h, int i, int j) {
  const int i_start = (i >> 4) << 4;
  const int i_end = (i_start + 16 < h ? i_start + 16 : h) - 1;
  const int j_start = (j >> 4) << 4;
  const int j_end = (j_start + 16 < w ? j_start + 16 : w) - 1;
  const int j_prev = j - 1 > j_start ? j - 1 : j_start;
  const int j_prev2 = j - 2 > j_start ? j - 2 : j_start;
  const int j_next = j + 1 < j_end ? j + 1 : j_end;
  const int j_next2 = j + 2 < j_end ? j + 2 : j_end;
  int temp = 42 * (tmp[i * w + j_next] - tmp[i * w + j_prev]) -
             5 * (tmp[i * w + j_next2] - tmp[i * w + j_prev2]);
  if (j + 1 > j_end || j - 1 < j_start) {
    temp <<= 1;
  }
  (void)i_start;
  (void)i_end;
  return round2_signed(temp, 7);
}

AV2ENC_CF_HD int opfl_y_grad(const int* tmp, int w, int h, int i, int j) {
  const int i_start = (i >> 4) << 4;
  const int i_end = (i_start + 16 < h ? i_start + 16 : h) - 1;
  const int j_start = (j >> 4) << 4;
  const int j_end = (j_start + 16 < w ? j_start + 16 : w) - 1;
  (void)j_start;
  (void)j_end;
  const int i_prev = i - 1 > i_start ? i - 1 : i_start;
  const int i_prev2 = i - 2 > i_start ? i - 2 : i_start;
  const int i_next = i + 1 < i_end ? i + 1 : i_end;
  const int i_next2 = i + 2 < i_end ? i + 2 : i_end;
  int temp = 42 * (tmp[i_next * w + j] - tmp[i_prev * w + j]) -
             5 * (tmp[i_next2 * w + j] - tmp[i_prev2 * w + j]);
  if (i + 1 > i_end || i - 1 < i_start) {
    temp <<= 1;
  }
  return round2_signed(temp, 7);
}

AV2ENC_CF_HD std::size_t opfl_index(int stride4, int row4, int col4,
                                           int list, int comp) {
  return ((static_cast<std::size_t>(row4) *
               static_cast<std::size_t>(stride4) +
           static_cast<std::size_t>(col4)) *
              2 +
          static_cast<std::size_t>(list)) *
             2 +
         static_cast<std::size_t>(comp);
}

// Runs one OPFL job. `tmp`/`pdiff` are caller-provided scratch of w*h ints
// (shared memory in the kernel, host memory in the emulation test).
AV2ENC_CF_HD void run_opfl_job_device(const OpflJob& job, int tid,
                                             int nthr, int* tmp, int* pdiff) {
  int dist[2] = {0, 0};
  const int d0 = get_relative_dist(job.order_hint, job.ref0_order_hint);
  const int d1 = get_relative_dist(job.order_hint, job.ref1_order_hint);
  if (!opfl_scale_distances(d0, d1, dist)) {
    return;  // §7.13.3.9 terminates immediately
  }
  const int w = job.w;
  const int h = job.h;
  const int total = w * h;
  const int shift = job.bit_depth - 8;

  for (int index = tid; index < total; index += nthr) {
    const int i = index / w;
    const int j = index - i * w;
    const int s0 = load_sample(job.p0,
                               static_cast<std::size_t>(i) *
                                       static_cast<std::size_t>(job.p0_stride) +
                                   static_cast<std::size_t>(j),
                               job.p0_16);
    const int s1 = load_sample(job.p1,
                               static_cast<std::size_t>(i) *
                                       static_cast<std::size_t>(job.p1_stride) +
                                   static_cast<std::size_t>(j),
                               job.p1_16);
    tmp[index] = round2_signed(dist[0] * s0 - dist[1] * s1, shift);
    pdiff[index] = s0 - s1;
  }
  AV2ENC_CF_SYNC();

  // §7.13.3.8 initial values; a det <= 0 block keeps them.
  const int cells = (h / 4) * (w / 4);
  for (int cell = tid; cell < cells; cell += nthr) {
    const int r4 = cell / (w / 4);
    const int c4 = cell - r4 * (w / 4);
    const int row4 = (job.opt_y >> 2) + r4;
    const int col4 = (job.opt_x >> 2) + c4;
    for (int list = 0; list < 2; ++list) {
      for (int comp = 0; comp < 2; ++comp) {
        const int base = job.mv[list][comp];
        job.opfl_mvs[opfl_index(job.opfl_stride4, row4, col4, list, comp)] =
            static_cast<std::int16_t>(
                clip3(-(1 << 17), (1 << 17) - 1, base * 2));
        job.mv_deltas[opfl_index(job.delta_stride4, row4, col4, list, comp)] =
            0;
      }
    }
  }
  AV2ENC_CF_SYNC();

  // §7.13.3.12 compute_opfl_mv().
  const int n = job.n;
  const int blocks = (w / n) * (h / n);
  for (int block = tid; block < blocks; block += nthr) {
    const int i_base = (block / (w / n)) * n;
    const int j_base = (block - (block / (w / n)) * (w / n)) * n;
    std::int64_t su2 = 0;
    std::int64_t sv2 = 0;
    std::int64_t suv = 0;
    std::int64_t suw = 0;
    std::int64_t svw = 0;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        const int gi = i_base + i;
        const int gj = j_base + j;
        const std::int64_t u = opfl_x_grad(tmp, w, h, gi, gj);
        const std::int64_t v = opfl_y_grad(tmp, w, h, gi, gj);
        const std::int64_t e = pdiff[gi * w + gj];
        su2 += u * u;
        suv += u * v;
        sv2 += v * v;
        suw += u * e;
        svw += v * e;
      }
    }
    su2 += n * n;
    sv2 += n * n;

    const int msb_su2 = 1 + get_msb(static_cast<std::uint32_t>(su2));
    const int msb_sv2 = 1 + get_msb(static_cast<std::uint32_t>(sv2));
    const int msb_suv = 1 + get_msb(static_cast<std::uint32_t>(
                                 abs_int(static_cast<int>(suv))));
    const int msb_suw = 1 + get_msb(static_cast<std::uint32_t>(
                                 abs_int(static_cast<int>(suw))));
    const int msb_svw = 1 + get_msb(static_cast<std::uint32_t>(
                                 abs_int(static_cast<int>(svw))));
    const int a1 = msb_su2 + msb_sv2;
    const int a2 = msb_sv2 + msb_suw;
    const int a3 = msb_suv + msb_svw;
    const int a4 = msb_su2 + msb_svw;
    const int a5 = msb_suv + msb_suw;
    const int m12 = a1 > a2 ? a1 : a2;
    const int m34 = a3 > a4 ? a3 : a4;
    const int max_mult_msb = m12 > m34 ? (m12 > a5 ? m12 : a5)
                                      : (m34 > a5 ? m34 : a5);
    const int raw_redbit = max_mult_msb - kOpflMaxLsBits + 3;
    const int redbit = (raw_redbit > 0 ? raw_redbit : 0) >> 1;
    su2 = round2_signed64(su2, redbit);
    sv2 = round2_signed64(sv2, redbit);
    suv = round2_signed64(suv, redbit);
    suw = round2_signed64(suw, redbit);
    svw = round2_signed64(svw, redbit);
    const std::int64_t det = su2 * sv2 - suv * suv;
    if (det <= 0) {
      continue;
    }
    const int bits = 4 - 1;
    const int sol[2] = {static_cast<int>(sv2 * suw - suv * svw),
                        static_cast<int>(su2 * svw - suv * suw)};
    int solved[2] = {0, 0};
    divide_and_round_array(sol, static_cast<int>(det), bits, solved);
    int vx0 = -solved[0];
    int vy0 = -solved[1];
    int vx1 = vx0 * dist[1];
    int vy1 = vy0 * dist[1];
    vx0 = vx0 * dist[0];
    vy0 = vy0 * dist[0];
    const int mv_delta[2][2] = {
        {clip3(-kOpflMvDeltaLimit, kOpflMvDeltaLimit, vy0),
         clip3(-kOpflMvDeltaLimit, kOpflMvDeltaLimit, vx0)},
        {clip3(-kOpflMvDeltaLimit, kOpflMvDeltaLimit, vy1),
         clip3(-kOpflMvDeltaLimit, kOpflMvDeltaLimit, vx1)}};
    const int row4 = (job.opt_y + i_base) >> 2;
    const int col4 = (job.opt_x + j_base) >> 2;
    for (int list = 0; list < 2; ++list) {
      for (int comp = 0; comp < 2; ++comp) {
        const int delta = mv_delta[list][comp];
        const std::size_t idx =
            opfl_index(job.delta_stride4, row4, col4, list, comp);
        job.mv_deltas[idx] = static_cast<std::int16_t>(delta);
        const int base = job.mv[list][comp];
        const int new_comp = base * 2 + delta;
        job.opfl_mvs[idx] = static_cast<std::int16_t>(
            clip3(-(1 << 17), (1 << 17) - 1, new_comp));
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Refine-MV search (§7.13.3.6)
// ---------------------------------------------------------------------------

AV2ENC_CF_DEV_TABLE int kRefinemvNeighbors[24][2] = {
    {-2, -2}, {-2, -1}, {-2, 0}, {-2, 1}, {-2, 2}, {-1, -2},
    {-1, -1}, {-1, 0},  {-1, 1}, {-1, 2}, {0, -2},  {0, -1},
    {0, 1},   {0, 2},   {1, -2}, {1, -1}, {1, 0},   {1, 1},
    {1, 2},   {2, -2},  {2, -1}, {2, 0},  {2, 1},   {2, 2}};

AV2ENC_CF_HD std::int64_t refine_candidate_sad(
    const RefineSearchJob& job, const int area[2][4], int dx, int dy, int tid,
    int nthr, std::int64_t* shared) {
  const int mv0_row = job.cand_mv[0][0] + dy * 8;
  const int mv0_col = job.cand_mv[0][1] + dx * 8;
  const int mv1_row = job.cand_mv[1][0] - dy * 8;
  const int mv1_col = job.cand_mv[1][1] - dx * 8;
  const int x = job.x - 2;
  const int y = job.y - 2;
  const int w = job.w + 4;
  const int h = job.h + 4;
  const int rows = (h + 1) / 2;
  const int total = rows * w;
  std::int64_t partial = 0;
  for (int index = tid; index < total; index += nthr) {
    const int rr = index / w;
    const int c = index - rr * w;
    const int r = rr * 2;
    const int p0 = predict_unscaled_u8(job.ref[0], area[0], x, y, w, h, mv0_row,
                                       mv0_col,
                                       static_cast<int>(mc::InterpFilter::kBilinear),
                                       job.bit_depth, r, c);
    const int p1 = predict_unscaled_u8(job.ref[1], area[1], x, y, w, h, mv1_row,
                                       mv1_col,
                                       static_cast<int>(mc::InterpFilter::kBilinear),
                                       job.bit_depth, r, c);
    partial += p0 > p1 ? p0 - p1 : p1 - p0;
  }
  const int shift = job.bit_depth - 8;
  if (shift > 0) {
    partial >>= shift;
  }
  shared[tid] = partial;
  AV2ENC_CF_SYNC();
  if (tid == 0) {
    std::int64_t sum = 0;
    for (int t = 0; t < nthr; ++t) {
      sum += shared[t];
    }
    shared[0] = sum;
  }
  AV2ENC_CF_SYNC();
  return shared[0];
}

// Runs one §7.13.3.6 search. `shared` must hold at least nthr int64 values.
AV2ENC_CF_HD void run_refine_search_device(const RefineSearchJob& job,
                                                  RefineResult* out, int tid,
                                                  int nthr,
                                                  std::int64_t* shared) {
  int area[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
  for (int list = 0; list < 2; ++list) {
    if (job.use_ref_area) {
      area[list][0] = job.area[list][0];
      area[list][1] = job.area[list][1];
      area[list][2] = job.area[list][2];
      area[list][3] = job.area[list][3];
    } else {
      area[list][0] = 0;
      area[list][1] = 0;
      area[list][2] = job.ref[list].width - 1;
      area[list][3] = job.ref[list].height - 1;
    }
  }

  RefineResult result{};
  result.dx = 0;
  result.dy = 0;
  result.best_sad = 0;
  for (int list = 0; list < 2; ++list) {
    result.offset_mv[list][0] = job.cand_mv[list][0];
    result.offset_mv[list][1] = job.cand_mv[list][1];
  }

  const bool range_ok =
      job.cand_mv[0][0] - 4 * 8 >= kCfMvLow + 1 &&
      job.cand_mv[0][0] + 2 * 8 <= kCfMvUpp - 1 &&
      job.cand_mv[0][1] - 4 * 8 >= kCfMvLow + 1 &&
      job.cand_mv[0][1] + 2 * 8 <= kCfMvUpp - 1 &&
      job.cand_mv[1][0] - 4 * 8 >= kCfMvLow + 1 &&
      job.cand_mv[1][0] + 2 * 8 <= kCfMvUpp - 1 &&
      job.cand_mv[1][1] - 4 * 8 >= kCfMvLow + 1 &&
      job.cand_mv[1][1] + 2 * 8 <= kCfMvUpp - 1;
  if (!range_ok) {
    if (tid == 0) {
      *out = result;
    }
    return;
  }

  const int w = job.w + 4;
  const int h = job.h + 4;
  const bool allow_centre = job.tip_pred || !job.switchable;
  int best_dy = 0;
  int best_dx = 0;
  std::int64_t best_sad = 0;
  const std::int64_t threshold =
      static_cast<std::int64_t>(w) * static_cast<std::int64_t>(h) * 2;

  if (allow_centre) {
    best_sad = refine_candidate_sad(job, area, 0, 0, tid, nthr, shared);
    best_sad -= best_sad >> 3;
    if (best_sad < threshold) {
      if (tid == 0) {
        result.best_sad = best_sad;
        *out = result;
      }
      return;
    }
  }

  for (int idx = 0; idx < 24; ++idx) {
    const int try_dy = kRefinemvNeighbors[idx][0];
    const int try_dx = kRefinemvNeighbors[idx][1];
    const std::int64_t sad =
        refine_candidate_sad(job, area, try_dx, try_dy, tid, nthr, shared);
    if ((idx == 0 && !allow_centre) || sad < best_sad) {
      best_dy = try_dy;
      best_dx = try_dx;
      best_sad = sad;
    }
  }

  if (tid == 0) {
    result.dx = best_dx;
    result.dy = best_dy;
    result.best_sad = best_sad;
    result.offset_mv[0][0] = job.cand_mv[0][0] + best_dy * 8;
    result.offset_mv[0][1] = job.cand_mv[0][1] + best_dx * 8;
    result.offset_mv[1][0] = job.cand_mv[1][0] - best_dy * 8;
    result.offset_mv[1][1] = job.cand_mv[1][1] - best_dx * 8;
    *out = result;
  }
}

// ---------------------------------------------------------------------------
// Final refine-MV prediction (§7.13.3.3): EIGHTTAP_SHARP 1/16 + average.
// ---------------------------------------------------------------------------

AV2ENC_CF_HD void run_refine_predict_device(
    const RefinePredictJob& job, int tid, int nthr) {
  int area[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
  for (int list = 0; list < 2; ++list) {
    if (job.use_ref_area) {
      area[list][0] = job.area[list][0];
      area[list][1] = job.area[list][1];
      area[list][2] = job.area[list][2];
      area[list][3] = job.area[list][3];
    } else {
      area[list][0] = 0;
      area[list][1] = 0;
      area[list][2] = job.ref[list].width - 1;
      area[list][3] = job.ref[list].height - 1;
    }
  }

  SampleLoc loc[2];
  for (int list = 0; list < 2; ++list) {
    loc[list] = location_scaled(job.ref[list], job.x, job.y,
                                job.offset_mv[list][0], job.offset_mv[list][1],
                                job.sub_x, job.sub_y, job.frame_w, job.frame_h);
  }
  const mc::InterpFilter fh_enum = mc::mc_pass_filter(
      mc::InterpFilter::kEighttapSharp, job.w <= 4);
  const mc::InterpFilter fv_enum = mc::mc_pass_filter(
      mc::InterpFilter::kEighttapSharp, job.h <= 4);
  const std::int16_t* taps_h =
      &mc::kSubpelFilters[static_cast<int>(fh_enum)][0][0];
  const std::int16_t* taps_v =
      &mc::kSubpelFilters[static_cast<int>(fv_enum)][0][0];

  const int total = job.w * job.h;
  for (int index = tid; index < total; index += nthr) {
    const int r = index / job.w;
    const int c = index - r * job.w;
    const int p0 = filter_sample(job.ref[0], area[0][0], area[0][1],
                                 area[0][2], area[0][3], loc[0], taps_h,
                                 taps_v, r, c, kCfInterRound1Compound);
    const int p1 = filter_sample(job.ref[1], area[1][0], area[1][1],
                                 area[1][2], area[1][3], loc[1], taps_h,
                                 taps_v, r, c, kCfInterRound1Compound);
    const int cw = job.cwp_weight;
    const int value = clip1_bd(
        round2(cw * p0 + (16 - cw) * p1, 4 + kCompoundPostRound),
        job.bit_depth);
    store_sample(job.dst,
                 static_cast<std::size_t>(r) *
                         static_cast<std::size_t>(job.dst_stride) +
                     static_cast<std::size_t>(c),
                 job.dst_16bit, value);
  }
}

}  // namespace detail

// Public wrapper declared in compound_full.h.
AV2ENC_CF_HD void ref_area_compute(const RefAreaInput& in,
                                          int out[2][4]) {
  detail::ref_area_compute(in, out);
}

}  // namespace av2enc::cuda::compound_full

#undef AV2ENC_CF_HD
#undef AV2ENC_CF_SYNC
#undef AV2ENC_CF_DEV_TABLE

#endif  // AV2ENC_CUDA_COMPOUND_FULL_COMPOUND_FULL_DEVICE_CUH_
