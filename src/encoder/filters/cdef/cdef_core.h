// AV2 §7.18 CDEF — shared host/device core.
//
// This header holds the normative arithmetic of the Constrained Directional
// Enhancement Filter (CDEF) in a form that compiles both as ordinary host C++
// and as CUDA device code.  The CPU reference (cdef.cpp) and the CUDA kernel
// (src/cuda/cdef) call the *same* inline functions, so CPU and GPU results are
// bit-exact by construction when the input planes and metadata match.
//
// Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
// (2026-05-28, commit 38f28e7):
//   §5.18.7.10 CDEF params syntax   (damping, strengths, skip flag)
//   §6.17.7.6  CDEF params semantics
//   §5.20.9.3  is_inside_filter_region
//   §7.18      CDEF process         (8x8 block traversal, 64x64 parameter grid)
//   §7.18.1    CDEF block process   (copy, skip, strength derivation)
//   §7.18.2    CDEF direction process
//   §7.18.3    CDEF filter process  (primary/secondary taps, constrain, clip)
//
// Scope: 8-bit 4:2:0 planar.  All arithmetic is integer; no floating point is
// used anywhere in the filter path.
//
// CDEF is a pure function of the *unfiltered* reconstructed frame: every tap
// reads CurrFrame, never the output CdefFrame.  Blocks therefore have no
// ordering dependency and both the CPU and GPU implementations may filter them
// in any order.

#ifndef AV2ENC_ENCODER_FILTERS_CDEF_CDEF_CORE_H_
#define AV2ENC_ENCODER_FILTERS_CDEF_CDEF_CORE_H_

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define AV2ENC_CDEF_HD __host__ __device__ __forceinline__
// During the device compilation pass the normative tables must live in device
// memory; the host pass sees ordinary inline constexpr arrays.
#if defined(__CUDA_ARCH__)
#define AV2ENC_CDEF_TABLE __device__ constexpr
#else
#define AV2ENC_CDEF_TABLE inline constexpr
#endif
#else
#define AV2ENC_CDEF_HD inline
#define AV2ENC_CDEF_TABLE inline constexpr
#endif

namespace av2enc::cdef {

// MI (mode-info) granularity and CDEF block geometry.
constexpr int kMiSize = 4;
constexpr int kMiSizeLog2 = 2;
constexpr int kCdefBlockSize = 64;   // one parameter set per 64x64 luma block
constexpr int kCdefSize4 = 16;       // 64 >> kMiSizeLog2
constexpr int kCdefBlock4 = 2;       // 8x8 CDEF block in MI units
constexpr int kMaxStrengths = 8;     // cdef_strengths_minus_1 f(3), plus one

// Cdef_Directions[dir][k] = (dy, dx); k = 0 is the distance-1 tap and k = 1
// the distance-2 tap.  Directions are 0, +22.5, ..., +157.5 degrees.
AV2ENC_CDEF_TABLE int kCdefDirections[8][2][2] = {
    {{-1, 1}, {-2, 2}},
    {{0, 1}, {-1, 2}},
    {{0, 1}, {0, 2}},
    {{0, 1}, {1, 2}},
    {{1, 1}, {2, 2}},
    {{1, 0}, {2, 1}},
    {{1, 0}, {2, 0}},
    {{1, 0}, {2, -1}},
};

// Cdef_Pri_Taps[parity][k], indexed by (priStr >> coeffShift) & 1.
AV2ENC_CDEF_TABLE int kCdefPriTaps[2][2] = {{4, 2}, {3, 3}};
// Cdef_Sec_Taps[parity][k]; the two rows are identical in AV2 v1.0.0.
AV2ENC_CDEF_TABLE int kCdefSecTaps[2][2] = {{2, 1}, {2, 1}};

// Cdef_Uv_Dir[SubsamplingX][SubsamplingY][yDir].
AV2ENC_CDEF_TABLE int kCdefUvDir[2][2][8] = {
    {{0, 1, 2, 3, 4, 5, 6, 7}, {1, 2, 2, 2, 3, 4, 6, 0}},
    {{7, 0, 2, 4, 5, 6, 6, 6}, {0, 1, 2, 3, 4, 5, 6, 7}},
};

// Div_Table[0..8] from §7.18.2.
AV2ENC_CDEF_TABLE int kCdefDivTable[9] = {0,   840, 420, 280, 210,
                                          168, 140, 120, 105};

// Region used by is_inside_filter_region (§5.20.9.3), in luma MI units.
struct CdefRegion {
  int col_start;
  int row_start;
  int col_end;
  int row_end;
};

AV2ENC_CDEF_HD constexpr int cdef_clip3(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}

AV2ENC_CDEF_HD constexpr int cdef_abs(int value) {
  return value < 0 ? -value : value;
}

// FloorLog2 for a positive integer; returns 0 for values <= 1.
AV2ENC_CDEF_HD constexpr int cdef_floor_log2(int value) {
  int result = 0;
  while (value > 1) {
    value >>= 1;
    ++result;
  }
  return result;
}

// constrain(diff, threshold, damping) from §7.18.3.
AV2ENC_CDEF_HD int cdef_constrain(int diff, int threshold,
                                         int damping) {
  if (threshold == 0) {
    return 0;
  }
  int damping_adj = damping - cdef_floor_log2(threshold);
  if (damping_adj < 0) {
    damping_adj = 0;
  }
  const int magnitude = cdef_abs(diff);
  int value = threshold - (magnitude >> damping_adj);
  if (value < 0) {
    value = 0;
  }
  if (value > magnitude) {
    value = magnitude;
  }
  return diff < 0 ? -value : value;
}

// cdef_get_at: fetch a tap sample and report availability.  `x0`/`y0` are the
// plane coordinates of the block origin, `i`/`j` the sample inside the block.
// Returns false when the candidate luma MI cell is outside the filter region;
// the caller must use the returned `value` only when the function is true.
AV2ENC_CDEF_HD bool cdef_get_at(const std::uint8_t* src,
                                       std::size_t stride, int x0, int y0,
                                       int i, int j, int dir, int k, int sign,
                                       int sub_x, int sub_y,
                                       const CdefRegion& region, int* value) {
  const int y = y0 + i + sign * kCdefDirections[dir][k][0];
  const int x = x0 + j + sign * kCdefDirections[dir][k][1];
  // Map the plane coordinate to a luma MI coordinate.  For chroma the shift
  // doubles the sample coordinate before the MI shift, matching
  // `(y << subY) >> MI_SIZE_LOG2` in cdef_get_at.
  const int candidate_r = (y * (1 << sub_y)) >> kMiSizeLog2;
  const int candidate_c = (x * (1 << sub_x)) >> kMiSizeLog2;
  if (candidate_c < region.col_start || candidate_c >= region.col_end ||
      candidate_r < region.row_start || candidate_r >= region.row_end) {
    return false;
  }
  *value = src[static_cast<std::size_t>(y) * stride +
               static_cast<std::size_t>(x)];
  return true;
}

// Filter one sample of one plane with the §7.18.3 process.  `pri_str` is the
// final primary strength (after the variance scaling for luma, unscaled for
// chroma); `damping` already includes CdefDamping + coeffShift adjustments.
AV2ENC_CDEF_HD int cdef_filter_sample(const std::uint8_t* src,
                                             std::size_t stride, int x0,
                                             int y0, int i, int j, int dir,
                                             int pri_str, int sec_str,
                                             int damping, int sub_x, int sub_y,
                                             const CdefRegion& region) {
  const int x = src[static_cast<std::size_t>(y0 + i) * stride +
                    static_cast<std::size_t>(x0 + j)];
  int sum = 0;
  int max_value = x;
  int min_value = x;
  const int tap_row = pri_str & 1;  // (priStr >> coeffShift) & 1, 8-bit
  for (int k = 0; k < 2; ++k) {
    for (int sign = -1; sign <= 1; sign += 2) {
      int p = 0;
      if (cdef_get_at(src, stride, x0, y0, i, j, dir, k, sign, sub_x, sub_y,
                      region, &p)) {
        sum += kCdefPriTaps[tap_row][k] *
               cdef_constrain(p - x, pri_str, damping);
        max_value = p > max_value ? p : max_value;
        min_value = p < min_value ? p : min_value;
      }
      for (int dir_off = -2; dir_off <= 2; dir_off += 4) {
        int s = 0;
        const int sec_dir = (dir + dir_off) & 7;
        if (cdef_get_at(src, stride, x0, y0, i, j, sec_dir, k, sign, sub_x,
                        sub_y, region, &s)) {
          sum += kCdefSecTaps[tap_row][k] *
                 cdef_constrain(s - x, sec_str, damping);
          max_value = s > max_value ? s : max_value;
          min_value = s < min_value ? s : min_value;
        }
      }
    }
  }
  const int adjusted = x + ((8 + sum - (sum < 0 ? 1 : 0)) >> 4);
  return cdef_clip3(min_value, max_value, adjusted);
}

// §7.18.2 CDEF direction process for the 8x8 luma block at (x0, y0).
// Outputs the direction index 0..7 and the scaled variance.
AV2ENC_CDEF_HD void cdef_direction(const std::uint8_t* luma,
                                          std::size_t stride, int x0, int y0,
                                          int bit_depth, int* out_dir,
                                          std::int64_t* out_var) {
  std::int64_t partial[8][15];
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 15; ++j) {
      partial[i][j] = 0;
    }
  }
  const int shift = bit_depth - 8;
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      const int sample =
          (luma[static_cast<std::size_t>(y0 + i) * stride +
                static_cast<std::size_t>(x0 + j)] >>
           shift) -
          128;
      const std::int64_t value = sample;
      partial[0][i + j] += value;
      partial[1][i + j / 2] += value;
      partial[2][i] += value;
      partial[3][3 + i - j / 2] += value;
      partial[4][7 + i - j] += value;
      partial[5][3 - i / 2 + j] += value;
      partial[6][j] += value;
      partial[7][i / 2 + j] += value;
    }
  }

  std::int64_t cost[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  const std::int64_t div8 = kCdefDivTable[8];
  for (int i = 0; i < 8; ++i) {
    cost[2] += partial[2][i] * partial[2][i];
    cost[6] += partial[6][i] * partial[6][i];
  }
  cost[2] *= div8;
  cost[6] *= div8;
  for (int i = 0; i < 7; ++i) {
    const std::int64_t div = kCdefDivTable[i + 1];
    cost[0] += (partial[0][i] * partial[0][i] +
                partial[0][14 - i] * partial[0][14 - i]) *
               div;
    cost[4] += (partial[4][i] * partial[4][i] +
                partial[4][14 - i] * partial[4][14 - i]) *
               div;
  }
  cost[0] += partial[0][7] * partial[0][7] * div8;
  cost[4] += partial[4][7] * partial[4][7] * div8;
  for (int i = 1; i < 8; i += 2) {
    for (int j = 0; j < 5; ++j) {
      cost[i] += partial[i][3 + j] * partial[i][3 + j];
    }
    cost[i] *= div8;
    for (int j = 0; j < 3; ++j) {
      const std::int64_t div = kCdefDivTable[2 * j + 2];
      cost[i] += (partial[i][j] * partial[i][j] +
                  partial[i][10 - j] * partial[i][10 - j]) *
                 div;
    }
  }

  std::int64_t best_cost = 0;
  int best_dir = 0;
  for (int i = 0; i < 8; ++i) {
    if (cost[i] > best_cost) {
      best_cost = cost[i];
      best_dir = i;
    }
  }
  *out_dir = best_dir;
  *out_var = (best_cost - cost[(best_dir + 4) & 7]) >> 10;
}

}  // namespace av2enc::cdef

#endif  // AV2ENC_ENCODER_FILTERS_CDEF_CDEF_CORE_H_
