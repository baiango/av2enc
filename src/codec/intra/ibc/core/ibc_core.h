// AV2 intra block copy — shared normative core (host + device).
//
// Spec-only: AV2 v1.0.0 §5.20.5.3/.4 (IBC mode info), §6.19.7.12
// (is_mv_valid), §6.19.5.4 (intrabc_mode semantics), §7.13.3.17 (motion
// vector scaling), §7.13.3.18 (integer-pel block copy), §7.13.3.26 (morph
// prediction, out of scope).
//
// This header is included by the CPU reference (`ibc.cpp`) and the CUDA
// kernels (`src/cuda/ibc/ibc.cu`) so the BV scaling, validity arithmetic and
// copy loops are identical on both paths.
//
// Scope: 8-bit 4:2:0, 64x64 superblocks, integer BVs (force_integer_mv=1).
// The local IBC buffer is four 64x64 regions in a ring (§5.20.5.1); the global
// range is the current and previous superblock rows minus INTRABC_DELAY_SB64.
//
// The ±1 interpolation-border check only applies to morph_pred
// (`is_offset_mv_valid`), which is not used here.

#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define AV2_IBC_HD __host__ __device__
#else
#define AV2_IBC_HD
#endif

namespace av2enc::ibc {

inline constexpr int kMiSize = 4;
inline constexpr int kMiSizeLog2 = 2;
inline constexpr int kIbcBufferSize = 64;
inline constexpr int kIbcBufferSizeLog2 = 6;
inline constexpr int kIbcNumBuffers = 4;
inline constexpr int kIntrabcDelayPixels = 256;
inline constexpr int kIntrabcDelaySb64 = 4;

// §7.13.3.17 motion vector scaling for a plane with the given subsampling.
// IBC BVs are in 1/8 luma samples; the decoder builds a 1/1024-sample position
// and rounds with the SCALE_SUBPEL offset.  The current-frame x/y cancel, so
// the result is a pure integer sample offset.
AV2_IBC_HD inline int plane_offset(int bv8, int subsampling) {
  if (subsampling < 0 || subsampling > 3) return 0;
  const long long two = 2LL * static_cast<long long>(bv8);
  long long v = 0;
  if (two >= 0) {
    v = two >> subsampling;
  } else {
    v = -(((-two) + (1LL << subsampling) - 1) >> subsampling);
  }
  const long long num = v * 64 + 32;
  if (num >= 0) return static_cast<int>(num / 1024);
  return static_cast<int>(-((-num + 1023) / 1024));
}

// §5.20.5.1 ibc_buffer_index().
AV2_IBC_HD inline int ibc_buffer_index(int buf_row, int buf_col,
                                       int sb_size) {
  if (sb_size == 64) return buf_col & 3;
  return (buf_col & 1) | ((buf_row & 1) << 1);
}

// §6.19.7.12 is_mv_valid() geometry part.  Returns false with a reason code:
//   1 = source outside the tile
//   2 = local check failed (caller must have evaluated check_valid_local_ibc)
//   3 = global delay window
//   4 = global wavefront constraint
// `local_ok` is the precomputed local result (ignored when local is off).
// `allow_extra_range` is AllowExtraIBCRange (0 for 64x64 SBs).
AV2_IBC_HD inline bool bv_geometry_valid(
    int mi_row, int mi_col, int bw, int bh, int tile_row_start,
    int tile_col_start, int tile_row_end, int tile_col_end, int bv_row,
    int bv_col, bool has_chroma, int chroma_mi_row, int chroma_mi_col,
    bool allow_local, bool allow_global, bool allow_extra_range,
    bool local_ok, int sb_size, int total_sb64_per_row, int plane_end_mi_rows,
    int plane_end_mi_cols, int* reason) {
  const int bottom_border = (bv_row & 7) != 0 ? 1 : 0;
  const int right_border = (bv_col & 7) != 0 ? 1 : 0;
  const int delta_row = bv_row >> 3;
  const int delta_col = bv_col >> 3;
  int src_top = mi_row * kMiSize + delta_row;
  int src_left = mi_col * kMiSize + delta_col;
  const int src_bottom = src_top + bh + bottom_border;
  const int src_right = src_left + bw + right_border;
  if (has_chroma) {
    src_left = chroma_mi_col * kMiSize + delta_col;
    src_top = chroma_mi_row * kMiSize + delta_row;
  }
  if (src_top < tile_row_start * kMiSize ||
      src_left < tile_col_start * kMiSize ||
      src_bottom > tile_row_end * kMiSize ||
      src_right > tile_col_end * kMiSize) {
    if (reason != nullptr) *reason = 1;
    return false;
  }
  // §6.19.7.12: a valid local source returns 1 immediately; the global checks
  // only run when local is disabled or does not cover the source.
  if (allow_local && local_ok) return true;
  if (!allow_global) {
    if (reason != nullptr) *reason = allow_local ? 2 : 3;
    return false;
  }
  const int sb_h = sb_size;
  const int active_sb_row = (mi_row * kMiSize) / sb_h;
  const int active_sb64_col = (mi_col * kMiSize) >> 6;
  const int src_sb_row = (src_bottom - 1) / sb_h;
  const int src_sb64_col = (src_right - 1) >> 6;
  const int active_sb64_row = (mi_row * kMiSize) >> 6;
  const bool is_bottom_left =
      (active_sb64_col & 1) == 0 && (active_sb64_row & 1) == 1;
  const int sb64_residual = (allow_extra_range && is_bottom_left) ? -1 : 0;
  const int active_sb64 =
      active_sb_row * total_sb64_per_row + active_sb64_col;
  const int src_sb64 = src_sb_row * total_sb64_per_row + src_sb64_col;
  if (src_sb64 >= active_sb64 - kIntrabcDelaySb64 - sb64_residual) {
    if (reason != nullptr) *reason = 3;
    return false;
  }
  const int gradient = kIntrabcDelaySb64 + (sb_size / 64);
  const int wf_offset = gradient * (active_sb_row - src_sb_row);
  if (src_sb_row > active_sb_row ||
      src_sb64_col >=
          active_sb64_col - kIntrabcDelaySb64 + wf_offset - sb64_residual) {
    if (reason != nullptr) *reason = 4;
    return false;
  }
  (void)plane_end_mi_rows;
  (void)plane_end_mi_cols;
  return true;
}

// §6.19.7.12 check_valid_local_ibc() for one source sample.  The caller owns
// the four ring-buffer entries, the current buffer position and the IBCCoded
// grid (indexed at MI resolution: y>>2, x>>2).
AV2_IBC_HD inline bool check_valid_local_ibc(
    int x, int y, int act_mi_col, int act_mi_row, const std::uint8_t* coded,
    int coded_stride, int cur_buf_row, int cur_buf_col,
    const int* entry_row, const int* entry_col, const bool* entry_valid,
    int sb_size) {
  if (x >= act_mi_col * kMiSize && y >= act_mi_row * kMiSize) return false;
  if (coded[static_cast<std::size_t>(y >> kMiSizeLog2) *
                static_cast<std::size_t>(coded_stride) +
            static_cast<std::size_t>(x >> kMiSizeLog2)] == 0) {
    return false;
  }
  const int buf_col = x >> kIbcBufferSizeLog2;
  const int buf_row = y >> kIbcBufferSizeLog2;
  const int buf_idx = ibc_buffer_index(buf_row, buf_col, sb_size);
  const bool in_current = buf_col == cur_buf_col && buf_row == cur_buf_row;
  if (!in_current) {
    if (!entry_valid[buf_idx] || buf_col != entry_col[buf_idx] ||
        buf_row != entry_row[buf_idx]) {
      return false;
    }
  }
  if (buf_idx == ibc_buffer_index(cur_buf_row, cur_buf_col, sb_size)) {
    if (!in_current) {
      const int colo_y = (y & (kIbcBufferSize - 1)) |
                         (cur_buf_row << kIbcBufferSizeLog2);
      const int colo_x = (x & (kIbcBufferSize - 1)) |
                         (cur_buf_col << kIbcBufferSizeLog2);
      if (coded[static_cast<std::size_t>(colo_y >> kMiSizeLog2) *
                    static_cast<std::size_t>(coded_stride) +
                static_cast<std::size_t>(colo_x >> kMiSizeLog2)] != 0) {
        return false;
      }
    }
  }
  return true;
}

AV2_IBC_HD inline void copy_plane(const std::uint8_t* src, int src_stride,
                                  std::uint8_t* dst, int dst_stride, int w,
                                  int h) {
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      dst[static_cast<std::size_t>(i) * static_cast<std::size_t>(dst_stride) +
          static_cast<std::size_t>(j)] =
          src[static_cast<std::size_t>(i) * static_cast<std::size_t>(src_stride) +
              static_cast<std::size_t>(j)];
    }
  }
}

AV2_IBC_HD inline std::int64_t sad_block(const std::uint8_t* a, int a_stride,
                                         const std::uint8_t* b, int b_stride,
                                         int w, int h) {
  std::int64_t sad = 0;
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      const int d = static_cast<int>(
                        a[static_cast<std::size_t>(i) *
                              static_cast<std::size_t>(a_stride) +
                          static_cast<std::size_t>(j)]) -
                    static_cast<int>(
                        b[static_cast<std::size_t>(i) *
                              static_cast<std::size_t>(b_stride) +
                          static_cast<std::size_t>(j)]);
      sad += d < 0 ? -d : d;
    }
  }
  return sad;
}

}  // namespace av2enc::ibc
