// Shared AV2 chroma loop-restoration arithmetic — host and device.
//
// This header is compiled three ways:
//   * by nvcc inside the kernels (`lr_chroma.cu`),
//   * by nvcc in the real parity test (`lr_chroma_test.cu`, host side),
//   * by the host compiler for the emulation test (no CUDA keywords needed:
//     this header does not include cuda_runtime.h).
//
// It is the single transcription of:
//   * §7.20.1 loop_restore_block_process context for a chroma plane
//     (unit index, stripe bounds, tile clamp, 8-bit/10-bit and 4:2:0/4:2:2/
//     4:4:4 geometry),
//   * §7.20.2 get_source_sample() (CurrFrame outside a 64-line stripe,
//     CdefFrame inside, two-line vertical clamp),
//   * §7.20.3 get_luma_sample() (subsampled luma, cfl_ds_filter_index),
//   * §7.20.3 non-separable Wiener with the 18-entry chroma coefficient
//     layout (12 chroma tap positions + 12 chroma-from-luma positions),
//   * §5.20.10.6 coefficient expansion: subset zeroing, uv_sym pair
//     mirroring, and bank/merged selection.
//
// All functions are `__host__ __device__`.  The header defines no mutable
// objects; the only tables it reads come from
// `encoder/filters/restoration/restoration_common.h`, whose
// `AV2ENC_LR_TABLE` macro gives them device storage in the nvcc pass.

#ifndef AV2ENC_CUDA_LR_CHROMA_LR_CHROMA_DEVICE_CUH_
#define AV2ENC_CUDA_LR_CHROMA_LR_CHROMA_DEVICE_CUH_

#include <cstddef>
#include <cstdint>

#include "codec/filters/loop_restoration/restoration_common.h"
#include "lr_chroma_types.h"

#if defined(__CUDACC__)
#define AV2ENC_LRC_HD __host__ __device__ __forceinline__
#else
#define AV2ENC_LRC_HD inline
#endif

namespace av2enc::cuda::lr_chroma {
namespace rest = av2enc::restoration;

// ---------------------------------------------------------------------------
// Geometry (§5.20.10.4 count_units_in_frame / §7.20.1 plane extents)
// ---------------------------------------------------------------------------

AV2ENC_LRC_HD int lrc_count_units(int unit_size, int frame_size) {
  if (unit_size <= 0) {
    return 1;
  }
  const int n = (frame_size + (unit_size >> 1)) / unit_size;
  return n > 1 ? n : 1;
}

AV2ENC_LRC_HD int lrc_plane_width(int mi_cols, int sub_x) {
  return (mi_cols * kMiSize) >> sub_x;
}

AV2ENC_LRC_HD int lrc_plane_height(int mi_rows, int sub_y) {
  return (mi_rows * kMiSize) >> sub_y;
}

// ---------------------------------------------------------------------------
// Sample access.  `Sample` is std::uint8_t (8-bit) or std::uint16_t (10-bit).
// ---------------------------------------------------------------------------

template <typename Sample>
AV2ENC_LRC_HD int lrc_read_sample(const void* plane, int stride, int x, int y) {
  const Sample* p = static_cast<const Sample*>(plane);
  return static_cast<int>(
      p[static_cast<std::ptrdiff_t>(y) * stride + static_cast<std::ptrdiff_t>(x)]);
}

template <typename Sample>
AV2ENC_LRC_HD void lrc_write_sample(void* plane, int stride, int x, int y,
                                    int value) {
  Sample* p = static_cast<Sample*>(plane);
  p[static_cast<std::ptrdiff_t>(y) * stride + static_cast<std::ptrdiff_t>(x)] =
      static_cast<Sample>(value);
}

// ---------------------------------------------------------------------------
// §7.20.1 block context (chroma planes only)
// ---------------------------------------------------------------------------
//
// Mirrors `restoration::lr_block_context()` but parameterised by bit depth and
// subsampling and with the chroma unit type supplied through `active` by the
// caller.  Returns false when the block must not be filtered (empty tile
// extent).  The output rectangle is always filled, because the caller still
// has to copy CdefFrame into LrFrame for it.
AV2ENC_LRC_HD bool lrc_block_context(const LrChromaCore& f, int chroma, int row,
                                     int col, LrChromaBlock* out) {
  if (out == nullptr || chroma < 0 || chroma > 1) {
    return false;
  }
  const int sub_x = f.sub_x;
  const int sub_y = f.sub_y;
  const int unit_size = f.unit_size[chroma];
  *out = LrChromaBlock{};
  out->x = (col * kMiSize) >> sub_x;
  out->y = (row * kMiSize) >> sub_y;
  out->w = kMiSize >> sub_x;
  out->h = kMiSize >> sub_y;
  out->unit_row = 0;
  out->unit_col = 0;
  out->unit_index = 0;

  if (unit_size <= 0) {
    return false;
  }

  int mi_col_start = 0;
  int mi_row_start = 0;
  int mi_col_end = f.mi_cols;
  int mi_row_end = f.mi_rows;
  if (f.disable_loopfilters_across_tiles != 0 && f.mi_col_start != nullptr &&
      f.mi_col_end != nullptr && f.mi_row_start != nullptr &&
      f.mi_row_end != nullptr) {
    const int idx = row * f.mi_cols + col;
    mi_col_start = static_cast<int>(f.mi_col_start[idx]);
    mi_col_end = static_cast<int>(f.mi_col_end[idx]);
    mi_row_start = static_cast<int>(f.mi_row_start[idx]);
    mi_row_end = static_cast<int>(f.mi_row_end[idx]);
  }
  if (mi_col_end <= mi_col_start || mi_row_end <= mi_row_start) {
    return false;
  }

  const int rel_row = row - mi_row_start;
  const int rel_col = col - mi_col_start;
  const int tile_mi_rows = mi_row_end - mi_row_start;
  const int tile_mi_cols = mi_col_end - mi_col_start;
  const int lr_row_offset =
      ((mi_row_start * kMiSize) >> sub_y) / unit_size;
  const int lr_col_offset =
      ((mi_col_start * kMiSize) >> sub_x) / unit_size;

  const int unit_rows =
      lrc_count_units(unit_size, (tile_mi_rows * kMiSize) >> sub_y);
  const int unit_cols =
      lrc_count_units(unit_size, (tile_mi_cols * kMiSize) >> sub_x);

  const int unit_row_local = ((rel_row * kMiSize) + 8) >> sub_y;
  const int unit_col_local = (rel_col * kMiSize) >> sub_x;
  const int local_row =
      (unit_row_local / unit_size) < (unit_rows - 1)
          ? (unit_row_local / unit_size)
          : (unit_rows - 1);
  const int local_col =
      (unit_col_local / unit_size) < (unit_cols - 1)
          ? (unit_col_local / unit_size)
          : (unit_cols - 1);
  const int unit_row = lr_row_offset + local_row;
  const int unit_col = lr_col_offset + local_col;
  if (unit_row < 0 || unit_row >= f.unit_rows[chroma] || unit_col < 0 ||
      unit_col >= f.unit_cols[chroma]) {
    return false;
  }
  out->unit_row = unit_row;
  out->unit_col = unit_col;
  out->unit_index = unit_row * f.unit_cols[chroma] + unit_col;

  // Allowed filtering extent (§7.20.1 start/end).
  int luma_start_x = 0;
  int luma_start_y = 0;
  int luma_end_x = f.mi_cols * kMiSize - 1;
  int luma_end_y = f.mi_rows * kMiSize - 1;
  if (f.disable_loopfilters_across_tiles != 0) {
    luma_start_x = mi_col_start * kMiSize;
    luma_end_x = mi_col_end * kMiSize - 1;
    luma_start_y = mi_row_start * kMiSize;
    luma_end_y = mi_row_end * kMiSize - 1;
  }
  out->luma_start_x = luma_start_x;
  out->luma_start_y = luma_start_y;
  out->luma_end_x = luma_end_x;
  out->luma_end_y = luma_end_y;

  const int luma_y = rel_row * kMiSize;
  const int stripe_num = (luma_y + 8) / 64;
  const int unclipped_stripe_start =
      mi_row_start * kMiSize + stripe_num * 64 - 8;
  const int unclipped_stripe_end = unclipped_stripe_start + 64;
  out->stripe_start_y = luma_start_y > unclipped_stripe_start
                            ? luma_start_y
                            : unclipped_stripe_start;
  out->stripe_end_y = luma_end_y < (unclipped_stripe_end - 1)
                          ? luma_end_y
                          : (unclipped_stripe_end - 1);
  return true;
}

// ---------------------------------------------------------------------------
// §7.20.2 get_source_sample()
// ---------------------------------------------------------------------------

template <typename Sample>
AV2ENC_LRC_HD int lrc_source_sample(const LrChromaCore& f,
                                    const LrChromaBlock& b, int chroma, int x,
                                    int y) {
  const int sub_x = f.sub_x;
  const int sub_y = f.sub_y;
  const LrChromaPlane& p = f.chroma[chroma];

  x = rest::clip3(b.luma_start_x >> sub_x, b.luma_end_x >> sub_x, x);
  y = rest::clip3(b.luma_start_y >> sub_y, b.luma_end_y >> sub_y, y);

  const int stripe_start = b.stripe_start_y >> sub_y;
  const int stripe_end = b.stripe_end_y >> sub_y;
  if (y < stripe_start) {
    const int yy = (stripe_start - 2 > y) ? (stripe_start - 2) : y;
    return lrc_read_sample<Sample>(p.curr, p.curr_stride, x, yy);
  }
  if (y > stripe_end) {
    const int yy = (stripe_end + 2 < y) ? (stripe_end + 2) : y;
    return lrc_read_sample<Sample>(p.curr, p.curr_stride, x, yy);
  }
  return lrc_read_sample<Sample>(p.cdef, p.cdef_stride, x, y);
}

// Luma-plane variant; `b` is the chroma block (its luma extents/stripe are in
// luma samples already).
template <typename Sample>
AV2ENC_LRC_HD int lrc_luma_source_sample(const LrChromaCore& f,
                                         const LrChromaBlock& b, int x, int y) {
  const LrChromaPlane& p = f.luma;
  x = rest::clip3(b.luma_start_x, b.luma_end_x, x);
  y = rest::clip3(b.luma_start_y, b.luma_end_y, y);
  if (y < b.stripe_start_y) {
    const int yy = (b.stripe_start_y - 2 > y) ? (b.stripe_start_y - 2) : y;
    return lrc_read_sample<Sample>(p.curr, p.curr_stride, x, yy);
  }
  if (y > b.stripe_end_y) {
    const int yy = (b.stripe_end_y + 2 < y) ? (b.stripe_end_y + 2) : y;
    return lrc_read_sample<Sample>(p.curr, p.curr_stride, x, yy);
  }
  return lrc_read_sample<Sample>(p.cdef, p.cdef_stride, x, y);
}

// §7.20.3 get_luma_sample(): luma fetch at a chroma position, with the
// 4:2:0 downsampling filter selected by cfl_ds_filter_index (3 -> 0).
template <typename Sample>
AV2ENC_LRC_HD int lrc_luma_sample(const LrChromaCore& f, const LrChromaBlock& b,
                                  int x, int y) {
  const int sub_x = f.sub_x;
  const int sub_y = f.sub_y;
  const int last_y = f.mi_rows * kMiSize - 1 - sub_y;
  const int last_x = b.luma_end_x - sub_x;
  x <<= sub_x;
  y <<= sub_y;
  y = rest::clip3(0, last_y, y);
  x = rest::clip3(b.luma_start_x, last_x, x);
  int filter_idx = f.cfl_ds_filter_index;
  if (filter_idx == 3) {
    filter_idx = 0;
  }
  if (sub_x != 0 && sub_y != 0 && filter_idx >= 0 && filter_idx <= 1) {
    int t = 0;
    for (int dy = 0; dy < 2; ++dy) {
      for (int dx = 0; dx < 2; ++dx) {
        t += static_cast<int>(rest::kWienerFilters420[filter_idx][dy][dx]) *
             lrc_luma_source_sample<Sample>(f, b, x + dx, y + dy);
      }
    }
    return t >> 2;
  }
  return lrc_luma_source_sample<Sample>(f, b, x, y);
}

// ---------------------------------------------------------------------------
// Copy + filter
// ---------------------------------------------------------------------------

template <typename Sample>
AV2ENC_LRC_HD void lrc_copy_block(const LrChromaCore& f, int chroma, int x,
                                  int y, int w, int h) {
  const LrChromaPlane& p = f.chroma[chroma];
  for (int r = 0; r < h; ++r) {
    const int sy = y + r;
    for (int c = 0; c < w; ++c) {
      const int v =
          lrc_read_sample<Sample>(p.cdef, p.cdef_stride, x + c, sy);
      lrc_write_sample<Sample>(p.dst, p.dst_stride, x + c, sy, v);
    }
  }
}

template <typename Sample>
AV2ENC_LRC_HD void lrc_filter_block(const LrChromaCore& f,
                                    const LrChromaBlock& b, int chroma,
                                    const std::int16_t* coeffs) {
  const int n_taps = kUvTaps;
  const int max_value = (1 << f.bit_depth) - 1;
  for (int r = 0; r < b.h; ++r) {
    const int sy = b.y + r;
    for (int c = 0; c < b.w; ++c) {
      const int sx = b.x + c;
      const int m = lrc_source_sample<Sample>(f, b, chroma, sx, sy);
      int s = m << kPrecBits;
      for (int i = 0; i < n_taps; ++i) {
        const int dy = rest::kWienerNsConfigUv[i][0];
        const int dx = rest::kWienerNsConfigUv[i][1];
        const int idx = rest::kWienerNsConfigUv[i][2];
        const int diff =
            lrc_source_sample<Sample>(f, b, chroma, sx + dx, sy + dy) - m;
        s += diff * static_cast<int>(coeffs[idx]);
      }
      const int m_luma = lrc_luma_sample<Sample>(f, b, sx, sy);
      for (int i = 0; i < n_taps; ++i) {
        const int coeff = static_cast<int>(coeffs[kShortCoeffs + i]);
        if (coeff != 0) {
          const int dy = rest::kWienerNsConfigUv[i][0];
          const int dx = rest::kWienerNsConfigUv[i][1];
          const int luma_diff =
              lrc_luma_sample<Sample>(f, b, sx + dx, sy + dy) - m_luma;
          s += luma_diff * coeff;
        }
      }
      const int v = rest::round2(s, kPrecBits);
      lrc_write_sample<Sample>(f.chroma[chroma].dst,
                               f.chroma[chroma].dst_stride, sx, sy,
                               rest::clip3(0, max_value, v));
    }
  }
}

// ---------------------------------------------------------------------------
// One MI block: copy CdefFrame -> LrFrame, then filter when active.
// ---------------------------------------------------------------------------
//
// Every MI block owns a disjoint rectangle, so the per-block copy reproduces
// the whole-plane §7.20 copy step without a second launch.  Blocks whose
// context is invalid (outside a tile) are still copied; only filtering is
// skipped.
template <typename Sample>
AV2ENC_LRC_HD void lrc_process_block(const LrChromaCore& f, int chroma,
                                     int row, int col,
                                     const std::int16_t* resolved,
                                     const std::uint8_t* active) {
  LrChromaBlock b;
  const bool context = lrc_block_context(f, chroma, row, col, &b);
  lrc_copy_block<Sample>(f, chroma, b.x, b.y, b.w, b.h);
  if (!context || resolved == nullptr || active == nullptr) {
    return;
  }
  if (active[b.unit_index] == 0) {
    return;
  }
  lrc_filter_block<Sample>(f, b, chroma,
                           resolved + b.unit_index * kCoeffStride);
}

// Whole-plane reference walk (host emulation and one-thread tests).
template <typename Sample>
AV2ENC_LRC_HD void lrc_process_frame(const LrChromaCore& f,
                                     const std::int16_t* const* resolved,
                                     const std::uint8_t* const* active) {
  for (int chroma = 0; chroma < 2; ++chroma) {
    for (int row = 0; row < f.mi_rows; ++row) {
      for (int col = 0; col < f.mi_cols; ++col) {
        lrc_process_block<Sample>(f, chroma, row, col, resolved[chroma],
                                  active[chroma]);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// §5.20.10.6 coefficient expansion
// ---------------------------------------------------------------------------

AV2ENC_LRC_HD int lrc_present_coeffs(int subset) {
  switch (subset) {
    case 0:
      return 6;
    case 1:
      return 10;
    case 2:
      return kCoeffStride;
    default:
      return 0;
  }
}

// Expands a unit stream in coding order.  `out` receives count*18 int16
// coefficients and `active` one byte per unit.  Mirrors the decoder:
//   * off units write zero coefficients and do not touch the bank;
//   * merged units copy the whole bank entry (including taps outside subset);
//   * non-merged units zero absent taps and mirror j/j+1 pairs for j >= 6 when
//     uv_sym is set.
AV2ENC_LRC_HD void lrc_resolve_plane(const LrChromaUnit* units, int count,
                                     std::int16_t* out, std::uint8_t* active,
                                     LrChromaBankState* bank) {
  if (units == nullptr || out == nullptr || active == nullptr ||
      bank == nullptr || count < 0) {
    return;
  }
  for (int s = 0; s < kBankSize; ++s) {
    for (int j = 0; j < kCoeffStride; ++j) {
      bank->bank[s][j] = 0;
    }
  }
  std::int32_t size = 0;
  std::int32_t ptr = 0;
  for (int i = 0; i < count; ++i) {
    const LrChromaUnit& u = units[i];
    std::int16_t* dst = out + i * kCoeffStride;
    active[i] = (u.on != 0) ? 1 : 0;
    if (u.on == 0) {
      for (int j = 0; j < kCoeffStride; ++j) {
        dst[j] = 0;
      }
      continue;
    }
    if (u.merged != 0) {
      if (size == 0) {
        // Invalid stream (merge before any coded filter).  The decoder leaves
        // the zero-initialised bank entry; stay in bounds and do the same.
        size = 1;
      }
      std::int32_t k = u.bank_k;
      if (k < 0) {
        k = 0;  // -1 selects the newest entry
      }
      if (k >= size) {
        k = size - 1;
      }
      const std::int32_t slot =
          ((ptr - k) % kBankSize + kBankSize) % kBankSize;
      for (int j = 0; j < kCoeffStride; ++j) {
        dst[j] = bank->bank[slot][j];
      }
      continue;
    }

    const std::int32_t new_ptr =
        (size < kBankSize) ? size : ((ptr + 1) % kBankSize);
    const std::int32_t new_size = (size < kBankSize) ? (size + 1) : kBankSize;
    const int present = lrc_present_coeffs(u.subset);
    for (int j = 0; j < kCoeffStride; ++j) {
      const int v = (j < present) ? static_cast<int>(u.coeffs[j]) : 0;
      dst[j] = static_cast<std::int16_t>(v);
      bank->bank[new_ptr][j] = static_cast<std::int16_t>(v);
      if (u.subset > 0 && u.uv_sym != 0 && j >= kShortCoeffs &&
          j + 1 < kCoeffStride) {
        dst[j + 1] = static_cast<std::int16_t>(v);
        bank->bank[new_ptr][j + 1] = static_cast<std::int16_t>(v);
        ++j;
      }
    }
    size = new_size;
    ptr = new_ptr;
  }
  bank->size = size;
  bank->ptr = ptr;
}

}  // namespace av2enc::cuda::lr_chroma

#endif  // AV2ENC_CUDA_LR_CHROMA_LR_CHROMA_DEVICE_CUH_
