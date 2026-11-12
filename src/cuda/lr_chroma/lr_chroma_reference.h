// Independent scalar transcription of the chroma loop-restoration pixel path.
//
// This header is used only by tests.  It deliberately does not call
// `lr_chroma_device.cuh`: it is the oracle for the shared device header and
// for the CUDA kernels.  Normative constants still come from
// `encoder/filters/restoration/restoration_common.h` (the same single source
// the CPU reference uses), but every loop, clamp and bank transition below is
// written out independently.
//
// Supports the same domain as the device header: 8/10-bit samples,
// 4:2:0/4:2:2/4:4:4 subsampling, unit sizes 32/64/128/256, tile clamps and
// the 64-line stripe sourcing rule.

#ifndef AV2ENC_CUDA_LR_CHROMA_LR_CHROMA_REFERENCE_H_
#define AV2ENC_CUDA_LR_CHROMA_LR_CHROMA_REFERENCE_H_

#include <cstddef>
#include <cstdint>

#include "encoder/filters/restoration/restoration_common.h"
#include "lr_chroma_types.h"

namespace av2enc::cuda::lr_chroma {
namespace ref {

namespace rest = av2enc::restoration;

struct Block {
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
  int unit_row = 0;
  int unit_col = 0;
  int unit_index = 0;
  int luma_start_x = 0;
  int luma_start_y = 0;
  int luma_end_x = 0;
  int luma_end_y = 0;
  int stripe_start_y = 0;
  int stripe_end_y = 0;
};

inline int count_units(int unit_size, int frame_size) {
  if (unit_size <= 0) return 1;
  const int n = (frame_size + (unit_size >> 1)) / unit_size;
  return n > 1 ? n : 1;
}

inline int chroma_width(const LrChromaConfig& c) {
  return (c.mi_cols * 4) >> c.sub_x;
}

inline int chroma_height(const LrChromaConfig& c) {
  return (c.mi_rows * 4) >> c.sub_y;
}

// §7.20.1 context, transcribed literally.  `unit_size` is the chroma unit size
// in plane samples.
inline Block block_context(const LrChromaFrame& f, int unit_size, int chroma,
                           int row, int col) {
  Block out;
  out.x = (col * 4) >> f.sub_x;
  out.y = (row * 4) >> f.sub_y;
  out.w = 4 >> f.sub_x;
  out.h = 4 >> f.sub_y;
  (void)chroma;

  int mi_col_start = 0;
  int mi_row_start = 0;
  int mi_col_end = f.mi_cols;
  int mi_row_end = f.mi_rows;
  if (f.disable_loopfilters_across_tiles != 0 && f.mi_col_start != nullptr &&
      f.mi_col_end != nullptr && f.mi_row_start != nullptr &&
      f.mi_row_end != nullptr) {
    const int idx = row * f.mi_cols + col;
    mi_col_start = f.mi_col_start[idx];
    mi_col_end = f.mi_col_end[idx];
    mi_row_start = f.mi_row_start[idx];
    mi_row_end = f.mi_row_end[idx];
  }

  const int tile_rows = mi_row_end - mi_row_start;
  const int tile_cols = mi_col_end - mi_col_start;
  const int local_unit_rows =
      count_units(unit_size, (tile_rows * 4) >> f.sub_y);
  const int local_unit_cols =
      count_units(unit_size, (tile_cols * 4) >> f.sub_x);
  const int rel_row = row - mi_row_start;
  const int rel_col = col - mi_col_start;
  const int row_offset = ((mi_row_start * 4) >> f.sub_y) / unit_size;
  const int col_offset = ((mi_col_start * 4) >> f.sub_x) / unit_size;
  const int local_row_y = ((rel_row * 4) + 8) >> f.sub_y;
  const int local_col_x = (rel_col * 4) >> f.sub_x;
  int lr = local_row_y / unit_size;
  int lc = local_col_x / unit_size;
  if (lr > local_unit_rows - 1) lr = local_unit_rows - 1;
  if (lc > local_unit_cols - 1) lc = local_unit_cols - 1;
  out.unit_row = row_offset + lr;
  out.unit_col = col_offset + lc;
  const int global_cols = count_units(unit_size, (f.mi_cols * 4) >> f.sub_x);
  out.unit_index = out.unit_row * global_cols + out.unit_col;

  out.luma_start_x = 0;
  out.luma_start_y = 0;
  out.luma_end_x = f.mi_cols * 4 - 1;
  out.luma_end_y = f.mi_rows * 4 - 1;
  if (f.disable_loopfilters_across_tiles != 0) {
    out.luma_start_x = mi_col_start * 4;
    out.luma_end_x = mi_col_end * 4 - 1;
    out.luma_start_y = mi_row_start * 4;
    out.luma_end_y = mi_row_end * 4 - 1;
  }

  const int stripe_num = (rel_row * 4 + 8) / 64;
  const int raw_start = mi_row_start * 4 + stripe_num * 64 - 8;
  const int raw_end = raw_start + 64;
  out.stripe_start_y =
      out.luma_start_y > raw_start ? out.luma_start_y : raw_start;
  out.stripe_end_y =
      out.luma_end_y < raw_end - 1 ? out.luma_end_y : raw_end - 1;
  return out;
}

template <typename Sample>
inline int read(const void* plane, int stride, int x, int y) {
  const Sample* p = static_cast<const Sample*>(plane);
  return static_cast<int>(
      p[static_cast<std::ptrdiff_t>(y) * stride + static_cast<std::ptrdiff_t>(x)]);
}

template <typename Sample>
inline void write(void* plane, int stride, int x, int y, int value) {
  Sample* p = static_cast<Sample*>(plane);
  p[static_cast<std::ptrdiff_t>(y) * stride + static_cast<std::ptrdiff_t>(x)] =
      static_cast<Sample>(value);
}

template <typename Sample>
inline int source_sample(const LrChromaFrame& f, const Block& b, int chroma,
                         int x, int y) {
  x = rest::clip3(b.luma_start_x >> f.sub_x, b.luma_end_x >> f.sub_x, x);
  y = rest::clip3(b.luma_start_y >> f.sub_y, b.luma_end_y >> f.sub_y, y);
  const int stripe_start = b.stripe_start_y >> f.sub_y;
  const int stripe_end = b.stripe_end_y >> f.sub_y;
  const void* plane = nullptr;
  int stride = 0;
  if (y < stripe_start) {
    if (stripe_start - 2 > y) y = stripe_start - 2;
    plane = f.chroma[chroma].curr;
    stride = f.chroma[chroma].curr_stride;
  } else if (y > stripe_end) {
    if (stripe_end + 2 < y) y = stripe_end + 2;
    plane = f.chroma[chroma].curr;
    stride = f.chroma[chroma].curr_stride;
  } else {
    plane = f.chroma[chroma].cdef;
    stride = f.chroma[chroma].cdef_stride;
  }
  return read<Sample>(plane, stride, x, y);
}

template <typename Sample>
inline int luma_source_sample(const LrChromaFrame& f, const Block& b, int x,
                              int y) {
  x = rest::clip3(b.luma_start_x, b.luma_end_x, x);
  y = rest::clip3(b.luma_start_y, b.luma_end_y, y);
  const void* plane = nullptr;
  int stride = 0;
  if (y < b.stripe_start_y) {
    if (b.stripe_start_y - 2 > y) y = b.stripe_start_y - 2;
    plane = f.luma.curr;
    stride = f.luma.curr_stride;
  } else if (y > b.stripe_end_y) {
    if (b.stripe_end_y + 2 < y) y = b.stripe_end_y + 2;
    plane = f.luma.curr;
    stride = f.luma.curr_stride;
  } else {
    plane = f.luma.cdef;
    stride = f.luma.cdef_stride;
  }
  return read<Sample>(plane, stride, x, y);
}

template <typename Sample>
inline int luma_sample(const LrChromaFrame& f, const Block& b, int x, int y) {
  const int last_y = f.mi_rows * 4 - 1 - f.sub_y;
  const int last_x = b.luma_end_x - f.sub_x;
  x <<= f.sub_x;
  y <<= f.sub_y;
  y = rest::clip3(0, last_y, y);
  x = rest::clip3(b.luma_start_x, last_x, x);
  int filter_idx = f.cfl_ds_filter_index;
  if (filter_idx == 3) filter_idx = 0;
  if (f.sub_x != 0 && f.sub_y != 0 && filter_idx >= 0 && filter_idx <= 1) {
    int t = 0;
    for (int dy = 0; dy < 2; ++dy) {
      for (int dx = 0; dx < 2; ++dx) {
        t += static_cast<int>(rest::kWienerFilters420[filter_idx][dy][dx]) *
             luma_source_sample<Sample>(f, b, x + dx, y + dy);
      }
    }
    return t >> 2;
  }
  return luma_source_sample<Sample>(f, b, x, y);
}

template <typename Sample>
inline void filter_block(const LrChromaFrame& f, const Block& b, int chroma,
                         const std::int16_t* coeffs) {
  const int max_value = (1 << f.bit_depth) - 1;
  for (int r = 0; r < b.h; ++r) {
    const int sy = b.y + r;
    for (int c = 0; c < b.w; ++c) {
      const int sx = b.x + c;
      const int m = source_sample<Sample>(f, b, chroma, sx, sy);
      int s = m << 7;
      for (int i = 0; i < 12; ++i) {
        const int dy = rest::kWienerNsConfigUv[i][0];
        const int dx = rest::kWienerNsConfigUv[i][1];
        const int idx = rest::kWienerNsConfigUv[i][2];
        s += (source_sample<Sample>(f, b, chroma, sx + dx, sy + dy) - m) *
             coeffs[idx];
      }
      const int m_luma = luma_sample<Sample>(f, b, sx, sy);
      for (int i = 0; i < 12; ++i) {
        const int coeff = coeffs[6 + i];
        if (coeff != 0) {
          const int dy = rest::kWienerNsConfigUv[i][0];
          const int dx = rest::kWienerNsConfigUv[i][1];
          s += (luma_sample<Sample>(f, b, sx + dx, sy + dy) - m_luma) * coeff;
        }
      }
      const int v = rest::round2(s, 7);
      write<Sample>(f.chroma[chroma].dst, f.chroma[chroma].dst_stride, sx, sy,
                    rest::clip3(0, max_value, v));
    }
  }
}

template <typename Sample>
inline void apply_frame_t(const LrChromaConfig& c, const LrChromaFrame& f,
                          const std::int16_t* const* resolved,
                          const std::uint8_t* const* active) {
  const int width = chroma_width(c);
  const int height = chroma_height(c);
  const int global_cols = count_units(c.unit_size, width);
  const int global_rows = count_units(c.unit_size, height);
  for (int chroma = 0; chroma < 2; ++chroma) {
    // §7.20 copy step for the whole plane.
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const int v = read<Sample>(f.chroma[chroma].cdef,
                                   f.chroma[chroma].cdef_stride, x, y);
        write<Sample>(f.chroma[chroma].dst, f.chroma[chroma].dst_stride, x, y,
                      v);
      }
    }
    for (int row = 0; row < c.mi_rows; ++row) {
      for (int col = 0; col < c.mi_cols; ++col) {
        const Block b = block_context(f, c.unit_size, chroma, row, col);
        // A last partial tile can be narrower than the restoration unit; the
        // CPU core then computes a frame-global unit index past the grid and
        // skips filtering (the sample stays a CdefFrame copy).
        if (b.unit_row < 0 || b.unit_row >= global_rows || b.unit_col < 0 ||
            b.unit_col >= global_cols) {
          continue;
        }
        if (active[chroma][b.unit_index] == 0) continue;
        filter_block<Sample>(
            f, b, chroma,
            resolved[chroma] + b.unit_index * kCoeffStride);
      }
    }
  }
}

inline void apply_frame(const LrChromaConfig& c, const LrChromaFrame& f,
                        const std::int16_t* const* resolved,
                        const std::uint8_t* const* active) {
  if (c.bit_depth == 8) {
    apply_frame_t<std::uint8_t>(c, f, resolved, active);
  } else {
    apply_frame_t<std::uint16_t>(c, f, resolved, active);
  }
}

// ---------------------------------------------------------------------------
// §5.20.10.6 expansion, independent transcription
// ---------------------------------------------------------------------------
inline void resolve_plane(const LrChromaUnit* units, int count,
                          std::int16_t* out, std::uint8_t* active) {
  std::int16_t bank[4][18];
  for (int s = 0; s < 4; ++s) {
    for (int j = 0; j < 18; ++j) bank[s][j] = 0;
  }
  int size = 0;
  int ptr = 0;
  for (int i = 0; i < count; ++i) {
    const LrChromaUnit& u = units[i];
    active[i] = static_cast<std::uint8_t>(u.on != 0);
    for (int j = 0; j < 18; ++j) out[i * 18 + j] = 0;
    if (u.on == 0) continue;
    if (u.merged != 0) {
      if (size == 0) size = 1;
      int k = (u.bank_k < 0) ? 0 : u.bank_k;
      if (k >= size) k = size - 1;
      int slot = (ptr - k) % 4;
      if (slot < 0) slot += 4;
      for (int j = 0; j < 18; ++j) out[i * 18 + j] = bank[slot][j];
      continue;
    }
    const int present =
        (u.subset <= 0) ? 6 : ((u.subset == 1) ? 10 : 18);
    const int new_ptr = (size < 4) ? size : ((ptr + 1) % 4);
    for (int j = 0; j < 18; ++j) {
      const int v = (j < present) ? u.coeffs[j] : 0;
      out[i * 18 + j] = static_cast<std::int16_t>(v);
      bank[new_ptr][j] = static_cast<std::int16_t>(v);
      if (u.subset > 0 && u.uv_sym != 0 && j >= 6 && j + 1 < 18) {
        out[i * 18 + j + 1] = static_cast<std::int16_t>(v);
        bank[new_ptr][j + 1] = static_cast<std::int16_t>(v);
        ++j;
      }
    }
    ptr = new_ptr;
    if (size < 4) ++size;
  }
}

}  // namespace ref
}  // namespace av2enc::cuda::lr_chroma

#endif  // AV2ENC_CUDA_LR_CHROMA_LR_CHROMA_REFERENCE_H_
