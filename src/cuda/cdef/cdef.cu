// AV2 §7.18 CDEF — CUDA kernel.
//
// One CUDA block per 8x8 luma CDEF block (64 threads).  Thread 0 runs the
// direction search and the strength/skip derivation once; every thread then
// copies its sample and, when the block is filtered, evaluates one filtered
// sample.  All arithmetic comes from the shared core header, so the device
// output is bit-exact with the CPU reference.
//
// CDEF is a pure function of the unfiltered source frame and blocks write
// disjoint destination regions, so no inter-block synchronisation is needed.

#include "cuda/cdef/cdef.h"

#include <cstddef>
#include <cstdint>

#include "encoder/filters/cdef/cdef_core.h"

namespace av2enc::cuda {
namespace {

using av2enc::cdef::CdefOptions;
using av2enc::cdef::CdefRegion;

__device__ int cdef_idx_cols(const CdefDeviceFrame& f) {
  return f.n64_cols > 0 ? f.n64_cols
                        : ((f.mi_cols + av2enc::cdef::kCdefSize4 - 1) /
                           av2enc::cdef::kCdefSize4);
}

// §5.20.9.3 filter region for the block at (r, c) in luma MI units.
__device__ void cdef_tile_region(const CdefDeviceFrame& f,
                                 const CdefOptions& options, int r, int c,
                                 CdefRegion* region) {
  if (!options.disable_loopfilters_across_tiles) {
    region->col_start = 0;
    region->row_start = 0;
    region->col_end = f.mi_cols;
    region->row_end = f.mi_rows;
    return;
  }
  int col_start = 0;
  int col_end = f.mi_cols;
  for (int i = 0; i < f.num_tile_cols; ++i) {
    const int start = static_cast<int>(f.tile_col_starts[i]);
    if (start <= c) {
      col_start = start;
    } else {
      col_end = start;
      break;
    }
  }
  int row_start = 0;
  int row_end = f.mi_rows;
  for (int i = 0; i < f.num_tile_rows; ++i) {
    const int start = static_cast<int>(f.tile_row_starts[i]);
    if (start <= r) {
      row_start = start;
    } else {
      row_end = start;
      break;
    }
  }
  region->col_start = col_start;
  region->row_start = row_start;
  region->col_end = col_end;
  region->row_end = row_end;
}

__global__ void cdef_kernel(CdefDeviceFrame f, CdefOptions options) {
  const int nblk_cols = f.mi_cols >> 1;
  const int blk = static_cast<int>(blockIdx.x);
  const int r = (blk / nblk_cols) << 1;
  const int c = (blk % nblk_cols) << 1;

  __shared__ int s_filtered;
  __shared__ int s_skip_chroma;
  __shared__ int s_pri[2];
  __shared__ int s_sec[2];
  __shared__ int s_dir[2];
  __shared__ int s_damping[2];
  __shared__ CdefRegion s_region;

  if (threadIdx.x == 0) {
    const int n64c = cdef_idx_cols(f);
    const int base_r = (r >> 4) << 4;
    const int base_c = (c >> 4) << 4;
    const int idx = static_cast<int>(
        f.cdef_idx[(base_r >> 4) * n64c + (base_c >> 4)]);

    cdef_tile_region(f, options, r, c, &s_region);

    bool skip = false;
    if (!options.on_skip_txfm) {
      const int o = r * f.mi_cols + c;
      skip = f.skip[o] != 0 && f.skip[o + 1] != 0 &&
             f.skip[o + f.mi_cols] != 0 && f.skip[o + f.mi_cols + 1] != 0;
    }
    bool skip_chroma = false;
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        const int o = (r + i) * f.mi_cols + (c + j);
        if (f.lossless[o] != 0) {
          skip = true;
        }
        if (f.lossless_chroma != nullptr && f.lossless_chroma[o] != 0) {
          skip_chroma = true;
        }
      }
    }
    s_skip_chroma = skip_chroma ? 1 : 0;
    s_filtered = 0;
    if (idx >= 0 && !skip) {
      int y_dir = 0;
      std::int64_t var = 0;
      av2enc::cdef::cdef_direction(
          f.src[0], static_cast<std::size_t>(f.stride[0]),
          c * av2enc::cdef::kMiSize, r * av2enc::cdef::kMiSize,
          options.bit_depth, &y_dir, &var);

      int var_str = 0;
      const std::int64_t shifted = var >> 6;
      if (shifted > 0) {
        var_str = av2enc::cdef::cdef_floor_log2(static_cast<int>(shifted));
        if (var_str > 12) {
          var_str = 12;
        }
      }

      const int y_pri_raw = options.strengths[idx].y_pri;
      const int y_sec = options.strengths[idx].y_sec;
      s_dir[0] = (y_pri_raw == 0) ? 0 : y_dir;
      s_pri[0] = (var != 0) ? (y_pri_raw * (4 + var_str) + 8) >> 4 : 0;
      s_sec[0] = y_sec;
      s_damping[0] = options.damping;

      const int uv_pri = options.strengths[idx].uv_pri;
      s_pri[1] = uv_pri;
      s_sec[1] = options.strengths[idx].uv_sec;
      s_dir[1] = (uv_pri == 0)
                     ? 0
                     : av2enc::cdef::kCdefUvDir[f.subsampling_x]
                                                [f.subsampling_y][y_dir];
      s_damping[1] = options.damping - 1;
      s_filtered = 1;
    }
  }
  __syncthreads();

  const int t = static_cast<int>(threadIdx.x);
  const int y0 = r * av2enc::cdef::kMiSize;
  const int x0 = c * av2enc::cdef::kMiSize;

  // §7.18.1 copy step runs for every block, filtered or not.
  {
    const int i = t >> 3;
    const int j = t & 7;
    const std::size_t o = static_cast<std::size_t>(y0 + i) *
                              static_cast<std::size_t>(f.stride[0]) +
                          static_cast<std::size_t>(x0 + j);
    f.dst[0][o] = f.src[0][o];
  }
  if (t < 16) {
    const int i = t >> 2;
    const int j = t & 3;
    for (int plane = 1; plane < 3; ++plane) {
      const std::size_t o = static_cast<std::size_t>((y0 >> 1) + i) *
                                static_cast<std::size_t>(f.stride[plane]) +
                            static_cast<std::size_t>((x0 >> 1) + j);
      f.dst[plane][o] = f.src[plane][o];
    }
  }

  if (s_filtered == 0) {
    return;
  }

  {
    const int i = t >> 3;
    const int j = t & 7;
    const int value = av2enc::cdef::cdef_filter_sample(
        f.src[0], static_cast<std::size_t>(f.stride[0]), x0, y0, i, j,
        s_dir[0], s_pri[0], s_sec[0], s_damping[0], 0, 0, s_region);
    const std::size_t o = static_cast<std::size_t>(y0 + i) *
                              static_cast<std::size_t>(f.stride[0]) +
                          static_cast<std::size_t>(x0 + j);
    f.dst[0][o] = static_cast<std::uint8_t>(value);
  }

  if (s_skip_chroma != 0 || t >= 16) {
    return;
  }
  const int i = t >> 2;
  const int j = t & 3;
  const int cx0 = x0 >> 1;
  const int cy0 = y0 >> 1;
  for (int plane = 1; plane < 3; ++plane) {
    const int value = av2enc::cdef::cdef_filter_sample(
        f.src[plane], static_cast<std::size_t>(f.stride[plane]), cx0, cy0, i,
        j, s_dir[1], s_pri[1], s_sec[1], s_damping[1], 1, 1, s_region);
    const std::size_t o = static_cast<std::size_t>(cy0 + i) *
                              static_cast<std::size_t>(f.stride[plane]) +
                          static_cast<std::size_t>(cx0 + j);
    f.dst[plane][o] = static_cast<std::uint8_t>(value);
  }
}

bool valid_sec_strength(int value) {
  return value == 0 || value == 1 || value == 2 || value == 4;
}

}  // namespace

cudaError_t cdef_frame(const CdefDeviceFrame& frame,
                       const CdefOptions& options, cudaStream_t stream) {
  if (options.bit_depth != 8) {
    return cudaErrorInvalidValue;
  }
  if (frame.subsampling_x != 1 || frame.subsampling_y != 1) {
    return cudaErrorInvalidValue;
  }
  if (frame.mi_cols <= 0 || frame.mi_rows <= 0 ||
      (frame.mi_cols & 1) != 0 || (frame.mi_rows & 1) != 0) {
    return cudaErrorInvalidValue;
  }
  for (int plane = 0; plane < 3; ++plane) {
    if (frame.src[plane] == nullptr || frame.dst[plane] == nullptr ||
        frame.src[plane] == frame.dst[plane]) {
      return cudaErrorInvalidValue;
    }
    const int width = (frame.mi_cols * av2enc::cdef::kMiSize) >>
                      ((plane == 0) ? 0 : frame.subsampling_x);
    if (frame.stride[plane] < width) {
      return cudaErrorInvalidValue;
    }
  }
  if (frame.skip == nullptr || frame.lossless == nullptr ||
      frame.cdef_idx == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (options.damping < 3 || options.damping > 6 ||
      options.num_strengths < 1 ||
      options.num_strengths > av2enc::cdef::kMaxStrengths) {
    return cudaErrorInvalidValue;
  }
  for (int i = 0; i < options.num_strengths; ++i) {
    const av2enc::cdef::CdefStrength& s = options.strengths[i];
    if (s.y_pri < 0 || s.y_pri > 15 || s.uv_pri < 0 || s.uv_pri > 15 ||
        !valid_sec_strength(s.y_sec) || !valid_sec_strength(s.uv_sec)) {
      return cudaErrorInvalidValue;
    }
  }
  const int need_cols = (frame.mi_cols + av2enc::cdef::kCdefSize4 - 1) /
                        av2enc::cdef::kCdefSize4;
  const int need_rows = (frame.mi_rows + av2enc::cdef::kCdefSize4 - 1) /
                        av2enc::cdef::kCdefSize4;
  if (frame.n64_cols < 0 || frame.n64_rows < 0 ||
      (frame.n64_cols != 0 && frame.n64_cols < need_cols) ||
      (frame.n64_rows != 0 && frame.n64_rows < need_rows)) {
    return cudaErrorInvalidValue;
  }
  if ((frame.num_tile_cols > 0 && frame.tile_col_starts == nullptr) ||
      (frame.num_tile_rows > 0 && frame.tile_row_starts == nullptr) ||
      frame.num_tile_cols < 0 || frame.num_tile_rows < 0) {
    return cudaErrorInvalidValue;
  }

  const int nblk_cols = frame.mi_cols >> 1;
  const int nblk_rows = frame.mi_rows >> 1;
  const long long blocks =
      static_cast<long long>(nblk_cols) * static_cast<long long>(nblk_rows);
  if (blocks <= 0 || blocks > 0x7fffffffLL) {
    return cudaErrorInvalidValue;
  }

  cdef_kernel<<<static_cast<unsigned int>(blocks), 64, 0, stream>>>(frame,
                                                                    options);
  return cudaGetLastError();
}

}  // namespace av2enc::cuda
