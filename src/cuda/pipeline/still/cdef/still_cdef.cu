// M1 GPU path — CDEF device implementation.
//
// Two kernels:
//   still_cdef_kernel        §7.18 filter, one CUDA block per 8x8 luma block
//                         (thread 0 decides, 64 threads filter/copy).
//   still_cdef_search_kernel encoder-side SSE, one thread per
//                         (64x64 block, candidate strength set).
//
// All sample arithmetic comes from encoder/filters/cdef/cdef_core.h and
// cdef_search.h, i.e. the same inline functions the CPU reference calls.
// Integer sums are associative, so the per-(block, candidate) SSE is exact and
// identical to the CPU search regardless of scheduling.

#include "cuda/pipeline/still/cdef/still_cdef_cuda.h"

#include <cstddef>
#include <cstdint>

#include "codec/filters/cdef/cdef_core.h"
#include "codec/filters/cdef/cdef_search.h"

namespace av2enc::cuda::still {
namespace {

using av2enc::cdef::CdefFilterParams;
using av2enc::cdef::CdefRegion;
using av2enc::cdef::kCdefBlock4;
using av2enc::cdef::kCdefSize4;
using av2enc::cdef::kMiSize;

__device__ int cdef_idx_cols(const CdefDeviceFrame& f) {
  return f.n64_cols > 0 ? f.n64_cols : ((f.mi_cols + kCdefSize4 - 1) /
                                        kCdefSize4);
}

__global__ void still_cdef_kernel(CdefDeviceFrame f,
                               av2enc::cdef::CdefOptions options) {
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

    s_region = av2enc::cdef::cdef_filter_region(
        f.mi_cols, f.mi_rows, f.tile_col_starts, f.num_tile_cols,
        f.tile_row_starts, f.num_tile_rows,
        options.disable_loopfilters_across_tiles, r, c);

    bool skip = false;
    bool skip_chroma = false;
    av2enc::cdef::cdef_block_skip_flags(f.skip, f.lossless, f.lossless_chroma,
                                        f.mi_cols, r, c, options.on_skip_txfm,
                                        &skip, &skip_chroma);
    s_skip_chroma = skip_chroma ? 1 : 0;
    s_filtered = 0;
    if (idx >= 0 && !skip) {
      int y_dir = 0;
      std::int64_t var = 0;
      av2enc::cdef::cdef_direction(
          f.src[0], static_cast<std::size_t>(f.stride[0]), c * kMiSize,
          r * kMiSize, options.bit_depth, &y_dir, &var);

      const CdefFilterParams luma = av2enc::cdef::cdef_luma_params(
          options.strengths[idx].y_pri, options.strengths[idx].y_sec, y_dir,
          var, options.damping);
      s_pri[0] = luma.pri;
      s_sec[0] = luma.sec;
      s_dir[0] = luma.dir;
      s_damping[0] = luma.damping;

      const CdefFilterParams chroma = av2enc::cdef::cdef_chroma_params(
          options.strengths[idx].uv_pri, options.strengths[idx].uv_sec, y_dir,
          /*sub_x=*/1, /*sub_y=*/1, options.damping);
      s_pri[1] = chroma.pri;
      s_sec[1] = chroma.sec;
      s_dir[1] = chroma.dir;
      s_damping[1] = chroma.damping;
      s_filtered = 1;
    }
  }
  __syncthreads();

  const int t = static_cast<int>(threadIdx.x);
  const int y0 = r * kMiSize;
  const int x0 = c * kMiSize;

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

__global__ void still_cdef_search_kernel(CdefDeviceFrame f,
                                      av2enc::cdef::CdefSearchFrame view,
                                      av2enc::cdef::CdefStrength* candidates,
                                      int candidate_count, int damping,
                                      int on_skip_txfm,
                                      int disable_across_tiles,
                                      std::int64_t* sse) {
  const int n64_cols =
      (f.mi_cols + kCdefSize4 - 1) / kCdefSize4;
  const int n64_rows =
      (f.mi_rows + kCdefSize4 - 1) / kCdefSize4;
  const int nblocks = n64_cols * n64_rows;
  const long long total =
      static_cast<long long>(nblocks) * static_cast<long long>(candidate_count);
  const long long tid64 = static_cast<long long>(blockIdx.x) *
                              static_cast<long long>(blockDim.x) +
                          static_cast<long long>(threadIdx.x);
  if (tid64 >= total || nblocks <= 0) {
    return;
  }
  const int tid = static_cast<int>(tid64);
  const int block = tid / candidate_count;
  const int k = tid - block * candidate_count;
  const int br = block / n64_cols;
  const int bc = block - br * n64_cols;

  std::uint8_t* const no_dst[3] = {nullptr, nullptr, nullptr};
  std::int64_t total_sse = 0;
  for (int r = br * kCdefSize4;
       r < (br + 1) * kCdefSize4 && r < f.mi_rows; r += kCdefBlock4) {
    for (int c = bc * kCdefSize4;
         c < (bc + 1) * kCdefSize4 && c < f.mi_cols; c += kCdefBlock4) {
      const av2enc::cdef::CdefBlockSse one = av2enc::cdef::cdef_eval_block(
          view, candidates[k], damping, on_skip_txfm != 0,
          disable_across_tiles != 0, r, c, no_dst);
      total_sse += one.luma + one.chroma;
    }
  }
  sse[static_cast<std::size_t>(k) * static_cast<std::size_t>(nblocks) +
      static_cast<std::size_t>(block)] = total_sse;
}

}  // namespace

cudaError_t still_cdef_filter(const CdefDeviceFrame& frame,
                           const av2enc::cdef::CdefOptions& options,
                           cudaStream_t stream) {
  if (frame.cdef_idx == nullptr) {
    return cudaErrorInvalidValue;
  }
  std::string error;
  if (!still_cdef_validate_device_frame(frame, options, &error)) {
    return cudaErrorInvalidValue;
  }
  const int nblk_cols = frame.mi_cols >> 1;
  const int nblk_rows = frame.mi_rows >> 1;
  const long long blocks =
      static_cast<long long>(nblk_cols) * static_cast<long long>(nblk_rows);
  if (blocks <= 0 || blocks > 0x7fffffffLL) {
    return cudaErrorInvalidValue;
  }
  still_cdef_kernel<<<static_cast<unsigned int>(blocks), 64, 0, stream>>>(frame,
                                                                     options);
  return cudaGetLastError();
}

cudaError_t still_cdef_search_sse(const CdefDeviceFrame& frame,
                               const std::uint8_t* const ref[3],
                               const int ref_stride[3],
                               const av2enc::cdef::CdefStrength* candidates,
                               int candidate_count, int damping,
                               bool on_skip_txfm,
                               bool disable_loopfilters_across_tiles,
                               std::int64_t* d_sse, cudaStream_t stream) {
  av2enc::cdef::CdefOptions options;
  options.damping = damping;
  options.num_strengths = 1;  // strength sets are supplied separately
  options.on_skip_txfm = on_skip_txfm;
  options.disable_loopfilters_across_tiles = disable_loopfilters_across_tiles;
  options.bit_depth = 8;
  std::string error;
  if (!still_cdef_validate_device_frame(frame, options, &error)) {
    return cudaErrorInvalidValue;
  }
  if (candidates == nullptr || candidate_count <= 0 || d_sse == nullptr ||
      ref == nullptr || ref_stride == nullptr) {
    return cudaErrorInvalidValue;
  }
  for (int plane = 0; plane < 3; ++plane) {
    if (ref[plane] == nullptr) {
      return cudaErrorInvalidValue;
    }
  }
  const int n64_cols = (frame.mi_cols + kCdefSize4 - 1) / kCdefSize4;
  const int n64_rows = (frame.mi_rows + kCdefSize4 - 1) / kCdefSize4;
  const long long blocks =
      static_cast<long long>(n64_cols) * static_cast<long long>(n64_rows);
  const long long total = blocks * static_cast<long long>(candidate_count);
  if (total <= 0 || total > 0x7fffffffLL) {
    return cudaErrorInvalidValue;
  }

  // The device view used by the shared evaluator.
  av2enc::cdef::CdefSearchFrame view{};
  for (int plane = 0; plane < 3; ++plane) {
    view.src[plane].data = frame.src[plane];
    view.src[plane].stride = frame.stride[plane];
    view.ref[plane].data = ref[plane];
    view.ref[plane].stride = ref_stride[plane];
  }
  view.width = frame.width;
  view.height = frame.height;
  view.mi_cols = frame.mi_cols;
  view.mi_rows = frame.mi_rows;
  view.subsampling_x = 1;
  view.subsampling_y = 1;
  view.skip = frame.skip;
  view.lossless = frame.lossless;
  view.lossless_chroma = frame.lossless_chroma;
  view.tile_col_starts = frame.tile_col_starts;
  view.num_tile_cols = frame.num_tile_cols;
  view.tile_row_starts = frame.tile_row_starts;
  view.num_tile_rows = frame.num_tile_rows;

  const int threads = 128;
  const unsigned int grid = static_cast<unsigned int>((total + threads - 1) /
                                                      threads);
  still_cdef_search_kernel<<<grid, threads, 0, stream>>>(
      frame, view, const_cast<av2enc::cdef::CdefStrength*>(candidates),
      candidate_count, damping, on_skip_txfm ? 1 : 0,
      disable_loopfilters_across_tiles ? 1 : 0, d_sse);
  return cudaGetLastError();
}

}  // namespace av2enc::cuda::still
