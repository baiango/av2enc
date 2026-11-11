// M1 GPU reconstruction + deblock-metadata kernels.  See recon.h.

#include "recon.h"

#include <algorithm>
#include <cstddef>

namespace av2enc::cuda {
namespace m1 {
namespace {

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ std::uint8_t clip1(int v) {
  return static_cast<std::uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
}

__device__ __host__ __forceinline__ int imin(int a, int b) {
  return a < b ? a : b;
}

__device__ __host__ __forceinline__ bool decoded_at(const std::uint8_t* map,
                                                    int cols, int rows, int x,
                                                    int y) {
  if (x < 0 || y < 0 || x >= cols || y >= rows) return false;
  return map[static_cast<std::ptrdiff_t>(y) * cols + x] != 0;
}

// CPU Encoder::count_top_right, plane 4x4 units.
__device__ __host__ int count_top_right_dev(
    const std::uint8_t* decoded, int decoded_cols, int decoded_rows,
    int plane_mi_cols, int sb_mi_col, int sb_mi_row, int plane,
    int sb_cols_plane, int abs_x4, int abs_y4, int w4) {
  const int sub = plane == 0 ? 0 : 1;
  const int sb_c0 = sb_mi_col >> sub;
  const int sb_r0 = sb_mi_row >> sub;
  const int rel_x = abs_x4 - sb_c0;
  const int rel_y = abs_y4 - sb_r0;
  const int border_w = (2 * (16 >> sub) < plane_mi_cols - sb_c0)
                           ? 2 * (16 >> sub)
                           : plane_mi_cols - sb_c0;
  int count = 0;
  for (int i = 0; i < w4; ++i) {
    const int x = rel_x + w4 + i;
    if (x >= (rel_y == 0 ? border_w : sb_cols_plane)) break;
    if (rel_y - 1 >= 0 &&
        !decoded_at(decoded, decoded_cols, decoded_rows, sb_c0 + x,
                    sb_r0 + rel_y - 1)) {
      break;
    }
    ++count;
  }
  return count;
}

// CPU Encoder::count_bottom_left, plane 4x4 units.
__device__ __host__ int count_bottom_left_dev(
    const std::uint8_t* decoded, int decoded_cols, int decoded_rows,
    int sb_mi_col, int sb_mi_row, int plane, int sb_rows_plane, int abs_x4,
    int abs_y4, int h4) {
  const int sub = plane == 0 ? 0 : 1;
  const int sb_c0 = sb_mi_col >> sub;
  const int sb_r0 = sb_mi_row >> sub;
  const int rel_x = abs_x4 - sb_c0;
  const int rel_y = abs_y4 - sb_r0;
  int count = 0;
  for (int i = 0; i < h4; ++i) {
    const int y = rel_y + h4 + i;
    if (y >= sb_rows_plane) break;
    if (rel_x - 1 >= 0 &&
        !decoded_at(decoded, decoded_cols, decoded_rows, sb_c0 + rel_x - 1,
                    sb_r0 + y)) {
      break;
    }
    ++count;
  }
  return count;
}

// ---------------------------------------------------------------------------
// Frame init
// ---------------------------------------------------------------------------

__global__ void init_mi_kernel(deblock::DeblockMi* mi0, deblock::DeblockMi* mi1,
                               deblock::DeblockMi* mi2, int count,
                               std::uint8_t qindex) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  deblock::DeblockMi neutral;
  neutral.qindex = qindex;
  mi0[i] = neutral;
  mi1[i] = neutral;
  mi2[i] = neutral;
}

__global__ void reset_decoded_kernel(std::uint8_t* d0, int n0,
                                     std::uint8_t* d1, int n1,
                                     std::uint8_t* d2, int n2) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n0) d0[i] = 0;
  if (i < n1) d1[i] = 0;
  if (i < n2) d2[i] = 0;
}

// ---------------------------------------------------------------------------
// Reconstruction write
// ---------------------------------------------------------------------------

__device__ __forceinline__ void write_one(const ReconWriteJob& j) {
  const int pels = j.w * j.h;
  for (int i = threadIdx.x; i < pels; i += blockDim.x) {
    const int x = i % j.w;
    const int y = i / j.w;
    const int px = j.x0 + x;
    const int py = j.y0 + y;
    // Same clamp as CPU Encoder::write_recon: the padded plane extent.
    if (px < 0 || px >= j.stride) continue;
    if (py < 0 || py >= j.height) continue;
    const int v = static_cast<int>(j.pred[i]) + static_cast<int>(j.residual[i]);
    j.plane[static_cast<std::ptrdiff_t>(py) * j.stride + px] = clip1(v);
  }
}

__global__ void write_blocks_kernel(const ReconWriteJob* jobs, int count) {
  const int job = blockIdx.x;
  if (job >= count) return;
  write_one(jobs[job]);
}

// Single-job variant: the descriptor is a kernel argument, so no device copy of
// the job array is needed.
__global__ void write_one_kernel(ReconWriteJob job) { write_one(job); }

// ---------------------------------------------------------------------------
// Prediction availability
// ---------------------------------------------------------------------------

__global__ void availability_kernel(const ReconPredState state,
                                    const ReconAvailRequest* requests,
                                    ReconAvail* out, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const ReconAvailRequest& r = requests[i];
  ReconAvail a;
  a.have_left = r.x4 > 0 ? 1 : 0;
  a.have_above = r.y4 > 0 ? 1 : 0;
  a.num4_above_right = 0;
  a.num4_below_left = 0;
  if (r.plane < 0 || r.plane > 2) {
    out[i] = a;
    return;
  }
  a.num4_above_right = count_top_right_dev(
      state.decoded[r.plane], state.decoded_cols[r.plane],
      state.decoded_rows[r.plane], state.decoded_cols[r.plane],
      state.sb_mi_col, state.sb_mi_row, r.plane, state.sb_cols[r.plane], r.x4,
      r.y4, r.w4);
  a.num4_below_left = count_bottom_left_dev(
      state.decoded[r.plane], state.decoded_cols[r.plane],
      state.decoded_rows[r.plane], state.sb_mi_col, state.sb_mi_row, r.plane,
      state.sb_rows[r.plane], r.x4, r.y4, r.h4);
  out[i] = a;
}

// ---------------------------------------------------------------------------
// Block state commit + deblock metadata fill
// ---------------------------------------------------------------------------

__device__ void commit_one(const ReconBlockInfo& b, const ReconPredState& s,
                           const ReconFrame& f) {
  const int mi_cols = s.mi_cols;
  const int mi_rows = s.mi_rows;

  // IntraJointModes over the luma MI extent (bounded like the CPU).
  for (int yy = 0; yy < b.h4; ++yy) {
    const int rr = b.mi_row + yy;
    if (rr >= mi_rows) break;
    for (int xx = 0; xx < b.w4; ++xx) {
      const int cc = b.mi_col + xx;
      if (cc >= mi_cols) break;
      s.intra_joint[static_cast<std::ptrdiff_t>(rr) * mi_cols + cc] = b.joint;
    }
  }

  // UV smooth context on the chroma 4x4 grid.
  const int cr = b.mi_row / 2;
  const int cc0 = b.mi_col / 2;
  const int chroma_mi_rows = mi_rows / 2;
  const int chroma_mi_cols = mi_cols / 2;
  for (int yy = 0; yy < b.chroma_h4; ++yy) {
    const int rr = cr + yy;
    if (rr >= chroma_mi_rows) break;
    for (int xx = 0; xx < b.chroma_w4; ++xx) {
      const int cc = cc0 + xx;
      if (cc >= chroma_mi_cols) break;
      const std::ptrdiff_t idx =
          static_cast<std::ptrdiff_t>(rr) * chroma_mi_cols + cc;
      s.uv_smooth[idx] = b.uv_smooth;
    }
  }

  // Decoded maps: luma at (mi_col, mi_row), chroma at half resolution.
  for (int yy = 0; yy < b.h4; ++yy) {
    for (int xx = 0; xx < b.w4; ++xx) {
      const int x = b.mi_col + xx;
      const int y = b.mi_row + yy;
      if (x < 0 || y < 0 || x >= s.decoded_cols[0] ||
          y >= s.decoded_rows[0]) {
        continue;
      }
      s.decoded[0][static_cast<std::ptrdiff_t>(y) * s.decoded_cols[0] + x] = 1;
    }
  }
  for (int plane = 1; plane <= 2; ++plane) {
    for (int yy = 0; yy < b.chroma_h4; ++yy) {
      for (int xx = 0; xx < b.chroma_w4; ++xx) {
        const int x = b.mi_col / 2 + xx;
        const int y = b.mi_row / 2 + yy;
        if (x < 0 || y < 0 || x >= s.decoded_cols[plane] ||
            y >= s.decoded_rows[plane]) {
          continue;
        }
        s.decoded[plane][static_cast<std::ptrdiff_t>(y) *
                             s.decoded_cols[plane] +
                         x] = 1;
      }
    }
  }

  // Deblock metadata.  Mirrors CPU Encoder::fill_deblock_mi.
  const deblock::TxSize luma =
      static_cast<deblock::TxSize>(b.tx_size_luma);
  const deblock::TxSize chroma =
      static_cast<deblock::TxSize>(b.tx_size_chroma);
  const int clamped_w4 = imin(b.w4, mi_cols - b.mi_col);
  const int clamped_h4 = imin(b.h4, mi_rows - b.mi_row);
  for (int yy = 0; yy < b.h4; ++yy) {
    const int rr = b.mi_row + yy;
    if (rr >= mi_rows) break;
    for (int xx = 0; xx < b.w4; ++xx) {
      const int cc = b.mi_col + xx;
      if (cc >= mi_cols) break;
      deblock::DeblockMi mi;
      mi.block_col_base = static_cast<std::uint16_t>(b.mi_col);
      mi.block_row_base = static_cast<std::uint16_t>(b.mi_row);
      mi.block_w4 = static_cast<std::uint8_t>(clamped_w4);
      mi.block_h4 = static_cast<std::uint8_t>(clamped_h4);
      mi.sub_pu_size = luma;
      mi.tx_size = luma;
      mi.tx_col_base = static_cast<std::uint16_t>(b.mi_col);
      mi.tx_row_base = static_cast<std::uint16_t>(b.mi_row);
      mi.sub_pu_col_base = static_cast<std::uint16_t>(b.mi_col);
      mi.sub_pu_row_base = static_cast<std::uint16_t>(b.mi_row);
      mi.skip = b.skip;
      mi.region_intra = b.region_intra;
      mi.lossless = b.lossless;
      mi.qindex = b.qindex;
      const std::ptrdiff_t idx =
          static_cast<std::ptrdiff_t>(rr) * mi_cols + cc;
      f.deblock_mi[0][idx] = mi;
      f.deblock_mi[1][idx] = mi;
      f.deblock_mi[2][idx] = mi;
      f.deblock_mi[1][idx].tx_size = chroma;
      f.deblock_mi[2][idx].tx_size = chroma;
    }
  }
}

__global__ void commit_blocks_kernel(const ReconBlockInfo* infos, int count,
                                     ReconPredState state, ReconFrame frame) {
  const int i = blockIdx.x;
  if (i >= count) return;
  if (threadIdx.x != 0) return;
  commit_one(infos[i], state, frame);
}

// Single-block variant: the descriptor is a kernel argument (no device copy).
__global__ void commit_one_kernel(ReconBlockInfo info, ReconPredState state,
                                  ReconFrame frame) {
  if (threadIdx.x != 0) return;
  commit_one(info, state, frame);
}

}  // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

cudaError_t recon_init_frame(const ReconFrame& frame, cudaStream_t stream) {
  for (int p = 0; p < 3; ++p) {
    if (frame.plane[p].data == nullptr || frame.plane[p].stride <= 0 ||
        frame.plane[p].height <= 0) {
      return cudaErrorInvalidValue;
    }
    const std::size_t bytes = static_cast<std::size_t>(frame.plane[p].stride) *
                              static_cast<std::size_t>(frame.plane[p].height);
    cudaError_t err = cudaMemsetAsync(frame.plane[p].data, 0, bytes, stream);
    if (err != cudaSuccess) return err;
  }
  if (frame.mi_cols <= 0 || frame.mi_rows <= 0) {
    return cudaErrorInvalidValue;
  }
  for (int p = 0; p < 3; ++p) {
    if (frame.deblock_mi[p] == nullptr) return cudaErrorInvalidValue;
  }
  const int count = frame.mi_cols * frame.mi_rows;
  constexpr int kThreads = 256;
  const int blocks = (count + kThreads - 1) / kThreads;
  init_mi_kernel<<<blocks, kThreads, 0, stream>>>(
      frame.deblock_mi[0], frame.deblock_mi[1], frame.deblock_mi[2], count,
      frame.base_q_idx);
  return cudaGetLastError();
}

cudaError_t recon_init_state(const ReconPredState& state,
                             cudaStream_t stream) {
  if (state.mi_cols <= 0 || state.mi_rows <= 0) {
    return cudaErrorInvalidValue;
  }
  for (int p = 0; p < 3; ++p) {
    if (state.decoded[p] == nullptr) return cudaErrorInvalidValue;
  }
  if (state.intra_joint == nullptr || state.uv_smooth == nullptr) {
    return cudaErrorInvalidValue;
  }
  for (int p = 0; p < 3; ++p) {
    const std::size_t n = static_cast<std::size_t>(state.decoded_cols[p]) *
                          static_cast<std::size_t>(state.decoded_rows[p]);
    cudaError_t err = cudaMemsetAsync(state.decoded[p], 0, n, stream);
    if (err != cudaSuccess) return err;
  }
  cudaError_t err = cudaMemsetAsync(
      state.intra_joint, 0,
      sizeof(int) * static_cast<std::size_t>(state.mi_cols) *
          static_cast<std::size_t>(state.mi_rows),
      stream);
  if (err != cudaSuccess) return err;
  const std::size_t uv_n = static_cast<std::size_t>(state.mi_cols / 2) *
                           static_cast<std::size_t>(state.mi_rows / 2);
  err = cudaMemsetAsync(state.uv_smooth, 0, uv_n, stream);
  if (err != cudaSuccess) return err;
  return cudaGetLastError();
}

cudaError_t recon_begin_sb(ReconPredState& state, int sb_mi_row, int sb_mi_col,
                           cudaStream_t stream) {
  if (state.mi_cols <= 0 || state.mi_rows <= 0) {
    return cudaErrorInvalidValue;
  }
  state.sb_mi_row = sb_mi_row;
  state.sb_mi_col = sb_mi_col;
  // Same bounds as the CPU encode loop (plane 1/2 share the chroma values).
  state.sb_cols[0] = std::min(16, state.mi_cols - sb_mi_col);
  state.sb_rows[0] = std::min(16, state.mi_rows - sb_mi_row);
  state.sb_cols[1] = std::min(8, state.mi_cols / 2 - sb_mi_col / 2);
  state.sb_rows[1] = std::min(8, state.mi_rows / 2 - sb_mi_row / 2);
  state.sb_cols[2] = state.sb_cols[1];
  state.sb_rows[2] = state.sb_rows[1];
  for (int p = 0; p < 3; ++p) {
    if (state.decoded[p] == nullptr) return cudaErrorInvalidValue;
  }
  const int n0 = state.decoded_cols[0] * state.decoded_rows[0];
  const int n1 = state.decoded_cols[1] * state.decoded_rows[1];
  const int n2 = state.decoded_cols[2] * state.decoded_rows[2];
  const int count = std::max(n0, std::max(n1, n2));
  if (count <= 0) return cudaErrorInvalidValue;
  constexpr int kThreads = 256;
  const int blocks = (count + kThreads - 1) / kThreads;
  reset_decoded_kernel<<<blocks, kThreads, 0, stream>>>(
      state.decoded[0], n0, state.decoded[1], n1, state.decoded[2], n2);
  return cudaGetLastError();
}

cudaError_t recon_write_blocks(const ReconWriteJob* d_jobs, int job_count,
                               cudaStream_t stream) {
  if (job_count <= 0) return cudaSuccess;
  if (d_jobs == nullptr) return cudaErrorInvalidValue;
  constexpr int kThreads = 128;
  write_blocks_kernel<<<job_count, kThreads, 0, stream>>>(d_jobs, job_count);
  return cudaGetLastError();
}

cudaError_t recon_write_block(const ReconWriteJob& job, cudaStream_t stream) {
  if (job.pred == nullptr || job.residual == nullptr || job.plane == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (job.w <= 0 || job.h <= 0 || job.stride <= 0 || job.height <= 0) {
    return cudaErrorInvalidValue;
  }
  constexpr int kThreads = 128;
  write_one_kernel<<<1, kThreads, 0, stream>>>(job);
  return cudaGetLastError();
}

cudaError_t recon_commit_blocks(const ReconBlockInfo* d_infos, int count,
                                const ReconPredState& state,
                                const ReconFrame& frame, cudaStream_t stream) {
  if (count <= 0) return cudaSuccess;
  if (d_infos == nullptr) return cudaErrorInvalidValue;
  commit_blocks_kernel<<<count, 32, 0, stream>>>(d_infos, count, state, frame);
  return cudaGetLastError();
}

cudaError_t recon_commit_block(const ReconBlockInfo& info,
                               const ReconPredState& state,
                               const ReconFrame& frame, cudaStream_t stream) {
  commit_one_kernel<<<1, 32, 0, stream>>>(info, state, frame);
  return cudaGetLastError();
}

cudaError_t recon_block_availability(const ReconPredState& state,
                                     const ReconAvailRequest* d_requests,
                                     ReconAvail* d_out, int count,
                                     cudaStream_t stream) {
  if (count <= 0) return cudaSuccess;
  if (d_requests == nullptr || d_out == nullptr) {
    return cudaErrorInvalidValue;
  }
  constexpr int kThreads = 128;
  const int blocks = (count + kThreads - 1) / kThreads;
  availability_kernel<<<blocks, kThreads, 0, stream>>>(state, d_requests, d_out,
                                                       count);
  return cudaGetLastError();
}

cudaError_t recon_deblock_frame(const ReconFrame& frame,
                                const deblock::DeblockOptions& options,
                                cudaStream_t stream) {
  DeblockDeviceFrame view;
  for (int p = 0; p < 3; ++p) {
    view.plane[p] = frame.plane[p].data;
    view.stride[p] = frame.plane[p].stride;
    view.mi[p] = frame.deblock_mi[p];
  }
  view.mi_cols = frame.mi_cols;
  view.mi_rows = frame.mi_rows;
  view.subsampling_x = 1;
  view.subsampling_y = 1;
  view.tile_col_starts = frame.tile_col_starts;
  view.num_tile_cols = frame.num_tile_cols;
  view.tile_row_starts = frame.tile_row_starts;
  view.num_tile_rows = frame.num_tile_rows;
  return deblock_frame(view, options, stream);
}

}  // namespace m1
}  // namespace av2enc::cuda
