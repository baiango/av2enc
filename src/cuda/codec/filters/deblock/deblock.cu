#include "deblock.h"

#include <cstddef>

namespace av2enc::cuda {
namespace {

using av2enc::deblock::DeblockCoreFrame;
using av2enc::deblock::DeblockMi;
using av2enc::deblock::DeblockOptions;

__global__ void deblock_pass_kernel(DeblockCoreFrame frame,
                                    DeblockOptions options, int pass,
                                    int count0, int count1, int count2,
                                    int cols0, int cols1, int cols2) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int plane = 0;
  int local = 0;
  int cols = 0;
  int row_step = 1;
  int col_step = 1;
  if (idx < count0) {
    plane = 0;
    local = idx;
    cols = cols0;
    row_step = 1;
    col_step = 1;
  } else if (idx < count0 + count1) {
    plane = 1;
    local = idx - count0;
    cols = cols1;
    row_step = 1 << frame.sub_y;
    col_step = 1 << frame.sub_x;
  } else if (idx < count0 + count1 + count2) {
    plane = 2;
    local = idx - count0 - count1;
    cols = cols2;
    row_step = 1 << frame.sub_y;
    col_step = 1 << frame.sub_x;
  } else {
    return;
  }

  const int flag_index = (plane == 0) ? pass : (plane + 1);
  if (!options.apply_deblocking_filter[flag_index]) {
    return;
  }
  if (cols <= 0) {
    return;
  }
  const int row = (local / cols) * row_step;
  const int col = (local % cols) * col_step;
  (void)av2enc::deblock::deblock_edge_core<true>(frame, options, plane, pass,
                                                 row, col);
}

int ceil_div(int a, int b) { return (a + b - 1) / b; }

bool validate(const DeblockDeviceFrame& f, const DeblockOptions& o) {
  if (o.bit_depth != 8) {
    return false;
  }
  if (f.subsampling_x != 1 || f.subsampling_y != 1) {
    return false;
  }
  if (f.mi_cols <= 0 || f.mi_rows <= 0) {
    return false;
  }
  if ((f.mi_cols & 1) != 0 || (f.mi_rows & 1) != 0) {
    return false;
  }
  for (int p = 0; p < 3; ++p) {
    if (f.plane[p] == nullptr || f.mi[p] == nullptr) {
      return false;
    }
    const int width = (p == 0) ? (f.mi_cols * 4) : (f.mi_cols * 2);
    if (f.stride[p] < width) {
      return false;
    }
  }
  if ((f.num_tile_cols > 0 && f.tile_col_starts == nullptr) ||
      (f.num_tile_rows > 0 && f.tile_row_starts == nullptr)) {
    return false;
  }
  if (f.num_tile_cols < 0 || f.num_tile_rows < 0) {
    return false;
  }
  return true;
}

}  // namespace

cudaError_t deblock_frame(const DeblockDeviceFrame& frame,
                          const DeblockOptions& options,
                          cudaStream_t stream) {
  if (!validate(frame, options)) {
    return cudaErrorInvalidValue;
  }

  DeblockCoreFrame core{};
  for (int p = 0; p < 3; ++p) {
    core.plane[p] = frame.plane[p];
    core.stride[p] = frame.stride[p];
    core.mi[p] = frame.mi[p];
  }
  core.mi_cols = frame.mi_cols;
  core.mi_rows = frame.mi_rows;
  core.sub_x = frame.subsampling_x;
  core.sub_y = frame.subsampling_y;
  core.tile_col_starts = frame.tile_col_starts;
  core.num_tile_cols = frame.num_tile_cols;
  core.tile_row_starts = frame.tile_row_starts;
  core.num_tile_rows = frame.num_tile_rows;

  const int col_step[3] = {1, 1 << frame.subsampling_x,
                           1 << frame.subsampling_x};
  const int row_step[3] = {1, 1 << frame.subsampling_y,
                           1 << frame.subsampling_y};
  const int cols[3] = {ceil_div(frame.mi_cols, col_step[0]),
                       ceil_div(frame.mi_cols, col_step[1]),
                       ceil_div(frame.mi_cols, col_step[2])};
  const int rows[3] = {ceil_div(frame.mi_rows, row_step[0]),
                       ceil_div(frame.mi_rows, row_step[1]),
                       ceil_div(frame.mi_rows, row_step[2])};
  const int counts[3] = {cols[0] * rows[0], cols[1] * rows[1],
                         cols[2] * rows[2]};

  constexpr int kThreads = 256;
  for (int pass = 0; pass < 2; ++pass) {
    const int total = counts[0] + counts[1] + counts[2];
    if (total <= 0) {
      continue;
    }
    const int blocks = ceil_div(total, kThreads);
    deblock_pass_kernel<<<blocks, kThreads, 0, stream>>>(
        core, options, pass, counts[0], counts[1], counts[2], cols[0], cols[1],
        cols[2]);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      return err;
    }
  }
  return cudaSuccess;
}

}  // namespace av2enc::cuda
