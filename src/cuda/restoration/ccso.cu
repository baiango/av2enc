#include "ccso.h"

#include <cstdint>

namespace av2enc::cuda {
namespace {

using av2enc::restoration::CcsoCoreFrame;

__global__ void ccso_kernel(CcsoCoreFrame frame, int count0, int count1,
                            int count2, int cols0, int cols1, int cols2) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int plane = 0;
  int local = 0;
  int cols = 0;
  if (idx < count0) {
    plane = 0;
    local = idx;
    cols = cols0;
  } else if (idx < count0 + count1) {
    plane = 1;
    local = idx - count0;
    cols = cols1;
  } else if (idx < count0 + count1 + count2) {
    plane = 2;
    local = idx - count0 - count1;
    cols = cols2;
  } else {
    return;
  }
  if (cols <= 0) {
    return;
  }
  const int y = local / cols;
  const int x = local - y * cols;
  (void)av2enc::restoration::ccso_apply_sample(frame, plane, y, x);
}

int ceil_div(int a, int b) { return (a + b - 1) / b; }

bool validate(const CcsoCoreFrame& f) {
  if (f.mi_cols <= 0 || f.mi_rows <= 0) {
    return false;
  }
  if (f.bit_depth != 8 || f.sub_x != 1 || f.sub_y != 1) {
    return false;
  }
  if (f.ccso_luma_size_log2 < 6 || f.ccso_luma_size_log2 > 8) {
    return false;
  }
  if (f.sb_width <= 0 || f.sb_height <= 0) {
    return false;
  }
  if (f.plane[0].curr_luma == nullptr || f.plane[0].curr_luma_stride <= 0) {
    return false;
  }
  for (int p = 0; p < 3; ++p) {
    const av2enc::restoration::CcsoCorePlane& c = f.plane[p];
    if (!c.enabled) {
      continue;
    }
    if (c.dst == nullptr || c.dst_stride <= 0 || c.blks == nullptr ||
        c.blk_cols <= 0 || c.blk_rows <= 0 || c.offsets == nullptr) {
      return false;
    }
    const int width = av2enc::restoration::ccso_plane_width(f, p);
    if (c.dst_stride < width) {
      return false;
    }
    const int sub_x = (p == 0) ? 0 : f.sub_x;
    const int sub_y = (p == 0) ? 0 : f.sub_y;
    int cols = 0;
    int rows = 0;
    av2enc::restoration::ccso_blk_grid_size(
        width, av2enc::restoration::ccso_plane_height(f, p),
        f.ccso_luma_size_log2, sub_x, sub_y, &cols, &rows);
    if (c.blk_cols < cols || c.blk_rows < rows) {
      return false;  // CcsoBlks must cover the plane
    }
    if (c.scale_idx < 0 || c.scale_idx > 3 || c.quant_idx < 0 ||
        c.quant_idx > 3 || c.ext_filter < 0 || c.ext_filter > 6 ||
        c.edge_clf < 0 || c.edge_clf > 2 || c.max_band_log2 < 0 ||
        c.max_band_log2 > 6) {
      return false;
    }
  }
  return true;
}

}  // namespace

cudaError_t ccso_frame(const av2enc::restoration::CcsoCoreFrame& frame,
                       cudaStream_t stream) {
  if (!validate(frame)) {
    return cudaErrorInvalidValue;
  }

  int width[3] = {0, 0, 0};
  int height[3] = {0, 0, 0};
  int counts[3] = {0, 0, 0};
  for (int p = 0; p < 3; ++p) {
    width[p] = av2enc::restoration::ccso_plane_width(frame, p);
    height[p] = av2enc::restoration::ccso_plane_height(frame, p);
    counts[p] = frame.plane[p].enabled ? width[p] * height[p] : 0;
  }
  const int total = counts[0] + counts[1] + counts[2];
  if (total <= 0) {
    return cudaSuccess;
  }
  constexpr int kThreads = 256;
  const int blocks = ceil_div(total, kThreads);
  ccso_kernel<<<blocks, kThreads, 0, stream>>>(frame, counts[0], counts[1],
                                               counts[2], width[0], width[1],
                                               width[2]);
  return cudaGetLastError();
}

}  // namespace av2enc::cuda
