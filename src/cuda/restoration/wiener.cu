#include "wiener.h"

#include <cstddef>

#include "encoder/filters/restoration/wiener_core.h"

namespace av2enc::cuda {
namespace {

using av2enc::restoration::LrBlock;
using av2enc::restoration::LrCoreFrame;

__global__ void wiener_kernel(LrCoreFrame f) {
  const int per = f.mi_cols * f.mi_rows;
  if (per <= 0) {
    return;
  }
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int plane = idx / per;
  if (plane >= 3) {
    return;
  }
  const int local = idx - plane * per;
  const int row = local / f.mi_cols;
  const int col = local - row * f.mi_cols;
  LrBlock b;
  if (!av2enc::restoration::lr_block_context(f, plane, row, col, &b)) {
    return;
  }
  if (b.type == av2enc::restoration::kRestoreWienerNs) {
    av2enc::restoration::wiener_ns_block(f, b, plane);
  }
}

int ceil_div(int a, int b) { return (a + b - 1) / b; }

bool validate(const av2enc::restoration::RestorationFrame& f,
              const av2enc::restoration::LoopRestorationParams& p,
              cudaError_t* err) {
  if (f.mi_cols <= 0 || f.mi_rows <= 0 || f.bit_depth != 8 ||
      f.subsampling_x != 1 || f.subsampling_y != 1) {
    *err = cudaErrorInvalidValue;
    return false;
  }
  if (f.sb_shift < 0 || f.sb_shift > 8) {
    *err = cudaErrorInvalidValue;
    return false;
  }
  if (p.gdf_frame_enable != 0) {
    *err = cudaErrorNotSupported;
    return false;
  }
  if (p.num_filter_classes < 1 ||
      p.num_filter_classes > av2enc::restoration::kWienerNsClasses) {
    *err = cudaErrorInvalidValue;
    return false;
  }
  for (int plane = 0; plane < 3; ++plane) {
    const int sub_x = (plane == 0) ? 0 : f.subsampling_x;
    const int sub_y = (plane == 0) ? 0 : f.subsampling_y;
    const int width = (f.mi_cols * 4) >> sub_x;
    const int height = (f.mi_rows * 4) >> sub_y;
    if (width <= 0 || height <= 0) {
      *err = cudaErrorInvalidValue;
      return false;
    }
    if (f.curr[plane].data == nullptr || f.cdef[plane].data == nullptr ||
        f.lr[plane].data == nullptr || f.curr[plane].stride < width ||
        f.cdef[plane].stride < width || f.lr[plane].stride < width) {
      *err = cudaErrorInvalidValue;
      return false;
    }
    if (f.lr[plane].data == f.cdef[plane].data) {
      *err = cudaErrorInvalidValue;
      return false;
    }
    const int type = static_cast<int>(p.frame_type[plane]);
    if (type < av2enc::restoration::kRestoreNone ||
        type > av2enc::restoration::kRestoreSwitchable) {
      *err = cudaErrorInvalidValue;
      return false;
    }
    if (type == av2enc::restoration::kRestorePcWiener) {
      *err = cudaErrorNotSupported;
      return false;
    }
    if (type == av2enc::restoration::kRestoreNone) {
      continue;
    }
    if (p.unit_size[plane] <= 0 || p.unit_type[plane] == nullptr ||
        p.unit_rows[plane] <= 0 || p.unit_cols[plane] <= 0) {
      *err = cudaErrorInvalidValue;
      return false;
    }
    if (p.frame_filters_on[plane] != 0) {
      if (p.frame_coeffs[plane] == nullptr) {
        *err = cudaErrorInvalidValue;
        return false;
      }
    } else if (p.unit_coeffs[plane] == nullptr) {
      *err = cudaErrorInvalidValue;
      return false;
    }
  }
  if (p.frame_filters_on[0] != 0 && p.num_filter_classes > 1 &&
      p.subclass_grid == nullptr) {
    *err = cudaErrorInvalidValue;
    return false;
  }
  return true;
}

}  // namespace

cudaError_t loop_restoration_frame(
    const av2enc::restoration::RestorationFrame& frame,
    const av2enc::restoration::LoopRestorationParams& params,
    cudaStream_t stream) {
  cudaError_t invalid = cudaErrorInvalidValue;
  if (!validate(frame, params, &invalid)) {
    return invalid;
  }

  // §7.20: LrFrame := CdefFrame.  The copy is byte-exact and leaves the guard
  // columns untouched (only `width` bytes per row are copied).
  for (int plane = 0; plane < 3; ++plane) {
    const int sub_x = (plane == 0) ? 0 : frame.subsampling_x;
    const int sub_y = (plane == 0) ? 0 : frame.subsampling_y;
    const int width = (frame.mi_cols * 4) >> sub_x;
    const int height = (frame.mi_rows * 4) >> sub_y;
    const cudaError_t err = cudaMemcpy2DAsync(
        frame.lr[plane].data, static_cast<std::size_t>(frame.lr[plane].stride),
        frame.cdef[plane].data,
        static_cast<std::size_t>(frame.cdef[plane].stride),
        static_cast<std::size_t>(width), static_cast<std::size_t>(height),
        cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) {
      return err;
    }
  }

  bool any = false;
  for (int plane = 0; plane < 3; ++plane) {
    if (params.frame_type[plane] != av2enc::restoration::kRestoreNone) {
      any = true;
    }
  }
  if (!any) {
    return cudaSuccess;
  }

  const LrCoreFrame core =
      av2enc::restoration::loop_restoration_to_core(frame, params);
  const int total = 3 * frame.mi_cols * frame.mi_rows;
  constexpr int kThreads = 256;
  wiener_kernel<<<ceil_div(total, kThreads), kThreads, 0, stream>>>(core);
  return cudaGetLastError();
}

}  // namespace av2enc::cuda
