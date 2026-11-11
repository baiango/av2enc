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
  const av2enc::restoration::RestorationStatus st =
      av2enc::restoration::loop_restoration_validate(f, p);
  if (st == av2enc::restoration::RestorationStatus::kOk) {
    return true;
  }
  *err = (st == av2enc::restoration::RestorationStatus::kUnsupportedGdf ||
          st == av2enc::restoration::RestorationStatus::kUnsupportedPcWiener)
             ? cudaErrorNotSupported
             : cudaErrorInvalidValue;
  return false;
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
