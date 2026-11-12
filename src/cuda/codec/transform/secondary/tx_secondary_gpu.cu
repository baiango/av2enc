// CUDA AV2 secondary transform (IST) kernels.
//
// The per-block arithmetic lives in tx_secondary_device.cuh, which is also
// compiled by the host emulation test.  This file only adds the table
// addresses, the one-thread-per-block copy/in-place wrapper and the launch
// plumbing.
#include "cuda/codec/transform/secondary/tx_secondary_gpu.h"

#include <cstdint>

#include "cuda/codec/transform/secondary/tx_secondary_device.cuh"
#include "codec/transform/secondary/tx_secondary_tables_dev.cuh"

namespace av2enc::cuda {
namespace {

using av2enc::txsec::device::SizeInfo;
using av2enc::txsec::device::Tables;

constexpr int kThreads = 256;

__device__ __forceinline__ Tables gpu_tables() {
  Tables t;
  t.ist4 = &av2enc::txsec::dev::kIst4x4Kernel[0][0][0][0];
  t.ist8 = &av2enc::txsec::dev::kIst8x8Kernel[0][0][0][0];
  t.scan_map = &av2enc::txsec::dev::kStxScanMap[0][0][0];
  t.tx_width = &av2enc::txsec::dev::kTxWidth[0];
  t.tx_height = &av2enc::txsec::dev::kTxHeight[0];
  t.tx_width_log2 = &av2enc::txsec::dev::kTxWidthLog2[0];
  t.scan_order4 = &av2enc::txsec::dev::kStxScanOrder4x4[0];
  t.scan_order8 = &av2enc::txsec::dev::kStxScanOrder8x8[0];
  return t;
}

__global__ void inverse_kernel(const std::int32_t* in, std::int32_t* out,
                               const TxSecondaryJob* jobs, int njobs) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= njobs) return;
  const TxSecondaryJob job = jobs[idx];
  const Tables tables = gpu_tables();
  SizeInfo info;
  if (!av2enc::txsec::device::size_info(tables, job.tx_size,
                                        job.plane_tx_type, &info)) {
    return;
  }
  const int elems = info.w * info.h;
  const std::int32_t* src = in + job.in_offset;
  std::int32_t* dst = out + job.out_offset;
  for (int i = 0; i < elems; ++i) dst[i] = src[i];
  av2enc::txsec::device::inverse(tables, info, job.kernel, job.sec_tx_type,
                                 job.transpose, job.bit_depth, dst);
}

__global__ void forward_kernel(const std::int32_t* in, std::int32_t* out,
                               const TxSecondaryJob* jobs, int njobs) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= njobs) return;
  const TxSecondaryJob job = jobs[idx];
  const Tables tables = gpu_tables();
  SizeInfo info;
  if (!av2enc::txsec::device::size_info(tables, job.tx_size,
                                        job.plane_tx_type, &info)) {
    return;
  }
  const int elems = info.w * info.h;
  const std::int32_t* src = in + job.in_offset;
  std::int32_t* dst = out + job.out_offset;
  for (int i = 0; i < elems; ++i) dst[i] = src[i];
  av2enc::txsec::device::forward(tables, info, job.kernel, job.sec_tx_type,
                                 job.transpose, job.bit_depth, dst);
}

}  // namespace

cudaError_t txsec_inverse_dequant(const std::int32_t* d_in,
                                  std::int32_t* d_out,
                                  const TxSecondaryJob* d_jobs, int njobs,
                                  cudaStream_t stream) {
  if (d_in == nullptr || d_out == nullptr || d_jobs == nullptr || njobs < 0) {
    return cudaErrorInvalidValue;
  }
  if (njobs == 0) return cudaSuccess;
  const int blocks = (njobs + kThreads - 1) / kThreads;
  inverse_kernel<<<blocks, kThreads, 0, stream>>>(d_in, d_out, d_jobs, njobs);
  return cudaGetLastError();
}

cudaError_t txsec_forward_coeffs(const std::int32_t* d_in,
                                 std::int32_t* d_out,
                                 const TxSecondaryJob* d_jobs, int njobs,
                                 cudaStream_t stream) {
  if (d_in == nullptr || d_out == nullptr || d_jobs == nullptr || njobs < 0) {
    return cudaErrorInvalidValue;
  }
  if (njobs == 0) return cudaSuccess;
  const int blocks = (njobs + kThreads - 1) / kThreads;
  forward_kernel<<<blocks, kThreads, 0, stream>>>(d_in, d_out, d_jobs, njobs);
  return cudaGetLastError();
}

}  // namespace av2enc::cuda
