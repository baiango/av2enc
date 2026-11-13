// Device block-variance cost kernel.  See plan_cost.h for the contract.

#include "cuda/pipeline/still/plan/plan_cost.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <string>

namespace av2enc::still::plan {
namespace {

__global__ void variance_kernel(const std::uint8_t* __restrict__ luma,
                                int stride,
                                const PlanCostRect* __restrict__ rects,
                                std::int64_t* __restrict__ costs, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const PlanCostRect rect = rects[index];
  const int w = rect.w;
  const int h = rect.h;
  if (w <= 0 || h <= 0) {
    costs[index] = 0;
    return;
  }
  std::int64_t sum = 0;
  for (int y = 0; y < h; ++y) {
    const std::uint8_t* row =
        luma + static_cast<std::size_t>(rect.y + y) *
                   static_cast<std::size_t>(stride) +
        static_cast<std::size_t>(rect.x);
    for (int x = 0; x < w; ++x) {
      sum += row[x];
    }
  }
  const double mean =
      static_cast<double>(sum) / static_cast<double>(w * h);
  double sse = 0.0;
  for (int y = 0; y < h; ++y) {
    const std::uint8_t* row =
        luma + static_cast<std::size_t>(rect.y + y) *
                   static_cast<std::size_t>(stride) +
        static_cast<std::size_t>(rect.x);
    for (int x = 0; x < w; ++x) {
      const double delta = __dsub_rn(static_cast<double>(row[x]), mean);
      sse = __dadd_rn(sse, __dmul_rn(delta, delta));
    }
  }
  costs[index] = static_cast<std::int64_t>(sse);
}

void set_error(std::string* error, const char* context,
               cudaError_t status) {
  if (error == nullptr) return;
  char detail[256];
  std::snprintf(detail, sizeof(detail), "%s: %s (CUDA %d)", context,
                cudaGetErrorString(status), static_cast<int>(status));
  *error = detail;
}

}  // namespace

bool plan_cost_launch_variance(const std::uint8_t* d_luma, int luma_stride,
                               const PlanCostRect* d_rects,
                               std::int64_t* d_costs, int count, void* stream,
                               std::string* error) {
  if (error != nullptr) error->clear();
  if (count < 0) {
    if (error != nullptr) *error = "variance kernel: negative rectangle count";
    return false;
  }
  if (count == 0) return true;
  if (d_luma == nullptr || d_rects == nullptr || d_costs == nullptr ||
      luma_stride <= 0) {
    if (error != nullptr) {
      *error = "variance kernel: null device pointer or invalid stride";
    }
    return false;
  }
  constexpr int kThreads = 128;
  const int blocks = (count + kThreads - 1) / kThreads;
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  variance_kernel<<<blocks, kThreads, 0, cuda_stream>>>(d_luma, luma_stride,
                                                        d_rects, d_costs,
                                                        count);
  const cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    set_error(error, "variance kernel launch failed", status);
    return false;
  }
  return true;
}

}  // namespace av2enc::still::plan
