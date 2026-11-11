// Host-side CUDA API shim used only by build_emulated_gpu_test.sh on machines
// without nvcc/CUDA. It allocates host memory and copies with memcpy, so the
// exact same alpha_gpu_test.cu runs against cuda_emulation.cpp.
//
// The real CUDA build does not include this directory in its include path.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

using cudaError_t = int;
using cudaStream_t = void*;

enum cudaMemcpyKind {
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3,
};

inline constexpr cudaError_t cudaSuccess = 0;
inline constexpr cudaError_t cudaErrorInvalidValue = 1;
inline constexpr cudaError_t cudaErrorMemoryAllocation = 2;
inline constexpr cudaError_t cudaErrorNotSupported = 3;

inline const char* cudaGetErrorString(cudaError_t error) {
  switch (error) {
    case cudaSuccess:
      return "no error";
    case cudaErrorInvalidValue:
      return "invalid value";
    case cudaErrorMemoryAllocation:
      return "out of memory";
    default:
      return "unknown error";
  }
}

inline cudaError_t cudaGetLastError() { return cudaSuccess; }

inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }

// Emulation-only allocation registry. A pointer that did not come from
// cudaMalloc() is a host pointer and would fault if a kernel dereferenced it;
// cuda_emulation.cpp uses this to reject host CflJob arrays. Ranges are stored
// so a pointer into a device allocation is accepted.
namespace av2enc_cuda_stub {
struct Allocation {
  void* pointer;
  std::size_t size;
};
inline std::vector<Allocation>& allocations() {
  static std::vector<Allocation> list;
  return list;
}
inline bool is_device_pointer(const void* pointer) {
  const std::uintptr_t address =
      reinterpret_cast<std::uintptr_t>(pointer);
  for (const Allocation& allocation : allocations()) {
    const std::uintptr_t base =
        reinterpret_cast<std::uintptr_t>(allocation.pointer);
    if (address >= base && address - base < allocation.size) {
      return true;
    }
  }
  return false;
}
}  // namespace av2enc_cuda_stub

inline cudaError_t cudaMalloc(void** pointer, std::size_t size) {
  if (pointer == nullptr) {
    return cudaErrorInvalidValue;
  }
  const std::size_t allocated = size == 0 ? 1 : size;
  *pointer = std::malloc(allocated);
  if (*pointer == nullptr) {
    return cudaErrorMemoryAllocation;
  }
  av2enc_cuda_stub::allocations().push_back(
      av2enc_cuda_stub::Allocation{*pointer, allocated});
  return cudaSuccess;
}

inline cudaError_t cudaFree(void* pointer) {
  if (pointer == nullptr) {
    return cudaSuccess;
  }
  std::vector<av2enc_cuda_stub::Allocation>& list =
      av2enc_cuda_stub::allocations();
  for (std::size_t i = 0; i < list.size(); ++i) {
    if (list[i].pointer == pointer) {
      list.erase(list.begin() + static_cast<std::ptrdiff_t>(i));
      break;
    }
  }
  std::free(pointer);
  return cudaSuccess;
}

inline cudaError_t cudaMemcpy(void* dst, const void* src, std::size_t size,
                              cudaMemcpyKind /*kind*/) {
  if ((dst == nullptr || src == nullptr) && size != 0) {
    return cudaErrorInvalidValue;
  }
  std::memcpy(dst, src, size);
  return cudaSuccess;
}
