// Small RAII device buffer for the restoration component tests.
//
// Tests are allowed to cudaMalloc (they are not the encoder hot path); the
// product path uses the persistent arena.

#ifndef AV2ENC_CUDA_RESTORATION_DEVICE_BUFFER_H_
#define AV2ENC_CUDA_RESTORATION_DEVICE_BUFFER_H_

#include <cuda_runtime.h>

#include <cstddef>

namespace av2enc::cuda::test {

template <typename T>
struct DevBuf {
  T* ptr = nullptr;
  std::size_t count = 0;

  DevBuf() = default;
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;
  ~DevBuf() { reset(); }

  void reset() {
    if (ptr != nullptr) {
      cudaFree(ptr);
      ptr = nullptr;
      count = 0;
    }
  }

  bool alloc(std::size_t n) {
    reset();
    if (n == 0) {
      return true;
    }
    if (cudaMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(T)) !=
        cudaSuccess) {
      ptr = nullptr;
      return false;
    }
    count = n;
    return true;
  }

  bool upload(const T* src, std::size_t n) {
    if (!alloc(n)) {
      return false;
    }
    if (n == 0) {
      return true;
    }
    return cudaMemcpy(ptr, src, n * sizeof(T), cudaMemcpyHostToDevice) ==
           cudaSuccess;
  }

  bool download(T* dst, std::size_t n) const {
    if (n > count || (n > 0 && ptr == nullptr)) {
      return false;
    }
    if (n == 0) {
      return true;
    }
    return cudaMemcpy(dst, ptr, n * sizeof(T), cudaMemcpyDeviceToHost) ==
           cudaSuccess;
  }
};

}  // namespace av2enc::cuda::test

#endif  // AV2ENC_CUDA_RESTORATION_DEVICE_BUFFER_H_
