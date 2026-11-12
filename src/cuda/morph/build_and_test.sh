#!/usr/bin/env bash
# Build/run the AV2 morphological prediction tests.
#
# With nvcc + a CUDA device: CPU-vs-GPU bit-exact parity (morph_test.cu).
# Without nvcc: the same shared device header is compiled as host C++, linked
# with the emulated entry points and the real arena/workspace host code, and
# compared against the CPU reference (morph_emulation_test.cpp).
#
#   ./build_and_test.sh                 # build + run
#   ./build_and_test.sh --no-run
#   ARCH=sm_120 ./build_gpu_test.sh     # secondary GPU target
#
# Self-contained: nothing here touches the root CMake build.

set -euo pipefail
cd "$(dirname "$0")"

HERE="$(pwd)"
ROOT="$(cd ../../.. && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-morph-emulation}"
RUN=1
for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    --arch=*) ;;  # forwarded builds use ARCH; ignored for emulation
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

if command -v nvcc >/dev/null 2>&1; then
  exec ./build_gpu_test.sh "$@"
fi

echo "nvcc not found: running host emulation parity only" >&2
echo "  (the real GPU gate is ./build_gpu_test.sh on the sm_86 box)" >&2

mkdir -p "$BUILD_DIR/cuda_stub"
cat > "$BUILD_DIR/cuda_stub/cuda_runtime.h" <<'EOF'
// Host-only CUDA runtime shim for the morph emulation test. Never used by the
// real GPU build.
#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>

using cudaError_t = int;
using cudaStream_t = void*;

enum cudaMemcpyKind {
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3,
  cudaMemcpyDefault = 4,
};

inline constexpr cudaError_t cudaSuccess = 0;
inline constexpr cudaError_t cudaErrorInvalidValue = 1;
inline constexpr cudaError_t cudaErrorMemoryAllocation = 2;
inline constexpr cudaError_t cudaErrorInvalidDevice = 3;
inline constexpr cudaError_t cudaErrorNoDevice = 4;
inline constexpr cudaError_t cudaErrorInsufficientDriver = 5;
inline constexpr cudaError_t cudaErrorInitializationError = 6;
inline constexpr cudaError_t cudaErrorNotReady = 7;
inline constexpr cudaError_t cudaErrorInvalidConfiguration = 8;
inline constexpr cudaError_t cudaErrorInvalidResourceHandle = 9;
inline constexpr cudaError_t cudaErrorLaunchOutOfResources = 10;
inline constexpr cudaError_t cudaErrorUnknown = 11;

inline const char* cudaGetErrorString(cudaError_t error) {
  switch (error) {
    case cudaSuccess: return "no error";
    case cudaErrorInvalidValue: return "invalid value";
    case cudaErrorMemoryAllocation: return "out of memory";
    default: return "cuda stub error";
  }
}

inline const char* cudaGetErrorName(cudaError_t error) {
  switch (error) {
    case cudaSuccess: return "cudaSuccess";
    case cudaErrorInvalidValue: return "cudaErrorInvalidValue";
    case cudaErrorMemoryAllocation: return "cudaErrorMemoryAllocation";
    default: return "cudaErrorUnknown";
  }
}

inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline cudaError_t cudaPeekAtLastError() { return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaStreamCreate(cudaStream_t* stream) {
  if (stream != nullptr) *stream = nullptr;
  return cudaSuccess;
}
inline cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaGetDevice(int* device) {
  if (device != nullptr) *device = 0;
  return cudaSuccess;
}
inline cudaError_t cudaSetDevice(int) { return cudaSuccess; }
inline cudaError_t cudaMemGetInfo(std::size_t* free_bytes,
                                  std::size_t* total_bytes) {
  if (free_bytes != nullptr) *free_bytes = std::size_t{1} << 30;
  if (total_bytes != nullptr) *total_bytes = std::size_t{2} << 30;
  return cudaSuccess;
}
inline cudaError_t cudaMalloc(void** pointer, std::size_t size) {
  if (pointer == nullptr) return cudaErrorInvalidValue;
  const std::size_t rounded = size == 0 ? 256 : ((size + 255) & ~std::size_t{255});
  *pointer = std::aligned_alloc(256, rounded);
  return *pointer == nullptr ? cudaErrorMemoryAllocation : cudaSuccess;
}
inline cudaError_t cudaFree(void* pointer) {
  std::free(pointer);
  return cudaSuccess;
}
inline cudaError_t cudaMemcpy(void* dst, const void* src, std::size_t size,
                              cudaMemcpyKind) {
  if ((dst == nullptr || src == nullptr) && size != 0) {
    return cudaErrorInvalidValue;
  }
  std::memcpy(dst, src, size);
  return cudaSuccess;
}
inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, std::size_t size,
                                   cudaMemcpyKind kind, cudaStream_t) {
  return cudaMemcpy(dst, src, size, kind);
}
inline cudaError_t cudaMemset(void* dst, int value, std::size_t size) {
  if (dst != nullptr) std::memset(dst, value, size);
  return cudaSuccess;
}
inline cudaError_t cudaMemsetAsync(void* dst, int value, std::size_t size,
                                   cudaStream_t) {
  return cudaMemset(dst, value, size);
}
EOF

CXX="${CXX:-c++}"
FLAGS=(-std=c++20 -O2 -g -Wall -Wextra -Wpedantic -Wconversion -Wshadow)

"$CXX" "${FLAGS[@]}" \
  -I"$BUILD_DIR/cuda_stub" -I"$ROOT/src" -I"$HERE" \
  "$HERE/morph_emulation_test.cpp" \
  "$HERE/morph_emulation.cpp" \
  "$HERE/morph_workspace.cpp" \
  "$ROOT/src/cuda/arena.cpp" \
  -o "$BUILD_DIR/morph_emulation_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/morph_emulation_test"
fi
