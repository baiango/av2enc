#!/usr/bin/env bash
# Host-emulation parity test for the shared warp-stack device header.
#
# Needs no GPU and no nvcc. It compiles the same `warp_stack_device.cuh` the
# kernels use with the CUDA execution-space keywords disabled, and links the
# real `Arena` against a small host cudaMalloc shim so the sizing tests run
# locally too.
set -euo pipefail
cd "$(dirname "$0")"

HERE="$(pwd)"
ROOT="$(cd ../../.. && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-warp-stack-emulation}"
RUN=1
for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$BUILD_DIR/cuda_stub"
cat > "$BUILD_DIR/cuda_stub/cuda_runtime.h" <<'EOF'
#pragma once
// Minimal host shim: the emulation test never launches a kernel. cudaMalloc is
// backed by 256-aligned host memory so Arena's alignment arithmetic is
// exercised exactly as it is on the device side.
#include <cstddef>
#include <cstdlib>
#include <cstring>

typedef int cudaError_t;
typedef struct CUstream_st* cudaStream_t;

static const cudaError_t cudaSuccess = 0;
static const cudaError_t cudaErrorInvalidValue = 1;
static const cudaError_t cudaErrorMemoryAllocation = 2;
static const cudaError_t cudaErrorInvalidDevice = 3;
static const cudaError_t cudaErrorNoDevice = 4;
static const cudaError_t cudaErrorInsufficientDriver = 5;
static const cudaError_t cudaErrorInitializationError = 6;
static const cudaError_t cudaErrorInvalidConfiguration = 7;
static const cudaError_t cudaErrorInvalidResourceHandle = 8;
static const cudaError_t cudaErrorLaunchOutOfResources = 9;
static const cudaError_t cudaErrorNotReady = 10;

static inline const char* cudaGetErrorName(cudaError_t) { return "host-shim"; }
static inline const char* cudaGetErrorString(cudaError_t) { return "host shim"; }
static inline cudaError_t cudaGetLastError() { return cudaSuccess; }

static inline cudaError_t cudaMalloc(void** out, std::size_t bytes) {
  if (out == nullptr) return cudaErrorInvalidValue;
  void* p = nullptr;
  if (posix_memalign(&p, 256, bytes) != 0) return cudaErrorMemoryAllocation;
  *out = p;
  return cudaSuccess;
}
static inline cudaError_t cudaFree(void* p) {
  std::free(p);
  return cudaSuccess;
}
static inline cudaError_t cudaMemGetInfo(std::size_t* free_bytes, std::size_t* total_bytes) {
  if (free_bytes == nullptr || total_bytes == nullptr) return cudaErrorInvalidValue;
  *free_bytes = std::size_t{1} << 30;
  *total_bytes = std::size_t{2} << 30;
  return cudaSuccess;
}
static inline cudaError_t cudaGetDevice(int* device) {
  if (device == nullptr) return cudaErrorInvalidValue;
  *device = 0;
  return cudaSuccess;
}
static inline cudaError_t cudaSetDevice(int) { return cudaSuccess; }

enum cudaMemcpyKind {
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3,
};
static inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, std::size_t bytes,
                                          cudaMemcpyKind, cudaStream_t) {
  std::memcpy(dst, src, bytes);
  return cudaSuccess;
}
EOF

# The module must not allocate device memory outside the persistent Arena.
if grep -nE '\bcuda(Malloc|Free|MallocHost|HostAlloc)[[:space:]]*\(' \
    "$HERE"/*.h "$HERE"/*.cuh "$HERE"/*.cu "$HERE"/*.cpp 2>/dev/null; then
  echo "error: direct device allocation found; use av2enc::cuda::Arena" >&2
  exit 3
fi

CXX="${CXX:-c++}"
printf 'host: %s\n' "$($CXX --version | head -1)"
printf 'build: %s\n' "$BUILD_DIR/warp_stack_emulation_test"

"$CXX" -std=c++20 -O2 -g \
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow \
  -I "$BUILD_DIR/cuda_stub" -I "$ROOT/src" -I "$HERE" \
  "$HERE/warp_stack_emulation_test.cpp" \
  "$HERE/warp_stack_host_ref.cpp" \
  "$ROOT/src/encoder/inter/advanced/warp.cpp" \
  "$ROOT/src/encoder/inter/warp_stack/warp_stack.cpp" \
  "$ROOT/src/cuda/arena.cpp" \
  -o "$BUILD_DIR/warp_stack_emulation_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/warp_stack_emulation_test"
fi
