#!/usr/bin/env bash
# Host-emulation parity test for the CUDA global-motion device core.
#
#   ./build_and_test.sh              # build + run (no GPU, no nvcc)
#   ./build_and_test.sh --no-run     # build only
#   ./build_and_test.sh --quick      # skip the real 720p pair 0/10
#   ./build_and_test.sh --asan       # AddressSanitizer/UBSan build
#   WERROR=1 ./build_and_test.sh     # promote warnings to errors
#   CXX=clang++ ./build_and_test.sh
#
# The test compiles the real shared device header (gme_device.cuh) with the
# host compiler and compares it with the CPU reference in
# src/codec/inter/gme/gme.cpp.  It also links the real av2enc::cuda::Arena
# and gme_plan.cpp against a tiny host CUDA runtime stub, so the exact byte
# budget, one-byte-short rejection, mark/rewind and staging paths are exercised
# without a GPU.  The real kernels are built by build_gpu_test.sh.
set -euo pipefail

cd "$(dirname "$0")"

HERE="$(pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
CXX="${CXX:-c++}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-gme-emulation}"
WERROR="${WERROR:-0}"
RUN=1
SANITIZE=0
RUN_ARGS=()

for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    --quick) RUN_ARGS+=(--quick) ;;
    --asan) SANITIZE=1 ;;
    -h|--help)
      sed -n '2,17p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

# Mandatory single-arena rule: this component must not allocate device memory
# on its own.  Call-shaped so prose in comments does not trip it.
if grep -nE '\b(cudaMalloc|cudaFree|cudaMallocHost|cudaHostAlloc|cudaHostRegister)[[:space:]]*\(' \
    gme.h gme_device.cuh gme_plan.cpp gme.cu; then
  echo "ERROR: direct device allocation in the component sources" >&2
  exit 1
fi

mkdir -p "$BUILD_DIR/cuda_stub"

cat > "$BUILD_DIR/cuda_stub/cuda_runtime.h" <<'EOF'
// Minimal host CUDA runtime for the no-nvcc emulation test.  cudaMalloc is
// posix_memalign(256) so Arena's base pointer is aligned exactly like
// cudaMalloc's; every "device" pointer is therefore a host pointer.
#pragma once
#include <cstddef>

typedef int cudaError_t;
typedef struct CUstream_st* cudaStream_t;

enum {
  cudaSuccess = 0,
  cudaErrorInvalidValue = 1,
  cudaErrorMemoryAllocation = 2,
  cudaErrorInvalidDevice = 3,
  cudaErrorNoDevice = 4,
  cudaErrorInsufficientDriver = 5,
  cudaErrorInitializationError = 6,
  cudaErrorInvalidConfiguration = 7,
  cudaErrorInvalidResourceHandle = 8,
  cudaErrorLaunchOutOfResources = 9,
  cudaErrorNotReady = 10,
  cudaErrorUnknown = 11
};

typedef enum cudaMemcpyKind {
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3
} cudaMemcpyKind;

cudaError_t cudaMalloc(void** ptr, std::size_t bytes);
cudaError_t cudaFree(void* ptr);
cudaError_t cudaMemcpy(void* dst, const void* src, std::size_t bytes,
                       cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void* dst, const void* src, std::size_t bytes,
                            cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemsetAsync(void* ptr, int value, std::size_t bytes,
                            cudaStream_t stream);
cudaError_t cudaMemGetInfo(std::size_t* free_bytes, std::size_t* total_bytes);
cudaError_t cudaGetDevice(int* device);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaGetLastError(void);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaDeviceSynchronize(void);
const char* cudaGetErrorString(cudaError_t error);
const char* cudaGetErrorName(cudaError_t error);
EOF

cat > "$BUILD_DIR/cuda_stub/cuda_stub.cpp" <<'EOF'
#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>

cudaError_t cudaMalloc(void** ptr, std::size_t bytes) {
  if (ptr == nullptr) return cudaErrorInvalidValue;
  void* memory = nullptr;
  if (posix_memalign(&memory, 256, bytes) != 0) return cudaErrorMemoryAllocation;
  *ptr = memory;
  return cudaSuccess;
}

cudaError_t cudaFree(void* ptr) {
  std::free(ptr);
  return cudaSuccess;
}

cudaError_t cudaMemcpy(void* dst, const void* src, std::size_t bytes,
                       cudaMemcpyKind) {
  if (bytes != 0) std::memcpy(dst, src, bytes);
  return cudaSuccess;
}

cudaError_t cudaMemcpyAsync(void* dst, const void* src, std::size_t bytes,
                            cudaMemcpyKind, cudaStream_t) {
  if (bytes != 0) std::memcpy(dst, src, bytes);
  return cudaSuccess;
}

cudaError_t cudaMemsetAsync(void* ptr, int value, std::size_t bytes,
                            cudaStream_t) {
  if (bytes != 0) std::memset(ptr, value, bytes);
  return cudaSuccess;
}

cudaError_t cudaMemGetInfo(std::size_t* free_bytes, std::size_t* total_bytes) {
  const std::size_t eight_gib = std::size_t{8} << 30;
  if (free_bytes != nullptr) *free_bytes = eight_gib;
  if (total_bytes != nullptr) *total_bytes = eight_gib;
  return cudaSuccess;
}

cudaError_t cudaGetDevice(int* device) {
  if (device != nullptr) *device = 0;
  return cudaSuccess;
}
cudaError_t cudaSetDevice(int) { return cudaSuccess; }
cudaError_t cudaGetLastError(void) { return cudaSuccess; }
cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
cudaError_t cudaDeviceSynchronize(void) { return cudaSuccess; }

const char* cudaGetErrorString(cudaError_t) { return "host-emulation stub"; }
const char* cudaGetErrorName(cudaError_t) { return "stub"; }
EOF

FLAGS=(-std=c++20 -O2 -g -ffp-contract=off -Wall -Wextra -Wpedantic
       -Wconversion -Wshadow -Wno-sign-conversion -Wno-missing-braces
       -I"$BUILD_DIR/cuda_stub" -I"$ROOT/src")
if [ "$SANITIZE" -eq 1 ]; then
  FLAGS+=(-O1 -fsanitize=address,undefined -fno-omit-frame-pointer)
fi
if [ "$WERROR" = "1" ]; then FLAGS+=(-Werror); fi

echo "compiler: $($CXX --version | head -1)"
echo "build:    $BUILD_DIR/gme_emulation_test"

"$CXX" "${FLAGS[@]}" \
  gme_emulation_test.cpp \
  gme_plan.cpp \
  "$ROOT/src/cuda/arena.cpp" \
  "$BUILD_DIR/cuda_stub/cuda_stub.cpp" \
  "$ROOT/src/codec/inter/gme/gme.cpp" \
  -o "$BUILD_DIR/gme_emulation_test"

if [ "$RUN" -eq 1 ]; then
  if [ "${#RUN_ARGS[@]}" -gt 0 ]; then
    "${BUILD_DIR}/gme_emulation_test" "$ROOT" "${RUN_ARGS[@]}"
  else
    "${BUILD_DIR}/gme_emulation_test" "$ROOT"
  fi
fi
