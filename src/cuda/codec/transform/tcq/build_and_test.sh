#!/usr/bin/env bash
# Build and run the CUDA TCQ / CCTX / parity-hiding component.
#
#   ./build_and_test.sh                 # host-emulation parity, then GPU if nvcc
#   ./build_and_test.sh --no-run
#   WERROR=1 ./build_and_test.sh
#   ARCH=sm_120 ./build_gpu_test.sh
#
# The emulation test needs no CUDA toolkit and no GPU: it compiles the exact
# shared device core (tcq_device.cuh) as host C++ and compares it bit for bit
# with src/codec/transform/tcq/.  When nvcc is present the real one-thread-per-block
# kernel test is built and run too.
set -euo pipefail
cd "$(dirname "$0")"

HERE="$(pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
BUILD_DIR="${BUILD_DIR:-${TMPDIR:-/tmp}/av2enc_tcq_cuda_emulation}"
CXX="${CXX:-c++}"
WERROR="${WERROR:-0}"
RUN=1
for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$BUILD_DIR/cuda_stub"
cat > "$BUILD_DIR/cuda_stub/cuda_runtime.h" <<'EOF'
#pragma once
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
static const cudaError_t cudaErrorNotReady = 7;
static const cudaError_t cudaErrorInvalidConfiguration = 8;
static const cudaError_t cudaErrorInvalidResourceHandle = 9;
static const cudaError_t cudaErrorLaunchOutOfResources = 10;
inline const char* cudaGetErrorName(cudaError_t) { return "stub"; }
inline const char* cudaGetErrorString(cudaError_t) { return "stub"; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline cudaError_t cudaMalloc(void** p, std::size_t n) {
  *p = std::malloc(n);
  return *p != nullptr ? cudaSuccess : cudaErrorMemoryAllocation;
}
inline cudaError_t cudaFree(void* p) { std::free(p); return cudaSuccess; }
inline cudaError_t cudaMemGetInfo(std::size_t* f, std::size_t* t) {
  *f = static_cast<std::size_t>(1) << 30;
  *t = *f;
  return cudaSuccess;
}
inline cudaError_t cudaGetDevice(int* d) { *d = 0; return cudaSuccess; }
inline cudaError_t cudaSetDevice(int) { return cudaSuccess; }
EOF

FLAGS=(-std=c++20 -O2 -ffp-contract=off -Wall -Wextra -Wpedantic
       -Wconversion -Wshadow -I"$BUILD_DIR/cuda_stub" -I"$ROOT/src" -I"$HERE")
if [ "$WERROR" = "1" ]; then FLAGS+=(-Werror); fi

echo "compiler: $("$CXX" --version | head -1)"
echo "build:    $BUILD_DIR/tcq_emulation_test"
"$CXX" "${FLAGS[@]}" \
  "$HERE/tcq_emulation_test.cpp" \
  "$HERE/tcq_host.cpp" \
  "$ROOT/src/cuda/arena.cpp" \
  "$ROOT/src/codec/transform/tcq/tcq.cpp" \
  "$ROOT/src/codec/transform/tcq/cctx.cpp" \
  "$ROOT/src/codec/transform/tcq/parity/parity_hiding.cpp" \
  -o "$BUILD_DIR/tcq_emulation_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/tcq_emulation_test"
fi

if command -v nvcc >/dev/null 2>&1; then
  ./build_gpu_test.sh "$@"
else
  echo "nvcc not found: skipped the real GPU test (host-emulation parity passed)."
fi
