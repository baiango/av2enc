#!/usr/bin/env bash
# Host-emulation parity test for src/cuda/codec/inter/compound.
# Needs no GPU and no nvcc.
#
#   ./build_emulation_test.sh
#   ./build_emulation_test.sh --no-run
set -euo pipefail
cd "$(dirname "$0")"

HERE="$(pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
BUILD_DIR="${BUILD_DIR:-$ROOT/build-compound-full-emulation}"
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

CXX="${CXX:-c++}"
"$CXX" -std=c++20 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow \
  -Wno-missing-braces \
  -I"$BUILD_DIR/cuda_stub" -I"$ROOT/src" -I"$HERE" \
  "$HERE/compound_full_emulation_test.cpp" \
  "$ROOT/src/cuda/arena.cpp" \
  "$ROOT/src/codec/inter/advanced/compound.cpp" \
  "$ROOT/src/codec/inter/temporal/opfl.cpp" \
  "$ROOT/src/codec/inter/temporal/refinemv.cpp" \
  "$ROOT/src/codec/inter/mc/mc.cpp" \
  "$ROOT/src/codec/inter/multiref/compound_predict.cpp" \
  -o "$BUILD_DIR/compound_full_emulation_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/compound_full_emulation_test"
fi
