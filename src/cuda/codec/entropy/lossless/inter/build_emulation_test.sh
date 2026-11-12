#!/usr/bin/env bash
# Host-emulation parity test for the shared AV2 lossless inter device header.
# Needs no GPU and no nvcc.
#
#   ./build_emulation_test.sh
#   ./build_emulation_test.sh --no-run
#   CXX=clang++ ./build_emulation_test.sh
#
# The CPU reference (src/codec/entropy/lossless/inter/lossless_inter.cpp) is compiled
# by the host compiler.  -ffunction-sections/-fdata-sections with the linker's
# section GC drops the bitstream writer/reader closure, so only the public
# reversible helpers and §5.18.2/§5.20.6.1 derivations are linked.
set -euo pipefail
cd "$(dirname "$0")"

HERE="$(pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
BUILD_DIR="${BUILD_DIR:-$ROOT/build-lossless-inter-emulation}"
RUN=1
for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$BUILD_DIR/cuda_stub"
cat > "$BUILD_DIR/cuda_stub/cuda_runtime.h" <<'EOF'
#pragma once
#include <cstddef>
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
static const cudaError_t cudaErrorNotSupported = 11;
static inline const char* cudaGetErrorName(cudaError_t) { return ""; }
static inline const char* cudaGetErrorString(cudaError_t) { return ""; }
static inline cudaError_t cudaGetLastError() { return cudaSuccess; }
EOF

if [[ "$(uname -s)" == "Darwin" ]]; then
  GC=(-Wl,-dead_strip)
else
  GC=(-Wl,--gc-sections)
fi

CXX="${CXX:-c++}"
# The CPU reference is an independent, warning-clean-at-its-own-strictness
# module; do not let its sign-conversion noise hide this component's warnings.
"$CXX" -std=c++20 -O2 -w -ffunction-sections -fdata-sections \
  -I"$ROOT/src" \
  -c "$ROOT/src/codec/entropy/lossless/inter/lossless_inter.cpp" \
  -o "$BUILD_DIR/lossless_inter_cpu.o"

"$CXX" -std=c++20 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow \
  -ffunction-sections -fdata-sections \
  -I"$BUILD_DIR/cuda_stub" -I"$ROOT/src" -I"$HERE" \
  "$HERE/lossless_inter_emulation_test.cpp" \
  "$BUILD_DIR/lossless_inter_cpu.o" \
  "${GC[@]}" \
  -o "$BUILD_DIR/lossless_inter_emulation_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/lossless_inter_emulation_test"
fi
