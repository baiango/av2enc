#!/usr/bin/env bash
# Host-emulation parity test for the shared AV2 TIP device header.
# Needs no GPU and no nvcc.
#
#   ./build_emulation_test.sh
#   ./build_emulation_test.sh --no-run
#   CXX=clang++ ./build_emulation_test.sh
set -euo pipefail
cd "$(dirname "$0")"

HERE="$(pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
BUILD_DIR="${BUILD_DIR:-$ROOT/build-tip-emulation}"
RUN=1
for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
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
inline const char* cudaGetErrorName(cudaError_t) { return "stub"; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline const char* cudaGetErrorString(cudaError_t) { return "stub"; }
EOF

CXX="${CXX:-c++}"
echo "CXX: $($CXX --version | head -1)"
echo "build: $BUILD_DIR/tip_emulation_test"
"$CXX" -std=c++20 -O2 -g -Wall -Wextra -Wpedantic -Wconversion -Wshadow \
  -Wno-sign-conversion -ffp-contract=off \
  -I"$BUILD_DIR/cuda_stub" -I"$ROOT/src" -I"$HERE" \
  "$HERE/tip_emulation_test.cpp" \
  "$ROOT/src/codec/inter/temporal/motion_field.cpp" \
  "$ROOT/src/codec/inter/temporal/temporal_common.cpp" \
  "$ROOT/src/codec/inter/temporal/tip.cpp" \
  -o "$BUILD_DIR/tip_emulation_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/tip_emulation_test"
fi
