#!/usr/bin/env bash
# Host-emulation parity test for the AV2 temporal motion-field device core.
#
#   ./build_and_test.sh              # build + run (no GPU, no nvcc)
#   ./build_and_test.sh --no-run     # build only
#   ./build_and_test.sh --asan       # AddressSanitizer/UBSan build
#
# The shared header (motion_field_device.cuh) is compiled by the host compiler
# with a CUDA stub and compared against the CPU temporal reference
# (src/encoder/inter/temporal/{motion_field,ref_mv_bank,temporal_common}.cpp).
# It is the same header the CUDA kernels include.
set -euo pipefail

cd "$(dirname "$0")"

HERE="$(pwd)"
ROOT="$(cd ../../.. && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-motion-field-emulation}"
CXX="${CXX:-c++}"
RUN=1
SANITIZE=0

for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    --asan) SANITIZE=1 ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
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
inline const char* cudaGetErrorName(cudaError_t) { return "stub"; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline const char* cudaGetErrorString(cudaError_t) { return "stub"; }
EOF

FLAGS=(-std=c++20 -O2 -g -Wall -Wextra -Wpedantic -Wconversion -Wshadow
       -Wno-sign-conversion -ffp-contract=off)
if [ "$SANITIZE" -eq 1 ]; then
  FLAGS+=(-fsanitize=address,undefined -fno-omit-frame-pointer)
fi

echo "host: $($CXX --version | head -1)"
echo "build: $BUILD_DIR/motion_field_emulation_test"

"$CXX" "${FLAGS[@]}" \
  -I "$BUILD_DIR/cuda_stub" -I "$ROOT/src" -I "$HERE" \
  "$HERE/motion_field_emulation_test.cpp" \
  "$ROOT/src/encoder/inter/temporal/motion_field.cpp" \
  "$ROOT/src/encoder/inter/temporal/ref_mv_bank.cpp" \
  "$ROOT/src/encoder/inter/temporal/temporal_common.cpp" \
  -o "$BUILD_DIR/motion_field_emulation_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/motion_field_emulation_test"
fi
