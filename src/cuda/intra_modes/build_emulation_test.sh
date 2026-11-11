#!/usr/bin/env bash
# Host-emulation test for the shared device intra predictor.
#
# The .cuh is the exact code the CUDA kernel runs. This script compiles it with
# CUDA keyword macros stubbed, one emulated thread, and compares every
# prediction byte with the CPU reference. It needs no GPU or nvcc.
#
#   ./build_emulation_test.sh            # build + run
#   ./build_emulation_test.sh --no-run
set -euo pipefail
cd "$(dirname "$0")"

ROOT="$(cd ../../.. && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-intra-emulation}"
RUN=1
for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
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
static const cudaError_t cudaErrorNotSupported = 2;
EOF

CXX="${CXX:-c++}"
"$CXX" -std=c++20 -O2 -Wall -Wextra -Wpedantic \
  -I"$BUILD_DIR/cuda_stub" -I"$ROOT/src" -I"$ROOT/src/cuda/intra_modes" \
  intra_device_emulation_test.cpp "$ROOT/src/encoder/intra_modes/intra_modes.cpp" \
  -o "$BUILD_DIR/intra_device_emulation_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/intra_device_emulation_test"
fi
