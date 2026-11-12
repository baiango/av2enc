#!/usr/bin/env bash
# Host-emulation parity test for the CUDA IBP device predictor.
#
# The .cuh is the exact code the CUDA kernel runs.  This script stubs the CUDA
# keywords, runs one emulated thread, and compares every byte against the CPU
# reference.  It needs no GPU and no nvcc.
#
#   ./build_emulation_test.sh            # build + run
#   ./build_emulation_test.sh --no-run
#   WERROR=1 ./build_emulation_test.sh
set -euo pipefail
cd "$(dirname "$0")"

ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
BUILD_DIR="${BUILD_DIR:-$ROOT/build-ibp-emulation}"
RUN=1
WERROR="${WERROR:-0}"
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
static const cudaError_t cudaErrorNotSupported = 2;
EOF

CXX="${CXX:-c++}"
CXXFLAGS=(-std=c++20 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
if [ "$WERROR" -eq 1 ]; then
  CXXFLAGS+=(-Werror)
fi

"$CXX" "${CXXFLAGS[@]}" \
  -I"$BUILD_DIR/cuda_stub" -I"$ROOT/src" -I"$ROOT/src/cuda/codec/intra/ibp" \
  -DAV2ENC_TESTDATA_DIR="\"$ROOT/tests/data\"" \
  ibp_device_emulation_test.cpp \
  "$ROOT/src/codec/intra/ibp/ibp.cpp" \
  "$ROOT/src/codec/intra/modes/intra_modes.cpp" \
  -o "$BUILD_DIR/ibp_device_emulation_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/ibp_device_emulation_test"
fi
