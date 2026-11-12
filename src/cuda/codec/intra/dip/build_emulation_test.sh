#!/usr/bin/env bash
# Host-emulation test for the shared AV2 DIP device predictor.
# Needs no GPU or nvcc.
set -euo pipefail
cd "$(dirname "$0")"

HERE="$(pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
BUILD_DIR="${BUILD_DIR:-$ROOT/build-dip-emulation}"
RUN=1
for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    -h|--help) sed -n '2,5p' "$0"; exit 0 ;;
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
"$CXX" -std=c++20 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow \
  -I"$BUILD_DIR/cuda_stub" -I"$ROOT/src" -I"$HERE" \
  "$HERE/dip_emulation_test.cpp" \
  "$ROOT/src/codec/intra/dip/dip.cpp" \
  "$ROOT/src/core/io/y4m.cpp" \
  -o "$BUILD_DIR/dip_emulation_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/dip_emulation_test" "$ROOT/testdata/shine_256x256_1f.y4m"
fi
