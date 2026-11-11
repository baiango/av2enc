#!/usr/bin/env bash
# Host-emulation parity test for the shared MHCCP device path.
# Needs no GPU or nvcc.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../../../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-mhccp-emulation}"
RUN=1
for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    -h|--help) sed -n '2,6p' "$0"; exit 0 ;;
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
EOF

CXX="${CXX:-c++}"
"$CXX" -std=c++20 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow \
  -ffp-contract=off \
  -I"$BUILD_DIR/cuda_stub" -I"$ROOT/src" -I"$HERE" \
  "$HERE/mhccp_emulation_test.cpp" \
  "$HERE/mhccp.cpp" \
  "$ROOT/src/encoder/cfl/cfl.cpp" \
  -o "$BUILD_DIR/mhccp_emulation_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/mhccp_emulation_test"
fi
