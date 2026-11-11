#!/usr/bin/env bash
# Build/run the DC/AC delta-Q RD measurement harness.
#
#   ./build_bench.sh testdata/m0_64x64_1f.y4m
#   ./build_bench.sh testdata/shine_256x256_1f.y4m 0 --max-passes 4
set -euo pipefail
cd "$(dirname "$0")"

ROOT="$(cd ../../../.. && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-dcdeltaq}"
mkdir -p "$BUILD_DIR"

CXX="${CXX:-c++}"
"$CXX" -std=c++20 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow \
  -I"$ROOT/src" \
  dcdeltaq_bench.cpp syntax.cpp quant.cpp search.cpp \
  "$ROOT/src/bitstream/bitwriter.cpp" \
  "$ROOT/src/encoder/qm/qm.cpp" \
  "$ROOT/src/bitstream/m0_headers.cpp" \
  "$ROOT/src/encoder/tx_variants/tx_variants.cpp" \
  "$ROOT/src/io/y4m.cpp" \
  -o "$BUILD_DIR/dcdeltaq_bench"

"$BUILD_DIR/dcdeltaq_bench" "$@"
