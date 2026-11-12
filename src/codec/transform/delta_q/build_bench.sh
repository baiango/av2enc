#!/usr/bin/env bash
# Build/run the DC/AC delta-Q RD measurement harness.
#
#   ./build_bench.sh testdata/basic_64x64_1f.y4m
#   ./build_bench.sh testdata/shine_256x256_1f.y4m 0 --max-passes 4
set -euo pipefail
cd "$(dirname "$0")"

ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
BUILD_DIR="${BUILD_DIR:-$ROOT/build-dcdeltaq}"
mkdir -p "$BUILD_DIR"

CXX="${CXX:-c++}"
"$CXX" -std=c++20 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow \
  -I"$ROOT/src" \
  dcdeltaq_bench.cpp syntax.cpp quant.cpp search.cpp \
  "$ROOT/src/core/bitstream/bitwriter.cpp" \
  "$ROOT/src/codec/transform/qm/qm.cpp" \
  "$ROOT/src/core/bitstream/basic_headers.cpp" \
  "$ROOT/src/codec/transform/variants/tx_variants.cpp" \
  "$ROOT/src/core/io/y4m.cpp" \
  -o "$BUILD_DIR/dcdeltaq_bench"

"$BUILD_DIR/dcdeltaq_bench" "$@"
