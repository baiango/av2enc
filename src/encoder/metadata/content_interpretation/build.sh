#!/usr/bin/env bash
# Standalone build for src/encoder/metadata/content_interpretation/.
#
# This agent owns only this directory, so there is no CMake target. The module
# reuses the shared Annex B.2 OBU framing and MSB-first bit writer from
# src/bitstream/ (the same helpers the M1 temporal unit uses).
#
# Usage: src/encoder/metadata/content_interpretation/build.sh [output-dir]
set -euo pipefail

CI_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$CI_DIR/../../../.." && pwd)
OUT=${1:-"$ROOT/build-ci"}
CXX=${CXX:-c++}

mkdir -p "$OUT"

WARNINGS=(-Wall -Wextra -Wpedantic -Wconversion -Wshadow)
INCLUDES=(-I "$ROOT/src")
SOURCES=(
  "$CI_DIR/content_interpretation.cpp"
  "$ROOT/src/bitstream/m0_headers.cpp"
  "$ROOT/src/bitstream/bitwriter.cpp"
)

"$CXX" -std=c++20 -O2 -g "${WARNINGS[@]}" "${INCLUDES[@]}" \
  "${SOURCES[@]}" "$CI_DIR/content_interpretation_test.cpp" \
  -o "$OUT/content_interpretation_test"

"$CXX" -std=c++20 -O2 -g "${WARNINGS[@]}" "${INCLUDES[@]}" \
  "${SOURCES[@]}" "$CI_DIR/append_content_interpretation.cpp" \
  -o "$OUT/append_content_interpretation"

echo "built: $OUT/content_interpretation_test"
echo "built: $OUT/append_content_interpretation"
