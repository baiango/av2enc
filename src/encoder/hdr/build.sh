#!/usr/bin/env bash
# Standalone build for src/encoder/hdr/ (no CMake target: this agent owns only
# this directory). Links the shared M0 OBU framing + bit writer it reuses.
#
# Usage: src/encoder/hdr/build.sh [output-dir]
set -euo pipefail

HDR_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HDR_DIR/../../.." && pwd)
OUT=${1:-"$ROOT/build-hdr"}
CXX=${CXX:-c++}

mkdir -p "$OUT"

WARNINGS=(-Wall -Wextra -Wpedantic -Wconversion -Wshadow)
INCLUDES=(-I "$ROOT/src")
SOURCES=(
  "$ROOT/src/encoder/hdr/hdr_metadata.cpp"
  "$ROOT/src/bitstream/m0_headers.cpp"
  "$ROOT/src/bitstream/bitwriter.cpp"
)

"$CXX" -std=c++20 -O2 -g "${WARNINGS[@]}" "${INCLUDES[@]}" \
  "${SOURCES[@]}" "$ROOT/src/encoder/hdr/hdr_metadata_test.cpp" \
  -o "$OUT/hdr_metadata_test"

"$CXX" -std=c++20 -O2 -g "${WARNINGS[@]}" "${INCLUDES[@]}" \
  "${SOURCES[@]}" "$ROOT/src/encoder/hdr/append_hdr_metadata.cpp" \
  -o "$OUT/append_hdr_metadata"

echo "built: $OUT/hdr_metadata_test"
echo "built: $OUT/append_hdr_metadata"
