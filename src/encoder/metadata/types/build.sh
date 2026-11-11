#!/usr/bin/env bash
# Standalone build for src/encoder/metadata/types/ (no CMake target: this agent
# owns only this directory). Reuses the shared OBU framing / bit writer and the
# HDR module's metadata OBU layer.
#
# Usage: src/encoder/metadata/types/build.sh [output-dir]
set -euo pipefail

TYPES_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$TYPES_DIR/../../../.." && pwd)
OUT=${1:-"$ROOT/build-metadata-types"}
CXX=${CXX:-c++}

mkdir -p "$OUT"

WARNINGS=(-Wall -Wextra -Wpedantic -Wconversion -Wshadow)
INCLUDES=(-I "$ROOT/src")
SOURCES=(
  "$ROOT/src/encoder/metadata/types/metadata_types.cpp"
  "$ROOT/src/encoder/hdr/hdr_metadata.cpp"
  "$ROOT/src/bitstream/m0_headers.cpp"
  "$ROOT/src/bitstream/bitwriter.cpp"
)

"$CXX" -std=c++20 -O2 -g "${WARNINGS[@]}" "${INCLUDES[@]}" \
  "${SOURCES[@]}" "$TYPES_DIR/metadata_types_test.cpp" \
  -o "$OUT/metadata_types_test"

"$CXX" -std=c++20 -O2 -g "${WARNINGS[@]}" "${INCLUDES[@]}" \
  "${SOURCES[@]}" "$TYPES_DIR/append_metadata_types.cpp" \
  -o "$OUT/append_metadata_types"

echo "built: $OUT/metadata_types_test"
echo "built: $OUT/append_metadata_types"
