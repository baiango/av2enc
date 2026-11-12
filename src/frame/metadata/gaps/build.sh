#!/usr/bin/env bash
# Standalone build for src/frame/metadata/gaps/ (no CMake target: this agent
# owns only this directory). Reuses the shared metadata/HDR unit layer and the
# OBU framing writer.
#
# Usage: src/frame/metadata/gaps/build.sh [output-dir]
set -euo pipefail

GAPS_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
OUT=${1:-"$ROOT/build-metadata-gaps"}
CXX=${CXX:-c++}

mkdir -p "$OUT"

WARNINGS=(-Wall -Wextra -Wpedantic -Wconversion -Wshadow)
INCLUDES=(-I "$ROOT/src")
SOURCES=(
  "$ROOT/src/frame/metadata/gaps/metadata_gaps.cpp"
  "$ROOT/src/frame/metadata/gaps/reference_state.cpp"
  "$ROOT/src/frame/metadata/types/metadata_types.cpp"
  "$ROOT/src/frame/metadata/hdr/hdr_metadata.cpp"
  "$ROOT/src/core/bitstream/basic_headers.cpp"
  "$ROOT/src/core/bitstream/bitwriter.cpp"
)

"$CXX" -std=c++20 -O2 -g "${WARNINGS[@]}" "${INCLUDES[@]}" \
  "${SOURCES[@]}" "$GAPS_DIR/metadata_gaps_test.cpp" \
  -o "$OUT/metadata_gaps_test"

"$CXX" -std=c++20 -O2 -g "${WARNINGS[@]}" "${INCLUDES[@]}" \
  "${SOURCES[@]}" "$GAPS_DIR/reference_state_test.cpp" \
  -o "$OUT/reference_state_test"

"$CXX" -std=c++20 -O2 -g "${WARNINGS[@]}" "${INCLUDES[@]}" \
  "${SOURCES[@]}" "$GAPS_DIR/gap_stream_gate.cpp" \
  -o "$OUT/gap_stream_gate"

echo "built: $OUT/metadata_gaps_test"
echo "built: $OUT/reference_state_test"
echo "built: $OUT/gap_stream_gate"
