#!/usr/bin/env bash
# Host-emulation parity test for the shared AV2 chroma-LR device code.
# Needs no GPU and no nvcc.
#
#   ./build_emulation_test.sh
#   ./build_emulation_test.sh --no-run
set -euo pipefail
cd "$(dirname "$0")"

HERE="$(pwd)"
ROOT="$(cd ../../.. && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-lr-chroma-emulation}"
RUN=1
for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    -h|--help) sed -n '2,6p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$BUILD_DIR"
CXX="${CXX:-c++}"

CORE=(
  "$ROOT/src/encoder/filters/lr_chroma/chroma_lr.cpp"
  "$ROOT/src/encoder/filters/restoration/restoration.cpp"
  "$ROOT/src/encoder/filters/restoration/classes/wiener_fit.cpp"
  "$ROOT/src/encoder/filters/restoration/classes/wiener_syntax.cpp"
  "$ROOT/src/encoder/filters/restoration/classes/wiener_class.cpp"
  "$ROOT/src/encoder/m1/filters_stream/filters_stream.cpp"
  "$ROOT/src/entropy/av2/av2_entropy.cpp"
  "$ROOT/src/bitstream/bitwriter.cpp"
)

"$CXX" -std=c++20 -O2 -g \
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wno-sign-conversion \
  -Wno-missing-braces \
  -I"$ROOT/src" -I"$HERE" \
  "$HERE/lr_chroma_emulation_test.cpp" "${CORE[@]}" \
  -o "$BUILD_DIR/lr_chroma_emulation_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/lr_chroma_emulation_test"
fi
