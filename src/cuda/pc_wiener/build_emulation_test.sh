#!/usr/bin/env bash
# Host-emulation parity test for the shared AV2 PC-Wiener device code.
# Needs no GPU and no nvcc.
set -euo pipefail
cd "$(dirname "$0")"

HERE="$(pwd)"
ROOT="$(cd ../../.. && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-pc-wiener-emulation}"
RUN=1
for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    -h|--help) sed -n '2,4p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$BUILD_DIR"
CXX="${CXX:-c++}"
"$CXX" -std=c++20 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow \
  -Wno-sign-conversion \
  -I"$ROOT/src" -I"$HERE" \
  "$HERE/pc_wiener_emulation_test.cpp" \
  "$ROOT/src/encoder/filters/spec_tables/pc_wiener.cpp" \
  "$ROOT/src/encoder/filters/spec_tables/luma_source.cpp" \
  -o "$BUILD_DIR/pc_wiener_emulation_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/pc_wiener_emulation_test"
fi
