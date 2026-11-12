#!/usr/bin/env bash
# Host-emulation parity test for the shared AV2 §7.16 TIP deblocking core.
# Needs no GPU and no nvcc.
#
#   ./build_emulation_test.sh            # build + run
#   ./build_emulation_test.sh --no-run   # build only
set -euo pipefail
cd "$(dirname "$0")"

HERE="$(pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
BUILD_DIR="${BUILD_DIR:-$ROOT/build-tip-deblock-emulation}"
CXX="${CXX:-c++}"
RUN=1
for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    -h|--help) sed -n '2,6p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$BUILD_DIR"

echo "CXX: $($CXX --version | head -1)"
echo "build: $BUILD_DIR/tip_deblock_emulation_test"

"$CXX" -std=c++20 -O2 -g \
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow \
  -I"$ROOT/src" -I"$HERE" \
  "$HERE/tip_deblock_emulation_test.cpp" \
  "$ROOT/src/codec/filters/tip_deblock/tip_deblock.cpp" \
  -o "$BUILD_DIR/tip_deblock_emulation_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/tip_deblock_emulation_test" "$ROOT"
fi
