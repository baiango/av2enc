#!/usr/bin/env bash
# Host-emulation parity test for the AV2 §7.20.5 GDF device core.
#
#   ./build_and_test.sh                # build + run (no GPU, no nvcc)
#   ./build_and_test.sh --no-run       # build only
#   ./build_and_test.sh --asan         # AddressSanitizer/UBSan build
#
# The shared device header (gdf_device.cuh) is compiled by the host compiler
# and compared with the CPU reference in
# src/codec/filters/tables/gdf.cpp.  This is the same header the CUDA
# kernel includes; it contains no CUDA-specific code.
set -euo pipefail

cd "$(dirname "$0")"

HERE="$(pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
BUILD_DIR="${BUILD_DIR:-$ROOT/build-gdf-emulation}"
CXX="${CXX:-c++}"
RUN=1
SANITIZE=0

for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    --asan) SANITIZE=1 ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$BUILD_DIR"

FLAGS=(-std=c++20 -O2 -g -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
if [ "$SANITIZE" -eq 1 ]; then
  FLAGS+=(-fsanitize=address,undefined -fno-omit-frame-pointer)
fi

echo "host: $($CXX --version | head -1)"
echo "build: $BUILD_DIR/gdf_emulation_test"

"$CXX" "${FLAGS[@]}" \
  -I "$ROOT/src" -I "$HERE" \
  "$HERE/gdf_emulation_test.cpp" \
  "$ROOT/src/codec/filters/tables/gdf.cpp" \
  "$ROOT/src/codec/filters/tables/luma_source.cpp" \
  -o "$BUILD_DIR/gdf_emulation_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/gdf_emulation_test"
fi
