#!/usr/bin/env bash
# Build and run the M2 post-processing layer tests and the tiled dav2d gate.
#
#   ./src/encoder/m2/post/build_and_test.sh
#   SANITIZE=1 ./src/encoder/m2/post/build_and_test.sh
#   WERROR=1   ./src/encoder/m2/post/build_and_test.sh
#   GATE=0     ./src/encoder/m2/post/build_and_test.sh   # unit tests only
#   DAV2D=/path/to/dav2d ./src/encoder/m2/post/build_and_test.sh
#
# Host-only CPU build.  No CUDA and no CMake changes; this directory is not
# wired into the root build yet.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../../../.." && pwd)"
CXX="${CXX:-c++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
GATE="${GATE:-1}"
BUILD="${BUILD:-${TMPDIR:-/tmp}/av2enc-m2-post}"

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
       -Wno-missing-braces -ffp-contract=off -I"$ROOT/src")
if [[ "${SANITIZE}" == "1" ]]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2)
fi
if [[ "${WERROR}" == "1" ]]; then
  # The untouched M2 core has pre-existing sign-conversion warnings; keep
  # -Werror useful for this directory without editing core files.
  FLAGS+=(-Werror -Wno-error=sign-conversion)
fi

mkdir -p "$BUILD"

SOURCES=(
  "$HERE/rc_adapter.cpp"
  "$HERE/bit_depth_checklist.cpp"
  "$HERE/loop_filters.cpp"
  "$HERE/tiled_frame.cpp"
  "$ROOT/src/encoder/rc/rc.cpp"
  "$ROOT/src/encoder/tiles/tile_layout.cpp"
  "$ROOT/src/encoder/tiles/tile_group.cpp"
  "$ROOT/src/encoder/filters/deblock/deblock.cpp"
  "$ROOT/src/encoder/filters/cdef/cdef.cpp"
  "$ROOT/src/encoder/filters/restoration/restoration.cpp"
  "$ROOT/src/encoder/m1/filters_stream/filters_stream.cpp"
  "$ROOT/src/encoder/m2/m2_headers.cpp"
  "$ROOT/src/encoder/m2/m2_syntax.cpp"
  "$ROOT/src/encoder/m2/merge/merge.cpp"
  "$ROOT/src/encoder/pipeline/pipeline.cpp"
  "$ROOT/src/multi_gpu/planner.cpp"
  "$ROOT/src/cuda/pipeline/pipeline_gpu.cpp"
  "$ROOT/src/encoder/partition/partition.cpp"
  "$ROOT/src/encoder/intra_modes/intra_modes.cpp"
  "$ROOT/src/encoder/tx_variants/tx_variants.cpp"
  "$ROOT/src/encoder/tx_variants/tx_select.cpp"
  "$ROOT/src/encoder/inter/me/me.cpp"
  "$ROOT/src/encoder/inter/mc/mc.cpp"
  "$ROOT/src/encoder/format/format.cpp"
  "$ROOT/src/entropy/av2/av2_entropy.cpp"
  "$ROOT/src/bitstream/bitwriter.cpp"
  "$ROOT/src/bitstream/obu.cpp"
  "$ROOT/src/bitstream/m0_headers.cpp"
)

echo "compiler: ${CXX}"
"${CXX}" --version 2>/dev/null | head -n 1 || true
echo "flags:    ${FLAGS[*]}"
echo

echo "building: m2_post_test"
"${CXX}" "${FLAGS[@]}" \
  "$HERE/m2_post_test.cpp" \
  "${SOURCES[@]}" \
  -o "$BUILD/m2_post_test"
echo "running: $BUILD/m2_post_test"
"$BUILD/m2_post_test"
echo

if [[ "${GATE}" != "1" ]]; then
  echo "GATE=0: tiled dav2d gate skipped"
  exit 0
fi

echo "building: tiled_gate"
"${CXX}" "${FLAGS[@]}" \
  "$HERE/tiled_gate.cpp" \
  "${SOURCES[@]}" \
  "$ROOT/src/io/y4m.cpp" \
  "$ROOT/src/io/ivf.cpp" \
  -o "$BUILD/tiled_gate"
echo "running: $BUILD/tiled_gate"
set +e
"$BUILD/tiled_gate" --out "$BUILD/m2_tiled_post_gate.ivf"
status=$?
set -e
if [[ $status -eq 77 ]]; then
  echo "tiled dav2d gate skipped (no decoder)"
  exit 0
fi
exit $status
