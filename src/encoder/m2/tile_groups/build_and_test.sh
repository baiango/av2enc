#!/usr/bin/env bash
# Build and run the M2 multi-tile-group framing tests and the dav2d gate.
#
#   ./src/encoder/m2/tile_groups/build_and_test.sh
#   SANITIZE=1 ./src/encoder/m2/tile_groups/build_and_test.sh
#   WERROR=1   ./src/encoder/m2/tile_groups/build_and_test.sh
#   GATE=0     ./src/encoder/m2/tile_groups/build_and_test.sh   # unit tests only
#   DAV2D=/path/to/dav2d ./src/encoder/m2/tile_groups/build_and_test.sh
#
# Host-only CPU build. No CUDA and no CMake changes; this directory is not
# wired into the root build yet.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../../../.." && pwd)"
CXX="${CXX:-c++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
GATE="${GATE:-1}"
BUILD="${BUILD:-${TMPDIR:-/tmp}/av2enc-m2-tile-groups}"

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
       -Wno-missing-braces -ffp-contract=off -I"$ROOT/src")
if [[ "${SANITIZE}" == "1" ]]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2)
fi
if [[ "${WERROR}" == "1" ]]; then
  # The M2 core has pre-existing sign-conversion warnings; keep -Werror useful
  # for this directory without editing core files.
  FLAGS+=(-Werror -Wno-error=sign-conversion)
fi

mkdir -p "$BUILD"

SOURCES=(
  "$HERE/tile_groups.cpp"
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
  "$ROOT/src/encoder/m2/post/tiled_frame.cpp"
  "$ROOT/src/encoder/m2/post/bit_depth_checklist.cpp"
  "$ROOT/src/encoder/m2/post/loop_filters.cpp"
  "$ROOT/src/encoder/m2/post/rc_adapter.cpp"
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

echo "building: tile_groups_test"
"${CXX}" "${FLAGS[@]}" \
  -DAV2ENC_TILE_GROUP_GOLDEN_DIR="\"$HERE/goldens\"" \
  "$HERE/tile_groups_test.cpp" \
  "${SOURCES[@]}" \
  -o "$BUILD/tile_groups_test"
echo "running: $BUILD/tile_groups_test"
"$BUILD/tile_groups_test"
echo

if [[ "${GATE}" != "1" ]]; then
  echo "GATE=0: dav2d multi-tile-group gate skipped"
  exit 0
fi

echo "building: tile_groups_gate"
"${CXX}" "${FLAGS[@]}" \
  "$HERE/tile_groups_gate.cpp" \
  "${SOURCES[@]}" \
  "$ROOT/src/io/y4m.cpp" \
  "$ROOT/src/io/ivf.cpp" \
  -o "$BUILD/tile_groups_gate"
echo "running: $BUILD/tile_groups_gate"
set +e
"$BUILD/tile_groups_gate" --out "$BUILD/m2_tile_groups_gate.ivf"
status=$?
set -e
if [[ $status -eq 77 ]]; then
  echo "multi-tile-group dav2d gate skipped (no decoder)"
  exit 0
fi
exit $status
