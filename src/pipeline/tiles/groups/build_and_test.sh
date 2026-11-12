#!/usr/bin/env bash
# Build and run the M2 multi-tile-group framing tests and the dav2d gate.
#
#   ./src/pipeline/tiles/groups/build_and_test.sh
#   SANITIZE=1 ./src/pipeline/tiles/groups/build_and_test.sh
#   WERROR=1   ./src/pipeline/tiles/groups/build_and_test.sh
#   GATE=0     ./src/pipeline/tiles/groups/build_and_test.sh   # unit tests only
#   DAV2D=/path/to/dav2d ./src/pipeline/tiles/groups/build_and_test.sh
#
# Host-only CPU build. No CUDA and no CMake changes; this directory is not
# wired into the root build yet.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
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
  "$ROOT/src/codec/rate/rc/rc.cpp"
  "$ROOT/src/pipeline/tiles/tile_layout.cpp"
  "$ROOT/src/pipeline/tiles/tile_group.cpp"
  "$ROOT/src/codec/filters/deblock/deblock.cpp"
  "$ROOT/src/codec/filters/cdef/cdef.cpp"
  "$ROOT/src/codec/filters/loop_restoration/restoration.cpp"
  "$ROOT/src/codec/filters/stream/filters_stream.cpp"
  "$ROOT/src/pipeline/video/video_headers.cpp"
  "$ROOT/src/pipeline/video/video_syntax.cpp"
  "$ROOT/src/codec/inter/merge/merge.cpp"
  "$ROOT/src/pipeline/video/post/tiled_frame.cpp"
  "$ROOT/src/pipeline/video/post/bit_depth_checklist.cpp"
  "$ROOT/src/pipeline/video/post/loop_filters.cpp"
  "$ROOT/src/pipeline/video/post/rc_adapter.cpp"
  "$ROOT/src/pipeline/planner/pipeline.cpp"
  "$ROOT/src/pipeline/dispatch/planner.cpp"
  "$ROOT/src/cuda/pipeline/planner/pipeline_gpu.cpp"
  "$ROOT/src/codec/partition/tree/partition.cpp"
  "$ROOT/src/codec/intra/modes/intra_modes.cpp"
  "$ROOT/src/codec/transform/variants/tx_variants.cpp"
  "$ROOT/src/codec/transform/variants/tx_select.cpp"
  "$ROOT/src/codec/inter/me/me.cpp"
  "$ROOT/src/codec/inter/mc/mc.cpp"
  "$ROOT/src/formats/format.cpp"
  "$ROOT/src/codec/entropy/av2/av2_entropy.cpp"
  "$ROOT/src/core/bitstream/bitwriter.cpp"
  "$ROOT/src/core/bitstream/obu.cpp"
  "$ROOT/src/core/bitstream/basic_headers.cpp"
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
  "$ROOT/src/core/io/y4m.cpp" \
  "$ROOT/src/core/io/ivf.cpp" \
  -o "$BUILD/tile_groups_gate"
echo "running: $BUILD/tile_groups_gate"
set +e
"$BUILD/tile_groups_gate" --out "$BUILD/video_tile_groups_gate.ivf"
status=$?
set -e
if [[ $status -eq 77 ]]; then
  echo "multi-tile-group dav2d gate skipped (no decoder)"
  exit 0
fi
exit $status
