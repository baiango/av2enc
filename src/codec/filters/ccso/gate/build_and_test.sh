#!/usr/bin/env bash
# Build and run the temporal CCSO reuse decoder gate.
#
#   ./build_and_test.sh                 # 256x192 3-frame gate
#   QUIET=0 ./build_and_test.sh         # keep intermediates
#   DAV2D=/path/to/dav2d ./build_and_test.sh
#   SANITIZE=1 ./build_and_test.sh
#
# The M2 core is not edited.  vendor_video_syntax.py copies video_syntax.cpp into the
# build directory and injects the single §5.20.10.2 read_ccso() call.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
CXX="${CXX:-c++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
BUILD="${BUILD:-${TMPDIR:-/tmp}/av2enc-ccso-reuse-gate}"

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
       -Wno-missing-braces -ffp-contract=off -I"$ROOT/src" -I"$BUILD")
if [[ "${SANITIZE}" == "1" ]]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2)
fi
if [[ "${WERROR}" == "1" ]]; then
  FLAGS+=(-Werror -Wno-error=sign-conversion -Wno-error=missing-braces)
fi

mkdir -p "$BUILD"
python3 "$HERE/vendor_video_syntax.py" --out "$BUILD"

SOURCES=(
  "$ROOT/src/codec/filters/ccso/ccso_reuse.cpp"
  "$ROOT/src/codec/filters/loop_restoration/restoration.cpp"
  "$ROOT/src/codec/filters/stream/filters_stream.cpp"
  "$BUILD/video_syntax_ccso.cpp"
  "$ROOT/src/pipeline/video/video_syntax.cpp"
  "$ROOT/src/pipeline/video/video_headers.cpp"
  "$ROOT/src/codec/inter/merge/merge.cpp"
  "$ROOT/src/pipeline/video/post/loop_filters.cpp"
  "$ROOT/src/pipeline/video/post/tiled_frame.cpp"
  "$ROOT/src/pipeline/video/post/bit_depth_checklist.cpp"
  "$ROOT/src/pipeline/video/post/rc_adapter.cpp"
  "$ROOT/src/codec/rate/rc/rc.cpp"
  "$ROOT/src/pipeline/tiles/tile_layout.cpp"
  "$ROOT/src/pipeline/tiles/tile_group.cpp"
  "$ROOT/src/codec/filters/deblock/deblock.cpp"
  "$ROOT/src/codec/filters/cdef/cdef.cpp"
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
  "$ROOT/src/core/io/y4m.cpp"
  "$ROOT/src/core/io/ivf.cpp"
)

echo "compiler: ${CXX}"
echo "building: $BUILD/ccso_reuse_gate"
"${CXX}" "${FLAGS[@]}" "$HERE/ccso_reuse_gate.cpp" "${SOURCES[@]}" \
  -o "$BUILD/ccso_reuse_gate"

if [[ -z "${DAV2D:-}" ]]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [[ -x "$candidate" ]]; then
      export DAV2D="$candidate"
      break
    fi
  done
fi

"$BUILD/ccso_reuse_gate" --out "$BUILD/ccso_reuse_gate.ivf" "$@"
