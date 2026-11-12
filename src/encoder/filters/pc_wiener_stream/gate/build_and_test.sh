#!/usr/bin/env bash
# Build and run the PC-Wiener decoder gate.
#
#   ./build_and_test.sh                     # 640x360 per-unit PC-Wiener
#   ./build_and_test.sh --frame-class       # frame-class translated mode
#   ./build_and_test.sh --width 1280 --height 720
#   DAV2D=/path/to/dav2d ./build_and_test.sh
#
# The M2 core is not edited.  vendor_m2_syntax_pc.py copies m2_syntax.cpp into
# the build directory and injects the single §5.20.10.4 read_lr() call; the LR
# symbols themselves come from the new pc_wiener_stream module.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STREAM_DIR="$(cd -- "${HERE}/.." && pwd)"
ROOT="$(cd -- "${HERE}/../../../../.." && pwd)"
CXX="${CXX:-c++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
BUILD="${BUILD:-${TMPDIR:-/tmp}/av2enc-pc-wiener-gate}"

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
       -Wno-missing-braces -Wno-sign-conversion -ffp-contract=off
       -I"$ROOT/src" -I"$BUILD")
if [[ "${SANITIZE}" == "1" ]]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2)
fi
if [[ "${WERROR}" == "1" ]]; then
  FLAGS+=(-Werror -Wno-error=sign-conversion -Wno-error=missing-braces)
fi

mkdir -p "$BUILD"
python3 "$HERE/vendor_m2_syntax_pc.py" --out "$BUILD"

SOURCES=(
  "$HERE/pc_wiener_gate.cpp"
  "$STREAM_DIR/pc_wiener_stream.cpp"
  "$ROOT/src/encoder/filters/spec_tables/pc_wiener.cpp"
  "$ROOT/src/encoder/filters/spec_tables/luma_source.cpp"
  "$ROOT/src/encoder/filters/restoration/restoration.cpp"
  "$ROOT/src/encoder/filters/restoration/classes/wiener_fit.cpp"
  "$ROOT/src/encoder/filters/restoration/classes/wiener_syntax.cpp"
  "$ROOT/src/encoder/filters/restoration/classes/wiener_class.cpp"
  "$ROOT/src/encoder/m1/filters_stream/filters_stream.cpp"
  "$BUILD/m2_syntax_pcw.cpp"
  "$ROOT/src/encoder/m2/m2_syntax.cpp"
  "$ROOT/src/encoder/m2/m2_headers.cpp"
  "$ROOT/src/encoder/m2/merge/merge.cpp"
  "$ROOT/src/encoder/m2/post/loop_filters.cpp"
  "$ROOT/src/encoder/m2/post/tiled_frame.cpp"
  "$ROOT/src/encoder/m2/post/bit_depth_checklist.cpp"
  "$ROOT/src/encoder/m2/post/rc_adapter.cpp"
  "$ROOT/src/encoder/rc/rc.cpp"
  "$ROOT/src/encoder/tiles/tile_layout.cpp"
  "$ROOT/src/encoder/tiles/tile_group.cpp"
  "$ROOT/src/encoder/filters/deblock/deblock.cpp"
  "$ROOT/src/encoder/filters/cdef/cdef.cpp"
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
  "$ROOT/src/io/y4m.cpp"
  "$ROOT/src/io/ivf.cpp"
)

echo "compiler: ${CXX}"
"${CXX}" "${FLAGS[@]}" "${SOURCES[@]}" -o "$BUILD/pc_wiener_gate"

if [[ -z "${DAV2D:-}" ]]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [[ -x "$candidate" ]]; then
      export DAV2D="$candidate"
      break
    fi
  done
fi
( cd "$ROOT" && "$BUILD/pc_wiener_gate" --out "$BUILD/pc_wiener_gate.ivf" "$@" )
