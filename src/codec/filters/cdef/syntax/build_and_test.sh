#!/usr/bin/env bash
# Build and run the M2 CDEF adapter tests and the dav2d stream gate.
#
#   ./build_and_test.sh                 # unit tests (+ gate when dav2d exists)
#   ./build_and_test.sh --no-gate       # unit tests only
#   ./build_and_test.sh --no-run        # build only
#   SANITIZE=1 ./build_and_test.sh
#   WERROR=1   ./build_and_test.sh
#   INPUT=testdata/shine_1280x720_60f.y4m ./build_and_test.sh
#
# No root CMake changes, no M2 core edits.  The gate links the untouched M2
# encoder, the verified filter/restoration header adapters and the M2 post
# layer's frame/tile writer.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
CXX="${CXX:-c++}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-m2-cdef}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
RUN=1
GATE=1
INPUT="${INPUT:-}"

for arg in "$@"; do
  case "$arg" in
    --no-gate) GATE=0 ;;
    --no-run) RUN=0 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$BUILD_DIR"
FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
       -Wno-missing-braces -ffp-contract=off -I"$ROOT/src")
if [[ "$SANITIZE" == "1" ]]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2 -g)
fi
# Temporary bridge: `src/pipeline/video/video_frame.cpp` (owned by the M2 core agent)
# currently uses rc::kQlookupQ0To255 without including encoder/rc/rc_tables.h.  Force
# the include instead of editing another agent's file; drop it once the core
# header is fixed.
GATE_FLAGS=("${FLAGS[@]}" -Wno-sign-conversion -include
            "$ROOT/src/codec/rate/rc/rc_tables.h")
if [[ "$WERROR" == "1" ]]; then
  FLAGS+=(-Werror)
  GATE_FLAGS+=(-Werror)
fi

CDEF_CORE=(
  "$ROOT/src/codec/filters/cdef/syntax/video_cdef.cpp"
  "$ROOT/src/core/bitstream/bitwriter.cpp"
  "$ROOT/src/cuda/pipeline/still/cdef/still_cdef.cpp"
  "$ROOT/src/codec/filters/cdef/cdef.cpp"
  "$ROOT/src/codec/filters/stream/filters_stream.cpp"
  "$ROOT/src/codec/entropy/av2/av2_entropy.cpp"
)

VIDEO_CORE=(
  "$ROOT/src/core/io/y4m.cpp"
  "$ROOT/src/core/io/ivf.cpp"
  "$ROOT/src/core/bitstream/obu.cpp"
  "$ROOT/src/core/bitstream/basic_headers.cpp"
  "$ROOT/src/codec/partition/tree/partition.cpp"
  "$ROOT/src/codec/intra/modes/intra_modes.cpp"
  "$ROOT/src/codec/transform/variants/tx_variants.cpp"
  "$ROOT/src/codec/transform/variants/tx_select.cpp"
  "$ROOT/src/codec/inter/me/me.cpp"
  "$ROOT/src/codec/inter/mc/mc.cpp"
  "$ROOT/src/codec/inter/merge/merge.cpp"
  "$ROOT/src/pipeline/video/video_headers.cpp"
  "$ROOT/src/pipeline/video/video_syntax.cpp"
  "$ROOT/src/pipeline/video/video_frame.cpp"
  "$ROOT/src/codec/inter/temporal/ref_mv_bank.cpp"
  "$ROOT/src/pipeline/video/post/tiled_frame.cpp"
  "$ROOT/src/pipeline/video/post/bit_depth_checklist.cpp"
  "$ROOT/src/pipeline/video/post/rc_adapter.cpp"
  "$ROOT/src/pipeline/video/post/loop_filters.cpp"
  "$ROOT/src/pipeline/tiles/tile_layout.cpp"
  "$ROOT/src/pipeline/tiles/tile_group.cpp"
  "$ROOT/src/formats/format.cpp"
  "$ROOT/src/codec/rate/rc/rc.cpp"
  "$ROOT/src/codec/filters/deblock/deblock.cpp"
  "$ROOT/src/codec/filters/loop_restoration/restoration.cpp"
  "${CDEF_CORE[@]}"
)

echo "CXX: $($CXX --version | head -1)"
echo "build: $BUILD_DIR"

"$CXX" "${FLAGS[@]}" \
  "$HERE/video_cdef_test.cpp" "${CDEF_CORE[@]}" \
  -o "$BUILD_DIR/video_cdef_test"

"$CXX" "${GATE_FLAGS[@]}" \
  "$HERE/video_cdef_gate.cpp" "${VIDEO_CORE[@]}" \
  -o "$BUILD_DIR/video_cdef_gate"

if [[ "$RUN" -eq 1 ]]; then
  "$BUILD_DIR/video_cdef_test" "$ROOT/decoded.y4m"
  if [[ "$GATE" -eq 1 ]]; then
    if [[ -z "${DAV2D:-}" ]]; then
      for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [[ -x "$candidate" ]]; then
          export DAV2D="$candidate"
          break
        fi
      done
    fi
    if [[ -z "${DAV2D:-}" ]]; then
      echo "no dav2d found; set DAV2D=/path/to/dav2d" >&2
      exit 77
    fi
    if [[ -z "$INPUT" ]]; then
      if [[ -f "$ROOT/testdata/shine_1280x720_60f.y4m" ]]; then
        INPUT="$ROOT/testdata/shine_1280x720_60f.y4m"
      else
        INPUT="$ROOT/tests/data/basic_640x360_1f.y4m"
      fi
    fi
    "$BUILD_DIR/video_cdef_gate" --input "$INPUT" --frames 2 \
      --out "$BUILD_DIR" --dav2d "$DAV2D"
  fi
fi
