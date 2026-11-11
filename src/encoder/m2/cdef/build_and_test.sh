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
ROOT="$(cd -- "$HERE/../../../.." && pwd)"
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
# Temporary bridge: `src/encoder/m2/m2_frame.cpp` (owned by the M2 core agent)
# currently uses rc::kQlookupQ0To255 without including encoder/rc/rc_tables.h.  Force
# the include instead of editing another agent's file; drop it once the core
# header is fixed.
GATE_FLAGS=("${FLAGS[@]}" -Wno-sign-conversion -include
            "$ROOT/src/encoder/rc/rc_tables.h")
if [[ "$WERROR" == "1" ]]; then
  FLAGS+=(-Werror)
  GATE_FLAGS+=(-Werror)
fi

CDEF_CORE=(
  "$ROOT/src/encoder/m2/cdef/m2_cdef.cpp"
  "$ROOT/src/bitstream/bitwriter.cpp"
  "$ROOT/src/cuda/m1/cdef/m1_cdef.cpp"
  "$ROOT/src/encoder/filters/cdef/cdef.cpp"
  "$ROOT/src/encoder/m1/filters_stream/filters_stream.cpp"
  "$ROOT/src/entropy/av2/av2_entropy.cpp"
)

M2_CORE=(
  "$ROOT/src/io/y4m.cpp"
  "$ROOT/src/io/ivf.cpp"
  "$ROOT/src/bitstream/obu.cpp"
  "$ROOT/src/bitstream/m0_headers.cpp"
  "$ROOT/src/encoder/partition/partition.cpp"
  "$ROOT/src/encoder/intra_modes/intra_modes.cpp"
  "$ROOT/src/encoder/tx_variants/tx_variants.cpp"
  "$ROOT/src/encoder/tx_variants/tx_select.cpp"
  "$ROOT/src/encoder/inter/me/me.cpp"
  "$ROOT/src/encoder/inter/mc/mc.cpp"
  "$ROOT/src/encoder/m2/merge/merge.cpp"
  "$ROOT/src/encoder/m2/m2_headers.cpp"
  "$ROOT/src/encoder/m2/m2_syntax.cpp"
  "$ROOT/src/encoder/m2/m2_frame.cpp"
  "$ROOT/src/encoder/m2/post/tiled_frame.cpp"
  "$ROOT/src/encoder/m2/post/bit_depth_checklist.cpp"
  "$ROOT/src/encoder/m2/post/rc_adapter.cpp"
  "$ROOT/src/encoder/m2/post/loop_filters.cpp"
  "$ROOT/src/encoder/tiles/tile_layout.cpp"
  "$ROOT/src/encoder/tiles/tile_group.cpp"
  "$ROOT/src/encoder/format/format.cpp"
  "$ROOT/src/encoder/rc/rc.cpp"
  "$ROOT/src/encoder/filters/deblock/deblock.cpp"
  "$ROOT/src/encoder/filters/restoration/restoration.cpp"
  "${CDEF_CORE[@]}"
)

echo "CXX: $($CXX --version | head -1)"
echo "build: $BUILD_DIR"

"$CXX" "${FLAGS[@]}" \
  "$HERE/m2_cdef_test.cpp" "${CDEF_CORE[@]}" \
  -o "$BUILD_DIR/m2_cdef_test"

"$CXX" "${GATE_FLAGS[@]}" \
  "$HERE/m2_cdef_gate.cpp" "${M2_CORE[@]}" \
  -o "$BUILD_DIR/m2_cdef_gate"

if [[ "$RUN" -eq 1 ]]; then
  "$BUILD_DIR/m2_cdef_test" "$ROOT/decoded.y4m"
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
        INPUT="$ROOT/tests/data/m0_640x360_1f.y4m"
      fi
    fi
    "$BUILD_DIR/m2_cdef_gate" --input "$INPUT" --frames 2 \
      --out "$BUILD_DIR" --dav2d "$DAV2D"
  fi
fi
