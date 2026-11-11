#!/usr/bin/env bash
# Build and run the multi-frame header tests and the pinned dav2d gate.
#
#   ./build_and_test.sh                  # unit tests + gate when dav2d exists
#   ./build_and_test.sh --no-gate        # unit tests only
#   ./build_and_test.sh --no-run         # build only
#   SANITIZE=1 ./build_and_test.sh
#   WERROR=1   ./build_and_test.sh
#   INPUT=some.y4m ./build_and_test.sh --frames 8
#
# Self-contained: no CMake changes, no M1/M2 core edits.  The gate links the
# untouched M2 encoder and only replaces the frame headers with the MFH-aware
# adapter in this directory.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../../../.." && pwd)"
CXX="${CXX:-c++}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-mfh}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
RUN=1
GATE=1
INPUT="${INPUT:-}"
FRAMES="${FRAMES:-8}"

for arg in "$@"; do
  case "$arg" in
    --no-gate) GATE=0 ;;
    --no-run) RUN=0 ;;
    --frames)
      if [ $# -lt 2 ]; then
        echo "--frames needs a value" >&2
        exit 2
      fi
      FRAMES=$2
      shift
      ;;
    -h|--help) sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
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
if [[ "$WERROR" == "1" ]]; then
  FLAGS+=(-Werror)
fi
# The M2 core uses rc::kQlookupQ0To255 without including rc_tables.h.  Force
# the include instead of editing another owner's file (the CDEF gate does the
# same).
GATE_FLAGS=("${FLAGS[@]}" -Wno-sign-conversion -include
            "$ROOT/src/encoder/rc/rc_tables.h")

MFH_CORE=(
  "$HERE/mfh.cpp"
  "$HERE/mfh_m2_adapter.cpp"
  "$ROOT/src/bitstream/bitwriter.cpp"
  "$ROOT/src/bitstream/m0_headers.cpp"
)

M2_CORE=(
  "$ROOT/src/io/y4m.cpp"
  "$ROOT/src/io/ivf.cpp"
  "$ROOT/src/bitstream/obu.cpp"
  "$ROOT/src/bitstream/m0_headers.cpp"
  "$ROOT/src/bitstream/bitwriter.cpp"
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
  "$ROOT/src/encoder/m1/filters_stream/filters_stream.cpp"
  "$ROOT/src/entropy/av2/av2_entropy.cpp"
  "$ROOT/src/cuda/m1/cdef/m1_cdef.cpp"
  "$ROOT/src/encoder/filters/cdef/cdef.cpp"
)

echo "CXX: $($CXX --version | head -1)"
echo "build: $BUILD_DIR"

"$CXX" "${FLAGS[@]}" \
  "$HERE/mfh_test.cpp" \
  "${MFH_CORE[@]}" \
  "$ROOT/src/encoder/m2/m2_headers.cpp" \
  -o "$BUILD_DIR/mfh_test"

if [[ "$RUN" -eq 1 ]]; then
  "$BUILD_DIR/mfh_test"
fi

if [[ "$GATE" -eq 1 ]]; then
  "$CXX" "${GATE_FLAGS[@]}" \
    "$HERE/mfh_gate.cpp" \
    "${M2_CORE[@]}" \
    "$HERE/mfh.cpp" \
    "$HERE/mfh_m2_adapter.cpp" \
    -o "$BUILD_DIR/mfh_gate"

  if [[ "$RUN" -eq 1 ]]; then
    if [[ -z "${DAV2D:-}" ]]; then
      for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [[ -x "$candidate" ]]; then
          export DAV2D="$candidate"
          break
        fi
      done
    fi
    if [[ -z "${DAV2D:-}" ]]; then
      echo "SKIP: no dav2d found; set DAV2D=/path/to/dav2d" >&2
      exit 77
    fi
    GATE_ARGS=(--frames "$FRAMES" --out "$BUILD_DIR/mfh_gate.ivf")
    if [[ -n "$INPUT" ]]; then
      GATE_ARGS=(--input "$INPUT" "${GATE_ARGS[@]}")
    fi
    "$BUILD_DIR/mfh_gate" "${GATE_ARGS[@]}"
  fi
fi
