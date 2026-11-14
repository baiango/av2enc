#!/usr/bin/env bash
# Build and run the frame-header reference-field tests and the dav2d gate.
#
#   ./build_and_test.sh                 # unit tests + gate when dav2d exists
#   ./build_and_test.sh --no-gate       # unit tests only
#   ./build_and_test.sh --no-run        # build only
#   SANITIZE=1 ./build_and_test.sh
#   WERROR=1   ./build_and_test.sh
#   DAV2D=/path/to/dav2d ./build_and_test.sh
#
# Self-contained: no CMake changes, no M1/M2/main edits.  The gate links the
# untouched M2 encoder for its tile payloads and only replaces the sequence and
# frame headers with the adapter in this directory.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
CXX="${CXX:-c++}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-ref-fields}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
RUN=1
GATE=1

for arg in "$@"; do
  case "$arg" in
    --no-gate) GATE=0 ;;
    --no-run) RUN=0 ;;
    -h|--help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
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
# The M2 core uses rc::kQlookupQ0To255 without including rc_tables.h.  Force the
# include instead of editing another owner's file (the MFH gate does the same).
GATE_FLAGS=("${FLAGS[@]}" -Wno-sign-conversion -include
            "$ROOT/src/codec/rate/rc/rc_tables.h")

VIDEO_CORE=(
  "$ROOT/src/core/io/y4m.cpp"
  "$ROOT/src/core/io/ivf.cpp"
  "$ROOT/src/core/bitstream/obu.cpp"
  "$ROOT/src/core/bitstream/basic_headers.cpp"
  "$ROOT/src/core/bitstream/bitwriter.cpp"
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
  "$ROOT/src/codec/filters/stream/filters_stream.cpp"
  "$ROOT/src/codec/entropy/av2/av2_entropy.cpp"
  "$ROOT/src/cuda/pipeline/still/cdef/still_cdef.cpp"
  "$ROOT/src/codec/filters/cdef/cdef.cpp"
)

echo "CXX: $($CXX --version | head -1)"
echo "build: $BUILD_DIR"

"$CXX" "${FLAGS[@]}" \
  "$HERE/ref_fields_test.cpp" \
  "$HERE/ref_fields.cpp" \
  "$HERE/ref_fields_video_adapter.cpp" \
  "$ROOT/src/core/bitstream/bitwriter.cpp" \
  "$ROOT/src/core/bitstream/basic_headers.cpp" \
  "$ROOT/src/pipeline/video/video_headers.cpp" \
  -o "$BUILD_DIR/ref_fields_test"

if [[ "$RUN" -eq 1 ]]; then
  "$BUILD_DIR/ref_fields_test"
fi

if [[ "$GATE" -eq 1 ]]; then
  "$CXX" "${GATE_FLAGS[@]}" \
    "$HERE/ref_fields_gate.cpp" \
    "$HERE/ref_fields.cpp" \
    "$HERE/ref_fields_video_adapter.cpp" \
    "${VIDEO_CORE[@]}" \
    -o "$BUILD_DIR/ref_fields_gate"

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
    "$BUILD_DIR/ref_fields_gate" --out "$BUILD_DIR/ref_fields_gate.ivf" \
      --dav2d "$DAV2D"
  fi
fi
