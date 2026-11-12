#!/usr/bin/env bash
# Build and run the long-term-reference / open-random-access layer.
#
#   ./build_and_test.sh                 # unit tests + dav2d gate
#   ./build_and_test.sh --no-gate       # unit tests only
#   ./build_and_test.sh --no-run        # build only
#   SANITIZE=1 ./build_and_test.sh
#   WERROR=1   ./build_and_test.sh
#   INPUT=testdata/shine_256x256_1f.y4m ./build_and_test.sh
#
# The gate links the untouched M2 encoder (src/encoder/m2/) and wraps its tile
# payloads in this layer's headers. No CMake changes, no M2 core edits.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
CXX="${CXX:-c++}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-longterm}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
RUN=1
GATE=1
INPUT="${INPUT:-}"

for arg in "$@"; do
  case "$arg" in
    --no-gate) GATE=0 ;;
    --no-run) RUN=0 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
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
# src/pipeline/video/video_frame.cpp currently relies on a transitive include of
# codec/rate/rc/rc_tables.h (owned by the M2 core agent); force it here instead of
# editing another agent's file. Same bridge the M2 CDEF gate uses.
VIDEO_FLAGS=("${FLAGS[@]}" -Wno-sign-conversion -include
          "$ROOT/src/codec/rate/rc/rc_tables.h")

LONGTERM_CORE=(
  "$HERE/longterm_headers.cpp"
  "$HERE/longterm_pool.cpp"
  "$HERE/longterm_stream.cpp"
  "$ROOT/src/core/bitstream/bitwriter.cpp"
  "$ROOT/src/core/bitstream/obu.cpp"
  "$ROOT/src/core/bitstream/basic_headers.cpp"
)

VIDEO_CORE=(
  "$ROOT/src/core/io/y4m.cpp"
  "$ROOT/src/core/io/ivf.cpp"
  "$ROOT/src/codec/partition/tree/partition.cpp"
  "$ROOT/src/codec/intra/modes/intra_modes.cpp"
  "$ROOT/src/codec/transform/variants/tx_variants.cpp"
  "$ROOT/src/codec/transform/variants/tx_select.cpp"
  "$ROOT/src/codec/entropy/av2/av2_entropy.cpp"
  "$ROOT/src/codec/inter/me/me.cpp"
  "$ROOT/src/codec/inter/mc/mc.cpp"
  "$ROOT/src/codec/inter/merge/merge.cpp"
  "$ROOT/src/codec/inter/drl_reorder/drl_reorder.cpp"
  "$ROOT/src/codec/inter/drl_reorder/video_stack_adapter.cpp"
  "$ROOT/src/codec/inter/mv_precision/mv_precision.cpp"
  "$ROOT/src/pipeline/video/video_headers.cpp"
  "$ROOT/src/pipeline/video/video_syntax.cpp"
  "$ROOT/src/pipeline/video/video_frame.cpp"
)

echo "CXX: $($CXX --version | head -1)"
echo "build: $BUILD_DIR"

"$CXX" "${FLAGS[@]}" \
  "$HERE/longterm_test.cpp" "${LONGTERM_CORE[@]}" \
  "$ROOT/src/pipeline/video/video_headers.cpp" \
  -o "$BUILD_DIR/longterm_test"

if [[ "$GATE" -eq 1 ]]; then
  "$CXX" "${VIDEO_FLAGS[@]}" \
    "$HERE/longterm_gate.cpp" "${LONGTERM_CORE[@]}" "${VIDEO_CORE[@]}" \
    -o "$BUILD_DIR/longterm_gate"
fi

if [[ "$RUN" -eq 1 ]]; then
  "$BUILD_DIR/longterm_test"
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
    if [[ -n "$INPUT" ]]; then
      "$BUILD_DIR/longterm_gate" --input "$INPUT" --frames 5 \
        --out "$BUILD_DIR" --dav2d "$DAV2D"
    else
      "$BUILD_DIR/longterm_gate" --frames 5 --out "$BUILD_DIR" --dav2d "$DAV2D"
    fi
  fi
fi
