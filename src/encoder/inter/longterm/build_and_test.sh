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
ROOT="$(cd -- "$HERE/../../../.." && pwd)"
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
# src/encoder/m2/m2_frame.cpp currently relies on a transitive include of
# encoder/rc/rc_tables.h (owned by the M2 core agent); force it here instead of
# editing another agent's file. Same bridge the M2 CDEF gate uses.
M2_FLAGS=("${FLAGS[@]}" -Wno-sign-conversion -include
          "$ROOT/src/encoder/rc/rc_tables.h")

LONGTERM_CORE=(
  "$HERE/longterm_headers.cpp"
  "$HERE/longterm_pool.cpp"
  "$HERE/longterm_stream.cpp"
  "$ROOT/src/bitstream/bitwriter.cpp"
  "$ROOT/src/bitstream/obu.cpp"
  "$ROOT/src/bitstream/m0_headers.cpp"
)

M2_CORE=(
  "$ROOT/src/io/y4m.cpp"
  "$ROOT/src/io/ivf.cpp"
  "$ROOT/src/encoder/partition/partition.cpp"
  "$ROOT/src/encoder/intra_modes/intra_modes.cpp"
  "$ROOT/src/encoder/tx_variants/tx_variants.cpp"
  "$ROOT/src/encoder/tx_variants/tx_select.cpp"
  "$ROOT/src/entropy/av2/av2_entropy.cpp"
  "$ROOT/src/encoder/inter/me/me.cpp"
  "$ROOT/src/encoder/inter/mc/mc.cpp"
  "$ROOT/src/encoder/m2/merge/merge.cpp"
  "$ROOT/src/encoder/m2/m2_headers.cpp"
  "$ROOT/src/encoder/m2/m2_syntax.cpp"
  "$ROOT/src/encoder/m2/m2_frame.cpp"
)

echo "CXX: $($CXX --version | head -1)"
echo "build: $BUILD_DIR"

"$CXX" "${FLAGS[@]}" \
  "$HERE/longterm_test.cpp" "${LONGTERM_CORE[@]}" \
  "$ROOT/src/encoder/m2/m2_headers.cpp" \
  -o "$BUILD_DIR/longterm_test"

if [[ "$GATE" -eq 1 ]]; then
  "$CXX" "${M2_FLAGS[@]}" \
    "$HERE/longterm_gate.cpp" "${LONGTERM_CORE[@]}" "${M2_CORE[@]}" \
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
