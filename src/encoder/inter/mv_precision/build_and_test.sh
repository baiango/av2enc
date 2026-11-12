#!/usr/bin/env bash
# Build and run the MV-precision / AMVD module tests and decoder gate.
#
#   ./build_and_test.sh                 # unit tests + dav2d gate
#   ./build_and_test.sh --no-gate       # unit tests only
#   ./build_and_test.sh --no-run        # build only
#   ./build_and_test.sh --gen-goldens   # rewrite goldens/mv_precision_goldens.txt
#   SANITIZE=1 ./build_and_test.sh      # ASan + UBSan
#
# Self-contained: no root CMake, src/main.cpp, or M2 core edits.  The M2 core
# translation units are compiled read-only as dependencies of the gate.
set -euo pipefail

cd "$(dirname "$0")"

CXX="${CXX:-c++}"
BUILD_DIR="${BUILD_DIR:-build-cpu}"
SANITIZE="${SANITIZE:-0}"
RUN=1
GATE=1
GEN=0

for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    --no-gate) GATE=0 ;;
    --gen-goldens) GEN=1 ;;
    -h|--help)
      sed -n '2,11p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

ROOT="$(cd ../../../.. && pwd)"
mkdir -p "$BUILD_DIR"

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
       -Wno-missing-braces -I "$ROOT/src")
if [[ "$SANITIZE" == "1" ]]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2 -g)
fi

UNIT_DEPS=(
  mv_precision.cpp
  "$ROOT/src/entropy/av2/av2_entropy.cpp"
  "$ROOT/src/bitstream/bitwriter.cpp"
)

echo "CXX: $($CXX --version | head -1)"
echo "build: $BUILD_DIR/mv_precision_test"
"$CXX" "${FLAGS[@]}" mv_precision_test.cpp "${UNIT_DEPS[@]}" \
  -o "$BUILD_DIR/mv_precision_test"

if [[ "$GEN" -eq 1 ]]; then
  "$BUILD_DIR/mv_precision_test" --emit-goldens goldens/mv_precision_goldens.txt
fi

if [[ "$RUN" -eq 1 ]]; then
  "$BUILD_DIR/mv_precision_test" --goldens goldens/mv_precision_goldens.txt
fi

if [[ "$GATE" -eq 0 ]]; then
  exit 0
fi

if [[ -z "${DAV2D:-}" ]]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [[ -x "$candidate" ]]; then
      export DAV2D="$candidate"
      break
    fi
  done
fi

GATE_DEPS=(
  "$ROOT/src/io/y4m.cpp"
  "$ROOT/src/io/ivf.cpp"
  "$ROOT/src/bitstream/bitwriter.cpp"
  "$ROOT/src/bitstream/obu.cpp"
  "$ROOT/src/bitstream/m0_headers.cpp"
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
  mv_precision.cpp
)

echo "build: $BUILD_DIR/mv_precision_gate"
"$CXX" "${FLAGS[@]}" gate/mv_precision_gate.cpp gate/mv_precision_stream.cpp \
  gate/mv_precision_tile.cpp "${GATE_DEPS[@]}" \
  -o "$BUILD_DIR/mv_precision_gate"

if [[ -z "${DAV2D:-}" ]]; then
  echo "gate: SKIPPED (no dav2d; set DAV2D=/path/to/dav2d)"
  exit 0
fi

if [[ "$RUN" -eq 1 ]]; then
  echo "gate: AMVD + adaptive precision vs pinned dav2d"
  "$BUILD_DIR/mv_precision_gate" --out "$BUILD_DIR/mv_precision_gate.ivf"
  SRC="$ROOT/testdata/shine_1280x720_60f.y4m"
  if [[ -f "$SRC" ]]; then
    echo "gate: real 720p clip (cropped to 1280x704)"
    "$BUILD_DIR/mv_precision_gate" --input "$SRC" \
      --out "$BUILD_DIR/mv_precision_gate_720p.ivf"
  fi
fi
