#!/usr/bin/env bash
# Build and run the AV2 palette intra CPU reference tests and the dav2d gate.
#
#   ./build_and_test.sh                 # build + tests + gate
#   ./build_and_test.sh --no-run        # build only
#   SANITIZE=1 ./build_and_test.sh      # ASan + UBSan
#   WERROR=1 ./build_and_test.sh        # warnings are errors
#   SKIP_GATE=1 ./build_and_test.sh     # tests only, no dav2d
#   CXX=clang++ ./build_and_test.sh
#
# Self-contained: does not touch the root CMake build, src/main.cpp, or any
# other component.  CPU-only; the CUDA kernel lives in src/cuda/palette/ and
# has its own test on the GPU box.
set -euo pipefail

cd "$(dirname "$0")"

CXX="${CXX:-c++}"
BUILD_DIR="${BUILD_DIR:-build}"
RUN=1
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
SKIP_GATE="${SKIP_GATE:-0}"

for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd ../.. && cd .. && pwd)"   # .../av2enc
mkdir -p "$BUILD_DIR"

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
if [ "$WERROR" -eq 1 ]; then FLAGS+=(-Werror); fi
if [ "$SANITIZE" -eq 1 ]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2)
fi
# m1_cdf_tables.h uses flat initialisers for 2-D rows; silence only that
# generated header's warning in the gate binary.
GATE_FLAGS=("${FLAGS[@]}" -Wno-missing-braces)

echo "CXX: $($CXX --version | head -1)"
echo "build: $BUILD_DIR/palette_test, $BUILD_DIR/palette_gate"

"$CXX" "${FLAGS[@]}" -I"$REPO_ROOT/src" \
  palette.cpp palette_test.cpp \
  "$REPO_ROOT/src/entropy/av2/av2_entropy.cpp" \
  -o "$BUILD_DIR/palette_test"

"$CXX" "${GATE_FLAGS[@]}" -I"$REPO_ROOT/src" \
  palette.cpp scc_stream.cpp palette_gate.cpp \
  "$REPO_ROOT/src/bitstream/bitwriter.cpp" \
  "$REPO_ROOT/src/bitstream/m0_headers.cpp" \
  "$REPO_ROOT/src/io/ivf.cpp" \
  "$REPO_ROOT/src/entropy/av2/av2_entropy.cpp" \
  "$REPO_ROOT/src/encoder/m1/m1_headers.cpp" \
  "$REPO_ROOT/src/encoder/intra_modes/intra_modes.cpp" \
  "$REPO_ROOT/src/encoder/partition/partition.cpp" \
  "$REPO_ROOT/src/encoder/tx_variants/tx_variants.cpp" \
  -o "$BUILD_DIR/palette_gate"

if [ "$RUN" -eq 0 ]; then exit 0; fi

"$BUILD_DIR/palette_test"

if [ "$SKIP_GATE" -eq 1 ]; then exit 0; fi

DAV2D="${DAV2D:-}"
if [ -z "$DAV2D" ]; then
  for candidate in "$REPO_ROOT/tools/bin/dav2d" "$REPO_ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$candidate" ]; then DAV2D="$candidate"; break; fi
  done
fi
if [ -z "$DAV2D" ] || [ ! -x "$DAV2D" ]; then
  echo "dav2d not found; set DAV2D=/path/to/dav2d (tests already passed)" >&2
  exit 127
fi

for mode in split single; do
  OUT="$BUILD_DIR/palette_gate_$mode.ivf"
  REF="$BUILD_DIR/palette_gate_${mode}_ref.y4m"
  DEC="$BUILD_DIR/palette_gate_${mode}_decoded.y4m"
  EXTRA=()
  if [ "$mode" = "single" ]; then EXTRA=(--single); fi
  "$BUILD_DIR/palette_gate" ${EXTRA[@]+"${EXTRA[@]}"} --out "$OUT" --ref "$REF"
  "$DAV2D" -q -i "$OUT" -o "$DEC" --threads 0
  if ! cmp -s "$REF" "$DEC"; then
    echo "FAIL: dav2d reconstruction differs for the $mode gate" >&2
    exit 1
  fi
done
echo "OK: dav2d decoded both palette gates bit-exactly ($($DAV2D --version | tail -1))"
