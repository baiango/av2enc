#!/usr/bin/env bash
# Build and run the M1 intra-block-copy integration tests.
#
#   ./build_and_test.sh                 # tests + dav2d gate when available
#   ./build_and_test.sh --no-run        # build only
#   SANITIZE=1 ./build_and_test.sh      # ASan + UBSan
#   WERROR=1 ./build_and_test.sh        # warnings are errors
#   SKIP_GATE=1 ./build_and_test.sh     # tests only
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
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd ../../../.. && pwd)"
mkdir -p "$BUILD_DIR"

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wno-missing-braces)
if [ "$WERROR" -eq 1 ]; then FLAGS+=(-Werror); fi
if [ "$SANITIZE" -eq 1 ]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2)
fi

echo "CXX: $($CXX --version | head -1)"
echo "build: $BUILD_DIR/ibc_stream_test"

"$CXX" "${FLAGS[@]}" -I"$REPO_ROOT/src" \
  ibc_syntax.cpp ibc_stack.cpp m1_ibc_stream.cpp ibc_stream_test.cpp \
  "$REPO_ROOT/src/bitstream/bitwriter.cpp" \
  "$REPO_ROOT/src/bitstream/m0_headers.cpp" \
  "$REPO_ROOT/src/encoder/ibc/ibc.cpp" \
  "$REPO_ROOT/src/encoder/intra_modes/intra_modes.cpp" \
  "$REPO_ROOT/src/encoder/m1/m1_headers.cpp" \
  "$REPO_ROOT/src/encoder/partition/partition.cpp" \
  "$REPO_ROOT/src/encoder/tx_variants/tx_variants.cpp" \
  "$REPO_ROOT/src/entropy/av2/av2_entropy.cpp" \
  "$REPO_ROOT/src/io/ivf.cpp" \
  -o "$BUILD_DIR/ibc_stream_test"

if [ "$RUN" -eq 0 ]; then exit 0; fi

OUT_PREFIX="$BUILD_DIR/ibc_stream_gate"
"$BUILD_DIR/ibc_stream_test" --out-prefix "$OUT_PREFIX"

if [ "$SKIP_GATE" -eq 1 ]; then exit 0; fi

DAV2D="${DAV2D:-}"
if [ -z "$DAV2D" ]; then
  for candidate in "$REPO_ROOT/tools/bin/dav2d" \
                   "$REPO_ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$candidate" ]; then DAV2D="$candidate"; break; fi
  done
fi
if [ -z "$DAV2D" ] || [ ! -x "$DAV2D" ]; then
  echo "dav2d not found; set DAV2D=/path/to/dav2d (tests already passed)" >&2
  exit 127
fi

for variant in "" ".near" ".resid" ".small"; do
  "$DAV2D" -q -i "$OUT_PREFIX${variant}.ivf" \
    -o "$OUT_PREFIX${variant}.decoded.y4m" --threads 0
  if ! cmp -s "$OUT_PREFIX${variant}.y4m" \
              "$OUT_PREFIX${variant}.decoded.y4m"; then
    echo "FAIL: dav2d reconstruction differs for the IBC stream gate${variant}" >&2
    exit 1
  fi
done
echo "OK: dav2d decoded all M1 IBC stream variants bit-exactly ($($DAV2D --version | tail -1))"
