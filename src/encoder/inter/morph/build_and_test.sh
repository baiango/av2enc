#!/usr/bin/env bash
# Build and run the morph_pred syntax + §7.13.3.26 prediction tests.
#
#   ./build_and_test.sh                 # tests + dav2d --strict 1 --threads 1 gate
#   ./build_and_test.sh --no-run        # build only
#   SANITIZE=1 ./build_and_test.sh      # ASan + UBSan
#   WERROR=1 ./build_and_test.sh        # warnings are errors
#   SKIP_GATE=1 ./build_and_test.sh     # tests only
#
# The gate streams are pinned:
#   morph_gate.ivf      MD5 b9246821ff00a2948011af36d0737b4e  (99 B)
#   morph_gate.off.ivf  MD5 6a633e910baa7fa070a6eb23c1fc1b3a  (95 B)
# Both must decode bit-exactly, and the two decoded Y4Ms must differ: that is
# the proof that dav2d actually runs the morphological prediction rather than
# silently treating morph_pred as 0.
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
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd ../../../.. && pwd)"
mkdir -p "$BUILD_DIR"

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
       -Wno-missing-braces -Wno-sign-conversion)
if [ "$WERROR" -eq 1 ]; then FLAGS+=(-Werror); fi
if [ "$SANITIZE" -eq 1 ]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2)
fi

echo "CXX: $($CXX --version | head -1)"
echo "build: $BUILD_DIR/morph_test"

"$CXX" "${FLAGS[@]}" -I"$REPO_ROOT/src" \
  morph_syntax.cpp morph_stream.cpp morph_test.cpp \
  "$REPO_ROOT/src/bitstream/bitwriter.cpp" \
  "$REPO_ROOT/src/bitstream/m0_headers.cpp" \
  "$REPO_ROOT/src/encoder/ibc/ibc.cpp" \
  "$REPO_ROOT/src/encoder/intra_modes/intra_modes.cpp" \
  "$REPO_ROOT/src/encoder/partition/partition.cpp" \
  "$REPO_ROOT/src/encoder/tx_variants/tx_variants.cpp" \
  "$REPO_ROOT/src/entropy/av2/av2_entropy.cpp" \
  "$REPO_ROOT/src/io/ivf.cpp" \
  "$REPO_ROOT/src/encoder/m1/ibc_stream/ibc_syntax.cpp" \
  "$REPO_ROOT/src/encoder/m1/ibc_stream/ibc_stack.cpp" \
  -o "$BUILD_DIR/morph_test"

if [ "$RUN" -eq 0 ]; then exit 0; fi

OUT_PREFIX="$BUILD_DIR/morph_gate"
"$BUILD_DIR/morph_test" --out-prefix "$OUT_PREFIX"

if [ "$SKIP_GATE" -eq 1 ]; then exit 0; fi

md5_of() {
  if command -v md5 >/dev/null 2>&1; then
    md5 -q "$1"
  else
    md5sum "$1" | awk '{print $1}'
  fi
}

check_md5() {
  local file="$1"
  local want="$2"
  local got
  got="$(md5_of "$file")"
  if [ "$got" != "$want" ]; then
    echo "FAIL: $file MD5 $got, pinned $want" >&2
    exit 1
  fi
}

check_md5 "$OUT_PREFIX.ivf" "b9246821ff00a2948011af36d0737b4e"
check_md5 "$OUT_PREFIX.off.ivf" "6a633e910baa7fa070a6eb23c1fc1b3a"

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

for variant in "" ".off"; do
  "$DAV2D" -q -i "$OUT_PREFIX${variant}.ivf" \
    -o "$OUT_PREFIX${variant}.decoded.y4m" --threads 1 --strict 1
  if ! cmp -s "$OUT_PREFIX${variant}.y4m" \
              "$OUT_PREFIX${variant}.decoded.y4m"; then
    echo "FAIL: dav2d reconstruction differs for morph gate${variant}" >&2
    exit 1
  fi
done

if cmp -s "$OUT_PREFIX.decoded.y4m" "$OUT_PREFIX.off.decoded.y4m"; then
  echo "FAIL: morph on/off decoded identically; morph_pred was not applied" >&2
  exit 1
fi

VERSION="$($DAV2D --version 2>&1 | head -1)"
echo "OK: dav2d --strict 1 --threads 1 decoded both morph streams bit-exactly"
echo "    and the on/off reconstructions differ ($VERSION)"
