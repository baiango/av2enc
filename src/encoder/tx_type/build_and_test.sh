#!/usr/bin/env bash
# Build and run the AV2 intra transform-type CPU reference test.
#
#   ./build_and_test.sh                 # -O2 build + run
#   ./build_and_test.sh --no-run        # build only
#   SANITIZE=1 ./build_and_test.sh      # ASan + UBSan
#   WERROR=1 ./build_and_test.sh        # promote warnings to errors
#   CXX=clang++ ./build_and_test.sh     # alternate compiler
#
# The test links the real reference implementations it depends on:
#   src/encoder/tx_variants/   transforms / quantisation
#   src/encoder/intra_modes/   DC prediction and wide_angle_mapping
#   src/encoder/rdo/           the shared QP->lambda helpers
#   src/io/                    the Y4M reader for the real-frame cases
#
# Self-contained: it does not touch the root CMake build, and no artifacts are
# written into the repository.
set -euo pipefail

cd "$(dirname "$0")"

CXX="${CXX:-c++}"
REPO="$(cd ../../.. && pwd)"
BUILD_DIR="${BUILD_DIR:-${TMPDIR:-/tmp}/av2enc_tx_type_build}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
RUN=1

for arg in "$@"; do
  case "$arg" in
    --no-run)
      RUN=0
      ;;
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$BUILD_DIR"

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
       -ffp-contract=off)
if [ "$WERROR" = "1" ]; then
  FLAGS+=(-Werror)
fi
if [ "$SANITIZE" = "1" ]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2)
fi

echo "compiler: $($CXX --version | head -1)"
echo "build:    $BUILD_DIR/tx_type_test"

"$CXX" "${FLAGS[@]}" -I"$REPO/src" \
  tx_type.cpp tx_type_test.cpp \
  "$REPO/src/encoder/tx_variants/tx_variants.cpp" \
  "$REPO/src/encoder/intra_modes/intra_modes.cpp" \
  "$REPO/src/encoder/rdo/rdo.cpp" \
  "$REPO/src/io/y4m.cpp" \
  -o "$BUILD_DIR/tx_type_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/tx_type_test" --data "$REPO/tests/data"
fi
