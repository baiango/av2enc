#!/usr/bin/env bash
# Build and run the AV2 secondary-transform (IST) CPU reference test.
#
#   ./build_and_test.sh                 # -O2 build + run
#   ./build_and_test.sh --no-run        # build only
#   SANITIZE=1 ./build_and_test.sh      # ASan + UBSan
#   WERROR=1 ./build_and_test.sh        # promote warnings to errors
#   CXX=clang++ ./build_and_test.sh
#
# Self-contained: no root CMake edit, no artifacts in the repository.
set -euo pipefail
cd "$(dirname "$0")"

CXX="${CXX:-c++}"
REPO="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${REPO}/CMakeLists.txt" ] && [ "${REPO}" != "/" ]; do
    REPO="$(dirname -- "${REPO}")"
done
BUILD_DIR="${BUILD_DIR:-${TMPDIR:-/tmp}/av2enc_tx_secondary_build}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
RUN=1

for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$BUILD_DIR"
FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
       -ffp-contract=off)
if [ "$WERROR" = "1" ]; then FLAGS+=(-Werror); fi
if [ "$SANITIZE" = "1" ]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2)
fi

echo "compiler: $($CXX --version | head -1)"
echo "build:    $BUILD_DIR/tx_secondary_test"

"$CXX" "${FLAGS[@]}" -I"$REPO/src" \
  tx_secondary.cpp tx_secondary_test.cpp \
  "$REPO/src/codec/transform/variants/tx_variants.cpp" \
  "$REPO/src/core/bitstream/bitwriter.cpp" \
  -o "$BUILD_DIR/tx_secondary_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/tx_secondary_test"
fi
