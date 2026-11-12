#!/usr/bin/env bash
# Build (and by default run) the AV2 intra prediction CPU reference test.
#
#   ./build_test.sh                 # -O2 build + run
#   ./build_test.sh --no-run        # build only
#   SANITIZE=1 ./build_test.sh      # ASan + UBSan build + run
#   WERROR=1 ./build_test.sh        # promote warnings to errors
#   CXX=clang++ ./build_test.sh     # alternate compiler
#
# Self-contained: does not touch the root CMake build or any other component.
set -euo pipefail

cd "$(dirname "$0")"

CXX="${CXX:-c++}"
BUILD_DIR="${BUILD_DIR:-build-intra-modes}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
RUN=1

for arg in "$@"; do
  case "$arg" in
    --no-run)
      RUN=0
      ;;
    -h|--help)
      sed -n '2,10p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$BUILD_DIR"

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
if [ "$WERROR" = "1" ]; then
  FLAGS+=(-Werror)
fi
if [ "$SANITIZE" = "1" ]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2)
fi

echo "compiler: $($CXX --version | head -1)"
echo "build: $BUILD_DIR/intra_modes_test"

"$CXX" "${FLAGS[@]}" -I../.. \
  intra_modes.cpp intra_modes_test.cpp \
  -o "$BUILD_DIR/intra_modes_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/intra_modes_test"
fi
