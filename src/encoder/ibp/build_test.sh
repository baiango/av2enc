#!/usr/bin/env bash
# Build (and by default run) the AV2 IBP CPU reference tests.
#
#   ./build_test.sh                 # -O2, build + run
#   ./build_test.sh --no-run        # build only
#   SANITIZE=1 ./build_test.sh      # ASan + UBSan
#   WERROR=1 ./build_test.sh        # warnings promoted to errors
#
# Self-contained: uses only src/encoder/ibp/ and the read-only intra reference
# module.  Does not touch the root CMake build.
set -euo pipefail

cd "$(dirname "$0")"

BUILD_DIR="${BUILD_DIR:-build}"
RUN=1
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"

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

REPO_ROOT="$(cd ../../.. && pwd)"
CXX="${CXX:-c++}"
CXXFLAGS=(-std=c++20 -O2 -g -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
if [ "$WERROR" -eq 1 ]; then
  CXXFLAGS+=(-Werror)
fi
if [ "$SANITIZE" -eq 1 ]; then
  CXXFLAGS+=(-fsanitize=address,undefined -fno-omit-frame-pointer)
fi

mkdir -p "$BUILD_DIR"

echo "c++: $($CXX --version | head -1)"
echo "build: $BUILD_DIR/ibp_test"

"$CXX" "${CXXFLAGS[@]}" \
  -I "$REPO_ROOT/src" \
  -DAV2ENC_TESTDATA_DIR="\"$REPO_ROOT/tests/data\"" \
  ibp.cpp ibp_test.cpp "$REPO_ROOT/src/encoder/intra_modes/intra_modes.cpp" \
  -o "$BUILD_DIR/ibp_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/ibp_test"
fi
