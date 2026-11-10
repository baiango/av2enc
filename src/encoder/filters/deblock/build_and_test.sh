#!/usr/bin/env bash
# Build (and by default run) the AV2 deblocking filter CPU reference test.
#
#   ./build_and_test.sh                 # build + run
#   ./build_and_test.sh --no-run        # build only
#   CXX=clang++ ./build_and_test.sh     # alternate compiler
#
# Self-contained: does not touch the root CMake build, src/main.cpp, or any
# other component.  CPU-only; the CUDA test lives in src/cuda/deblock/.
set -euo pipefail

cd "$(dirname "$0")"

CXX="${CXX:-c++}"
BUILD_DIR="${BUILD_DIR:-build-cpu}"
RUN=1

for arg in "$@"; do
  case "$arg" in
    --no-run)
      RUN=0
      ;;
    -h|--help)
      sed -n '2,9p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

REPO_ROOT="$(cd ../../../.. && pwd)"
mkdir -p "$BUILD_DIR"

echo "CXX: $($CXX --version | head -1)"
echo "build: $BUILD_DIR/deblock_test"

"$CXX" -std=c++20 -O2 -g \
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow \
  -I "$REPO_ROOT/src" \
  deblock.cpp deblock_test.cpp \
  -o "$BUILD_DIR/deblock_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/deblock_test" "$REPO_ROOT"
fi
