#!/usr/bin/env bash
# Build (and by default run) the AV2 restoration CPU reference tests.
#
#   ./build_and_test.sh                 # build + run both test binaries
#   ./build_and_test.sh --no-run        # build only
#   CXX=clang++ ./build_and_test.sh     # alternate compiler
#
# Self-contained: does not touch the root CMake build, src/main.cpp, or any
# other component.  CPU-only; the CUDA tests live in src/cuda/codec/filters/loop_restoration/.
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
      sed -n '2,10p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

REPO_ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${REPO_ROOT}/CMakeLists.txt" ] && [ "${REPO_ROOT}" != "/" ]; do
    REPO_ROOT="$(dirname -- "${REPO_ROOT}")"
done
mkdir -p "$BUILD_DIR"

echo "CXX: $($CXX --version | head -1)"

FLAGS=(-std=c++20 -O2 -g -Wall -Wextra -Wpedantic -Wconversion -Wshadow
       -I "$REPO_ROOT/src")

for target in ccso wiener; do
  echo "build: $BUILD_DIR/${target}_test"
  "$CXX" "${FLAGS[@]}" restoration.cpp "${target}_test.cpp" \
    -o "$BUILD_DIR/${target}_test"
done

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/ccso_test"
  "$BUILD_DIR/wiener_test"
fi
