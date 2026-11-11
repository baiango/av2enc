#!/usr/bin/env bash
# Build (and by default run) the AV2 restoration CPU-vs-CUDA tests.
#
#   ./build_and_test.sh                 # sm_86, build + run both binaries
#   ./build_and_test.sh --no-run        # build only
#   ARCH=sm_120 ./build_and_test.sh     # secondary build target
#
# Self-contained: does not touch the root CMake build or any other component.
# The CPU reference object is compiled by the host compiler; kernels and tests
# are compiled by nvcc, then linked together.
set -euo pipefail

cd "$(dirname "$0")"

ARCH="${ARCH:-sm_86}"
BUILD_DIR="${BUILD_DIR:-build-cuda}"
CXX="${CXX:-c++}"
RUN=1

for arg in "$@"; do
  case "$arg" in
    --no-run)
      RUN=0
      ;;
    --arch=*)
      ARCH="${arg#--arch=}"
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
mkdir -p "$BUILD_DIR"

echo "nvcc: $(nvcc --version | tail -1)"
echo "host: $($CXX --version | head -1)"
echo "arch: $ARCH"

"$CXX" -std=c++20 -O2 -g \
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow \
  -I "$REPO_ROOT/src" \
  -c "$REPO_ROOT/src/encoder/filters/restoration/restoration.cpp" \
  -o "$BUILD_DIR/restoration_cpu.o"

for target in ccso wiener; do
  echo "build: $BUILD_DIR/${target}_test"
  nvcc -std=c++20 -O3 -lineinfo -arch="$ARCH" \
    -Xcompiler -Wall,-Wextra,-Wconversion,-Wshadow \
    --ptxas-options=-v \
    -I "$REPO_ROOT/src" -I . \
    "${target}.cu" "${target}_test.cu" "$BUILD_DIR/restoration_cpu.o" \
    -o "$BUILD_DIR/${target}_test"
done

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/ccso_test"
  "$BUILD_DIR/wiener_test"
fi
