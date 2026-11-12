#!/usr/bin/env bash
# Build (and by default run) the CUDA CfL component test.
#
#   ./build_cfl_test.sh                 # sm_86, build + run
#   ./build_cfl_test.sh --no-run        # build only
#   ARCH=sm_120 ./build_cfl_test.sh     # secondary build target
#
# Self-contained: does not touch the root CMake build or any other component.
# The CPU reference is compiled from src/codec/intra/cfl/cfl.cpp, so this test is
# a true CPU-vs-GPU bit-exactness check, not two kernel copies.
set -euo pipefail

cd "$(dirname "$0")"

ARCH="${ARCH:-sm_86}"
BUILD_DIR="${BUILD_DIR:-build-cfl}"
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
      sed -n '2,9p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$BUILD_DIR"

echo "nvcc: $(nvcc --version | tail -1)"
echo "arch: $ARCH"
echo "build: $BUILD_DIR/cfl_test"

nvcc -std=c++20 -O3 -lineinfo -arch="$ARCH" \
  -Xcompiler -Wall,-Wextra,-Wconversion,-Wshadow \
  cfl.cu cfl_test.cu ../../../../codec/intra/cfl/cfl.cpp \
  -o "$BUILD_DIR/cfl_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/cfl_test"
fi
