#!/usr/bin/env bash
# Build (and by default run) the CUDA advanced-inter component test.
#
#   ./build_advanced_test.sh                 # sm_86, build + run
#   ./build_advanced_test.sh --no-run        # build only
#   ARCH=sm_120 ./build_advanced_test.sh     # secondary build target
#
# Self-contained: does not touch the root CMake build or any other component.
# The CPU reference is compiled from src/encoder/inter/advanced/*.cpp, so this is
# a true CPU-vs-GPU bit-exactness check.
#
# Requires nvcc and a visible CUDA device. `--no-run` only needs nvcc.
set -euo pipefail

cd "$(dirname "$0")"

ARCH="${ARCH:-sm_86}"
BUILD_DIR="${BUILD_DIR:-build-advanced}"
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
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

CPU_DIR="../../../encoder/inter/advanced"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found: the CUDA component test needs the Linux+CUDA target." >&2
  echo "The CPU reference builds and runs on the Mac via:" >&2
  echo "  ${CPU_DIR}/build_and_test.sh" >&2
  exit 1
fi

mkdir -p "$BUILD_DIR"

echo "nvcc: $(nvcc --version | tail -1)"
echo "arch: $ARCH"
echo "build: $BUILD_DIR/advanced_cuda_test"

nvcc -std=c++20 -O3 -lineinfo -arch="$ARCH" \
  -Xcompiler -Wall,-Wextra,-Wconversion,-Wshadow \
  advanced.cu advanced_test.cu \
  "${CPU_DIR}/compound.cpp" "${CPU_DIR}/warp.cpp" \
  "${CPU_DIR}/bawp.cpp" "${CPU_DIR}/b_frame.cpp" \
  "${CPU_DIR}/obmc.cpp" \
  -o "$BUILD_DIR/advanced_cuda_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/advanced_cuda_test"
fi
