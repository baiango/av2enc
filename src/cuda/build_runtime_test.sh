#!/usr/bin/env bash
# Build and run the av2enc CUDA runtime-layer tests.
#
# Target box: RTX 3060 (sm_86), CUDA 12.8. The component test uses sm_86 only;
# the fat sm_86;sm_120 build happens in the final integration target.
#
# Usage:
#   ./build_runtime_test.sh                 # build + run the normal suite
#   ./build_runtime_test.sh --full-cap      # also allocate the 7244 MiB arena
#   CUDA_ARCH=sm_120 ./build_runtime_test.sh
#   BUILD_DIR=/tmp/rt ./build_runtime_test.sh
set -euo pipefail
cd "$(dirname "$0")"

NVCC="${NVCC:-nvcc}"
ARCH="${CUDA_ARCH:-sm_86}"
BUILD_DIR="${BUILD_DIR:-build-runtime}"
# -Wpedantic is intentionally omitted: nvcc's generated line directives trip it
# for every source line. Host-only CMake builds still use the full warning set.
WARNINGS="-Wall,-Wextra,-Wconversion,-Wshadow"

if ! command -v "$NVCC" >/dev/null 2>&1; then
  echo "error: nvcc not found; set NVCC=/path/to/nvcc" >&2
  exit 1
fi

mkdir -p "$BUILD_DIR"

echo "nvcc: $("$NVCC" --version | grep -m1 'release')"
echo "arch: $ARCH"
echo "build: $BUILD_DIR/context_test"

"$NVCC" \
  -std=c++20 \
  -arch="$ARCH" \
  -lineinfo \
  -O2 \
  -g \
  -Xcompiler="$WARNINGS" \
  -I. \
  -o "$BUILD_DIR/context_test" \
  context.cpp \
  arena.cpp \
  context_test.cu

echo
"$BUILD_DIR/context_test" "$@"
