#!/usr/bin/env bash
# Build and run the palette CUDA test on an sm_86 GPU box.
#
#   ./build_and_test.sh
#   GPU_ARCH=sm_89 ./build_and_test.sh
#
# The CPU reference build is in src/codec/intra/palette/core/; this script is only the
# device-parity check.  It does not touch the root CMake build.
set -euo pipefail
cd "$(dirname "$0")"

ARCH="${GPU_ARCH:-sm_86}"
NVCC="${NVCC:-nvcc}"
if ! command -v "$NVCC" >/dev/null 2>&1; then
  echo "nvcc not found; palette CUDA test requires a CUDA toolchain" >&2
  exit 127
fi
REPO_ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${REPO_ROOT}/CMakeLists.txt" ] && [ "${REPO_ROOT}" != "/" ]; do
    REPO_ROOT="$(dirname -- "${REPO_ROOT}")"
done
mkdir -p build

FLAGS=(-std=c++20 -O2 -arch="$ARCH" -lineinfo
       -Xcompiler -Wall,-Wextra,-Wpedantic,-Wconversion,-Wshadow)

echo "nvcc: $($NVCC --version | tail -2 | head -1)"
echo "arch: $ARCH"

"$NVCC" "${FLAGS[@]}" -I"$REPO_ROOT/src" \
  palette.cu palette_test.cu \
  "$REPO_ROOT/src/codec/intra/palette/core/palette.cpp" \
  "$REPO_ROOT/src/codec/entropy/av2/av2_entropy.cpp" \
  -o build/palette_gpu_test

./build/palette_gpu_test
