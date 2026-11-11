#!/usr/bin/env bash
# Build and run the IBC CUDA test on an sm_86 GPU box.
#
#   ./build_and_test.sh
#   GPU_ARCH=sm_89 ./build_and_test.sh
#
# The CPU reference build is in src/encoder/ibc/; this script is only the
# device-parity check.  It does not touch the root CMake build.
set -euo pipefail
cd "$(dirname "$0")"

ARCH="${GPU_ARCH:-sm_86}"
NVCC="${NVCC:-nvcc}"
if ! command -v "$NVCC" >/dev/null 2>&1; then
  echo "nvcc not found; IBC CUDA test requires a CUDA toolchain" >&2
  exit 127
fi
REPO_ROOT="$(cd ../../.. && pwd)"
mkdir -p build

FLAGS=(-std=c++17 -O2 -arch="$ARCH" -lineinfo
       -Xcompiler -Wall,-Wextra,-Wpedantic,-Wconversion,-Wshadow)

echo "nvcc: $($NVCC --version | tail -2 | head -1)"
echo "arch: $ARCH"

"$NVCC" "${FLAGS[@]}" -I"$REPO_ROOT/src" \
  ibc.cu ibc_test.cu \
  "$REPO_ROOT/src/encoder/ibc/ibc.cpp" \
  "$REPO_ROOT/src/encoder/partition/partition.cpp" \
  -o build/ibc_gpu_test

./build/ibc_gpu_test
