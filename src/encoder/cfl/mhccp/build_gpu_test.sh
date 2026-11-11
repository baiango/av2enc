#!/usr/bin/env bash
# Build and run the CUDA MHCCP component test (Linux + CUDA box only).
#
#   ./build_gpu_test.sh                 # sm_86, build + run
#   ./build_gpu_test.sh --no-run        # build only
#   ARCH=sm_120 ./build_gpu_test.sh     # secondary target
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../../../.." && pwd)"
ARCH="${ARCH:-sm_86}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-mhccp-gpu}"
RUN=1

for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    --arch=*) ARCH="${arg#--arch=}" ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$BUILD_DIR"
echo "nvcc: $(nvcc --version | tail -1)"
echo "arch: $ARCH"

nvcc -std=c++20 -O3 -lineinfo -arch="$ARCH" --fmad=false \
  -I"$ROOT/src" -I"$HERE" \
  -Xcompiler -Wall,-Wextra,-Wconversion,-Wshadow \
  "$HERE/mhccp_gpu.cu" \
  "$HERE/mhccp_gpu_test.cu" \
  "$HERE/mhccp.cpp" \
  "$ROOT/src/encoder/cfl/cfl.cpp" \
  -o "$BUILD_DIR/mhccp_cuda_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/mhccp_cuda_test"
fi
