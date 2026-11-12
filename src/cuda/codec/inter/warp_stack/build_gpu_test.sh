#!/usr/bin/env bash
# Build/run the CUDA warp-stack CPU-vs-GPU parity test.
#
#   ./build_gpu_test.sh                 # sm_86, build + run
#   ./build_gpu_test.sh --no-run
#   ARCH=sm_120 ./build_gpu_test.sh
#
# Needs nvcc and a CUDA device. On hosts without nvcc use
# ./build_emulation_test.sh, which runs the same shared device header against
# the CPU reference. The GPU test allocates every device buffer through
# av2enc::cuda::Arena; it contains no cudaMalloc/cudaFree/cudaMallocHost.
set -euo pipefail
cd "$(dirname "$0")"

HERE="$(pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
ARCH="${ARCH:-sm_86}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-warp-stack-gpu}"
CXX="${CXX:-c++}"
RUN=1
for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    --arch=*) ARCH="${arg#--arch=}" ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found; use ./build_emulation_test.sh for host parity" >&2
  exit 2
fi

NVCC_DIR="$(dirname "$(command -v nvcc)")"
CUDA_INC="$NVCC_DIR/../include"
mkdir -p "$BUILD_DIR"

# The module must not allocate device memory outside the persistent Arena.
if grep -nE '\bcuda(Malloc|Free|MallocHost|HostAlloc)[[:space:]]*\(' \
    "$HERE"/*.h "$HERE"/*.cuh "$HERE"/*.cu "$HERE"/*.cpp 2>/dev/null; then
  echo "error: direct device allocation found; use av2enc::cuda::Arena" >&2
  exit 3
fi

echo "nvcc: $(nvcc --version | tail -1)"
echo "host: $($CXX --version | head -1)"
echo "arch: $ARCH"
echo "build: $BUILD_DIR/warp_stack_gpu_test"

# Host-side translation units: the shared device header compiled for emulation
# and the real Arena. They are compiled by the host compiler (not nvcc) so the
# `__CUDACC__` branch stays off in the reference.
"$CXX" -std=c++20 -O2 -g \
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow \
  -I "$ROOT/src" -I "$HERE" \
  -c "$HERE/warp_stack_host_ref.cpp" -o "$BUILD_DIR/warp_stack_host_ref.o"

"$CXX" -std=c++20 -O2 -g \
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow \
  -I "$CUDA_INC" -I "$ROOT/src" -I "$HERE" \
  -c "$ROOT/src/cuda/arena.cpp" -o "$BUILD_DIR/arena.o"

for src in warp.cpp compound.cpp bawp.cpp b_frame.cpp obmc.cpp; do
  "$CXX" -std=c++20 -O2 -g -Wall -Wextra -Wpedantic -Wconversion -Wshadow \
    -I "$ROOT/src" -I "$HERE" \
    -c "$ROOT/src/codec/inter/advanced/$src" \
    -o "$BUILD_DIR/adv_$src.o"
done

"$CXX" -std=c++20 -O2 -g -I "$ROOT/src" -I "$HERE" \
  -c "$ROOT/src/codec/inter/warp_stack/warp_stack.cpp" \
  -o "$BUILD_DIR/cpu_warp_stack.o"

nvcc -std=c++20 -O3 -lineinfo -arch="$ARCH" \
  -Xcompiler -Wall,-Wextra,-Wconversion,-Wshadow \
  -I "$ROOT/src" -I "$HERE" \
  "$HERE/warp_stack.cu" "$HERE/warp_stack_gpu_test.cu" \
  "$BUILD_DIR/cpu_warp_stack.o" \
  "$BUILD_DIR/warp_stack_host_ref.o" "$BUILD_DIR/arena.o" \
  "$BUILD_DIR"/adv_*.o \
  -o "$BUILD_DIR/warp_stack_gpu_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/warp_stack_gpu_test"
fi
