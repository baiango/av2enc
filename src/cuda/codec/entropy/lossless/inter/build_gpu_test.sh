#!/usr/bin/env bash
# Build/run the AV2 lossless inter CPU-vs-GPU parity + arena test.
#
#   ./build_gpu_test.sh                 # sm_86, build + run
#   ./build_gpu_test.sh --no-run
#   ARCH=sm_120 ./build_gpu_test.sh
#
# Needs nvcc and a CUDA device.  All device buffers come from the persistent
# av2enc::cuda::Arena; the test sizes it from required_device_bytes() and
# checks that two launches do not grow it.
#
# The CPU reference is compiled by the host compiler with function sections and
# the unused syntax writer/reader closure is dropped by the linker's section
# GC, so only the reversible helpers and derivations are linked.
set -euo pipefail
cd "$(dirname "$0")"

HERE="$(pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
ARCH="${ARCH:-sm_86}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-lossless-inter-gpu}"
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

CXX="${CXX:-g++}"
CUDA_HOME="${CUDA_HOME:-$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)}"
NVCC_FLAGS=(-std=c++20 -O3 -arch="$ARCH" -lineinfo
            -Xcompiler -Wall,-Wextra,-Wconversion,-Wshadow,-Wno-sign-conversion
            -I"$ROOT/src")

mkdir -p "$BUILD_DIR"

echo "nvcc: $(nvcc --version | tail -1)"
echo "arch: $ARCH"
echo "build: $BUILD_DIR/lossless_inter_test"

nvcc "${NVCC_FLAGS[@]}" -c "$HERE/lossless_inter.cu" \
  -o "$BUILD_DIR/lossless_inter.o"
nvcc "${NVCC_FLAGS[@]}" -c "$HERE/lossless_inter_test.cu" \
  -o "$BUILD_DIR/lossless_inter_test.o"

# Host-only translation units: workspace accounting and the Arena itself.
"$CXX" -std=c++20 -O2 -Wall -Wextra -Wconversion -Wshadow \
  -I"$ROOT/src" -I"$CUDA_HOME/include" \
  -c "$HERE/lossless_inter_host.cpp" -o "$BUILD_DIR/lossless_inter_host.o"
"$CXX" -std=c++20 -O2 -Wall -Wextra -Wconversion -Wshadow \
  -I"$ROOT/src" -I"$CUDA_HOME/include" \
  -c "$ROOT/src/cuda/arena.cpp" -o "$BUILD_DIR/arena.o"

# CPU reference: keep sections so the linker can discard the syntax closure.
"$CXX" -std=c++20 -O2 -ffunction-sections -fdata-sections \
  -I"$ROOT/src" \
  -c "$ROOT/src/codec/entropy/lossless/inter/lossless_inter.cpp" \
  -o "$BUILD_DIR/lossless_inter_cpu.o"

nvcc -std=c++20 -arch="$ARCH" \
  "$BUILD_DIR/lossless_inter.o" \
  "$BUILD_DIR/lossless_inter_test.o" \
  "$BUILD_DIR/lossless_inter_host.o" \
  "$BUILD_DIR/arena.o" \
  "$BUILD_DIR/lossless_inter_cpu.o" \
  -Xlinker --gc-sections \
  -o "$BUILD_DIR/lossless_inter_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/lossless_inter_test"
fi
