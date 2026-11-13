#!/usr/bin/env bash
# Real-GPU parity + arena test for the DC/AC delta-Q quantizer (sm_86).
#
#   ./build_gpu_test.sh
#   ARCH=sm_120 ./build_gpu_test.sh
#
# Requires nvcc and a visible CUDA device.  The host CPU reference is compiled
# with -ffp-contract=off and the device objects with -fmad=false, so the
# quantizer arithmetic can be compared bit for bit.
set -euo pipefail
cd "$(dirname "$0")"

HERE="$(pwd)"
REPO="$HERE"
while [ ! -f "${REPO}/CMakeLists.txt" ] && [ "${REPO}" != "/" ]; do
    REPO="$(dirname -- "${REPO}")"
done
ARCH="${ARCH:-${CUDA_ARCH:-sm_86}}"
CXX="${CXX:-g++}"
BUILD_DIR="${BUILD_DIR:-${TMPDIR:-/tmp}/av2enc_deltaq_cuda_gpu}"
RUN=1
for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done
if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found: the real GPU test needs the CUDA toolkit" >&2
  exit 2
fi
mkdir -p "$BUILD_DIR"

# Device-visibility / arena audit for this component.
if grep -nE '(cudaMalloc|cudaFree|cudaMallocAsync|cudaFreeAsync)[[:space:]]*\(' \
    deltaq_gpu.h deltaq_gpu.cu deltaq_device.cuh deltaq_types.h; then
  echo "delta_q: direct device allocation in a production source" >&2
  exit 1
fi

CPU_FLAGS=(-std=c++20 -O3 -ffp-contract=off -Wall -Wextra -Wconversion
           -Wshadow -I"${REPO}/src")
CPU_OBJS=()
compile_cpu() {
  local src="$1"
  local obj="$BUILD_DIR/$(basename "${src%.*}").o"
  "$CXX" "${CPU_FLAGS[@]}" -c "$src" -o "$obj"
  CPU_OBJS+=("$obj")
}
compile_cpu "$REPO/src/codec/transform/delta_q/quant.cpp"
compile_cpu "$REPO/src/codec/transform/delta_q/syntax.cpp"
compile_cpu "$REPO/src/codec/transform/qm/qm.cpp"
compile_cpu "$REPO/src/core/bitstream/bitwriter.cpp"
compile_cpu "$REPO/src/core/bitstream/basic_headers.cpp"
# arena.cpp includes cuda_runtime.h; compile it with nvcc.
nvcc -std=c++20 -arch="$ARCH" -O3 -fmad=false -I"${REPO}/src" \
  -Xcompiler=-Wall,-Wextra,-Wconversion,-Wshadow,-ffp-contract=off \
  -c "$REPO/src/cuda/arena.cpp" -o "$BUILD_DIR/arena.o"
CPU_OBJS+=("$BUILD_DIR/arena.o")

echo "compiler: $($CXX --version | head -1)"
echo "nvcc:     $(nvcc --version | tail -1)"
echo "arch:     $ARCH"
echo "build:    $BUILD_DIR/deltaq_gpu_test"
nvcc -std=c++20 -arch="$ARCH" -O3 -lineinfo -fmad=false -I"${REPO}/src" \
  -Xcompiler=-Wall,-Wextra,-Wconversion,-Wshadow,-ffp-contract=off \
  -Xptxas=-v \
  deltaq_gpu.cu deltaq_gpu_test.cu "${CPU_OBJS[@]}" \
  -o "$BUILD_DIR/deltaq_gpu_test"
if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/deltaq_gpu_test"
fi
