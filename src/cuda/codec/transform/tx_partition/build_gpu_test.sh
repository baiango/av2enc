#!/usr/bin/env bash
# Build/run the real-GPU parity test for the CUDA TX_MODE_SELECT transform
# partition.  Needs nvcc and a CUDA device.
#
#   ./build_gpu_test.sh                 # sm_86, build + run
#   ./build_gpu_test.sh --no-run
#   ARCH=sm_120 ./build_gpu_test.sh
#
# On hosts without nvcc use ./build_and_test.sh, which runs the shared device
# core against the CPU reference with the arena emulated on the host.
#
# Every device buffer in the test comes from one persistent av2enc::cuda::Arena;
# the component itself never allocates.  The component sources are checked for
# direct allocation calls below.
set -euo pipefail
cd "$(dirname "$0")"

ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
ARCH="${ARCH:-sm_86}"
BUILD_DIR="${BUILD_DIR:-${TMPDIR:-/tmp}/av2enc_tx_partition_gpu}"
CXX="${CXX:-g++}"
RUN=1
for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    --arch=*) ARCH="${arg#--arch=}" ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found; use ./build_and_test.sh for host-emulation parity" >&2
  exit 2
fi

if grep -nE '\b(cudaMalloc|cudaFree|cudaMallocHost|cudaHostAlloc|cudaHostRegister)[[:space:]]*\(' \
    tx_partition_device.cuh tx_partition_plan.h tx_partition_plan.cpp \
    tx_partition_gpu.h tx_partition_gpu.cu; then
  echo "ERROR: direct device allocation in the component sources" >&2
  exit 1
fi

mkdir -p "$BUILD_DIR"

CPU_FLAGS=(-std=c++20 -O3 -ffp-contract=off -Wall -Wextra -Wconversion
           -Wshadow -I"$ROOT/src")
compile_cpu() {
  local src="$1"
  local obj="$BUILD_DIR/$(basename "${src%.*}").o"
  "$CXX" "${CPU_FLAGS[@]}" -c "$src" -o "$obj"
  echo "$obj"
}

VIDEO_PARTITION="$(compile_cpu "$ROOT/src/codec/transform/tx_partition/tx_partition.cpp")"
VIDEO_SCAN="$(compile_cpu "$ROOT/src/codec/transform/tx_partition/tx_scan.cpp")"
CPU_TXV="$(compile_cpu "$ROOT/src/codec/transform/variants/tx_variants.cpp")"

NVCC_FLAGS=(-std=c++20 -arch="$ARCH" -O3 -lineinfo -fmad=false
            --expt-relaxed-constexpr -I"$ROOT/src"
            -Xcompiler=-Wall,-Wextra,-Wconversion,-Wshadow,-ffp-contract=off
            -Xptxas=-v)

echo "nvcc:  $(nvcc --version | tail -1)"
echo "arch:  $ARCH"
echo "build: $BUILD_DIR/tx_partition_gpu_test"

nvcc "${NVCC_FLAGS[@]}" -c tx_partition_gpu.cu -o "$BUILD_DIR/tx_partition_gpu.o"
nvcc "${NVCC_FLAGS[@]}" -c tx_partition_gpu_test.cu \
  -o "$BUILD_DIR/tx_partition_gpu_test.o"
# Plan/host helpers and the arena are compiled with __CUDACC__ defined so the
# shared inline core has one definition across TUs.
nvcc "${NVCC_FLAGS[@]}" -x cu -c tx_partition_plan.cpp \
  -o "$BUILD_DIR/tx_partition_plan.o"
nvcc "${NVCC_FLAGS[@]}" -x cu -c "$ROOT/src/cuda/arena.cpp" \
  -o "$BUILD_DIR/arena.o"
nvcc "${NVCC_FLAGS[@]}" -c "$ROOT/src/cuda/codec/transform/variants/tx_variants.cu" \
  -o "$BUILD_DIR/txv_gpu.o"

nvcc "${NVCC_FLAGS[@]}" \
  "$BUILD_DIR/tx_partition_gpu.o" \
  "$BUILD_DIR/tx_partition_gpu_test.o" \
  "$BUILD_DIR/tx_partition_plan.o" \
  "$BUILD_DIR/arena.o" \
  "$BUILD_DIR/txv_gpu.o" \
  "$VIDEO_PARTITION" "$VIDEO_SCAN" "$CPU_TXV" \
  -o "$BUILD_DIR/tx_partition_gpu_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/tx_partition_gpu_test"
fi
