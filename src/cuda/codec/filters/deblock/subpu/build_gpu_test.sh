#!/usr/bin/env bash
# Build/run the real-GPU bit-exact parity test for the AV2 §7.13.3.15 sub-PU
# grid.
#
#   ./build_gpu_test.sh                 # sm_86, build + run
#   ./build_gpu_test.sh --no-run        # build only
#   ARCH=sm_120 ./build_gpu_test.sh     # secondary build target
#
# Needs nvcc and a CUDA device.  On hosts without nvcc use ./build_and_test.sh,
# which runs the same shared device header against the CPU reference on the
# host.
#
# All device buffers in the test come from one av2enc::cuda::Arena; the test
# itself calls no cudaMalloc/cudaFree/cudaMallocHost.
set -euo pipefail

cd "$(dirname "$0")"

HERE="$(pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
ARCH="${ARCH:-sm_86}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-subpu-gpu}"
CXX="${CXX:-c++}"
RUN=1

for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    --arch=*) ARCH="${arg#--arch=}" ;;
    -h|--help)
      sed -n '2,15p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found; use ./build_and_test.sh for host-emulation parity" >&2
  exit 2
fi

# Mandatory single-arena rule: the component must not allocate device memory on
# its own.  Call-shaped so prose in comments does not trip it.
if grep -nE '\b(cudaMalloc|cudaFree|cudaMallocHost|cudaHostAlloc|cudaHostRegister)[[:space:]]*\(' \
    subpu.h subpu_device.cuh subpu_plan.cpp subpu.cu; then
  echo "ERROR: direct device allocation in the component sources" >&2
  exit 1
fi

mkdir -p "$BUILD_DIR"

echo "nvcc: $(nvcc --version | tail -1)"
echo "host: $($CXX --version | head -1)"
echo "arch: $ARCH"
echo "build: $BUILD_DIR/subpu_test"

# CPU oracle (the §7.13.3.15 reference), compiled by the host compiler.
"$CXX" -std=c++20 -O2 -g \
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow \
  -I "$ROOT/src" \
  -c "$ROOT/src/codec/intra/supplemental/save_subpu.cpp" \
  -o "$BUILD_DIR/save_subpu_cpu.o"

# Kernels + test + the persistent arena runtime.  No --expt-relaxed-constexpr:
# the device core must not depend on host namespace-scope constexpr objects.
nvcc -std=c++20 -O3 -lineinfo -arch="$ARCH" \
  -Xcompiler -Wall,-Wextra,-Wconversion,-Wshadow \
  --ptxas-options=-v \
  -I "$ROOT/src" -I "$HERE" \
  "$HERE/subpu.cu" "$HERE/subpu_plan.cpp" "$HERE/subpu_test.cu" \
  "$ROOT/src/cuda/arena.cpp" "$BUILD_DIR/save_subpu_cpu.o" \
  -o "$BUILD_DIR/subpu_test"

if [ "$RUN" -eq 1 ]; then
  CUDA_DISABLE_PTX_JIT=1 "$BUILD_DIR/subpu_test"
fi
