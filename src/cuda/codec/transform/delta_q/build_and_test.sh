#!/usr/bin/env bash
# Build and run the DC/AC delta-Q component.
#
#   ./build_and_test.sh                 # host emulation always; GPU test if nvcc
#   ./build_and_test.sh --no-run
#   WERROR=1 ./build_and_test.sh
#   CUDA_ARCH=sm_120 ./build_and_test.sh
#
# The emulation test compiles the exact shared device core
# (deltaq_device.cuh) with the host compiler and compares it bit for bit with
# the CPU reference in src/codec/transform/delta_q/quant.cpp.  When nvcc is
# available the real one-thread-per-block kernel test is built and run too.
set -euo pipefail
cd "$(dirname "$0")"

HERE="$(pwd)"
REPO="$HERE"
while [ ! -f "${REPO}/CMakeLists.txt" ] && [ "${REPO}" != "/" ]; do
    REPO="$(dirname -- "${REPO}")"
done
ARCH="${CUDA_ARCH:-sm_86}"
CXX="${CXX:-c++}"
BUILD_DIR="${BUILD_DIR:-${TMPDIR:-/tmp}/av2enc_deltaq_cuda_build}"
WERROR="${WERROR:-0}"
RUN=1
for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$BUILD_DIR"

# Single-arena guard: production sources must not allocate device memory.
if grep -nE '(cudaMalloc|cudaFree|cudaMallocAsync|cudaFreeAsync)[[:space:]]*\(' \
    deltaq_gpu.h deltaq_gpu.cu deltaq_device.cuh deltaq_types.h; then
  echo "delta_q: direct device allocation in a production source" >&2
  exit 1
fi
FLAGS=(-std=c++20 -O2 -ffp-contract=off -Wall -Wextra -Wpedantic
       -Wconversion -Wshadow -I"${REPO}/src")
if [ "$WERROR" = "1" ]; then FLAGS+=(-Werror); fi

CPU_SOURCES=(
  "$HERE/deltaq_emulation_test.cpp"
  "$REPO/src/codec/transform/delta_q/quant.cpp"
  "$REPO/src/codec/transform/delta_q/syntax.cpp"
  "$REPO/src/codec/transform/qm/qm.cpp"
  "$REPO/src/core/bitstream/bitwriter.cpp"
  "$REPO/src/core/bitstream/basic_headers.cpp"
)

echo "compiler: $($CXX --version | head -1)"
echo "build:    $BUILD_DIR/deltaq_emulation_test"
"$CXX" "${FLAGS[@]}" "${CPU_SOURCES[@]}" \
  -o "$BUILD_DIR/deltaq_emulation_test"
if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/deltaq_emulation_test"
fi

if command -v nvcc >/dev/null 2>&1; then
  echo "nvcc:     $(nvcc --version | tail -1)"
  echo "arch:     $ARCH"
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

  echo "build:    $BUILD_DIR/deltaq_gpu_test"
  nvcc -std=c++20 -arch="$ARCH" -O3 -lineinfo -fmad=false -I"${REPO}/src" \
    -Xcompiler=-Wall,-Wextra,-Wconversion,-Wshadow,-ffp-contract=off \
    deltaq_gpu.cu deltaq_gpu_test.cu "${CPU_OBJS[@]}" \
    -o "$BUILD_DIR/deltaq_gpu_test"
  if [ "$RUN" -eq 1 ]; then
    "$BUILD_DIR/deltaq_gpu_test"
  fi
else
  echo "nvcc not found: skipped the real GPU test (emulation core passed)."
fi
