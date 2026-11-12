#!/usr/bin/env bash
# Build/run the real-GPU DDT parity test on the sm_86 box.
#
#   ./build_gpu_test.sh                    # sm_86, build + run
#   ./build_gpu_test.sh --no-run           # build only
#   CUDA_ARCH=sm_120 ./build_gpu_test.sh   # secondary target
#
# The CPU reference (src/codec/transform/ddt/ddt_transform.cpp) and
# src/codec/transform/variants/tx_variants.cpp are compiled by the host compiler
# with -ffp-contract=off, then linked into the nvcc binary.  arena.cpp is passed
# to nvcc so it sees the toolkit's cuda_runtime.h.  This component's arithmetic
# is integer-only; -fmad=false is kept for consistency with the other transform
# parity tests.
set -euo pipefail
cd "$(dirname "$0")"

REPO="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${REPO}/CMakeLists.txt" ] && [ "${REPO}" != "/" ]; do
    REPO="$(dirname -- "${REPO}")"
done
ARCH="${CUDA_ARCH:-sm_86}"
CXX="${CXX:-g++}"
NVCC="${NVCC:-nvcc}"
BUILD_DIR="${BUILD_DIR:-${TMPDIR:-/tmp}/av2enc_ddt_gpu_build}"
RUN=1
for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    --arch=*) ARCH="${arg#--arch=}" ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if ! command -v "$NVCC" >/dev/null 2>&1; then
  echo "error: nvcc not found (run on the Linux + CUDA box)" >&2
  exit 127
fi

mkdir -p "$BUILD_DIR"

WARN=(-Wall -Wextra -Wconversion -Wshadow -Wno-sign-conversion)
CXXFLAGS=(-std=c++20 -O3 -ffp-contract=off "${WARN[@]}" -I"${REPO}/src")

compile_host() {
  local src=$1
  local obj=$2
  "${CXX}" "${CXXFLAGS[@]}" -c "${src}" -o "${obj}"
}

compile_host "${REPO}/src/codec/transform/variants/tx_variants.cpp" \
  "${BUILD_DIR}/tx_variants_cpu.o"
compile_host "${REPO}/src/codec/transform/ddt/ddt_transform.cpp" \
  "${BUILD_DIR}/ddt_transform_cpu.o"

echo "nvcc: $("$NVCC" --version | tail -1)"
echo "arch: $ARCH"

"$NVCC" -std=c++17 -arch="${ARCH}" -O3 -lineinfo -fmad=false \
  -I"${REPO}/src" \
  -Xcompiler=-Wall,-Wextra,-Wconversion,-Wshadow,-ffp-contract=off \
  -Xptxas=-v \
  ddt_gpu.cu ddt_gpu_test.cu "${REPO}/src/cuda/arena.cpp" \
  "${BUILD_DIR}/tx_variants_cpu.o" "${BUILD_DIR}/ddt_transform_cpu.o" \
  -o "${BUILD_DIR}/ddt_cuda_test"

echo "built ${BUILD_DIR}/ddt_cuda_test"

if [[ "$RUN" == "1" ]]; then
  "${BUILD_DIR}/ddt_cuda_test"
fi
