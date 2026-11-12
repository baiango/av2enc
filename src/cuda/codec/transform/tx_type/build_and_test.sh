#!/usr/bin/env bash
# Build and run the CUDA intra transform-type batch test.
#
#   ./build_and_test.sh                    # sm_86 (RTX 3060 target)
#   CUDA_ARCH=sm_120 ./build_and_test.sh
#   ./build_and_test.sh --no-run
#
# The CPU reference (src/codec/transform/tx_type + src/codec/transform/tx and the
# components they use) is compiled by the host compiler with
# -ffp-contract=off, then linked into the nvcc test binary.  Device code is
# built with -fmad=false so the double forward transform matches the CPU
# bit-for-bit.
set -euo pipefail
cd "$(dirname "$0")"

REPO="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${REPO}/CMakeLists.txt" ] && [ "${REPO}" != "/" ]; do
    REPO="$(dirname -- "${REPO}")"
done
ARCH="${CUDA_ARCH:-sm_86}"
CXX="${CXX:-g++}"
BUILD_DIR="${BUILD_DIR:-${TMPDIR:-/tmp}/av2enc_tx_type_cuda_build}"
RUN=1

for arg in "$@"; do
  case "$arg" in
    --no-run)
      RUN=0
      ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

mkdir -p "${BUILD_DIR}"

CPU_FLAGS=(-std=c++20 -O3 -ffp-contract=off
           -Wall -Wextra -Wconversion -Wshadow -I"${REPO}/src")
CPU_OBJS=()
compile_cpu() {
  local src="$1"
  local obj="${BUILD_DIR}/$(basename "${src%.*}").o"
  "${CXX}" "${CPU_FLAGS[@]}" -c "${src}" -o "${obj}"
  CPU_OBJS+=("${obj}")
}

echo "compiler: $(${CXX} --version | head -1)"
echo "nvcc:     $(nvcc --version | tail -1)"
echo "arch:     ${ARCH}"
echo "build:    ${BUILD_DIR}/tx_type_cuda_test"

compile_cpu "${REPO}/src/codec/transform/tx_type/tx_type.cpp"
compile_cpu "${REPO}/src/codec/transform/variants/tx_variants.cpp"
compile_cpu "${REPO}/src/codec/intra/modes/intra_modes.cpp"
compile_cpu "${REPO}/src/codec/rate/rdo/rdo.cpp"
compile_cpu "${REPO}/src/core/io/y4m.cpp"

nvcc -std=c++20 \
  -arch="${ARCH}" \
  -O3 \
  -lineinfo \
  -fmad=false \
  -I"${REPO}/src" \
  -Xcompiler=-Wall,-Wextra,-Wconversion,-Wshadow,-ffp-contract=off \
  tx_type.cu tx_type_test.cu \
  "$REPO/src/cuda/codec/transform/variants/tx_variants.cu" \
  "${CPU_OBJS[@]}" \
  -o "${BUILD_DIR}/tx_type_cuda_test"

if [ "$RUN" -eq 1 ]; then
  "${BUILD_DIR}/tx_type_cuda_test" --data "${REPO}/tests/data"
fi
