#!/usr/bin/env bash
# Build the CUDA transform-variants component test.
#
#   ./build_tx_variants_test.sh              # sm_86 (RTX 3060 target)
#   CUDA_ARCH=sm_120 ./build_tx_variants_test.sh
#
# The CPU reference is compiled by the host compiler (not nvcc) with
# -ffp-contract=off, then linked into the nvcc test binary.  -fmad=false keeps
# device FMA contraction off as well, so the double forward transform can be
# compared bit-for-bit.  -lineinfo is on for Nsight Systems.
set -euo pipefail
cd "$(dirname "$0")"

REPO="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${REPO}/CMakeLists.txt" ] && [ "${REPO}" != "/" ]; do
    REPO="$(dirname -- "${REPO}")"
done
ARCH="${CUDA_ARCH:-sm_86}"
CXX="${CXX:-g++}"
BUILD_DIR="${BUILD_DIR:-${TMPDIR:-/tmp}/av2enc_tx_variants_build}"
mkdir -p "${BUILD_DIR}"

CPU_OBJ="${BUILD_DIR}/tx_variants_cpu.o"

"${CXX}" -std=c++20 -O3 -ffp-contract=off \
  -Wall -Wextra -Wconversion -Wshadow \
  -I"${REPO}/src" \
  -c "${REPO}/src/codec/transform/variants/tx_variants.cpp" \
  -o "${CPU_OBJ}"

nvcc -std=c++17 \
  -arch="${ARCH}" \
  -O3 \
  -lineinfo \
  -fmad=false \
  -I"${REPO}/src" \
  -Xcompiler=-Wall,-Wextra,-Wconversion,-Wshadow,-ffp-contract=off \
  -Xptxas=-v \
  tx_variants.cu tx_variants_test.cu "${CPU_OBJ}" \
  -o "${BUILD_DIR}/tx_variants_cuda_test"

echo "built ${BUILD_DIR}/tx_variants_cuda_test (arch ${ARCH})"
