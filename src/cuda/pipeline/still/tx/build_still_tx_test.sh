#!/usr/bin/env bash
# Build the batched M1 transform/quant stage test.
#
#   ./build_still_tx_test.sh                    # sm_86 (RTX 3060 target)
#   CUDA_ARCH=sm_120 ./build_still_tx_test.sh
#
# The CPU reference (`src/codec/transform/variants/tx_variants.cpp`) is compiled by
# the host compiler with -ffp-contract=off; the CUDA side uses -fmad=false and
# explicit __dadd_rn/__dmul_rn so the binary64 forward transform can be
# compared bit-for-bit.  BUILD_DIR defaults to a private directory so this
# component never collides with another agent's build tree.
set -euo pipefail
cd "$(dirname "$0")"

REPO="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${REPO}/CMakeLists.txt" ] && [ "${REPO}" != "/" ]; do
    REPO="$(dirname -- "${REPO}")"
done
ARCH="${CUDA_ARCH:-sm_86}"
CXX="${CXX:-g++}"
BUILD_DIR="${BUILD_DIR:-${TMPDIR:-/tmp}/av2enc_still_tx_build}"
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
  -DAV2ENC_TEST_ROOT="\"${REPO}\"" \
  -Xcompiler=-Wall,-Wextra,-Wconversion,-Wshadow,-ffp-contract=off \
  -Xptxas=-v \
  still_tx.cu still_tx_test.cu "${CPU_OBJ}" \
  -o "${BUILD_DIR}/still_tx_cuda_test"

echo "built ${BUILD_DIR}/still_tx_cuda_test (arch ${ARCH})"
