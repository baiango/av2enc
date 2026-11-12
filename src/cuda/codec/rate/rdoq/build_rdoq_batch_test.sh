#!/usr/bin/env bash
# Build and run the CUDA RDOQ batch parity test on the sm_86 box.
#
#   ./build_rdoq_batch_test.sh
#   CUDA_ARCH=sm_86 ./build_rdoq_batch_test.sh
#
# The CPU reference is compiled by the host compiler with -ffp-contract=off;
# nvcc uses -fmad=false so the double DP arithmetic matches bit for bit.
set -euo pipefail
cd "$(dirname "$0")"

REPO="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${REPO}/CMakeLists.txt" ] && [ "${REPO}" != "/" ]; do
    REPO="$(dirname -- "${REPO}")"
done
ARCH="${CUDA_ARCH:-sm_86}"
CXX="${CXX:-g++}"
BUILD_DIR="${BUILD_DIR:-${TMPDIR:-/tmp}/av2enc_rdoq_cuda_build}"
mkdir -p "${BUILD_DIR}"

CPU_OBJ="${BUILD_DIR}/rdoq_cpu.o"

"${CXX}" -std=c++20 -O3 -ffp-contract=off \
  -Wall -Wextra -Wconversion -Wshadow \
  -I"${REPO}/src" \
  -c "${REPO}/src/codec/rate/rdoq/rdoq.cpp" \
  -o "${CPU_OBJ}"

nvcc -std=c++20 \
  -arch="${ARCH}" \
  -O3 \
  -lineinfo \
  -fmad=false \
  -I"${REPO}/src" \
  -I"${REPO}/src/cuda/codec/rate/rdoq" \
  -Xcompiler=-Wall,-Wextra,-Wconversion,-Wshadow,-ffp-contract=off \
  -Xptxas=-v \
  rdoq_batch.cu rdoq_batch_test.cu "${CPU_OBJ}" \
  -o "${BUILD_DIR}/rdoq_batch_test"

echo "built ${BUILD_DIR}/rdoq_batch_test (arch ${ARCH})"
"${BUILD_DIR}/rdoq_batch_test"
