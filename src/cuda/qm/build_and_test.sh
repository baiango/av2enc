#!/usr/bin/env bash
# Build/run the CUDA quantizer-matrix parity test on an sm_86 box.
#
#   ./build_and_test.sh
#   CUDA_ARCH=sm_86 ./build_and_test.sh
#
# The CPU reference is compiled from src/encoder/qm/qm.cpp; both paths call the
# same integer helpers.  No FP math is involved.
set -euo pipefail
cd "$(dirname "$0")"

REPO="$(cd ../../.. && pwd)"
ARCH="${CUDA_ARCH:-sm_86}"
BUILD_DIR="${BUILD_DIR:-${TMPDIR:-/tmp}/av2enc_qm_cuda_build}"
mkdir -p "${BUILD_DIR}"

nvcc -std=c++20 \
  -arch="${ARCH}" \
  -O3 \
  -lineinfo \
  -fmad=false \
  -I"${REPO}/src" \
  -I"${REPO}/src/cuda/qm" \
  -Xcompiler=-Wall,-Wextra,-Wconversion,-Wshadow,-ffp-contract=off \
  qm_gpu.cu qm_gpu_test.cu \
  "${REPO}/src/encoder/qm/qm.cpp" \
  "${REPO}/src/bitstream/bitwriter.cpp" \
  "${REPO}/src/bitstream/obu.cpp" \
  "${REPO}/src/bitstream/m0_headers.cpp" \
  -o "${BUILD_DIR}/qm_gpu_test"

echo "built ${BUILD_DIR}/qm_gpu_test (arch ${ARCH})"
"${BUILD_DIR}/qm_gpu_test"
