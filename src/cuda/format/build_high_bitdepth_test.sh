#!/usr/bin/env bash
# Build the CUDA high-bit-depth sample-path component test.
#
#   ./build_high_bitdepth_test.sh
#   CUDA_ARCH=sm_120 ./build_high_bitdepth_test.sh
#
# The CPU reference (src/encoder/format/format.cpp) is compiled by the host
# compiler and linked into the nvcc test binary so the comparison uses the real
# component, not a copy.  sm_86 is the primary target; no Blackwell-only code.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
ARCH="${CUDA_ARCH:-sm_86}"
CXX="${CXX:-g++}"
BUILD_DIR="${BUILD_DIR:-${TMPDIR:-/tmp}/av2enc_format_build}"
mkdir -p "${BUILD_DIR}"

CPU_OBJ="${BUILD_DIR}/format_cpu_ref.o"

"${CXX}" -std=c++20 -O2 \
  -Wall -Wextra -Wconversion -Wshadow -Wpedantic \
  -I"${REPO}/src" \
  -c "${REPO}/src/encoder/format/format.cpp" \
  -o "${CPU_OBJ}"

nvcc -std=c++17 \
  -arch="${ARCH}" \
  -O2 \
  -lineinfo \
  -I"${REPO}/src" \
  -Xcompiler=-Wall,-Wextra,-Wconversion,-Wshadow \
  "${HERE}/high_bitdepth.cu" \
  "${HERE}/high_bitdepth_test.cu" \
  "${CPU_OBJ}" \
  -o "${BUILD_DIR}/high_bitdepth_cuda_test"

echo "built ${BUILD_DIR}/high_bitdepth_cuda_test (arch ${ARCH})"
