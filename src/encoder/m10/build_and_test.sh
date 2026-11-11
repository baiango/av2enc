#!/usr/bin/env bash
# Standalone build/test for the 10-bit M10 still picture.
#
#   ./build_and_test.sh
#
# CPU reference test and encoder tool always build.  If nvcc is on PATH the
# CPU-vs-GPU bit-exact test is built for CUDA_ARCH (default sm_86) and run too.
# Build artifacts go to ${TMPDIR:-/tmp}/av2enc_m10_build; nothing is committed.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${TMPDIR:-/tmp}/av2enc_m10_build}"
CXX="${CXX:-c++}"
mkdir -p "${BUILD_DIR}"

WARN=(-Wall -Wextra -Wconversion -Wshadow -Wpedantic -Wno-sign-conversion)
COMMON=(
  "${REPO}/src/encoder/m10/m10_headers.cpp"
  "${REPO}/src/encoder/m10/m10_entropy.cpp"
  "${REPO}/src/encoder/m10/m10_still.cpp"
  "${REPO}/src/encoder/m10/m10_y4m.cpp"
  "${REPO}/src/encoder/format/format.cpp"
  "${REPO}/src/encoder/tx_variants/tx_variants.cpp"
  "${REPO}/src/cuda/m1/hbd/hbd_cpu.cpp"
  "${REPO}/src/entropy/av2/av2_entropy.cpp"
  "${REPO}/src/bitstream/bitwriter.cpp"
  "${REPO}/src/bitstream/obu.cpp"
  "${REPO}/src/bitstream/m0_headers.cpp"
  "${REPO}/src/io/ivf.cpp"
)

echo "building M10 CPU test"
"${CXX}" -std=c++20 -O2 -g -ffp-contract=off \
  "${WARN[@]}" -I"${REPO}/src" \
  "${COMMON[@]}" "${HERE}/m10_test.cpp" \
  -o "${BUILD_DIR}/m10_test"
"${BUILD_DIR}/m10_test"

echo
echo "building M10 encoder tool"
"${CXX}" -std=c++20 -O2 -g -ffp-contract=off \
  "${WARN[@]}" -I"${REPO}/src" \
  "${COMMON[@]}" "${HERE}/m10_encode.cpp" \
  -o "${BUILD_DIR}/m10_encode"
echo "tool: ${BUILD_DIR}/m10_encode"

if command -v nvcc >/dev/null 2>&1; then
  echo
  echo "nvcc found: building and running the CPU-vs-GPU bit-exact test"
  BUILD_DIR="${BUILD_DIR}" "${REPO}/src/cuda/format/build_m10_gpu_test.sh"
else
  echo
  echo "nvcc not found: CPU reference/tool only."
  echo "Run src/cuda/format/build_m10_gpu_test.sh on the sm_86 box."
fi
