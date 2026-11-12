#!/usr/bin/env bash
# Standalone build/test for the 10-bit M10 still picture.
#
#   ./build_and_test.sh
#
# CPU reference test and encoder tool always build.  If nvcc is on PATH the
# CPU-vs-GPU bit-exact test is built for CUDA_ARCH (default sm_86) and run too.
# Build artifacts go to ${TMPDIR:-/tmp}/av2enc_high_bitdepth_build; nothing is committed.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${REPO}/CMakeLists.txt" ] && [ "${REPO}" != "/" ]; do
    REPO="$(dirname -- "${REPO}")"
done
BUILD_DIR="${BUILD_DIR:-${TMPDIR:-/tmp}/av2enc_high_bitdepth_build}"
CXX="${CXX:-c++}"
mkdir -p "${BUILD_DIR}"

WARN=(-Wall -Wextra -Wconversion -Wshadow -Wpedantic -Wno-sign-conversion)
COMMON=(
  "${REPO}/src/pipeline/still/high_bitdepth/high_bitdepth_headers.cpp"
  "${REPO}/src/pipeline/still/high_bitdepth/high_bitdepth_entropy.cpp"
  "${REPO}/src/pipeline/still/high_bitdepth/high_bitdepth_still.cpp"
  "${REPO}/src/pipeline/still/high_bitdepth/high_bitdepth_y4m.cpp"
  "${REPO}/src/formats/format.cpp"
  "${REPO}/src/codec/transform/variants/tx_variants.cpp"
  "${REPO}/src/cuda/pipeline/still/hbd/hbd_cpu.cpp"
  "${REPO}/src/codec/entropy/av2/av2_entropy.cpp"
  "${REPO}/src/core/bitstream/bitwriter.cpp"
  "${REPO}/src/core/bitstream/obu.cpp"
  "${REPO}/src/core/bitstream/basic_headers.cpp"
  "${REPO}/src/core/io/ivf.cpp"
)

echo "building M10 CPU test"
"${CXX}" -std=c++20 -O2 -g -ffp-contract=off \
  "${WARN[@]}" -I"${REPO}/src" \
  "${COMMON[@]}" "${HERE}/high_bitdepth_test.cpp" \
  -o "${BUILD_DIR}/high_bitdepth_test"
"${BUILD_DIR}/high_bitdepth_test"

echo
echo "building M10 encoder tool"
"${CXX}" -std=c++20 -O2 -g -ffp-contract=off \
  "${WARN[@]}" -I"${REPO}/src" \
  "${COMMON[@]}" "${HERE}/high_bitdepth_encode.cpp" \
  -o "${BUILD_DIR}/high_bitdepth_encode"
echo "tool: ${BUILD_DIR}/high_bitdepth_encode"

if command -v nvcc >/dev/null 2>&1; then
  echo
  echo "nvcc found: building and running the CPU-vs-GPU bit-exact test"
  BUILD_DIR="${BUILD_DIR}" "${REPO}/src/cuda/formats/build_high_bitdepth_gpu_test.sh"
else
  echo
  echo "nvcc not found: CPU reference/tool only."
  echo "Run src/cuda/formats/build_high_bitdepth_gpu_test.sh on the sm_86 box."
fi
