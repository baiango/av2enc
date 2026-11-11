#!/usr/bin/env bash
# Standalone build/test for the 10-bit M1 transform + 4:2:0 sample path.
#
#   ./build_hbd_test.sh
#   CXX=g++ ./build_hbd_test.sh
#
# The CPU test always runs.  If nvcc is on PATH the CPU-vs-GPU bit-exact test
# is built for sm_86 and run too.  Build artifacts go to a temporary directory;
# nothing is committed and the root CMake build is untouched.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "${HERE}/../../../.." && pwd)"
ARCH="${CUDA_ARCH:-sm_86}"
CXX="${CXX:-c++}"
BUILD_DIR="${BUILD_DIR:-${TMPDIR:-/tmp}/av2enc_hbd_build}"
mkdir -p "${BUILD_DIR}"

WARN=(-Wall -Wextra -Wconversion -Wshadow -Wpedantic)

echo "building CPU reference + test"
"${CXX}" -std=c++20 -O2 -g -ffp-contract=off \
  "${WARN[@]}" \
  -I"${REPO}/src" \
  "${HERE}/hbd_cpu.cpp" \
  "${HERE}/hbd_test.cpp" \
  "${REPO}/src/encoder/tx_variants/tx_variants.cpp" \
  "${REPO}/src/encoder/format/format.cpp" \
  -o "${BUILD_DIR}/hbd_cpu_test"

"${BUILD_DIR}/hbd_cpu_test"

if command -v nvcc >/dev/null 2>&1; then
  echo
  echo "nvcc found: building the CPU-vs-GPU bit-exact test (${ARCH})"
  CPU_OBJ="${BUILD_DIR}/hbd_cpu_ref.o"
  TXV_OBJ="${BUILD_DIR}/tx_variants_cpu.o"
  FMT_OBJ="${BUILD_DIR}/format_cpu.o"

  "${CXX}" -std=c++20 -O3 -ffp-contract=off \
    -Wall -Wextra -Wconversion -Wshadow \
    -I"${REPO}/src" \
    -c "${HERE}/hbd_cpu.cpp" -o "${CPU_OBJ}"
  "${CXX}" -std=c++20 -O3 -ffp-contract=off \
    -Wall -Wextra -Wconversion -Wshadow \
    -I"${REPO}/src" \
    -c "${REPO}/src/encoder/tx_variants/tx_variants.cpp" -o "${TXV_OBJ}"
  "${CXX}" -std=c++20 -O3 \
    -Wall -Wextra -Wconversion -Wshadow \
    -I"${REPO}/src" \
    -c "${REPO}/src/encoder/format/format.cpp" -o "${FMT_OBJ}"

  nvcc -std=c++17 \
    -arch="${ARCH}" \
    -O3 \
    -lineinfo \
    -fmad=false \
    -I"${REPO}/src" \
    -Xcompiler=-Wall,-Wextra,-Wconversion,-Wshadow,-ffp-contract=off \
    "${HERE}/hbd_frame.cu" \
    "${HERE}/hbd_tx.cu" \
    "${HERE}/hbd_test.cu" \
    "${CPU_OBJ}" "${TXV_OBJ}" "${FMT_OBJ}" \
    -o "${BUILD_DIR}/hbd_cuda_test"

  "${BUILD_DIR}/hbd_cuda_test"
else
  echo
  echo "nvcc not found: CPU reference only (CUDA test must run on the sm_86 box)"
fi
