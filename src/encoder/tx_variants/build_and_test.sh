#!/usr/bin/env bash
# Standalone build/test for the AV2 transform-variants CPU reference.
#
#   ./build_and_test.sh
#   CXX=g++ ./build_and_test.sh
#
# If nvcc is on PATH this also builds and runs the CPU-vs-CUDA bit-exact test
# (src/cuda/tx_variants/build_tx_variants_test.sh).  The CPU test is the only
# part that runs on macOS; the CUDA half requires the Linux/CUDA box.
#
# Build artifacts go to a temporary directory so nothing is committed.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
CXX="${CXX:-c++}"
BUILD_DIR="${BUILD_DIR:-${TMPDIR:-/tmp}/av2enc_tx_variants_build}"

mkdir -p "${BUILD_DIR}"

WARN=(-Wall -Wextra -Wconversion -Wshadow -Wpedantic)

"${CXX}" -std=c++20 -O2 -g \
  "${WARN[@]}" \
  -ffp-contract=off \
  -I"${REPO}/src" \
  "${HERE}/tx_variants.cpp" \
  "${HERE}/tx_select.cpp" \
  "${HERE}/tx_variants_test.cpp" \
  -o "${BUILD_DIR}/tx_variants_cpu_test"

echo "built ${BUILD_DIR}/tx_variants_cpu_test"
"${BUILD_DIR}/tx_variants_cpu_test"

if command -v nvcc >/dev/null 2>&1; then
  echo
  echo "nvcc found: building the CUDA component test"
  BUILD_DIR="${BUILD_DIR}" \
    "${REPO}/src/cuda/tx_variants/build_tx_variants_test.sh"
  "${BUILD_DIR}/tx_variants_cuda_test"
else
  echo
  echo "nvcc not found: CPU reference only (CUDA test must run on the sm_86 box)"
fi
