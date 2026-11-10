#!/usr/bin/env bash
# Standalone build/test for the AV2 format / high-bit-depth CPU reference.
#
#   ./build_and_test.sh
#   CXX=g++ ./build_and_test.sh
#
# This does not touch the root CMake build and does not produce a bitstream.
# Build artifacts go to a temporary directory so nothing is committed.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
CXX="${CXX:-c++}"
BUILD_DIR="${BUILD_DIR:-${TMPDIR:-/tmp}/av2enc_format_build}"

mkdir -p "${BUILD_DIR}"

# Keep the generated Subsampled_Size table in sync with the spec dump.
if command -v python3 >/dev/null 2>&1 &&
   [[ -f "${REPO}/spec/tables/block_tables.json" ]]; then
  python3 "${HERE}/generate_format_tables.py" --check
fi

WARN=(-Wall -Wextra -Wconversion -Wshadow -Wpedantic)

"${CXX}" -std=c++20 -O2 -g \
  "${WARN[@]}" \
  -I"${REPO}/src" \
  "${HERE}/format.cpp" \
  "${HERE}/format_test.cpp" \
  -o "${BUILD_DIR}/format_cpu_test"

echo "built ${BUILD_DIR}/format_cpu_test"
"${BUILD_DIR}/format_cpu_test"

if command -v nvcc >/dev/null 2>&1; then
  echo
  echo "nvcc found: building the CPU-vs-CUDA sample-path test"
  BUILD_DIR="${BUILD_DIR}" \
    "${REPO}/src/cuda/format/build_high_bitdepth_test.sh"
  "${BUILD_DIR}/high_bitdepth_cuda_test"
else
  echo
  echo "nvcc not found: CPU reference only (CUDA test must run on the sm_86 box)"
fi
