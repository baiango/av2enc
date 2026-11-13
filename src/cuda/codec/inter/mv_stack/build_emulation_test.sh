#!/usr/bin/env bash
# Host-emulation parity test for src/cuda/codec/inter/mv_stack (no GPU/nvcc).
#
#   ./build_emulation_test.sh
#   BUILD_DIR=/tmp/x ./build_emulation_test.sh
set -euo pipefail

HERE="$(cd -- "$(dirname -- "$0")" && pwd)"
ROOT="$(cd -- "${HERE}" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
  ROOT="$(dirname -- "${ROOT}")"
done

BUILD_DIR="${BUILD_DIR:-${ROOT}/build-mv-stack-emulation}"
CXX="${CXX:-c++}"
mkdir -p "${BUILD_DIR}"

"${CXX}" -std=c++20 -O2 -g -ffp-contract=off \
  -Wall -Wextra -Wconversion -Wshadow \
  -I"${ROOT}/src" -I"${HERE}" \
  "${HERE}/mv_stack_emulation_test.cpp" \
  "${ROOT}/src/codec/inter/merge/merge.cpp" \
  "${ROOT}/src/codec/inter/multiref/mvstack_multi.cpp" \
  "${ROOT}/src/codec/inter/multiref/compound_predict.cpp" \
  "${ROOT}/src/codec/inter/mc/mc.cpp" \
  "${ROOT}/src/codec/inter/advanced/compound.cpp" \
  "${ROOT}/src/codec/partition/tree/partition.cpp" \
  -o "${BUILD_DIR}/mv_stack_emulation_test"

"${BUILD_DIR}/mv_stack_emulation_test"
