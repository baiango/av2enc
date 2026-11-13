#!/usr/bin/env bash
# Host-only parity test for the shared device frame/filter geometry.
#
#   ./build_format_device_test.sh
#
# Links the product format reference (`src/formats/format.cpp`) and the 8-bit
# 4:2:0 deblock geometry (`src/codec/filters/deblock/deblock.cpp`).  No GPU or
# nvcc is needed.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd -- "${HERE}/../../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${TMPDIR:-/tmp}/av2enc_format_device_build}"
CXX="${CXX:-c++}"
mkdir -p "${BUILD_DIR}"

WARN=(-Wall -Wextra -Wconversion -Wshadow -Wpedantic)

"${CXX}" -std=c++20 -O2 -g -ffp-contract=off "${WARN[@]}" \
  -I"${ROOT}/src" \
  "${ROOT}/src/cuda/formats/format_device.cpp" \
  "${ROOT}/src/cuda/formats/format_device_test.cpp" \
  "${ROOT}/src/formats/format.cpp" \
  "${ROOT}/src/codec/filters/deblock/deblock.cpp" \
  -o "${BUILD_DIR}/format_device_test"

"${BUILD_DIR}/format_device_test"
