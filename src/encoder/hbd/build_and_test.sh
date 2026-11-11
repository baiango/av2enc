#!/usr/bin/env bash
# Standalone build/test for the 10-bit `src/encoder/hbd/` component.
#
#   ./build_and_test.sh              # unit tests + encoder tool
#   GATE=1 ./build_and_test.sh       # also run the dav2d gate (64x64/640x360/4K)
#   SANITIZE=1 ./build_and_test.sh   # ASan/UBSan unit test
#   WERROR=1 ./build_and_test.sh     # -Werror
#
# No CMake changes: this component is not wired into the product build.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${TMPDIR:-/tmp}/av2enc_hbd_build}"
CXX="${CXX:-c++}"
mkdir -p "${BUILD_DIR}"

WARN=(-Wall -Wextra -Wconversion -Wshadow -Wpedantic -Wno-sign-conversion)
if [ "${WERROR:-0}" = "1" ]; then
  WARN+=(-Werror)
fi
OPT=(-O2 -g -ffp-contract=off)
if [ "${SANITIZE:-0}" = "1" ]; then
  OPT=(-O1 -g -ffp-contract=off -fsanitize=address,undefined
       -fno-omit-frame-pointer)
fi

COMMON=(
  "${HERE}/hbd_depth.cpp"
  "${HERE}/hbd_headers.cpp"
  "${HERE}/hbd_reader.cpp"
  "${HERE}/hbd_still.cpp"
  "${REPO}/src/encoder/m10/m10_still.cpp"
  "${REPO}/src/encoder/m10/m10_entropy.cpp"
  "${REPO}/src/encoder/m10/m10_headers.cpp"
  "${REPO}/src/encoder/format/format.cpp"
  "${REPO}/src/encoder/tx_variants/tx_variants.cpp"
  "${REPO}/src/cuda/m1/hbd/hbd_cpu.cpp"
  "${REPO}/src/entropy/av2/av2_entropy.cpp"
  "${REPO}/src/bitstream/bitwriter.cpp"
  "${REPO}/src/bitstream/obu.cpp"
  "${REPO}/src/bitstream/m0_headers.cpp"
)

echo "building hbd unit test"
"${CXX}" -std=c++20 "${OPT[@]}" "${WARN[@]}" -I"${REPO}/src" \
  "${COMMON[@]}" "${HERE}/hbd_test.cpp" \
  -o "${BUILD_DIR}/hbd_test"
"${BUILD_DIR}/hbd_test"

echo
echo "building hbd encoder tool"
"${CXX}" -std=c++20 "${OPT[@]}" "${WARN[@]}" -I"${REPO}/src" \
  "${COMMON[@]}" "${REPO}/src/encoder/m10/m10_y4m.cpp" \
  "${REPO}/src/io/ivf.cpp" "${HERE}/hbd_encode.cpp" \
  -o "${BUILD_DIR}/hbd_encode"
echo "tool: ${BUILD_DIR}/hbd_encode"

if [ "${GATE:-0}" = "1" ]; then
  echo
  HBD_ENCODE="${BUILD_DIR}/hbd_encode" "${HERE}/gate_hbd_still.sh"
fi
