#!/usr/bin/env bash
# Standalone build/test for the 10-bit `src/formats/high_bitdepth/` component.
#
#   ./build_and_test.sh              # unit tests + encoder tool
#   GATE=1 ./build_and_test.sh       # also run the dav2d gate (64x64/640x360/4K)
#   SANITIZE=1 ./build_and_test.sh   # ASan/UBSan unit test
#   WERROR=1 ./build_and_test.sh     # -Werror
#
# No CMake changes: this component is not wired into the product build.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${REPO}/CMakeLists.txt" ] && [ "${REPO}" != "/" ]; do
    REPO="$(dirname -- "${REPO}")"
done
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
  "${REPO}/src/pipeline/still/high_bitdepth/high_bitdepth_still.cpp"
  "${REPO}/src/pipeline/still/high_bitdepth/high_bitdepth_entropy.cpp"
  "${REPO}/src/pipeline/still/high_bitdepth/high_bitdepth_headers.cpp"
  "${REPO}/src/formats/format.cpp"
  "${REPO}/src/codec/transform/variants/tx_variants.cpp"
  "${REPO}/src/codec/filters/deblock/deblock.cpp"
  "${REPO}/src/codec/filters/cdef/cdef.cpp"
  "${REPO}/src/codec/filters/loop_restoration/restoration.cpp"
  "${REPO}/src/codec/filters/pc_wiener/pc_wiener_stream.cpp"
  "${REPO}/src/codec/filters/tables/pc_wiener.cpp"
  "${REPO}/src/codec/filters/tables/luma_source.cpp"
  "${REPO}/src/codec/filters/gdf/gdf_syntax.cpp"
  "${REPO}/src/codec/filters/gdf/gdf_decision.cpp"
  "${REPO}/src/codec/filters/tables/gdf.cpp"
  "${REPO}/src/cuda/pipeline/still/hbd/hbd_cpu.cpp"
  "${REPO}/src/codec/entropy/av2/av2_entropy.cpp"
  "${REPO}/src/core/bitstream/bitwriter.cpp"
  "${REPO}/src/core/bitstream/obu.cpp"
  "${REPO}/src/core/bitstream/basic_headers.cpp"
)

echo "building hbd unit test"
"${CXX}" -std=c++20 "${OPT[@]}" "${WARN[@]}" -I"${REPO}/src" \
  "${COMMON[@]}" "${HERE}/hbd_test.cpp" \
  -o "${BUILD_DIR}/hbd_test"
"${BUILD_DIR}/hbd_test"

echo
echo "building hbd encoder tool"
"${CXX}" -std=c++20 "${OPT[@]}" "${WARN[@]}" -I"${REPO}/src" \
  "${COMMON[@]}" "${REPO}/src/pipeline/still/high_bitdepth/high_bitdepth_y4m.cpp" \
  "${REPO}/src/core/io/ivf.cpp" "${HERE}/hbd_encode.cpp" \
  -o "${BUILD_DIR}/hbd_encode"
echo "tool: ${BUILD_DIR}/hbd_encode"

if [ "${GATE:-0}" = "1" ]; then
  echo
  HBD_ENCODE="${BUILD_DIR}/hbd_encode" "${HERE}/gate_hbd_still.sh"
fi
