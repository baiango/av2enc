#!/usr/bin/env bash
# Build and run src/codec/intra/supplemental/ (tests + dav2d gate).
#
#   ./src/codec/intra/supplemental/build_and_test.sh
#   SANITIZE=1 WERROR=1 ./src/codec/intra/supplemental/build_and_test.sh
#   SKIP_GATE=1 ./src/codec/intra/supplemental/build_and_test.sh
#   ./src/codec/intra/supplemental/build_and_test.sh --dump-goldens
#
# Adds files only: no root CMake, src/main.cpp or M1 core source is modified.
# The gate requires tools/bin/dav2d or tools/bin/dav2d-macos (override DAV2D).
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
CXX="${CXX:-c++}"
BUILD_DIR="${BUILD_DIR:-${HERE}/build}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
SKIP_GATE="${SKIP_GATE:-0}"
DUMP=0

for arg in "$@"; do
  case "$arg" in
    --dump-goldens) DUMP=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
       -Wno-missing-braces)
OUR_FLAGS=("${FLAGS[@]}")
if [[ "${WERROR}" == "1" ]]; then OUR_FLAGS+=(-Werror); fi
if [[ "${SANITIZE}" == "1" ]]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
  OUR_FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2)
  OUR_FLAGS+=(-O2)
fi

mkdir -p "${BUILD_DIR}"

CORE=(
  "${ROOT}/src/codec/intra/modes/intra_modes.cpp"
  "${ROOT}/src/codec/entropy/av2/av2_entropy.cpp"
)
MISSING=(
  "${HERE}/dc_intra_subsampled.cpp"
  "${HERE}/intra_filter_corner.cpp"
  "${HERE}/dc_sign_horz_vert.cpp"
  "${HERE}/save_subpu.cpp"
)

echo "CXX: $($CXX --version | head -1)"
echo "build: ${BUILD_DIR}/missing_bits_test"

"$CXX" "${OUR_FLAGS[@]}" -I"${ROOT}/src" \
  "${MISSING[@]}" "${CORE[@]}" "${HERE}/missing_bits_test.cpp" \
  -o "${BUILD_DIR}/missing_bits_test"

if [[ "${DUMP}" -eq 1 ]]; then
  "${BUILD_DIR}/missing_bits_test" --dump-goldens
  exit 0
fi

"${BUILD_DIR}/missing_bits_test"

if [[ "${SKIP_GATE}" == "1" ]]; then
  exit 0
fi

DAV2D="${DAV2D:-}"
if [[ -z "${DAV2D}" ]]; then
  for candidate in "${ROOT}/tools/bin/dav2d" "${ROOT}/tools/bin/dav2d-macos"; do
    if [[ -x "${candidate}" ]]; then DAV2D="${candidate}"; break; fi
  done
fi
if [[ -z "${DAV2D}" || ! -x "${DAV2D}" ]]; then
  echo "dav2d not found; set DAV2D=/path/to/dav2d (tests already passed)" >&2
  exit 77
fi

echo "== dc_sign_horz_vert dav2d gate =="
GATE_DIR="${BUILD_DIR}/gate_out"
rm -rf "${GATE_DIR}"
mkdir -p "${GATE_DIR}"

GATE_CORE=(
  "${ROOT}/src/core/bitstream/bitwriter.cpp"
  "${ROOT}/src/core/bitstream/obu.cpp"
  "${ROOT}/src/core/bitstream/basic_headers.cpp"
  "${ROOT}/src/codec/partition/tree/partition.cpp"
  "${ROOT}/src/codec/partition/ext/partition_ext.cpp"
  "${ROOT}/src/codec/partition/ext/ext_syntax.cpp"
  "${ROOT}/src/codec/partition/stream/partition_stream.cpp"
  "${ROOT}/src/pipeline/still/still_headers.cpp"
  "${ROOT}/src/core/io/ivf.cpp"
  "${ROOT}/src/codec/entropy/av2/av2_entropy.cpp"
  "${ROOT}/src/codec/intra/modes/intra_modes.cpp"
)
"$CXX" "${FLAGS[@]}" -I"${ROOT}/src" \
  "${MISSING[@]}" "${GATE_CORE[@]}" "${HERE}/gate_dc_sign_horz_vert.cpp" \
  -o "${BUILD_DIR}/gate_dc_sign_horz_vert"

"${BUILD_DIR}/gate_dc_sign_horz_vert" --out "${GATE_DIR}"
python3 "${HERE}/gate_dav2d.py" --dir "${GATE_DIR}" --decoder "${DAV2D}"
