#!/usr/bin/env bash
# Build and test the M1 palette syntax adapter and its minimal dav2d gate.
#
#   ./src/codec/intra/palette/stream/build_and_test.sh
#   ./src/codec/intra/palette/stream/build_and_test.sh --no-run
#   SKIP_GATE=1 ./src/codec/intra/palette/stream/build_and_test.sh
#   SANITIZE=1 WERROR=1 ./src/codec/intra/palette/stream/build_and_test.sh
#   CXX=clang++ ./src/codec/intra/palette/stream/build_and_test.sh
#
# Self-contained: does not touch the root CMake build, src/main.cpp, or the M1
# core sources.  CPU-only correctness oracle.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
CXX="${CXX:-c++}"
BUILD_DIR="${BUILD_DIR:-${HERE}/build}"
RUN=1
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
SKIP_GATE="${SKIP_GATE:-0}"

for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
       -Wno-missing-braces)
if [[ "${WERROR}" == "1" ]]; then FLAGS+=(-Werror); fi
if [[ "${SANITIZE}" == "1" ]]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2)
fi

mkdir -p "${BUILD_DIR}"

CORE=(
  "${ROOT}/src/codec/intra/palette/core/palette.cpp"
  "${ROOT}/src/codec/intra/palette/core/scc_stream.cpp"
  "${ROOT}/src/core/bitstream/bitwriter.cpp"
  "${ROOT}/src/core/bitstream/basic_headers.cpp"
  "${ROOT}/src/core/io/ivf.cpp"
  "${ROOT}/src/codec/entropy/av2/av2_entropy.cpp"
  "${ROOT}/src/pipeline/still/still_headers.cpp"
  "${ROOT}/src/codec/intra/modes/intra_modes.cpp"
  "${ROOT}/src/codec/partition/tree/partition.cpp"
  "${ROOT}/src/codec/transform/variants/tx_variants.cpp"
  "${HERE}/palette_stream.cpp"
)

echo "CXX: $($CXX --version | head -1)"
echo "build: ${BUILD_DIR}/palette_stream_test, ${BUILD_DIR}/palette_stream_gate"

"$CXX" "${FLAGS[@]}" -I"${ROOT}/src" \
  "${CORE[@]}" "${HERE}/palette_stream_test.cpp" \
  -o "${BUILD_DIR}/palette_stream_test"

"$CXX" "${FLAGS[@]}" -I"${ROOT}/src" \
  "${CORE[@]}" "${HERE}/palette_stream_gate.cpp" \
  -o "${BUILD_DIR}/palette_stream_gate"

if [[ "${RUN}" -eq 0 ]]; then exit 0; fi

"${BUILD_DIR}/palette_stream_test"

if [[ "${SKIP_GATE}" == "1" ]]; then exit 0; fi

DAV2D="${DAV2D:-}"
if [[ -z "${DAV2D}" ]]; then
  for candidate in "${ROOT}/tools/bin/dav2d" "${ROOT}/tools/bin/dav2d-macos"; do
    if [[ -x "${candidate}" ]]; then DAV2D="${candidate}"; break; fi
  done
fi
if [[ -z "${DAV2D}" || ! -x "${DAV2D}" ]]; then
  echo "dav2d not found; set DAV2D=/path/to/dav2d (tests already passed)" >&2
  exit 127
fi

for mode in single split vert wide; do
  OUT="${BUILD_DIR}/palette_stream_gate_${mode}.ivf"
  REF="${BUILD_DIR}/palette_stream_gate_${mode}_ref.y4m"
  DEC="${BUILD_DIR}/palette_stream_gate_${mode}_decoded.y4m"
  "${BUILD_DIR}/palette_stream_gate" --mode "${mode}" --out "${OUT}" --ref "${REF}"
  "${DAV2D}" -q -i "${OUT}" -o "${DEC}" --threads 0
  if ! cmp -s "${REF}" "${DEC}"; then
    echo "FAIL: dav2d reconstruction differs for the ${mode} palette gate" >&2
    exit 1
  fi
done
echo "OK: dav2d decoded all four M1-palette gates bit-exactly ($(${DAV2D} --version | tail -1))"
