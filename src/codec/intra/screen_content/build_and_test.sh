#!/usr/bin/env bash
# Build and test the §5.4.7/§5.18.3.3 SCC signalling module.
#
#   ./src/codec/intra/screen_content/build_and_test.sh
#   ./src/codec/intra/screen_content/build_and_test.sh --no-run
#   SKIP_GATE=1 ./src/codec/intra/screen_content/build_and_test.sh
#   SANITIZE=1 WERROR=1 ./src/codec/intra/screen_content/build_and_test.sh
#   CXX=clang++ ./src/codec/intra/screen_content/build_and_test.sh
#
# Self-contained: does not touch the root CMake build, src/main.cpp, or any
# other module's sources.
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

# scc_flags + the two existing tile coders it validates.  No other module is
# modified; these are the documented call targets.
CORE=(
  "${HERE}/scc_flags.cpp"
  "${HERE}/scc_flags_bridge.cpp"
  "${ROOT}/src/core/bitstream/bitwriter.cpp"
  "${ROOT}/src/core/bitstream/basic_headers.cpp"
  "${ROOT}/src/pipeline/still/still_headers.cpp"
  "${ROOT}/src/codec/intra/modes/intra_modes.cpp"
  "${ROOT}/src/codec/partition/tree/partition.cpp"
  "${ROOT}/src/codec/transform/variants/tx_variants.cpp"
  "${ROOT}/src/codec/entropy/av2/av2_entropy.cpp"
  "${ROOT}/src/core/io/ivf.cpp"
  "${ROOT}/src/codec/intra/palette/core/palette.cpp"
  "${ROOT}/src/codec/intra/palette/core/scc_stream.cpp"
  "${ROOT}/src/codec/intra/palette/stream/palette_stream.cpp"
  "${ROOT}/src/codec/intra/ibc/core/ibc.cpp"
  "${HERE}/../ibc_stream/ibc_syntax.cpp"
  "${HERE}/../ibc_stream/ibc_stack.cpp"
  "${HERE}/../ibc_stream/still_ibc_stream.cpp"
)

echo "CXX: $($CXX --version | head -1)"
echo "build: ${BUILD_DIR}/scc_flags_test, ${BUILD_DIR}/scc_flags_gate"

"$CXX" "${FLAGS[@]}" -I"${ROOT}/src" \
  "${CORE[@]}" "${HERE}/scc_flags_test.cpp" \
  -o "${BUILD_DIR}/scc_flags_test"

"$CXX" "${FLAGS[@]}" -I"${ROOT}/src" \
  "${CORE[@]}" "${HERE}/scc_flags_gate.cpp" \
  -o "${BUILD_DIR}/scc_flags_gate"

if [[ "${RUN}" -eq 0 ]]; then exit 0; fi

"${BUILD_DIR}/scc_flags_test"

if [[ "${SKIP_GATE}" == "1" ]]; then exit 0; fi

"${BUILD_DIR}/scc_flags_gate" --out-prefix "${BUILD_DIR}/scc_flags_gate"

DAV2D="${DAV2D:-}"
if [[ -z "${DAV2D}" ]]; then
  for candidate in "${ROOT}/tools/bin/dav2d" "${ROOT}/tools/bin/dav2d-macos"; do
    if [[ -x "${candidate}" ]]; then DAV2D="${candidate}"; break; fi
  done
fi
if [[ -z "${DAV2D}" || ! -x "${DAV2D}" ]]; then
  echo "dav2d not found; set DAV2D=/path/to/dav2d (unit tests passed)" >&2
  exit 127
fi

for mode in off integer intrabc palette; do
  IVF="${BUILD_DIR}/scc_flags_gate.${mode}.ivf"
  REF="${BUILD_DIR}/scc_flags_gate.${mode}.y4m"
  DEC="${BUILD_DIR}/scc_flags_gate.${mode}.decoded.y4m"
  "${DAV2D}" -q --strict 1 --threads 1 -i "${IVF}" -o "${DEC}"
  if ! cmp -s "${REF}" "${DEC}"; then
    echo "FAIL: dav2d reconstruction differs for the ${mode} SCC gate" >&2
    exit 1
  fi
done
echo "OK: dav2d --strict 1 decoded all four SCC gates bit-exactly ($(${DAV2D} --version 2>&1 | tail -1))"
