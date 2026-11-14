#!/usr/bin/env bash
# Build and run the temporal-layer planner tests and decoder gate.
#
# The unit test is standalone (C++20, no CUDA). The decoder gate links the
# host M2 encoder as a single-reference carrier and then re-wraps its tile
# payloads with tlayers' sequence/frame headers, so it needs the M2 source
# list (still no CMake changes).
#
# Usage:
#   ./src/codec/inter/tlayers/build_and_test.sh
#   ./src/codec/inter/tlayers/build_and_test.sh --no-gate
#   SANITIZE=1 ./src/codec/inter/tlayers/build_and_test.sh --no-gate
#   WERROR=1   ./src/codec/inter/tlayers/build_and_test.sh --no-gate
#   OUT=/tmp/tlayers ./src/codec/inter/tlayers/build_and_test.sh
#   DAV2D=/path/to/dav2d ./src/codec/inter/tlayers/build_and_test.sh
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
CXX="${CXX:-c++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
OUT="${OUT:-${TMPDIR:-/tmp}/tlayers}"
BUILD="${BUILD:-${TMPDIR:-/tmp}/tlayers_build}"

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
       -Wno-missing-braces -I"${ROOT}/src")
if [[ "${SANITIZE}" == "1" ]]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2)
fi
if [[ "${WERROR}" == "1" ]]; then
  FLAGS+=(-Werror)
fi

RUN_GATE=1
for arg in "$@"; do
  case "$arg" in
    --no-gate) RUN_GATE=0 ;;
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

mkdir -p "${BUILD}" "${OUT}"

TEST_BIN="${BUILD}/tlayers_test"
"${CXX}" "${FLAGS[@]}" \
  "${HERE}/tlayers_test.cpp" \
  "${HERE}/tlayers.cpp" \
  "${HERE}/tlayers_syntax.cpp" \
  "${ROOT}/src/pipeline/video/video_headers.cpp" \
  "${ROOT}/src/core/bitstream/bitwriter.cpp" \
  "${ROOT}/src/core/bitstream/basic_headers.cpp" \
  -o "${TEST_BIN}"
"${TEST_BIN}"

if [[ "${RUN_GATE}" == "0" ]]; then
  exit 0
fi

GATE_BIN="${BUILD}/tlayers_gate"
VIDEO_CORE=(
  "${ROOT}/src/core/io/y4m.cpp"
  "${ROOT}/src/core/io/ivf.cpp"
  "${ROOT}/src/core/bitstream/bitwriter.cpp"
  "${ROOT}/src/core/bitstream/obu.cpp"
  "${ROOT}/src/core/bitstream/basic_headers.cpp"
  "${ROOT}/src/codec/partition/tree/partition.cpp"
  "${ROOT}/src/codec/intra/modes/intra_modes.cpp"
  "${ROOT}/src/codec/transform/variants/tx_variants.cpp"
  "${ROOT}/src/codec/transform/variants/tx_select.cpp"
  "${ROOT}/src/codec/entropy/av2/av2_entropy.cpp"
  "${ROOT}/src/codec/inter/me/me.cpp"
  "${ROOT}/src/codec/inter/mc/mc.cpp"
  "${ROOT}/src/codec/inter/merge/merge.cpp"
  "${ROOT}/src/pipeline/video/video_headers.cpp"
  "${ROOT}/src/pipeline/video/video_syntax.cpp"
  "${ROOT}/src/pipeline/video/video_frame.cpp"
  "${ROOT}/src/codec/inter/temporal/ref_mv_bank.cpp"
)
# M2 core is not -Werror clean; the gate build keeps the warnings visible.
"${CXX}" "${FLAGS[@]}" \
  "${HERE}/tlayers_gate.cpp" \
  "${HERE}/tlayers.cpp" \
  "${HERE}/tlayers_syntax.cpp" \
  "${VIDEO_CORE[@]}" \
  -o "${GATE_BIN}"

if [[ -z "${DAV2D:-}" ]]; then
  for candidate in "${ROOT}/tools/bin/dav2d" "${ROOT}/tools/bin/dav2d-macos"; do
    if [[ -x "${candidate}" ]]; then
      export DAV2D="${candidate}"
      break
    fi
  done
fi
if [[ -z "${DAV2D:-}" ]]; then
  echo "tlayers gate skipped: pinned dav2d not found (set DAV2D)" >&2
  exit 0
fi

cd "${ROOT}"
"${GATE_BIN}" --width 64 --height 64 --frames 8 --out "${OUT}/64x64"
"${GATE_BIN}" --width 640 --height 360 --frames 8 --out "${OUT}/640x360"
