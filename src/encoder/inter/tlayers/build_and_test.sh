#!/usr/bin/env bash
# Build and run the temporal-layer planner tests and decoder gate.
#
# The unit test is standalone (C++20, no CUDA). The decoder gate links the
# host M2 encoder as a single-reference carrier and then re-wraps its tile
# payloads with tlayers' sequence/frame headers, so it needs the M2 source
# list (still no CMake changes).
#
# Usage:
#   ./src/encoder/inter/tlayers/build_and_test.sh
#   ./src/encoder/inter/tlayers/build_and_test.sh --no-gate
#   SANITIZE=1 ./src/encoder/inter/tlayers/build_and_test.sh --no-gate
#   WERROR=1   ./src/encoder/inter/tlayers/build_and_test.sh --no-gate
#   OUT=/tmp/tlayers ./src/encoder/inter/tlayers/build_and_test.sh
#   DAV2D=/path/to/dav2d ./src/encoder/inter/tlayers/build_and_test.sh
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${HERE}/../../../.." && pwd)"
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
  "${ROOT}/src/encoder/m2/m2_headers.cpp" \
  "${ROOT}/src/bitstream/bitwriter.cpp" \
  "${ROOT}/src/bitstream/m0_headers.cpp" \
  -o "${TEST_BIN}"
"${TEST_BIN}"

if [[ "${RUN_GATE}" == "0" ]]; then
  exit 0
fi

GATE_BIN="${BUILD}/tlayers_gate"
M2_CORE=(
  "${ROOT}/src/io/y4m.cpp"
  "${ROOT}/src/io/ivf.cpp"
  "${ROOT}/src/bitstream/bitwriter.cpp"
  "${ROOT}/src/bitstream/obu.cpp"
  "${ROOT}/src/bitstream/m0_headers.cpp"
  "${ROOT}/src/encoder/partition/partition.cpp"
  "${ROOT}/src/encoder/intra_modes/intra_modes.cpp"
  "${ROOT}/src/encoder/tx_variants/tx_variants.cpp"
  "${ROOT}/src/encoder/tx_variants/tx_select.cpp"
  "${ROOT}/src/entropy/av2/av2_entropy.cpp"
  "${ROOT}/src/encoder/inter/me/me.cpp"
  "${ROOT}/src/encoder/inter/mc/mc.cpp"
  "${ROOT}/src/encoder/m2/merge/merge.cpp"
  "${ROOT}/src/encoder/m2/m2_headers.cpp"
  "${ROOT}/src/encoder/m2/m2_syntax.cpp"
  "${ROOT}/src/encoder/m2/m2_frame.cpp"
)
# M2 core is not -Werror clean; the gate build keeps the warnings visible.
"${CXX}" "${FLAGS[@]}" \
  "${HERE}/tlayers_gate.cpp" \
  "${HERE}/tlayers.cpp" \
  "${HERE}/tlayers_syntax.cpp" \
  "${M2_CORE[@]}" \
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
