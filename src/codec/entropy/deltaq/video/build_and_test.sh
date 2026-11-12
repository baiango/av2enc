#!/usr/bin/env bash
# Build and run the M2 delta-Q / segmentation adapter tests.
#
# Host-only CPU reference build.  The normative syntax tests live in
# src/codec/entropy/deltaq/still/build_and_test.sh; this script covers the M2
# frame/block glue.
#
# Usage:
#   ./src/codec/entropy/deltaq/video/build_and_test.sh
#   SANITIZE=1 ./src/codec/entropy/deltaq/video/build_and_test.sh
#   WERROR=1   ./src/codec/entropy/deltaq/video/build_and_test.sh
#   OUT=/tmp/video_deltaq_test ./src/codec/entropy/deltaq/video/build_and_test.sh
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
CXX="${CXX:-c++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
OUT="${OUT:-${TMPDIR:-/tmp}/video_deltaq_test}"

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow -I"$ROOT/src")
if [[ "${SANITIZE}" == "1" ]]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2)
fi
if [[ "${WERROR}" == "1" ]]; then
  FLAGS+=(-Werror)
fi

"${CXX}" "${FLAGS[@]}" \
  "$HERE/video_deltaq_test.cpp" \
  "$HERE/video_deltaq.cpp" \
  "$ROOT/src/codec/entropy/deltaq/still/deltaq.cpp" \
  "$ROOT/src/codec/partition/tree/partition.cpp" \
  "$ROOT/src/core/bitstream/bitwriter.cpp" \
  "$ROOT/src/codec/entropy/av2/av2_entropy.cpp" \
  "$ROOT/src/codec/rate/rc/rc.cpp" \
  -o "${OUT}"

"${OUT}"
