#!/usr/bin/env bash
# Build and run the chroma loop-restoration unit tests.
#
#   ./build_and_test.sh
#   SANITIZE=1 ./build_and_test.sh
#   WERROR=1   ./build_and_test.sh
#
# The decoder gate lives in gate/build_and_test.sh; the quality probe lives in
# gate/run_quality.sh.  This module is host-only and does not touch CMake.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
CXX="${CXX:-c++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
OUT="${OUT:-${TMPDIR:-/tmp}/av2enc-lr-chroma-test}"

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
       -Wno-missing-braces -I"$ROOT/src")
if [[ "${SANITIZE}" == "1" ]]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2)
fi
if [[ "${WERROR}" == "1" ]]; then
  FLAGS+=(-Werror)
fi

CORE=(
  "$HERE/chroma_lr.cpp"
  "$ROOT/src/codec/filters/loop_restoration/restoration.cpp"
  "$ROOT/src/codec/filters/loop_restoration/classes/wiener_fit.cpp"
  "$ROOT/src/codec/filters/loop_restoration/classes/wiener_syntax.cpp"
  "$ROOT/src/codec/filters/loop_restoration/classes/wiener_class.cpp"
  "$ROOT/src/codec/filters/stream/filters_stream.cpp"
  "$ROOT/src/codec/entropy/av2/av2_entropy.cpp"
  "$ROOT/src/core/bitstream/bitwriter.cpp"
)

"${CXX}" "${FLAGS[@]}" "$HERE/chroma_lr_test.cpp" "${CORE[@]}" -o "$OUT"

"$OUT"
