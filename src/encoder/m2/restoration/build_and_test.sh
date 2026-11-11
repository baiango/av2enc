#!/usr/bin/env bash
# Build and run the M2 loop-restoration adapter gates.
#
#   ./build_and_test.sh              # CPU-only, optimised
#   SANITIZE=1 ./build_and_test.sh   # ASan + UBSan
#   WERROR=1 ./build_and_test.sh     # warnings as errors
#
# Self-contained: no CUDA, no root CMake changes.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../../../.." && pwd)"
CXX="${CXX:-c++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
OUT="${OUT:-${TMPDIR:-/tmp}/m2_restoration_test}"

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
  "$ROOT/src/bitstream/bitwriter.cpp"
  "$ROOT/src/entropy/av2/av2_entropy.cpp"
  "$ROOT/src/encoder/filters/restoration/restoration.cpp"
  "$ROOT/src/encoder/m1/filters_stream/filters_stream.cpp"
  "$ROOT/src/encoder/m2/restoration/m2_restoration.cpp"
)

"${CXX}" "${FLAGS[@]}" \
  "$ROOT/src/encoder/m2/restoration/m2_restoration_test.cpp" \
  "${CORE[@]}" \
  -o "${OUT}"

"${OUT}"
