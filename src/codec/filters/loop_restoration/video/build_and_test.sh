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
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
CXX="${CXX:-c++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
OUT="${OUT:-${TMPDIR:-/tmp}/video_restoration_test}"

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
  "$ROOT/src/core/bitstream/bitwriter.cpp"
  "$ROOT/src/codec/entropy/av2/av2_entropy.cpp"
  "$ROOT/src/codec/filters/loop_restoration/restoration.cpp"
  "$ROOT/src/codec/filters/stream/filters_stream.cpp"
  "$ROOT/src/codec/filters/loop_restoration/video/video_restoration.cpp"
)

"${CXX}" "${FLAGS[@]}" \
  "$ROOT/src/codec/filters/loop_restoration/video/video_restoration_test.cpp" \
  "${CORE[@]}" \
  -o "${OUT}"

"${OUT}"
