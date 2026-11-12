#!/usr/bin/env bash
# Build and run the PC-Wiener stream unit tests.
#
#   ./build_and_test.sh              # unit tests
#   SANITIZE=1 ./build_and_test.sh   # ASan/UBSan
#   WERROR=1 ./build_and_test.sh
#
# Nothing outside this directory is modified.  The tests link the existing
# bit writer, §8.2 entropy coder and the recovered spec_tables PC-Wiener
# reference; no production core file is edited.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
CXX="${CXX:-c++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
OUT_DIR="${OUT_DIR:-${TMPDIR:-/tmp}/pc_wiener_stream_build}"

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
       -Wno-missing-braces -Wno-sign-conversion -I"$ROOT/src")
if [[ "${SANITIZE}" == "1" ]]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2)
fi
if [[ "${WERROR}" == "1" ]]; then
  FLAGS+=(-Werror -Wno-error=sign-conversion -Wno-error=missing-braces)
fi

SOURCES=(
  "$HERE/pc_wiener_stream_test.cpp"
  "$HERE/pc_wiener_stream.cpp"
  "$ROOT/src/codec/filters/tables/pc_wiener.cpp"
  "$ROOT/src/codec/filters/tables/luma_source.cpp"
  "$ROOT/src/core/bitstream/bitwriter.cpp"
  "$ROOT/src/codec/entropy/av2/av2_entropy.cpp"
)

mkdir -p "$OUT_DIR"
"${CXX}" "${FLAGS[@]}" "${SOURCES[@]}" -o "$OUT_DIR/pc_wiener_stream_test"
"$OUT_DIR/pc_wiener_stream_test"
