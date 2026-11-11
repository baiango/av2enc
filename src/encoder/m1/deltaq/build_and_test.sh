#!/usr/bin/env bash
# Build and run the delta-Q / segmentation syntax tests.
#
# Host-only CPU reference build (the module has no CUDA dependency).
#
# Usage:
#   ./src/encoder/m1/deltaq/build_and_test.sh
#   SANITIZE=1 ./src/encoder/m1/deltaq/build_and_test.sh
#   WERROR=1   ./src/encoder/m1/deltaq/build_and_test.sh
#   OUT=/tmp/deltaq_test ./src/encoder/m1/deltaq/build_and_test.sh
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../../../.." && pwd)"
CXX="${CXX:-c++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
OUT="${OUT:-${TMPDIR:-/tmp}/deltaq_test}"

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow -I"$ROOT/src")
if [[ "${SANITIZE}" == "1" ]]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2)
fi
if [[ "${WERROR}" == "1" ]]; then
  FLAGS+=(-Werror)
fi

python3 "$HERE/generate_deltaq_tables.py" --check

"${CXX}" "${FLAGS[@]}" \
  "$HERE/deltaq_test.cpp" \
  "$HERE/deltaq.cpp" \
  "$ROOT/src/bitstream/bitwriter.cpp" \
  "$ROOT/src/entropy/av2/av2_entropy.cpp" \
  "$ROOT/src/encoder/rc/rc.cpp" \
  -o "${OUT}"

"${OUT}"
