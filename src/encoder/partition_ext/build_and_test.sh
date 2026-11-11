#!/usr/bin/env bash
# Build and run the extended-partition reference and syntax tests.
#
#   ./src/encoder/partition_ext/build_and_test.sh
#   CXX=clang++ ./src/encoder/partition_ext/build_and_test.sh
#   SANITIZE=1 ./src/encoder/partition_ext/build_and_test.sh
#   WERROR=1 ./src/encoder/partition_ext/build_and_test.sh
#   OUT=/tmp/partition_ext ./src/encoder/partition_ext/build_and_test.sh
#
# Self-contained: does not touch the root CMake build or any other component.
# The target is Linux + CUDA; this CPU binary is the correctness oracle only.

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${HERE}/../../.." && pwd)"
CXX="${CXX:-c++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
OUT="${OUT:-${TMPDIR:-/tmp}/partition_ext_test}"

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
if [[ "${SANITIZE}" == "1" ]]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2)
fi
if [[ "${WERROR}" == "1" ]]; then
  FLAGS+=(-Werror)
fi

COMMON=(
  "${ROOT}/src/encoder/partition/partition.cpp"
  "${HERE}/partition_ext.cpp"
)

"${CXX}" "${FLAGS[@]}" -I"${ROOT}/src" -I"${HERE}" \
  "${COMMON[@]}" "${HERE}/partition_ext_test.cpp" \
  -o "${OUT}"
"${OUT}"

"${CXX}" "${FLAGS[@]}" -I"${ROOT}/src" -I"${HERE}" \
  "${COMMON[@]}" "${HERE}/ext_syntax.cpp" \
  "${ROOT}/src/entropy/av2/av2_entropy.cpp" \
  "${HERE}/ext_syntax_test.cpp" \
  -o "${OUT}_syntax"
"${OUT}_syntax"
