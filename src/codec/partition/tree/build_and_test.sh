#!/usr/bin/env bash
# Build and run the CPU reference for the AV2 recursive partition tree.
#
#   ./src/codec/partition/tree/build_and_test.sh
#   CXX=clang++ ./src/codec/partition/tree/build_and_test.sh
#   SANITIZE=1 ./src/codec/partition/tree/build_and_test.sh
#   WERROR=1 ./src/codec/partition/tree/build_and_test.sh
#   OUT=/tmp/partition_test ./src/codec/partition/tree/build_and_test.sh
#
# Self-contained: does not touch the root CMake build or any other component.
# The target is Linux + CUDA; this CPU binary is the correctness oracle only.

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CXX="${CXX:-c++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
OUT="${OUT:-${TMPDIR:-/tmp}/partition_test}"

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow)

if [[ "${SANITIZE}" == "1" ]]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2)
fi

if [[ "${WERROR}" == "1" ]]; then
  FLAGS+=(-Werror)
fi

"${CXX}" "${FLAGS[@]}" -I"${HERE}" \
  "${HERE}/partition.cpp" \
  "${HERE}/partition_test.cpp" \
  -o "${OUT}"

"${OUT}"
