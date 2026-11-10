#!/usr/bin/env bash
# Build and run the CPU correctness reference for AV2 §7.13.5 CfL.
#
# This is intentionally a plain C++20 build: the product target is CUDA on
# Linux and the CPU binary only has to prove correctness. The CUDA kernel in
# src/cuda/cfl/ includes the same integer helpers from cfl_common.h.
#
# Usage:
#   ./src/encoder/cfl/build_and_test.sh
#   CXX=g++-13 ./src/encoder/cfl/build_and_test.sh
#   SANITIZE=1 ./src/encoder/cfl/build_and_test.sh
#   WERROR=1 ./src/encoder/cfl/build_and_test.sh
#   OUT=/tmp/cfl_test ./src/encoder/cfl/build_and_test.sh

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CXX="${CXX:-g++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
OUT="${OUT:-${TMPDIR:-/tmp}/cfl_test}"

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
  "${HERE}/cfl.cpp" \
  "${HERE}/cfl_test.cpp" \
  -o "${OUT}"

"${OUT}"
