#!/usr/bin/env bash
# Build and run the CPU correctness reference for the AV2 §8.2 entropy coder.
#
# This is intentionally a plain g++ C++20 build: the final target is CUDA on
# Linux, and the CPU binary only has to prove correctness.  The same header is
# compiled by the future .cu implementation; do not add platform-specific code
# here.
#
# Usage:
#   ./src/codec/entropy/av2/build_and_test.sh
#   CXX=g++-13 ./src/codec/entropy/av2/build_and_test.sh
#   SANITIZE=1 ./src/codec/entropy/av2/build_and_test.sh
#   WERROR=1 ./src/codec/entropy/av2/build_and_test.sh
#   OUT=/tmp/av2_entropy_test ./src/codec/entropy/av2/build_and_test.sh

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CXX="${CXX:-g++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
OUT="${OUT:-${TMPDIR:-/tmp}/av2_entropy_test}"

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow)

if [[ "${SANITIZE}" == "1" ]]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2)
fi

if [[ "${WERROR}" == "1" ]]; then
  FLAGS+=(-Werror)
fi

"${CXX}" "${FLAGS[@]}" -I"${HERE}" -I"${HERE}/../../.." \
  "${HERE}/av2_entropy.cpp" \
  "${HERE}/av2_entropy_test.cpp" \
  -o "${OUT}"

"${OUT}"
