#!/usr/bin/env bash
# CPU build/test for the explicit CfL alpha path.
#
#   ./build_and_test.sh
#   SANITIZE=1 WERROR=1 ./build_and_test.sh
#   ALPHA_PRINT_GOLDENS=1 ./build_and_test.sh      # print byte goldens
#
# Links the shared CfL core (../cfl.cpp) and the normative §8.2 entropy coder
# (src/entropy/av2), so the syntax tests run against the real range coder.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CXX="${CXX:-g++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
OUT="${OUT:-${TMPDIR:-/tmp}/cfl_alpha_test}"

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
if [[ "${SANITIZE}" == "1" ]]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2)
fi
if [[ "${WERROR}" == "1" ]]; then
  FLAGS+=(-Werror)
fi

"${CXX}" "${FLAGS[@]}" \
  -I"${HERE}" \
  "${HERE}/alpha.cpp" \
  "${HERE}/syntax.cpp" \
  "${HERE}/alpha_test.cpp" \
  "${HERE}/../cfl.cpp" \
  "${HERE}/../../../entropy/av2/av2_entropy.cpp" \
  -o "${OUT}"

"${OUT}"
