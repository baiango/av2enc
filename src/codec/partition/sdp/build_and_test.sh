#!/usr/bin/env bash
# Build and run the SDP (semi-decoupled partitioning) CPU reference tests.
#
#   ./src/codec/partition/sdp/build_and_test.sh
#   CXX=clang++ ./src/codec/partition/sdp/build_and_test.sh
#   SANITIZE=1 ./src/codec/partition/sdp/build_and_test.sh
#   WERROR=1 ./src/codec/partition/sdp/build_and_test.sh
#   OUT=/tmp/sdp_test ./src/codec/partition/sdp/build_and_test.sh
#
# Self-contained: reuses codec/partition/tree and codec/partition/ext but does
# not touch the root CMake build or any other component.  The product target is
# Linux + CUDA; this CPU binary is the correctness oracle only.

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
CXX="${CXX:-c++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
OUT="${OUT:-${TMPDIR:-/tmp}/sdp_test}"

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
if [[ "${SANITIZE}" == "1" ]]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2)
fi
if [[ "${WERROR}" == "1" ]]; then
  FLAGS+=(-Werror)
fi

SOURCES=(
  "${ROOT}/src/codec/partition/tree/partition.cpp"
  "${ROOT}/src/codec/partition/ext/partition_ext.cpp"
  "${HERE}/sdp.cpp"
  "${HERE}/sdp_validate.cpp"
  "${HERE}/sdp_syntax.cpp"
  "${HERE}/sdp_test.cpp"
)

"${CXX}" "${FLAGS[@]}" -I"${ROOT}/src" -I"${HERE}" \
  "${SOURCES[@]}" -o "${OUT}"
"${OUT}"
