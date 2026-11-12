#!/usr/bin/env bash
# Build and run the CPU correctness tests for AV2 MHCCP (§7.13.6-§7.13.8).
#
#   ./build_and_test.sh
#   CXX=g++-13 ./build_and_test.sh
#   SANITIZE=1 WERROR=1 ./build_and_test.sh
#   OUT=/tmp/mhccp_test ./build_and_test.sh
#
# Plain C++20; no CUDA toolkit or GPU required. The GPU kernel is tested by
# build_emulation_test.sh (host emulation) and build_gpu_test.sh (nvcc).
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
CXX="${CXX:-c++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
OUT="${OUT:-${TMPDIR:-/tmp}/mhccp_test}"

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
       -ffp-contract=off -I"$ROOT/src")
if [[ "${SANITIZE}" == "1" ]]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2)
fi
if [[ "${WERROR}" == "1" ]]; then
  FLAGS+=(-Werror)
fi

"${CXX}" "${FLAGS[@]}" \
  "$HERE/mhccp.cpp" \
  "$HERE/mhccp_syntax.cpp" \
  "$HERE/mhccp_test.cpp" \
  "$ROOT/src/codec/intra/cfl/cfl.cpp" \
  "$ROOT/src/core/io/y4m.cpp" \
  "$ROOT/src/codec/entropy/av2/av2_entropy.cpp" \
  -o "$OUT"

cd "$ROOT"
"$OUT" "$ROOT/tests/data/basic_64x64_1f.y4m" \
       "$ROOT/tests/data/basic_640x360_1f.y4m"
