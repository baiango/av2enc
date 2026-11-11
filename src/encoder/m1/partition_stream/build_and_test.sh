#!/usr/bin/env bash
# Build and run the M1 <-> partition_ext adapter tests, then the dav2d gate.
#
#   ./src/encoder/m1/partition_stream/build_and_test.sh
#   CXX=clang++ ./src/encoder/m1/partition_stream/build_and_test.sh
#   SANITIZE=1 WERROR=1 ./src/encoder/m1/partition_stream/build_and_test.sh
#   SKIP_GATE=1 ./src/encoder/m1/partition_stream/build_and_test.sh
#   OUT=/tmp/partition_stream ./src/encoder/m1/partition_stream/build_and_test.sh
#
# Self-contained: does not touch the root CMake build or any other component.
# The target is Linux + CUDA; this CPU binary is the correctness oracle only.

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${HERE}/../../../.." && pwd)"
CXX="${CXX:-c++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
SKIP_GATE="${SKIP_GATE:-0}"
OUT="${OUT:-${TMPDIR:-/tmp}/partition_stream_test}"

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
  "${ROOT}/src/encoder/partition_ext/partition_ext.cpp"
  "${ROOT}/src/encoder/partition_ext/ext_syntax.cpp"
  "${ROOT}/src/entropy/av2/av2_entropy.cpp"
  "${HERE}/partition_stream.cpp"
)

echo "== adapter tests =="
"${CXX}" "${FLAGS[@]}" -I"${ROOT}/src" -I"${HERE}" \
  "${COMMON[@]}" "${HERE}/partition_stream_test.cpp" \
  -o "${OUT}"
"${OUT}"

if [[ "${SKIP_GATE}" == "0" ]]; then
  echo "== minimal dav2d gate =="
  # m1_cdf_tables.h uses flat initialisers for 2-D rows; keep the project's
  # warnings for our own code and silence that third-party header only.
  "${CXX}" "${FLAGS[@]}" -Wno-missing-braces -I"${ROOT}/src" -I"${HERE}" \
    "${COMMON[@]}" \
    "${ROOT}/src/encoder/m1/m1_headers.cpp" \
    "${ROOT}/src/bitstream/m0_headers.cpp" \
    "${ROOT}/src/bitstream/bitwriter.cpp" \
    "${ROOT}/src/encoder/tx_variants/tx_variants.cpp" \
    "${ROOT}/src/io/ivf.cpp" \
    "${HERE}/gate_minimal.cpp" \
    -o "${OUT}_gate"
  "${OUT}_gate" --out "${OUT}_gate_out"
  python3 "${HERE}/gate_dav2d.py" --dir "${OUT}_gate_out"
fi
