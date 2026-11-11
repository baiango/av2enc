#!/usr/bin/env bash
# Build and run the M1 GPU -> syntax bridge unit test.
#
# Host-only (no CUDA): the bridge consumes the plain PODs from
# src/cuda/m1/m1_gpu.h and the syntax writer is CPU code.  This mirrors the
# other per-component scripts in the tree (src/entropy/av2/build_and_test.sh).
#
# Usage:
#   ./src/encoder/m1/stream/build_and_test.sh
#   CXX=g++-13 ./src/encoder/m1/stream/build_and_test.sh
#   SANITIZE=1 ./src/encoder/m1/stream/build_and_test.sh
#   WERROR=1    ./src/encoder/m1/stream/build_and_test.sh
#   OUT=/tmp/m1_stream_bridge_test ./src/encoder/m1/stream/build_and_test.sh
#
# Note: the test includes src/encoder/m1/m1_still.cpp so it can see the CPU
# decision stream.  m1_still.cpp is therefore *not* on the compile line (that
# would duplicate symbols); m1_syntax.cpp is, because the core library builds it
# as its own translation unit.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../../../.." && pwd)"
CXX="${CXX:-g++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
OUT="${OUT:-${TMPDIR:-/tmp}/m1_stream_bridge_test}"

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow -I"$ROOT/src")

if [[ "${SANITIZE}" == "1" ]]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2)
fi
if [[ "${WERROR}" == "1" ]]; then
  FLAGS+=(-Werror)
fi

CORE=(
  "$ROOT/src/bitstream/bitwriter.cpp"
  "$ROOT/src/bitstream/obu.cpp"
  "$ROOT/src/bitstream/m0_headers.cpp"
  "$ROOT/src/encoder/partition/partition.cpp"
  "$ROOT/src/encoder/intra_modes/intra_modes.cpp"
  "$ROOT/src/encoder/tx_variants/tx_variants.cpp"
  "$ROOT/src/encoder/tx_variants/tx_select.cpp"
  "$ROOT/src/encoder/cfl/cfl.cpp"
  "$ROOT/src/encoder/filters/deblock/deblock.cpp"
  "$ROOT/src/encoder/m1/m1_headers.cpp"
  "$ROOT/src/encoder/m1/m1_syntax.cpp"
  "$ROOT/src/cuda/m1/plan/plan.cpp"
  "$ROOT/src/entropy/av2/av2_entropy.cpp"
)

"${CXX}" "${FLAGS[@]}" \
  "${HERE}/m1_stream_bridge.cpp" \
  "${HERE}/m1_stream_bridge_test.cpp" \
  "${CORE[@]}" \
  -o "${OUT}"

"${OUT}"
