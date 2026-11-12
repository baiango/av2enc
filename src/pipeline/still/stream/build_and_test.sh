#!/usr/bin/env bash
# Build and run the M1 GPU -> syntax bridge unit test.
#
# Host-only (no CUDA): the bridge consumes the plain PODs from
# src/cuda/pipeline/still/still_gpu.h and the syntax writer is CPU code.  This mirrors the
# other per-component scripts in the tree (src/codec/entropy/av2/build_and_test.sh).
#
# Usage:
#   ./src/pipeline/still/stream/build_and_test.sh
#   CXX=g++-13 ./src/pipeline/still/stream/build_and_test.sh
#   SANITIZE=1 ./src/pipeline/still/stream/build_and_test.sh
#   WERROR=1    ./src/pipeline/still/stream/build_and_test.sh
#   OUT=/tmp/still_stream_bridge_test ./src/pipeline/still/stream/build_and_test.sh
#
# Note: the test includes src/pipeline/still/still_encoder.cpp so it can see the CPU
# decision stream.  still_encoder.cpp is therefore *not* on the compile line (that
# would duplicate symbols); still_syntax.cpp is, because the core library builds it
# as its own translation unit.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
CXX="${CXX:-g++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
OUT="${OUT:-${TMPDIR:-/tmp}/still_stream_bridge_test}"

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
  "$ROOT/src/core/bitstream/bitwriter.cpp"
  "$ROOT/src/core/bitstream/obu.cpp"
  "$ROOT/src/core/bitstream/basic_headers.cpp"
  "$ROOT/src/codec/partition/tree/partition.cpp"
  "$ROOT/src/codec/intra/modes/intra_modes.cpp"
  "$ROOT/src/codec/transform/variants/tx_variants.cpp"
  "$ROOT/src/codec/transform/variants/tx_select.cpp"
  "$ROOT/src/codec/intra/cfl/cfl.cpp"
  "$ROOT/src/codec/filters/deblock/deblock.cpp"
  "$ROOT/src/pipeline/still/still_headers.cpp"
  "$ROOT/src/pipeline/still/still_syntax.cpp"
  "$ROOT/src/cuda/pipeline/still/plan/plan.cpp"
  "$ROOT/src/codec/entropy/av2/av2_entropy.cpp"
)

"${CXX}" "${FLAGS[@]}" \
  "${HERE}/still_stream_bridge.cpp" \
  "${HERE}/still_stream_bridge_test.cpp" \
  "${CORE[@]}" \
  -o "${OUT}"

"${OUT}"
