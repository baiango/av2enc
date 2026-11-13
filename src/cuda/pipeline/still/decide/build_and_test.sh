#!/usr/bin/env bash
# Host parity test for the still decision core (no nvcc, no GPU).
#
# Compares `decide_core.cuh` (compiled as host code) with the shipped CPU
# references: the §7.16 av2 rate models, `rdoq::quantize` and `tcq::quantize`.
set -euo pipefail

SELF=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SELF/../../../../.." && pwd)
OUT=${OUT:-${TMPDIR:-/tmp}/av2enc_decide_core_test}
CXX=${CXX:-c++}
WERROR=${WERROR:-0}

mkdir -p "$(dirname "$OUT")"

SOURCES=(
    "$SELF/decide_core_test.cpp"
    "$ROOT/src/codec/rate/rdoq/rdoq.cpp"
    "$ROOT/src/codec/rate/rdoq/rdoq_av2_model.cpp"
    "$ROOT/src/codec/transform/tcq/tcq.cpp"
    "$ROOT/src/codec/transform/tcq/tcq_av2_model.cpp"
    "$ROOT/src/codec/transform/variants/tx_variants.cpp"
    "$ROOT/src/codec/rate/rdo/rdo.cpp"
)

FLAGS=(-std=c++20 -O2 -ffp-contract=off -fno-fast-math
       -I"$ROOT/src" -I"$ROOT/src/codec" -I"$ROOT/src/pipeline")
if [ "$WERROR" = "1" ]; then
    FLAGS+=(-Wall -Wextra -Werror)
fi

"$CXX" "${FLAGS[@]}" "${SOURCES[@]}" -o "$OUT"
"$OUT"
