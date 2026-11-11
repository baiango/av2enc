#!/usr/bin/env bash
# Build and run the M2 inter-residual component tests and the decoder gate.
#
# Host-only CPU reference build (no CUDA needed).  The component is standalone:
# it is not wired into CMake or src/main.cpp, and the gate applies its
# m2_syntax.cpp patch to a *copy* in the build directory, never to the M2 core.
#
# Usage:
#   ./src/encoder/m2/residual/build_and_test.sh              # unit + parity tests
#   ./src/encoder/m2/residual/build_and_test.sh --gate       # also run dav2d gate
#   ./src/encoder/m2/residual/build_and_test.sh --gate-only
#   SANITIZE=1 ./src/encoder/m2/residual/build_and_test.sh
#   WERROR=1   ./src/encoder/m2/residual/build_and_test.sh
#   OUT=/tmp/m2_residual ./src/encoder/m2/residual/build_and_test.sh
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../../../../" && pwd)"
CXX="${CXX:-c++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
OUT="${OUT:-${TMPDIR:-/tmp}/m2_residual}"

RUN_TESTS=1
RUN_GATE=0
GATE_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --gate) RUN_GATE=1 ;;
    --gate-only) RUN_TESTS=0; RUN_GATE=1 ;;
    --no-gate) RUN_GATE=0 ;;
    --help)
      sed -n '2,16p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *) GATE_ARGS+=("$arg") ;;
  esac
done

mkdir -p "$OUT"

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
       -Wno-missing-braces -I"$ROOT/src")
if [[ "${SANITIZE}" == "1" ]]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2)
fi
if [[ "${WERROR}" == "1" ]]; then
  FLAGS+=(-Werror)
fi

CORE=(
  "$ROOT/src/io/y4m.cpp"
  "$ROOT/src/io/ivf.cpp"
  "$ROOT/src/bitstream/bitwriter.cpp"
  "$ROOT/src/bitstream/obu.cpp"
  "$ROOT/src/bitstream/m0_headers.cpp"
  "$ROOT/src/encoder/partition/partition.cpp"
  "$ROOT/src/encoder/intra_modes/intra_modes.cpp"
  "$ROOT/src/encoder/tx_variants/tx_variants.cpp"
  "$ROOT/src/encoder/tx_variants/tx_select.cpp"
  "$ROOT/src/encoder/tx_type/tx_type.cpp"
  "$ROOT/src/encoder/rdo/rdo.cpp"
  "$ROOT/src/encoder/rdoq/rdoq.cpp"
  "$ROOT/src/encoder/rdoq/rdoq_av2_model.cpp"
  "$ROOT/src/entropy/av2/av2_entropy.cpp"
  "$ROOT/src/encoder/inter/me/me.cpp"
  "$ROOT/src/encoder/inter/mc/mc.cpp"
  "$ROOT/src/encoder/m2/merge/merge.cpp"
)

COMPONENT=(
  "$ROOT/src/encoder/m2/residual/inter_tx_type.cpp"
  "$ROOT/src/encoder/m2/residual/inter_coeff_syntax.cpp"
  "$ROOT/src/encoder/m2/residual/inter_residual.cpp"
)

# The component calls av2enc::m2::syntax::build_scan(), which lives in the M2
# syntax writer.  The gate compiles a *patched copy* of that file instead, so it
# must not also link this one.
TEST_CORE=("${CORE[@]}" "$ROOT/src/encoder/m2/m2_headers.cpp"
           "$ROOT/src/encoder/m2/m2_syntax.cpp")

if [[ "$RUN_TESTS" == "1" ]]; then
  echo "== unit tests =="
  "${CXX}" "${FLAGS[@]}" \
    "$ROOT/src/encoder/m2/residual/inter_residual_test.cpp" \
    "${COMPONENT[@]}" "${TEST_CORE[@]}" \
    -o "$OUT/inter_residual_test"
  "$OUT/inter_residual_test"

  echo "== device emulation parity =="
  "${CXX}" "${FLAGS[@]}" \
    "$ROOT/src/encoder/m2/residual/inter_residual_emulation_test.cpp" \
    "${COMPONENT[@]}" "${TEST_CORE[@]}" \
    -o "$OUT/inter_residual_emulation_test"
  "$OUT/inter_residual_emulation_test"
fi

if [[ "$RUN_GATE" == "1" ]]; then
  echo "== decoder gate =="
  "$HERE/gate/build_gate.sh" --out "$OUT" "${GATE_ARGS[@]+"${GATE_ARGS[@]}"}"
fi

echo "m2 residual: OK"
