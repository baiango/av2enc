#!/usr/bin/env bash
# Build and run the M2 inter-DDT CPU reference tests.
#
# The component is standalone: it is not wired into CMake or src/main.cpp.
# The optional --gate build uses build-directory copies of M2 files and never
# edits the core (see gate/README.md).
#
# Usage:
#   ./src/encoder/m2/ddt/build_and_test.sh            # unit tests
#   ./src/encoder/m2/ddt/build_and_test.sh --gate     # unit tests + decoder gate
#   ./src/encoder/m2/ddt/build_and_test.sh --gate-only
#   SANITIZE=1 ./src/encoder/m2/ddt/build_and_test.sh
#   WERROR=1   ./src/encoder/m2/ddt/build_and_test.sh
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../../../../" && pwd)"
CXX="${CXX:-c++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
OUT="${OUT:-${TMPDIR:-/tmp}/m2_ddt}"

RUN_TESTS=1
RUN_GATE=0
GATE_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --gate) RUN_GATE=1 ;;
    --gate-only) RUN_TESTS=0; RUN_GATE=1 ;;
    --no-gate) RUN_GATE=0 ;;
    --help)
      sed -n '2,14p' "${BASH_SOURCE[0]}"
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

COMPONENT=(
  "$HERE/ddt_transform.cpp"
  "$HERE/ddt_constraints.cpp"
  "$HERE/ddt_syntax.cpp"
)

DEPS=(
  "$ROOT/src/bitstream/bitwriter.cpp"
  "$ROOT/src/encoder/tx_variants/tx_variants.cpp"
  "$ROOT/src/encoder/m2/residual/inter_tx_type.cpp"
)

if [[ "$RUN_TESTS" == "1" ]]; then
  echo "== DDT unit tests =="
  "${CXX}" "${FLAGS[@]}" "$HERE/ddt_test.cpp" "${COMPONENT[@]}" "${DEPS[@]}" \
    -o "$OUT/ddt_test"
  "$OUT/ddt_test"
fi

if [[ "$RUN_GATE" == "1" ]]; then
  echo "== DDT decoder gate =="
  OUT="$OUT" "$HERE/gate/build_gate.sh" "${GATE_ARGS[@]+"${GATE_ARGS[@]}"}"
fi
