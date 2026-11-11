#!/usr/bin/env bash
# Build and run the M2 P-frame rate-aware decision layer self-tests.
#
# Host-only CPU reference build.  The layer is not wired into CMake; the
# decoder experiment is a separate opt-in run:
#
#   ./src/encoder/m2/rdo/build_and_test.sh
#   ./src/encoder/m2/rdo/build_and_test.sh --experiment
#   SANITIZE=1 ./src/encoder/m2/rdo/build_and_test.sh
#   WERROR=1   ./src/encoder/m2/rdo/build_and_test.sh
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../../../../" && pwd)"
CXX="${CXX:-c++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
OUT="${OUT:-${TMPDIR:-/tmp}/m2_rdo}"
RUN_EXPERIMENT=0

for arg in "$@"; do
  case "$arg" in
    --experiment) RUN_EXPERIMENT=1 ;;
    --help)
      sed -n '2,10p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "build_and_test: unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$OUT"

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
       -Wno-missing-braces -Wno-sign-conversion -I"$ROOT/src")
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
  "$ROOT/src/encoder/rdo/intra_decision.cpp"
  "$ROOT/src/encoder/rdoq/rdoq.cpp"
  "$ROOT/src/encoder/rdoq/rdoq_av2_model.cpp"
  "$ROOT/src/entropy/av2/av2_entropy.cpp"
  "$ROOT/src/encoder/inter/me/me.cpp"
  "$ROOT/src/encoder/inter/mc/mc.cpp"
  "$ROOT/src/encoder/m2/merge/merge.cpp"
  "$ROOT/src/encoder/m2/m2_headers.cpp"
  "$ROOT/src/encoder/m2/m2_syntax.cpp"
  "$ROOT/src/encoder/m2/residual/inter_tx_type.cpp"
  "$ROOT/src/encoder/m2/residual/inter_coeff_syntax.cpp"
  "$ROOT/src/encoder/m2/residual/inter_residual.cpp"
)

M1_RDO=(
  "$ROOT/src/encoder/m1/rdo_decide/config.cpp"
  "$ROOT/src/encoder/m1/rdo_decide/transform_stream.cpp"
  "$ROOT/src/encoder/m1/rdo_decide/intra_rdo.cpp"
  "$ROOT/src/encoder/m1/rdo_decide/chroma_rdo.cpp"
  "$ROOT/src/encoder/m1/rdo_decide/partition_rdo.cpp"
)

LAYER=(
  "$ROOT/src/encoder/m2/rdo/config.cpp"
  "$ROOT/src/encoder/m2/rdo/context.cpp"
  "$ROOT/src/encoder/m2/rdo/rate_model.cpp"
  "$ROOT/src/encoder/m2/rdo/residual_decision.cpp"
  "$ROOT/src/encoder/m2/rdo/inter_decision.cpp"
  "$ROOT/src/encoder/m2/rdo/intra_fallback.cpp"
  "$ROOT/src/encoder/m2/rdo/partition_decision.cpp"
)

echo "== M2 RDO unit tests =="
"${CXX}" "${FLAGS[@]}" \
  "$ROOT/src/encoder/m2/rdo/m2_rdo_test.cpp" \
  "${LAYER[@]}" "${M1_RDO[@]}" "${CORE[@]}" \
  -o "$OUT/m2_rdo_test"
"$OUT/m2_rdo_test"

if [[ "$RUN_EXPERIMENT" == "1" ]]; then
  "$HERE/experiment/run_m2_rdo_experiment.sh" --out "$OUT/experiment"
fi

echo "m2 rdo: OK"
