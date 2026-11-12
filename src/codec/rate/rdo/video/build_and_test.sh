#!/usr/bin/env bash
# Build and run the M2 P-frame rate-aware decision layer self-tests.
#
# Host-only CPU reference build.  The layer is not wired into CMake; the
# decoder experiment is a separate opt-in run:
#
#   ./src/codec/rate/rdo/video/build_and_test.sh
#   ./src/codec/rate/rdo/video/build_and_test.sh --experiment
#   SANITIZE=1 ./src/codec/rate/rdo/video/build_and_test.sh
#   WERROR=1   ./src/codec/rate/rdo/video/build_and_test.sh
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
CXX="${CXX:-c++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
OUT="${OUT:-${TMPDIR:-/tmp}/video_rdo}"
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
  "$ROOT/src/core/io/y4m.cpp"
  "$ROOT/src/core/io/ivf.cpp"
  "$ROOT/src/core/bitstream/bitwriter.cpp"
  "$ROOT/src/core/bitstream/obu.cpp"
  "$ROOT/src/core/bitstream/basic_headers.cpp"
  "$ROOT/src/codec/partition/tree/partition.cpp"
  "$ROOT/src/codec/intra/modes/intra_modes.cpp"
  "$ROOT/src/codec/transform/variants/tx_variants.cpp"
  "$ROOT/src/codec/transform/variants/tx_select.cpp"
  "$ROOT/src/codec/transform/tx_type/tx_type.cpp"
  "$ROOT/src/codec/rate/rdo/rdo.cpp"
  "$ROOT/src/codec/rate/rdo/intra_decision.cpp"
  "$ROOT/src/codec/rate/rdoq/rdoq.cpp"
  "$ROOT/src/codec/rate/rdoq/rdoq_av2_model.cpp"
  "$ROOT/src/codec/entropy/av2/av2_entropy.cpp"
  "$ROOT/src/codec/inter/me/me.cpp"
  "$ROOT/src/codec/inter/mc/mc.cpp"
  "$ROOT/src/codec/inter/merge/merge.cpp"
  "$ROOT/src/pipeline/video/video_headers.cpp"
  "$ROOT/src/pipeline/video/video_syntax.cpp"
  "$ROOT/src/codec/transform/inter_residual/inter_tx_type.cpp"
  "$ROOT/src/codec/transform/inter_residual/inter_coeff_syntax.cpp"
  "$ROOT/src/codec/transform/inter_residual/inter_residual.cpp"
)

STILL_RDO=(
  "$ROOT/src/codec/rate/decide/config.cpp"
  "$ROOT/src/codec/rate/decide/transform_stream.cpp"
  "$ROOT/src/codec/rate/decide/intra_rdo.cpp"
  "$ROOT/src/codec/rate/decide/chroma_rdo.cpp"
  "$ROOT/src/codec/rate/decide/partition_rdo.cpp"
)

LAYER=(
  "$ROOT/src/codec/rate/rdo/video/config.cpp"
  "$ROOT/src/codec/rate/rdo/video/context.cpp"
  "$ROOT/src/codec/rate/rdo/video/rate_model.cpp"
  "$ROOT/src/codec/rate/rdo/video/residual_decision.cpp"
  "$ROOT/src/codec/rate/rdo/video/inter_decision.cpp"
  "$ROOT/src/codec/rate/rdo/video/intra_fallback.cpp"
  "$ROOT/src/codec/rate/rdo/video/partition_decision.cpp"
)

echo "== M2 RDO unit tests =="
"${CXX}" "${FLAGS[@]}" \
  "$ROOT/src/codec/rate/rdo/video/video_rdo_test.cpp" \
  "${LAYER[@]}" "${STILL_RDO[@]}" "${CORE[@]}" \
  -o "$OUT/video_rdo_test"
"$OUT/video_rdo_test"

if [[ "$RUN_EXPERIMENT" == "1" ]]; then
  "$HERE/experiment/run_video_rdo_experiment.sh" --out "$OUT/experiment"
fi

echo "m2 rdo: OK"
