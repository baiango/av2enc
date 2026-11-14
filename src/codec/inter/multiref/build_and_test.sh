#!/usr/bin/env bash
# Build and run the multi-reference / compound prediction component.
#
#   ./src/codec/inter/multiref/build_and_test.sh
#   ./src/codec/inter/multiref/build_and_test.sh --720p
#   SANITIZE=1 ./src/codec/inter/multiref/build_and_test.sh
#   WERROR=1   ./src/codec/inter/multiref/build_and_test.sh
#
# The decoder gate needs tools/bin/dav2d[-macos] (or DAV2D=...).  It is skipped
# when no decoder is available; unit and library tests still run.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
CXX="${CXX:-c++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
OUT_DIR="${OUT_DIR:-${TMPDIR:-/tmp}/video_multiref}"

mkdir -p "$OUT_DIR"

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
  "$ROOT/src/core/io/y4m.cpp"
  "$ROOT/src/core/io/ivf.cpp"
  "$ROOT/src/core/bitstream/bitwriter.cpp"
  "$ROOT/src/core/bitstream/obu.cpp"
  "$ROOT/src/core/bitstream/basic_headers.cpp"
  "$ROOT/src/codec/partition/tree/partition.cpp"
  "$ROOT/src/codec/intra/modes/intra_modes.cpp"
  "$ROOT/src/codec/transform/variants/tx_variants.cpp"
  "$ROOT/src/codec/transform/variants/tx_select.cpp"
  "$ROOT/src/codec/entropy/av2/av2_entropy.cpp"
  "$ROOT/src/codec/inter/me/me.cpp"
  "$ROOT/src/codec/inter/mc/mc.cpp"
  "$ROOT/src/codec/inter/advanced/compound.cpp"
  "$ROOT/src/codec/inter/merge/merge.cpp"
  "$ROOT/src/codec/inter/drl_reorder/drl_reorder.cpp"
  "$ROOT/src/codec/inter/drl_reorder/video_stack_adapter.cpp"
  "$ROOT/src/codec/inter/mv_precision/mv_precision.cpp"
  "$ROOT/src/pipeline/video/video_headers.cpp"
  "$ROOT/src/pipeline/video/video_syntax.cpp"
  "$ROOT/src/pipeline/video/video_frame.cpp"
  "$ROOT/src/codec/inter/temporal/ref_mv_bank.cpp"
)

MULTIREF=(
  "$ROOT/src/codec/inter/multiref/ref_map.cpp"
  "$ROOT/src/codec/inter/multiref/compound_predict.cpp"
  "$ROOT/src/codec/inter/multiref/mvstack_multi.cpp"
  "$ROOT/src/codec/inter/multiref/decide.cpp"
  "$ROOT/src/codec/inter/multiref/gpu_batch.cpp"
)

GATE=(
  "$ROOT/src/codec/inter/multiref/gate/compound_tile.cpp"
  "$ROOT/src/codec/inter/multiref/gate/compound_stream.cpp"
)

echo "== build: unit tests =="
"${CXX}" "${FLAGS[@]}" \
  "$ROOT/src/codec/inter/multiref/multiref_test.cpp" \
  "${MULTIREF[@]}" "${CORE[@]}" \
  -o "${OUT_DIR}/multiref_test"

echo "== build: compound gate =="
"${CXX}" "${FLAGS[@]}" \
  "$ROOT/src/codec/inter/multiref/gate/compound_gate.cpp" \
  "${GATE[@]}" "${MULTIREF[@]}" "${CORE[@]}" \
  -o "${OUT_DIR}/compound_gate"

echo "== unit tests =="
"${OUT_DIR}/multiref_test"

if [[ -z "${DAV2D:-}" ]]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [[ -x "$candidate" ]]; then
      export DAV2D="$candidate"
      break
    fi
  done
fi

if [[ -z "${DAV2D:-}" ]]; then
  echo "== gate: SKIPPED (no dav2d; set DAV2D=/path/to/dav2d) =="
  exit 0
fi

echo "== gate: synthetic 128x128 =="
"${OUT_DIR}/compound_gate" --out "${OUT_DIR}/compound_gate.ivf"

if [[ "${1:-}" == "--720p" || "${ALL:-0}" == "1" ]]; then
  SRC="${ROOT}/testdata/shine_1280x720_60f.y4m"
  if [[ -f "$SRC" ]]; then
    echo "== gate: real 720p (cropped to 1280x704) =="
    "${OUT_DIR}/compound_gate" --input "$SRC" \
      --out "${OUT_DIR}/compound_gate_720p.ivf"
  else
    echo "== gate: 720p media missing, skipped =="
  fi
fi
