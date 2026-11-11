#!/usr/bin/env bash
# Build and run the multi-reference / compound prediction component.
#
#   ./src/encoder/m2/multiref/build_and_test.sh
#   ./src/encoder/m2/multiref/build_and_test.sh --720p
#   SANITIZE=1 ./src/encoder/m2/multiref/build_and_test.sh
#   WERROR=1   ./src/encoder/m2/multiref/build_and_test.sh
#
# The decoder gate needs tools/bin/dav2d[-macos] (or DAV2D=...).  It is skipped
# when no decoder is available; unit and library tests still run.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../../../.." && pwd)"
CXX="${CXX:-c++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
OUT_DIR="${OUT_DIR:-${TMPDIR:-/tmp}/m2_multiref}"

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
  "$ROOT/src/io/y4m.cpp"
  "$ROOT/src/io/ivf.cpp"
  "$ROOT/src/bitstream/bitwriter.cpp"
  "$ROOT/src/bitstream/obu.cpp"
  "$ROOT/src/bitstream/m0_headers.cpp"
  "$ROOT/src/encoder/partition/partition.cpp"
  "$ROOT/src/encoder/intra_modes/intra_modes.cpp"
  "$ROOT/src/encoder/tx_variants/tx_variants.cpp"
  "$ROOT/src/encoder/tx_variants/tx_select.cpp"
  "$ROOT/src/entropy/av2/av2_entropy.cpp"
  "$ROOT/src/encoder/inter/me/me.cpp"
  "$ROOT/src/encoder/inter/mc/mc.cpp"
  "$ROOT/src/encoder/inter/advanced/compound.cpp"
  "$ROOT/src/encoder/m2/merge/merge.cpp"
  "$ROOT/src/encoder/m2/m2_headers.cpp"
  "$ROOT/src/encoder/m2/m2_syntax.cpp"
  "$ROOT/src/encoder/m2/m2_frame.cpp"
)

MULTIREF=(
  "$ROOT/src/encoder/m2/multiref/ref_map.cpp"
  "$ROOT/src/encoder/m2/multiref/compound_predict.cpp"
  "$ROOT/src/encoder/m2/multiref/mvstack_multi.cpp"
  "$ROOT/src/encoder/m2/multiref/decide.cpp"
  "$ROOT/src/encoder/m2/multiref/gpu_batch.cpp"
)

GATE=(
  "$ROOT/src/encoder/m2/multiref/gate/compound_tile.cpp"
  "$ROOT/src/encoder/m2/multiref/gate/compound_stream.cpp"
)

echo "== build: unit tests =="
"${CXX}" "${FLAGS[@]}" \
  "$ROOT/src/encoder/m2/multiref/multiref_test.cpp" \
  "${MULTIREF[@]}" "${CORE[@]}" \
  -o "${OUT_DIR}/multiref_test"

echo "== build: compound gate =="
"${CXX}" "${FLAGS[@]}" \
  "$ROOT/src/encoder/m2/multiref/gate/compound_gate.cpp" \
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
