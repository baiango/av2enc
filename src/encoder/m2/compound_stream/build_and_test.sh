#!/usr/bin/env bash
# Build and run the M2 compound / skip-mode inter syntax gate.
#
#   ./src/encoder/m2/compound_stream/build_and_test.sh
#   SANITIZE=1 ./src/encoder/m2/compound_stream/build_and_test.sh
#   WERROR=1   ./src/encoder/m2/compound_stream/build_and_test.sh
#   DAV2D=/path/to/dav2d ./src/encoder/m2/compound_stream/build_and_test.sh
#
# The decoder gate needs tools/bin/dav2d[-macos] (or DAV2D=...).  The unit
# tests and the all-disabled byte-identity control always run.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../../../.." && pwd)"
CXX="${CXX:-c++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
OUT_DIR="${OUT_DIR:-${TMPDIR:-/tmp}/m2_compound_stream}"

mkdir -p "$OUT_DIR"

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
       -Wno-missing-braces -I"$ROOT/src")
if [[ "${SANITIZE}" == "1" ]]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2)
fi
if [[ "${WERROR}" == "1" ]]; then
  FLAGS+=(-Werror -Wno-sign-conversion)
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
  "$ROOT/src/encoder/inter/temporal/opfl.cpp"
  "$ROOT/src/encoder/inter/temporal/temporal_common.cpp"
  "$ROOT/src/encoder/m2/merge/merge.cpp"
  "$ROOT/src/encoder/m2/m2_headers.cpp"
  "$ROOT/src/encoder/m2/m2_syntax.cpp"
  "$ROOT/src/encoder/m2/m2_frame.cpp"
  "$ROOT/src/encoder/m2/multiref/ref_map.cpp"
  "$ROOT/src/encoder/m2/multiref/compound_predict.cpp"
)

COMPONENT=(
  "$ROOT/src/encoder/m2/compound_stream/compound_syntax.cpp"
  "$ROOT/src/encoder/m2/compound_stream/compound_reader.cpp"
  "$ROOT/src/encoder/m2/compound_stream/compound_stream.cpp"
)

echo "== build: unit tests =="
"${CXX}" "${FLAGS[@]}" \
  "$ROOT/src/encoder/m2/compound_stream/compound_test.cpp" \
  "${COMPONENT[@]}" "${CORE[@]}" \
  -o "${OUT_DIR}/compound_test"

echo "== build: decoder gate =="
"${CXX}" "${FLAGS[@]}" \
  "$ROOT/src/encoder/m2/compound_stream/compound_gate.cpp" \
  "${COMPONENT[@]}" "${CORE[@]}" \
  -o "${OUT_DIR}/compound_gate"

echo "== unit tests =="
"${OUT_DIR}/compound_test"

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

echo "== gate: synthetic 320x128 =="
"${OUT_DIR}/compound_gate" --out "${OUT_DIR}/compound_gate.ivf"

SRC="${ROOT}/testdata/shine_1280x720_60f.y4m"
if [[ "${1:-}" == "--720p" && -f "$SRC" ]]; then
  echo "== gate: real 720p (cropped) =="
  "${OUT_DIR}/compound_gate" --input "$SRC" \
    --out "${OUT_DIR}/compound_gate_720p.ivf"
fi
