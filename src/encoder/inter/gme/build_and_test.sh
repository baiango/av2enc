#!/usr/bin/env bash
# Build and run the AV2 global motion CPU reference and (optionally) the
# minimal decoder gate.
#
# Usage:
#   ./src/encoder/inter/gme/build_and_test.sh            # unit tests
#   ./src/encoder/inter/gme/build_and_test.sh --gate     # unit tests + dav2d gate
#   SANITIZE=1 ./src/encoder/inter/gme/build_and_test.sh
#   WERROR=1 ./src/encoder/inter/gme/build_and_test.sh
#   CXX=clang++ ./src/encoder/inter/gme/build_and_test.sh
#   ./src/encoder/inter/gme/build_and_test.sh --no-run
#
# The gate needs tools/bin/dav2d (Linux) or tools/bin/dav2d-macos (macOS), or
# DAV2D=/path/to/dav2d. If no decoder is present the gate exits 77 (skip); the
# unit tests still run and pass.
#
# This component owns only src/encoder/inter/gme/. The gate compiles the
# existing M2/advanced sources read-only; it never edits them.

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../../../.." && pwd)"
CXX="${CXX:-c++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
BUILD="${BUILD:-${TMPDIR:-/tmp}/gme_build}"
OUT="${OUT:-${BUILD}/gme_test}"

RUN=1
GATE=0
for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    --gate) GATE=1 ;;
    -h|--help)
      sed -n '2,22p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$BUILD"

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
       -Wno-sign-conversion -Wno-missing-braces -ffp-contract=off
       -I"$ROOT/src" -I"$HERE" -I"$ROOT/src/encoder/inter/advanced"
       -I"$ROOT/src/encoder/inter/advanced/tables")
if [[ "$SANITIZE" == "1" ]]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2 -g)
fi
if [[ "$WERROR" == "1" ]]; then
  FLAGS+=(-Werror)
fi

echo "compiler: $("$CXX" --version | head -1)"
echo "build:    $BUILD"

"$CXX" "${FLAGS[@]}" \
  "$HERE/gme.cpp" "$HERE/gme_syntax.cpp" \
  "$ROOT/src/bitstream/bitwriter.cpp" "$HERE/gme_test.cpp" \
  -o "$OUT"

if [[ "$RUN" -eq 1 ]]; then
  "$OUT" "$ROOT"
fi

if [[ "$GATE" -eq 0 ]]; then
  exit 0
fi

GATE_OUT="${GATE_OUT:-${BUILD}/gme_gate}"
M2_CORE=(
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
  "$ROOT/src/encoder/m2/merge/merge.cpp"
  "$ROOT/src/encoder/m2/m2_headers.cpp"
  "$ROOT/src/encoder/m2/m2_syntax.cpp"
  "$ROOT/src/encoder/m2/m2_frame.cpp"
)

"$CXX" "${FLAGS[@]}" \
  "$HERE/gme_gate.cpp" "$HERE/gme.cpp" "$HERE/gme_syntax.cpp" \
  "$ROOT/src/encoder/inter/advanced/warp.cpp" \
  "${M2_CORE[@]}" \
  -o "$GATE_OUT"

if [[ -z "${DAV2D:-}" ]]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [[ -x "$candidate" ]]; then
      DAV2D="$candidate"
      break
    fi
  done
fi
export DAV2D

cd "$ROOT"
STATUS=0
"$GATE_OUT" --out "${BUILD}/gme_gate.ivf" --decoded "${BUILD}/gme_gate_decoded.y4m" || {
  code=$?
  if [[ "$code" -ne 77 ]]; then STATUS=1; fi
}
"$GATE_OUT" --translation --out "${BUILD}/gme_gate_translation.ivf" \
  --decoded "${BUILD}/gme_gate_translation_decoded.y4m" || {
  code=$?
  if [[ "$code" -ne 77 ]]; then STATUS=1; fi
}
"$GATE_OUT" --affine --out "${BUILD}/gme_gate_affine.ivf" \
  --decoded "${BUILD}/gme_gate_affine_decoded.y4m" || {
  code=$?
  if [[ "$code" -ne 77 ]]; then STATUS=1; fi
}
"$GATE_OUT" --shear --out "${BUILD}/gme_gate_shear.ivf" \
  --decoded "${BUILD}/gme_gate_shear_decoded.y4m" || {
  code=$?
  if [[ "$code" -ne 77 ]]; then STATUS=1; fi
}
"$GATE_OUT" --base-diff --out "${BUILD}/gme_gate_base_diff.ivf" \
  --decoded "${BUILD}/gme_gate_base_diff_decoded.y4m" || {
  code=$?
  if [[ "$code" -ne 77 ]]; then STATUS=1; fi
}
if [[ -f "$ROOT/testdata/shine_1280x720_60f.y4m" ]]; then
  "$GATE_OUT" --input "$ROOT/testdata/shine_1280x720_60f.y4m" \
    --out "${BUILD}/gme_gate_720.ivf" \
    --decoded "${BUILD}/gme_gate_720_decoded.y4m" || {
    code=$?
    if [[ "$code" -ne 77 ]]; then STATUS=1; fi
  }
fi
exit "$STATUS"
