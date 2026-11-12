#!/usr/bin/env bash
# Build and run the AV2 global motion CPU reference and (optionally) the
# minimal decoder gate.
#
# Usage:
#   ./src/codec/inter/gme/build_and_test.sh            # unit tests
#   ./src/codec/inter/gme/build_and_test.sh --gate     # unit tests + dav2d gate
#   SANITIZE=1 ./src/codec/inter/gme/build_and_test.sh
#   WERROR=1 ./src/codec/inter/gme/build_and_test.sh
#   CXX=clang++ ./src/codec/inter/gme/build_and_test.sh
#   ./src/codec/inter/gme/build_and_test.sh --no-run
#
# The gate needs tools/bin/dav2d (Linux) or tools/bin/dav2d-macos (macOS), or
# DAV2D=/path/to/dav2d. If no decoder is present the gate exits 77 (skip); the
# unit tests still run and pass.
#
# This component owns only src/codec/inter/gme/. The gate compiles the
# existing M2/advanced sources read-only; it never edits them.

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
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
       -I"$ROOT/src" -I"$HERE" -I"$ROOT/src/codec/inter/advanced"
       -I"$ROOT/src/codec/inter/advanced/tables")
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
  "$ROOT/src/core/bitstream/bitwriter.cpp" "$HERE/gme_test.cpp" \
  -o "$OUT"

if [[ "$RUN" -eq 1 ]]; then
  "$OUT" "$ROOT"
fi

if [[ "$GATE" -eq 0 ]]; then
  exit 0
fi

GATE_OUT="${GATE_OUT:-${BUILD}/gme_gate}"
VIDEO_CORE=(
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
  "$ROOT/src/codec/inter/merge/merge.cpp"
  "$ROOT/src/pipeline/video/video_headers.cpp"
  "$ROOT/src/pipeline/video/video_syntax.cpp"
  "$ROOT/src/pipeline/video/video_frame.cpp"
)

"$CXX" "${FLAGS[@]}" \
  "$HERE/gme_gate.cpp" "$HERE/gme.cpp" "$HERE/gme_syntax.cpp" \
  "$ROOT/src/codec/inter/advanced/warp.cpp" \
  "${VIDEO_CORE[@]}" \
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
