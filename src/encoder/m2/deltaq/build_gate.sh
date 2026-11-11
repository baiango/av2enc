#!/usr/bin/env bash
# Build and run the delta-Q / segmentation decoder gate.
#
# The gate builds a 2-frame M2 stream (intra key + all-skip P), replaces the P
# frame header with one that signals segmentation and/or delta-Q, and decodes
# it with the pinned dav2d.  It requires no edits to m1/m2 core writers.
#
# Usage:
#   ./src/encoder/m2/deltaq/build_gate.sh
#   DAV2D=/path/to/dav2d ./src/encoder/m2/deltaq/build_gate.sh
#   ./src/encoder/m2/deltaq/build_gate.sh -- --no-decode   # bytes only
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../../../.." && pwd)"
CXX="${CXX:-c++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
OUT="${OUT:-${TMPDIR:-/tmp}/m2_deltaq_gate}"

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
  "$ROOT/src/io/ivf.cpp"
  "$ROOT/src/io/y4m.cpp"
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
  "$ROOT/src/encoder/m2/deltaq/m2_deltaq.cpp"
  "$ROOT/src/encoder/m1/deltaq/deltaq.cpp"
  "$ROOT/src/encoder/rc/rc.cpp"
)

"${CXX}" "${FLAGS[@]}" \
  "$HERE/deltaq_gate.cpp" \
  "${CORE[@]}" \
  -o "${OUT}"

if [[ -z "${DAV2D:-}" ]]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [[ -x "$candidate" ]]; then
      export DAV2D="$candidate"
      break
    fi
  done
fi

cd "$ROOT"
"${OUT}" "$@"
