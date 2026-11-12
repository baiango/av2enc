#!/usr/bin/env bash
# Build and run the delta-Q / segmentation decoder gate.
#
# The gate builds a 2-frame M2 stream (intra key + all-skip P), replaces the P
# frame header with one that signals segmentation and/or delta-Q, and decodes
# it with the pinned dav2d.  It requires no edits to m1/m2 core writers.
#
# Usage:
#   ./src/codec/entropy/deltaq/video/build_gate.sh
#   DAV2D=/path/to/dav2d ./src/codec/entropy/deltaq/video/build_gate.sh
#   ./src/codec/entropy/deltaq/video/build_gate.sh -- --no-decode   # bytes only
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
CXX="${CXX:-c++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
OUT="${OUT:-${TMPDIR:-/tmp}/video_deltaq_gate}"

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
  "$ROOT/src/core/io/ivf.cpp"
  "$ROOT/src/core/io/y4m.cpp"
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
  "$ROOT/src/codec/entropy/deltaq/video/video_deltaq.cpp"
  "$ROOT/src/codec/entropy/deltaq/still/deltaq.cpp"
  "$ROOT/src/codec/rate/rc/rc.cpp"
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
