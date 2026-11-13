#!/usr/bin/env bash
# Build and run the M2 compound / skip-mode inter syntax gate.
#
#   ./src/codec/inter/compound/build_and_test.sh
#   SANITIZE=1 ./src/codec/inter/compound/build_and_test.sh
#   WERROR=1   ./src/codec/inter/compound/build_and_test.sh
#   DAV2D=/path/to/dav2d ./src/codec/inter/compound/build_and_test.sh
#
# The decoder gate needs tools/bin/dav2d[-macos] (or DAV2D=...).  The unit
# tests and the all-disabled byte-identity control always run.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
CXX="${CXX:-c++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
OUT_DIR="${OUT_DIR:-${TMPDIR:-/tmp}/video_compound_stream}"

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

# The product core source list lives in CMakeLists.txt.  Deriving it keeps this
# standalone gate linkable as the shared core grows (a stale explicit list fails
# with undefined symbols long before the gate itself is wrong).
CORE=()
while IFS= read -r src; do
  CORE+=("$ROOT/$src")
done < <(python3 - "$ROOT/CMakeLists.txt" <<'PY'
import re
import sys

text = open(sys.argv[1]).read()
start = text.index("add_library(av2enc_core STATIC")
end = text.index("target_include_directories(av2enc_core PUBLIC", start)
for path in re.findall(r"^\s*(src/[^\s)]+\.cpp)", text[start:end], re.M):
    print(path)
PY
)
# Sources added by the later `target_sources(av2enc_core ...)` blocks and the
# multi-reference/compound components the gate links directly.  Keep this list
# to the modules a standalone `encode_video_sequence()` consumer needs.
CORE+=(
  "$ROOT/src/codec/inter/multiref/ref_map.cpp"
  "$ROOT/src/codec/inter/temporal/opfl.cpp"
  "$ROOT/src/codec/inter/temporal/temporal_common.cpp"
  "$ROOT/src/formats/format.cpp"
  "$ROOT/src/pipeline/tiles/tile_layout.cpp"
  "$ROOT/src/pipeline/tiles/tile_group.cpp"
  "$ROOT/src/pipeline/tiles/stream/tiles_stream.cpp"
  "$ROOT/src/pipeline/video/post/rc_adapter.cpp"
  "$ROOT/src/codec/filters/stream/filters_stream.cpp"
  "$ROOT/src/pipeline/video/post/bit_depth_checklist.cpp"
  "$ROOT/src/pipeline/video/post/loop_filters.cpp"
  "$ROOT/src/pipeline/video/post/tiled_frame.cpp"
  "$ROOT/src/pipeline/video/post/video_loop_filters.cpp"
)

COMPONENT=(
  "$ROOT/src/codec/inter/compound/compound_syntax.cpp"
  "$ROOT/src/codec/inter/compound/compound_reader.cpp"
  "$ROOT/src/codec/inter/compound/compound_stream.cpp"
)

echo "== build: unit tests =="
"${CXX}" "${FLAGS[@]}" \
  "$ROOT/src/codec/inter/compound/compound_test.cpp" \
  "${COMPONENT[@]}" "${CORE[@]}" \
  -o "${OUT_DIR}/compound_test"

echo "== build: decoder gate =="
"${CXX}" "${FLAGS[@]}" \
  "$ROOT/src/codec/inter/compound/compound_gate.cpp" \
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
echo "== gate: wedge-only 320x128 =="
"${OUT_DIR}/compound_gate" --mode wedge --out "${OUT_DIR}/compound_gate_wedge.ivf"
echo "== gate: diffwtd-only 320x128 =="
"${OUT_DIR}/compound_gate" --mode diffwtd --out "${OUT_DIR}/compound_gate_diffwtd.ivf"

SRC="${ROOT}/testdata/shine_1280x720_60f.y4m"
if [[ "${1:-}" == "--720p" && -f "$SRC" ]]; then
  echo "== gate: real 720p (cropped) =="
  "${OUT_DIR}/compound_gate" --input "$SRC" \
    --out "${OUT_DIR}/compound_gate_720p.ivf"
fi
