#!/usr/bin/env bash
# Build and run the AV2 warp-stack machinery (§7.11.4, §7.12.2.3/.4/.9/.11/
# .20, §7.12.3, §7.13.3.23) plus the LOCALWARP dav2d gate.
#
# Usage:
#   ./src/codec/inter/warp_stack/build_and_test.sh             # tests + gate
#   ./src/codec/inter/warp_stack/build_and_test.sh --no-gate   # unit tests
#   SANITIZE=1 ./src/codec/inter/warp_stack/build_and_test.sh
#   WERROR=1   ./src/codec/inter/warp_stack/build_and_test.sh
#   CXX=clang++ ./src/codec/inter/warp_stack/build_and_test.sh
#   DAV2D=/path/to/dav2d ./src/codec/inter/warp_stack/build_and_test.sh
#
# The gate needs tools/bin/dav2d (Linux) or tools/bin/dav2d-macos (macOS), or
# DAV2D=... .  If no decoder is present the gate exits 77 (skip) and the unit
# tests still count.
#
# This component owns only src/codec/inter/warp_stack/.  The gate compiles
# the M2 core and the advanced warp predictor read-only; it never edits them.

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
CXX="${CXX:-c++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
BUILD="${BUILD:-${TMPDIR:-/tmp}/warp_stack_build}"
OUT="${OUT:-${BUILD}/warp_stack_test}"

RUN=1
GATE=1
for arg in "$@"; do
  case "$arg" in
    --no-gate) GATE=0 ;;
    --no-run) RUN=0 ;;
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

UNITS=(
  "$HERE/warp_stack.cpp"
  "$HERE/warp_stack_mirror.cpp"
  "$HERE/warp_stack_io.cpp"
  "$HERE/warp_stack_reader.cpp"
  "$HERE/warp_stack_test.cpp"
  # Reference stack for the derive_wrl == 0 parity test (read-only).
  "$ROOT/src/codec/inter/merge/merge.cpp"
  "$ROOT/src/codec/inter/mc/mc.cpp"
  "$ROOT/src/codec/partition/tree/partition.cpp"
)

echo "== build: unit tests =="
"$CXX" "${FLAGS[@]}" "${UNITS[@]}" -o "$OUT"

if [[ "$RUN" -eq 1 ]]; then
  echo "== unit tests =="
  "$OUT"
fi

if [[ "$GATE" -eq 0 ]]; then
  exit 0
fi

GATE_OUT="${GATE_OUT:-${BUILD}/warp_gate}"
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
  "$ROOT/src/codec/inter/advanced/warp.cpp"
  "$ROOT/src/codec/inter/merge/merge.cpp"
  "$ROOT/src/pipeline/video/video_headers.cpp"
  "$ROOT/src/pipeline/video/video_syntax.cpp"
  "$ROOT/src/pipeline/video/video_frame.cpp"
  "$ROOT/src/codec/inter/temporal/ref_mv_bank.cpp"
)

echo "== build: LOCALWARP gate =="
"$CXX" "${FLAGS[@]}" \
  "$HERE/gate/warp_gate.cpp" "$HERE/gate/warp_gate_tile.cpp" \
  "$HERE/warp_stack.cpp" "${VIDEO_CORE[@]}" -o "$GATE_OUT"

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
echo "== gate: dav2d --strict 1 --threads 1 =="
set +e
"$GATE_OUT" --out "${BUILD}/warp_gate.ivf" \
  --decoded "${BUILD}/warp_gate_decoded.y4m"
status=$?
set -e
if [[ "$status" -eq 77 ]]; then
  echo "warp_stack: dav2d not available (gate skipped)"
  exit 0
fi
if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

echo "== gate: negative control (wrong TileIsWarpCdf row) =="
"$GATE_OUT" --negative-context --out "${BUILD}/warp_gate_neg.ivf" \
  --decoded "${BUILD}/warp_gate_neg_decoded.y4m"
