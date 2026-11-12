#!/usr/bin/env bash
# Build and test the complete §7.12.2 MV stack, allow_warpmv_mode, the §7.7
# implicit reference map (R77) and the stack mirror.
#
#   ./build_and_test.sh              # unit tests + dav2d gate
#   ./build_and_test.sh --no-gate    # unit tests only
#   ./build_and_test.sh --no-run     # build only
#   SANITIZE=1 ./build_and_test.sh
#   WERROR=1   ./build_and_test.sh
#   CXX=clang++ ./build_and_test.sh
#
# The gate needs tools/bin/dav2d (Linux) or tools/bin/dav2d-macos (macOS), or
# DAV2D=/path/to/dav2d.  When no decoder is present the gate exits 127 and this
# script reports it as a skip; the unit tests still run and pass.
#
# This component owns only src/codec/inter/mv_stack/.  The gate compiles
# the existing M2 / advanced sources read-only; it never edits them.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
CXX="${CXX:-c++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
OUT_DIR="${OUT_DIR:-${TMPDIR:-/tmp}/mv_stack_full}"
RUN=1
GATE=1

for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    --no-gate) GATE=0 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$OUT_DIR"

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
       -Wno-missing-braces -Wno-sign-conversion -ffp-contract=off -I"$ROOT/src")
if [[ "$SANITIZE" == "1" ]]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2 -g)
fi
if [[ "$WERROR" == "1" ]]; then
  FLAGS+=(-Werror)
fi

MODULE=(
  "$HERE/mv_stack_full.cpp"
  "$HERE/frame_flags.cpp"
  "$HERE/ref_map.cpp"
  "$HERE/stack_io.cpp"
  "$HERE/stack_mirror.cpp"
)

# Dependencies used by the unit tests: the wired merge component and the
# §7.7/§7.8 reference-list component for parity, plus shared infrastructure.
TEST_DEPS=(
  "$ROOT/src/codec/inter/merge/merge.cpp"
  "$ROOT/src/codec/inter/advanced/video/advanced_syntax.cpp"
  "$ROOT/src/codec/inter/advanced/b_frame.cpp"
  "$ROOT/src/codec/partition/tree/partition.cpp"
  "$ROOT/src/codec/inter/mc/mc.cpp"
  "$ROOT/src/core/bitstream/bitwriter.cpp"
  "$ROOT/src/codec/entropy/av2/av2_entropy.cpp"
)

# M2 core used by the decoder gate (read-only).
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
  "$ROOT/src/codec/inter/advanced/warp.cpp"
  "$ROOT/src/codec/inter/merge/merge.cpp"
  "$ROOT/src/pipeline/video/video_headers.cpp"
  "$ROOT/src/pipeline/video/video_syntax.cpp"
  "$ROOT/src/pipeline/video/video_frame.cpp"
  "$ROOT/src/codec/inter/advanced/video/advanced_syntax.cpp"
)

echo "== build: unit tests =="
"$CXX" "${FLAGS[@]}" \
  "$HERE/mv_stack_full_test.cpp" "${MODULE[@]}" "${TEST_DEPS[@]}" \
  -o "${OUT_DIR}/mv_stack_full_test"

echo "== build: decoder gate =="
"$CXX" "${FLAGS[@]}" \
  "$HERE/gate/mv_stack_gate.cpp" "$HERE/gate/gate_stream.cpp" \
  "${MODULE[@]}" "${CORE[@]}" \
  -o "${OUT_DIR}/mv_stack_gate"

if [[ "$RUN" -eq 1 ]]; then
  echo "== unit tests =="
  "${OUT_DIR}/mv_stack_full_test"
fi

if [[ "$GATE" -eq 0 ]]; then
  exit 0
fi

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

echo "== gate: multi-reference stream, allow_warpmv_mode = 1 =="
"${OUT_DIR}/mv_stack_gate" --out "${OUT_DIR}/mv_stack_gate.ivf"

echo "== probe: every MV-stack index (byte-level ordering) =="
for k in 0 1 2 3 4 5; do
  echo "   GATE_DRL_INDEX=$k"
  GATE_DRL_INDEX="$k" GATE_SKIP_COVERAGE=1 \
    "${OUT_DIR}/mv_stack_gate" --out "${OUT_DIR}/drl${k}.ivf" \
    | grep -E "PASS|MISMATCH"
done

echo "== gate: warp-block stream (documented pinned-decoder limitation) =="
if GATE_WARP_BLOCKS=1 "${OUT_DIR}/mv_stack_gate" \
     --out "${OUT_DIR}/mv_stack_gate_warp.ivf" >/dev/null 2>&1; then
  echo "   unexpected PASS: the pinned decoder accepted is_warp blocks"
else
  echo "   expected FAIL: the pinned dav2d rejects is_warp blocks (README)"
fi
