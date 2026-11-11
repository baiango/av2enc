#!/usr/bin/env bash
# Build and run the M2 GPU-pipeline gate against the CPU-emulation backend.
#
# This needs no GPU and no nvcc.  It builds the exact orchestration that
# m2_backend_cuda.cu plugs into, links the CPU component references in place of
# the device components, and checks:
#   * CPU M2 vs GPU-path M2: temporal units byte-identical, recon 0 diffs
#   * dav2d decodes the GPU-path stream to the same reconstruction
#   * the inter residual TX/quant/inverse chain matches the CPU transform
#
# The same m2_gate.cpp builds against the real CUDA backend with
# build_gpu_gate.sh.
#
# Usage:
#   ./build_and_test.sh                 # build + full gate
#   ./build_and_test.sh --no-run
#   ./build_and_test.sh --frames 8 --full-tree
#   OUT=/tmp/m2gpu ./build_and_test.sh
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../../.." && pwd)"
CXX="${CXX:-c++}"
BUILD_DIR="${BUILD_DIR:-${TMPDIR:-/tmp}/m2_gpu_emulation}"
OUT="${OUT:-$BUILD_DIR/m2_gate}"
RUN=1
ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-run) RUN=0; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

# Optional CPU-reference snapshot.  If the working tree's src/encoder/m2 is
# mid-refactor, point M2_CPU_REF at a checkout root (containing src/encoder/m2)
# to build the gate against that reference without touching the tree.
M2_CPU_REF="${M2_CPU_REF:-}"
M2_SRC="$ROOT/src"
M2_DIR="$ROOT/src/encoder/m2"
EXTRA_INCLUDES=()
if [[ -n "$M2_CPU_REF" ]]; then
  M2_SRC="$M2_CPU_REF/src"
  M2_DIR="$M2_CPU_REF/src/encoder/m2"
  EXTRA_INCLUDES=(-I"$M2_SRC")
fi

FLAGS=(-std=c++20 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
       -Wno-missing-braces ${EXTRA_INCLUDES[@]+"${EXTRA_INCLUDES[@]}"} -I"$ROOT/src" -I"$HERE")
if [[ "${WERROR:-0}" == "1" ]]; then
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
  "$M2_DIR/m2_headers.cpp"
  "$M2_DIR/m2_syntax.cpp"
  "$M2_DIR/m2_frame.cpp"
)

# The extended CPU M2 reference grows optional modules.  Link the merge helper
# when present (the CPU gate lists it too).  M2_CPU_REF snapshots are expected
# to be self-contained.
if [[ -f "$M2_DIR/merge/merge.cpp" ]]; then
  CORE+=("$M2_DIR/merge/merge.cpp")
fi

mkdir -p "$BUILD_DIR"
"${CXX}" "${FLAGS[@]}" \
  "$HERE/m2_gate.cpp" \
  "$HERE/m2_pipeline.cpp" \
  "$HERE/m2_backend_cpu.cpp" \
  "$HERE/m2_selftest.cpp" \
  "${CORE[@]}" \
  -o "$OUT"

if [[ "$RUN" -eq 0 ]]; then
  echo "built $OUT"
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

cd "$ROOT"
"$OUT" --selftest 64
"$OUT" ${ARGS[@]+"${ARGS[@]}"}
