#!/usr/bin/env bash
# Build and run the M2 GPU-pipeline gate against the CPU-emulation backend.
#
# This needs no GPU and no nvcc.  It builds the exact orchestration that
# video_backend_cuda.cu plugs into, links the CPU component references in place of
# the device components, and checks:
#   * CPU M2 vs GPU-path M2: temporal units byte-identical, recon 0 diffs
#   * dav2d decodes the GPU-path stream to the same reconstruction
#   * the inter residual TX/quant/inverse chain matches the CPU transform
#
# The same video_gate.cpp builds against the real CUDA backend with
# build_gpu_gate.sh.
#
# Usage:
#   ./build_and_test.sh                 # build + full gate
#   ./build_and_test.sh --no-run
#   ./build_and_test.sh --frames 8 --full-tree
#   OUT=/tmp/video_gpu ./build_and_test.sh
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
CXX="${CXX:-c++}"
BUILD_DIR="${BUILD_DIR:-${TMPDIR:-/tmp}/video_gpu_emulation}"
OUT="${OUT:-$BUILD_DIR/video_gate}"
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
# mid-refactor, point VIDEO_CPU_REF at a checkout root (containing src/encoder/m2)
# to build the gate against that reference without touching the tree.
VIDEO_CPU_REF="${VIDEO_CPU_REF:-}"
VIDEO_SRC="$ROOT/src"
VIDEO_DIR="$ROOT/src/pipeline/video"
EXTRA_INCLUDES=()
if [[ -n "$VIDEO_CPU_REF" ]]; then
  VIDEO_SRC="$VIDEO_CPU_REF/src"
  VIDEO_DIR="$VIDEO_CPU_REF/src/pipeline/video"
  EXTRA_INCLUDES=(-I"$VIDEO_SRC")
fi

FLAGS=(-std=c++20 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
       -Wno-missing-braces -DAV2ENC_VIDEO_FRAMES=1 -DAV2ENC_VIDEO_CDF_CHAIN=1
       ${EXTRA_INCLUDES[@]+"${EXTRA_INCLUDES[@]}"} -I"$ROOT/src" -I"$HERE")
if [[ "${WERROR:-0}" == "1" ]]; then
  FLAGS+=(-Werror)
fi

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
  "$ROOT/src/codec/inter/drl_reorder/drl_reorder.cpp"
  "$ROOT/src/codec/inter/drl_reorder/video_stack_adapter.cpp"
  "$ROOT/src/codec/inter/mv_precision/mv_precision.cpp"
  "$ROOT/src/codec/inter/longterm/longterm_headers.cpp"
  "$ROOT/src/codec/inter/longterm/longterm_pool.cpp"
  "$ROOT/src/codec/inter/longterm/longterm_stream.cpp"
  "$ROOT/src/codec/inter/leading/leading.cpp"
  "$ROOT/src/codec/inter/sef/sef.cpp"
  "$ROOT/src/codec/inter/random_access/random_access.cpp"
  "$ROOT/src/codec/inter/tlayers/tlayers.cpp"
  "$ROOT/src/codec/inter/tlayers/tlayers_syntax.cpp"
  "$ROOT/src/codec/inter/advanced/b_frame.cpp"
  "$ROOT/src/codec/inter/bframes/bframes.cpp"
  "$ROOT/src/codec/inter/multiref/compound_predict.cpp"
  "$ROOT/src/codec/inter/advanced/compound.cpp"
  "$ROOT/src/pipeline/video/post/tiled_frame.cpp"
  "$ROOT/src/pipeline/video/post/bit_depth_checklist.cpp"
  "$ROOT/src/pipeline/video/post/loop_filters.cpp"
  "$ROOT/src/pipeline/video/post/rc_adapter.cpp"
  "$ROOT/src/pipeline/tiles/tile_layout.cpp"
  "$ROOT/src/pipeline/tiles/tile_group.cpp"
  "$ROOT/src/codec/filters/stream/filters_stream.cpp"
  "$ROOT/src/codec/filters/deblock/deblock.cpp"
  "$ROOT/src/codec/filters/cdef/cdef.cpp"
  "$ROOT/src/codec/filters/loop_restoration/restoration.cpp"
  "$ROOT/src/codec/rate/rc/rc.cpp"
  "$ROOT/src/frame/mfh/mfh.cpp"
  "$ROOT/src/codec/entropy/avg_cdf/avg_cdf_bank.cpp"
  "$ROOT/src/frame/metadata/content_interpretation/content_interpretation.cpp"
  "$ROOT/src/formats/format.cpp"
  "$VIDEO_DIR/video_headers.cpp"
  "$VIDEO_DIR/video_syntax.cpp"
  "$VIDEO_DIR/video_frame.cpp"
  "$VIDEO_DIR/video_frames.cpp"
)

# The extended CPU M2 reference grows optional modules.  Link the merge helper
# when present (the CPU gate lists it too).  VIDEO_CPU_REF snapshots are expected
# to be self-contained.
if [[ -f "$VIDEO_SRC/codec/inter/merge/merge.cpp" ]]; then
  CORE+=("$VIDEO_SRC/codec/inter/merge/merge.cpp")
fi

# The M2 writer now calls into drl_reorder / mv_precision / video post
# (tiles, RC adapter) and the rest of the integrated core.  Take the canonical
# `av2enc_core` list instead of chasing undefined references by hand.
source "$ROOT/src/cuda/pipeline/cpu_reference_sources.sh"
CORE_RELATIVE=()
for src in "${CORE[@]}"; do
  case "$src" in
    "$ROOT"/*) CORE_RELATIVE+=("${src#"$ROOT"/}") ;;
    "$VIDEO_SRC"/*) CORE_RELATIVE+=("src/${src#"$VIDEO_SRC"/}") ;;
    *) CORE_RELATIVE+=("$src") ;;
  esac
done
av2enc_cpu_core_append CORE "$ROOT/" \
  "${CORE_RELATIVE[@]+"${CORE_RELATIVE[@]}"}"

mkdir -p "$BUILD_DIR"
"${CXX}" "${FLAGS[@]}" \
  "$HERE/video_gate.cpp" \
  "$HERE/video_pipeline.cpp" \
  "$HERE/video_frame_modes.cpp" \
  "$HERE/video_backend_cpu.cpp" \
  "$HERE/video_selftest.cpp" \
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
