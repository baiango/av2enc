#!/usr/bin/env bash
# Build and run the M2 GPU-pipeline gate against the real CUDA backend.
#
# Requires nvcc + a CUDA device.  Everything except the CUDA backend and the
# device component sources is compiled by the host C++ compiler; nvcc links the
# final binary so libcudart comes from the toolkit.
#
# Usage:
#   ./build_gpu_gate.sh
#   ARCH=89 ./build_gpu_gate.sh
#   ./build_gpu_gate.sh --selftest 128
#   ./build_gpu_gate.sh --frames 8 --full-tree
#   SANITIZE=1 ./build_gpu_gate.sh      # memcheck selftest + 64x64 smoke
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
CXX="${CXX:-c++}"
NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-86}"
BUILD_DIR="${BUILD_DIR:-${TMPDIR:-/tmp}/video_gpu_cuda}"
OUT="${OUT:-$BUILD_DIR/video_gate_cuda}"

mkdir -p "$BUILD_DIR"

VIDEO_CPU_REF="${VIDEO_CPU_REF:-}"
VIDEO_SRC="$ROOT/src"
VIDEO_DIR="$ROOT/src/pipeline/video"
EXTRA_INCLUDES=()
if [[ -n "$VIDEO_CPU_REF" ]]; then
  VIDEO_SRC="$VIDEO_CPU_REF/src"
  VIDEO_DIR="$VIDEO_CPU_REF/src/pipeline/video"
  EXTRA_INCLUDES=(-I"$VIDEO_CPU_REF/src")
fi

HOST_FLAGS=(-std=c++20 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
            -Wno-missing-braces ${EXTRA_INCLUDES[@]+"${EXTRA_INCLUDES[@]}"} -I"$ROOT/src" -I"$HERE")
NVCC_FLAGS=(-std=c++20 -O2 -lineinfo -arch="sm_${ARCH}" --fmad=false
            -Xcompiler "-Wall,-Wextra,-Wconversion,-Wshadow"
            -I"$ROOT/src" -I"$HERE")

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
  "$ROOT/src/formats/format.cpp"
  "$VIDEO_DIR/video_headers.cpp"
  "$VIDEO_DIR/video_syntax.cpp"
  "$VIDEO_DIR/video_frame.cpp"
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

CUDA_SUPPORT=(
  "$ROOT/src/cuda/arena.cpp"
  "$ROOT/src/cuda/codec/inter/me/me.cu"
  "$ROOT/src/cuda/codec/inter/mc/mc.cu"
  "$ROOT/src/cuda/codec/inter/compound/compound_full.cu"
  "$ROOT/src/cuda/codec/intra/modes/intra_modes.cu"
  "$ROOT/src/cuda/codec/transform/variants/tx_variants.cu"
)

OBJECTS=()
index=0
for src in "${CORE[@]}" "$HERE/video_pipeline.cpp" "$HERE/video_frame_modes.cpp" \
           "$HERE/video_selftest.cpp" "$HERE/video_gate.cpp"; do
  obj="$BUILD_DIR/host_${index}.o"
  "$CXX" "${HOST_FLAGS[@]}" -c "$src" -o "$obj"
  OBJECTS+=("$obj")
  index=$((index + 1))
done

index=0
for src in "${CUDA_SUPPORT[@]}" "$HERE/video_backend_cuda.cu"; do
  obj="$BUILD_DIR/cuda_${index}.o"
  "$NVCC" "${NVCC_FLAGS[@]}" -c "$src" -o "$obj"
  OBJECTS+=("$obj")
  index=$((index + 1))
done

"$NVCC" -arch="sm_${ARCH}" "${OBJECTS[@]}" -o "$OUT"

if [[ -z "${DAV2D:-}" ]]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [[ -x "$candidate" ]]; then
      export DAV2D="$candidate"
      break
    fi
  done
fi

cd "$ROOT"

# SANITIZE=1 runs the selftest and a two-frame smoke gate under
# compute-sanitizer memcheck.  The full 720p gate is not run under the
# sanitizer (it is 50-100x slower); run it normally afterwards.
if [[ "${SANITIZE:-0}" == "1" ]]; then
  SAN=(compute-sanitizer --tool memcheck --target-processes all
       --print-limit 20)
  "${SAN[@]}" "$OUT" --selftest 128 || exit 1
  "${SAN[@]}" "$OUT" --frames 2 --no-hd \
    --out "${TMPDIR:-/tmp}/video_gpu_sanitize.ivf" || exit 1
  echo "compute-sanitizer smoke PASS; run without SANITIZE=1 for the full gate"
  exit 0
fi

"$OUT" --selftest 128
"$OUT" "$@"
