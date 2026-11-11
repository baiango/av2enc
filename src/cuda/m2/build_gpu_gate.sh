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
ROOT="$(cd -- "$HERE/../../.." && pwd)"
CXX="${CXX:-c++}"
NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-86}"
BUILD_DIR="${BUILD_DIR:-${TMPDIR:-/tmp}/m2_gpu_cuda}"
OUT="${OUT:-$BUILD_DIR/m2_gate_cuda}"

mkdir -p "$BUILD_DIR"

M2_CPU_REF="${M2_CPU_REF:-}"
M2_DIR="$ROOT/src/encoder/m2"
EXTRA_INCLUDES=()
if [[ -n "$M2_CPU_REF" ]]; then
  M2_DIR="$M2_CPU_REF/src/encoder/m2"
  EXTRA_INCLUDES=(-I"$M2_CPU_REF/src")
fi

HOST_FLAGS=(-std=c++20 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
            -Wno-missing-braces ${EXTRA_INCLUDES[@]+"${EXTRA_INCLUDES[@]}"} -I"$ROOT/src" -I"$HERE")
NVCC_FLAGS=(-std=c++20 -O2 -lineinfo -arch="sm_${ARCH}" --fmad=false
            -Xcompiler "-Wall,-Wextra,-Wconversion,-Wshadow"
            -I"$ROOT/src" -I"$HERE")

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

CUDA_SUPPORT=(
  "$ROOT/src/cuda/inter/me/me.cu"
  "$ROOT/src/cuda/inter/mc/mc.cu"
  "$ROOT/src/cuda/intra_modes/intra_modes.cu"
  "$ROOT/src/cuda/tx_variants/tx_variants.cu"
)

OBJECTS=()
index=0
for src in "${CORE[@]}" "$HERE/m2_pipeline.cpp" "$HERE/m2_selftest.cpp" \
           "$HERE/m2_gate.cpp"; do
  obj="$BUILD_DIR/host_${index}.o"
  "$CXX" "${HOST_FLAGS[@]}" -c "$src" -o "$obj"
  OBJECTS+=("$obj")
  index=$((index + 1))
done

index=0
for src in "${CUDA_SUPPORT[@]}" "$HERE/m2_backend_cuda.cu"; do
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
    --out "${TMPDIR:-/tmp}/m2_gpu_sanitize.ivf" || exit 1
  echo "compute-sanitizer smoke PASS; run without SANITIZE=1 for the full gate"
  exit 0
fi

"$OUT" --selftest 128
"$OUT" "$@"
