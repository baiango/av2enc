#!/usr/bin/env bash
# Build (and by default run) the real-GPU intra-full test.
#
#   ./build_gpu_test.sh                  # sm_86 (8 GB Ampere primary target)
#   CUDA_ARCH=sm_120 ./build_gpu_test.sh # secondary target
#   ./build_gpu_test.sh --no-run         # build only
#
# The CPU reference components are compiled by the host compiler with
# -ffp-contract=off; nvcc builds the device code with -fmad=false so the
# binary64 forward transform can be compared bit-for-bit.  All device memory in
# the test comes from src/cuda/arena.cpp (no cudaMalloc in the test or module).
set -euo pipefail
cd "$(dirname "$0")"

REPO="$(cd ../../.. && pwd)"
CUDA_ARCH="${CUDA_ARCH:-${ARCH:-sm_86}}"
CXX="${CXX:-g++}"
NVCC="${NVCC:-nvcc}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
BUILD_DIR="${BUILD_DIR:-${TMPDIR:-/tmp}/av2enc_intra_full_build}"
RUN=1
for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    --arch=*) CUDA_ARCH="${arg#--arch=}" ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$BUILD_DIR"

CUDA_INC=()
if [ -d "$CUDA_HOME/include" ]; then
  CUDA_INC=(-I"$CUDA_HOME/include")
fi

HOST_FLAGS=(-std=c++20 -O3 -ffp-contract=off -I"$REPO/src" "${CUDA_INC[@]}" -w)
REF_SOURCES=(
  "$REPO/src/cuda/arena.cpp"
  "$REPO/src/encoder/m1/missing_bits/dc_intra_subsampled.cpp"
  "$REPO/src/encoder/m1/missing_bits/intra_filter_corner.cpp"
  "$REPO/src/encoder/tx_variants/tx_variants.cpp"
  "$REPO/src/encoder/transform/idtx_intra/idtx_intra.cpp"
  "$REPO/src/encoder/tx_type/tx_type.cpp"
  "$REPO/src/encoder/intra_modes/intra_modes.cpp"
)

echo "nvcc:   $("$NVCC" --version | tail -1)"
echo "arch:   $CUDA_ARCH"
echo "build:  $BUILD_DIR/intra_full_test"

OBJECTS=()
index=0
for source in "${REF_SOURCES[@]}"; do
  object="$BUILD_DIR/ref_${index}.o"
  "$CXX" "${HOST_FLAGS[@]}" -c "$source" -o "$object"
  OBJECTS+=("$object")
  index=$((index + 1))
done

"$NVCC" -std=c++20 -arch="$CUDA_ARCH" -O3 -lineinfo -fmad=false \
  -I"$REPO/src" -I"$REPO/src/cuda/intra_full" \
  -Xcompiler=-Wall,-Wextra,-Wconversion,-Wshadow,-ffp-contract=off \
  -Xptxas=-v \
  intra_full.cu intra_full_test.cu "${OBJECTS[@]}" \
  -o "$BUILD_DIR/intra_full_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/intra_full_test"
fi
