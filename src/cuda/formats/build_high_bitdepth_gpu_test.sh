#!/usr/bin/env bash
# Build/run the M10 CPU-vs-GPU bit-exact test on the sm_86 box.
#
#   ./build_high_bitdepth_gpu_test.sh
#   CUDA_ARCH=sm_86 CXX=g++ ./build_high_bitdepth_gpu_test.sh
#
# Requires nvcc.  Host objects are compiled with -ffp-contract=off; CUDA
# translation units use -fmad=false, the same rule as the M1/hbd parity tests.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${REPO}/CMakeLists.txt" ] && [ "${REPO}" != "/" ]; do
    REPO="$(dirname -- "${REPO}")"
done
BUILD_DIR="${BUILD_DIR:-${TMPDIR:-/tmp}/av2enc_high_bitdepth_build}"
ARCH="${CUDA_ARCH:-sm_86}"
CXX="${CXX:-c++}"
mkdir -p "${BUILD_DIR}"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "error: nvcc not found (run on the Linux + CUDA box)" >&2
  exit 127
fi

WARN=(-Wall -Wextra -Wconversion -Wshadow -Wno-sign-conversion)

compile_host() {
  local src=$1
  local obj=$2
  "${CXX}" -std=c++20 -O3 -ffp-contract=off "${WARN[@]}" \
    -I"${REPO}/src" -c "${src}" -o "${obj}"
}

HOST_SOURCES=(
  "${REPO}/src/pipeline/still/high_bitdepth/high_bitdepth_headers.cpp"
  "${REPO}/src/pipeline/still/high_bitdepth/high_bitdepth_entropy.cpp"
  "${REPO}/src/pipeline/still/high_bitdepth/high_bitdepth_still.cpp"
  "${REPO}/src/formats/format.cpp"
  "${REPO}/src/cuda/formats/format_device.cpp"
  "${REPO}/src/formats/high_bitdepth/hbd_depth.cpp"
  "${REPO}/src/formats/high_bitdepth/hbd_headers.cpp"
  "${REPO}/src/formats/high_bitdepth/hbd_still.cpp"
  "${REPO}/src/codec/transform/variants/tx_variants.cpp"
  "${REPO}/src/cuda/pipeline/still/hbd/hbd_cpu.cpp"
  "${REPO}/src/codec/filters/deblock/deblock.cpp"
  "${REPO}/src/codec/entropy/av2/av2_entropy.cpp"
  "${REPO}/src/core/bitstream/bitwriter.cpp"
  "${REPO}/src/core/bitstream/obu.cpp"
  "${REPO}/src/core/bitstream/basic_headers.cpp"
)

HOST_OBJECTS=()
for src in "${HOST_SOURCES[@]}"; do
  obj="${BUILD_DIR}/$(basename "${src}").o"
  compile_host "${src}" "${obj}"
  HOST_OBJECTS+=("${obj}")
done

CUDA_SOURCES=(
  "${REPO}/src/cuda/arena.cpp"
  "${REPO}/src/cuda/formats/high_bitdepth_gpu.cu"
  "${REPO}/src/cuda/formats/high_bitdepth_gpu_test.cu"
  "${REPO}/src/cuda/pipeline/still/hbd/hbd_tx.cu"
  "${REPO}/src/cuda/codec/filters/deblock/deblock.cu"
)

nvcc -std=c++20 -arch="${ARCH}" -O3 -lineinfo -fmad=false \
  -I"${REPO}/src" \
  -Xcompiler=-Wall,-Wextra,-Wconversion,-Wshadow,-ffp-contract=off \
  "${CUDA_SOURCES[@]}" "${HOST_OBJECTS[@]}" \
  -o "${BUILD_DIR}/high_bitdepth_cuda_test"

"${BUILD_DIR}/high_bitdepth_cuda_test"
