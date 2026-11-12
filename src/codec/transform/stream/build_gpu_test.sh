#!/usr/bin/env bash
# Build and run the CPU-vs-GPU parity test for the M1 transform-stream adapter.
# Run this on the Linux + CUDA target (RTX 3060, sm_86):
#
#   ./build_gpu_test.sh
#   CUDA_ARCH=sm_86 ./build_gpu_test.sh
#   CXX=g++-13 ./build_gpu_test.sh
#
# The CPU adapter and reference components are compiled with -ffp-contract=off;
# nvcc uses -fmad=false.  This checkout (macOS) has no nvcc, so the binary is
# not built here.  The glue logic is additionally exercised by a host emulation
# of the CUDA entry points; see README.md.
set -euo pipefail
cd "$(dirname "$0")"

REPO="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${REPO}/CMakeLists.txt" ] && [ "${REPO}" != "/" ]; do
    REPO="$(dirname -- "${REPO}")"
done
ARCH="${CUDA_ARCH:-sm_86}"
CXX="${CXX:-g++}"
BUILD_DIR="${BUILD_DIR:-${TMPDIR:-/tmp}/av2enc_transform_stream_gpu_build}"
DATA="${DATA:-${REPO}/tests/data}"
mkdir -p "${BUILD_DIR}"

CPU_SOURCES=(
  "${REPO}/src/codec/transform/stream/transform_stream.cpp"
  "${REPO}/src/codec/transform/tx_type/tx_type.cpp"
  "${REPO}/src/codec/transform/variants/tx_variants.cpp"
  "${REPO}/src/codec/rate/rdoq/rdoq.cpp"
  "${REPO}/src/codec/rate/rdoq/rdoq_av2_model.cpp"
  "${REPO}/src/codec/intra/modes/intra_modes.cpp"
  "${REPO}/src/codec/rate/rdo/rdo.cpp"
  "${REPO}/src/core/io/y4m.cpp"
)

OBJECTS=()
index=0
for source in "${CPU_SOURCES[@]}"; do
  object="${BUILD_DIR}/cpu_${index}.o"
  "${CXX}" -std=c++20 -O3 -ffp-contract=off -w -I"${REPO}/src" \
    -c "${source}" -o "${object}"
  OBJECTS+=("${object}")
  index=$((index + 1))
done

nvcc -std=c++20 \
  -arch="${ARCH}" \
  -O3 \
  -lineinfo \
  -fmad=false \
  -I"${REPO}/src" \
  -Xcompiler=-ffp-contract=off \
  transform_stream_gpu.cu transform_stream_gpu_test.cu \
  "${REPO}/src/cuda/codec/transform/tx_type/tx_type.cu" \
  "${REPO}/src/cuda/codec/rate/rdoq/rdoq_batch.cu" \
  "${REPO}/src/cuda/codec/transform/variants/tx_variants.cu" \
  "${OBJECTS[@]}" \
  -o "${BUILD_DIR}/transform_stream_gpu_test"

echo "built ${BUILD_DIR}/transform_stream_gpu_test (arch ${ARCH})"
"${BUILD_DIR}/transform_stream_gpu_test" --data "${DATA}"
