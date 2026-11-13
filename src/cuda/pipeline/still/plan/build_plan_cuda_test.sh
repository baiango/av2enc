#!/usr/bin/env bash
# Build and run the CPU-vs-GPU frame-plan parity test (sm_86).
#
#   ./src/cuda/pipeline/still/plan/build_plan_cuda_test.sh
#   OUT=/tmp/plan_cost_test ./src/cuda/pipeline/still/plan/build_plan_cuda_test.sh
#
# Self-contained: does not touch the root CMake build.  Device buffers come
# from one av2enc::cuda::Arena; the variance kernel never allocates.

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done

CXX="${CXX:-c++}"
NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-sm_86}"
OUT="${OUT:-${TMPDIR:-/tmp}/still_plan_cost_test}"
BUILD="${BUILD:-${TMPDIR:-/tmp}/still_plan_cost_build}"
mkdir -p "${BUILD}"

if ! command -v "${NVCC}" >/dev/null 2>&1; then
  echo "build_plan_cuda_test: nvcc not found (Linux + CUDA target only)" >&2
  exit 2
fi
CUDA_HOME="${CUDA_HOME:-$(dirname -- "$(dirname -- "$(readlink -f "$(command -v "${NVCC}")")")")}"

CFLAGS=(-std=c++20 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
        -ffp-contract=off -I"${ROOT}/src" -I"${CUDA_HOME}/include")
NVFLAGS=(-std=c++20 -O2 -arch="${ARCH}" -fmad=false -I"${ROOT}/src")

HOST_SOURCES=(
  "${HERE}/plan.cpp"
  "${ROOT}/src/cuda/arena.cpp"
  "${ROOT}/src/codec/partition/tree/partition.cpp"
  "${ROOT}/src/codec/partition/ext/partition_ext.cpp"
  "${ROOT}/src/codec/partition/sdp/sdp.cpp"
  "${ROOT}/src/codec/partition/sdp/sdp_validate.cpp"
  "${ROOT}/src/codec/partition/sdp/sdp_syntax.cpp"
  "${ROOT}/src/codec/transform/variants/tx_variants.cpp"
)

objects=()
for source in "${HOST_SOURCES[@]}"; do
  obj="${BUILD}/$(basename "${source}" .cpp).o"
  "${CXX}" "${CFLAGS[@]}" -c "${source}" -o "${obj}"
  objects+=("${obj}")
done

for source in "${HERE}/plan_cost.cu" "${HERE}/plan_cost_test.cu"; do
  obj="${BUILD}/$(basename "${source}" .cu).o"
  "${NVCC}" "${NVFLAGS[@]}" -c "${source}" -o "${obj}"
  objects+=("${obj}")
done

"${NVCC}" "${NVFLAGS[@]}" "${objects[@]}" -o "${OUT}"
"${OUT}"
