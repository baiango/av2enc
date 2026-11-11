#!/usr/bin/env bash
# Real CPU-vs-GPU bit-exactness test for the explicit CfL alpha path.
#
# Run on the Linux + CUDA box (nvcc, sm_86):
#   ./build_cuda_test.sh
#   ARCH=sm_120 ./build_cuda_test.sh
#   ./build_cuda_test.sh --no-run
#
# Host-only .cpp files are compiled with the host C++ compiler and the two .cu
# files with nvcc, then linked with nvcc. That keeps the §8.2 coder and the
# search code out of nvcc's device pass. Self-contained: nothing touches the
# root CMake build.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ARCH="${ARCH:-sm_86}"
CXX="${CXX:-g++}"
BUILD_DIR="${BUILD_DIR:-${HERE}/build-alpha-cuda}"
RUN=1

for arg in "$@"; do
  case "$arg" in
    --no-run)
      RUN=0
      ;;
    --arch=*)
      ARCH="${arg#--arch=}"
      ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

mkdir -p "${BUILD_DIR}"

echo "nvcc: $(nvcc --version | tail -1)"
echo "host: $(${CXX} --version | head -1)"
echo "arch: ${ARCH}"

HOST_FLAGS=(-std=c++20 -O3 -Wall -Wextra -Wconversion -Wshadow)

"${CXX}" "${HOST_FLAGS[@]}" -c "${HERE}/alpha.cpp" \
  -o "${BUILD_DIR}/alpha.o"
"${CXX}" "${HOST_FLAGS[@]}" -c "${HERE}/syntax.cpp" \
  -o "${BUILD_DIR}/syntax.o"
"${CXX}" "${HOST_FLAGS[@]}" -c "${HERE}/../cfl.cpp" \
  -o "${BUILD_DIR}/cfl_cpu.o"
"${CXX}" "${HOST_FLAGS[@]}" -c "${HERE}/../../../entropy/av2/av2_entropy.cpp" \
  -o "${BUILD_DIR}/av2_entropy.o"

nvcc -std=c++20 -O3 -lineinfo -arch="${ARCH}" \
  -c "${HERE}/alpha_gpu_test.cu" -o "${BUILD_DIR}/alpha_gpu_test.o"
nvcc -std=c++20 -O3 -lineinfo -arch="${ARCH}" \
  -c "${HERE}/../../../cuda/cfl/cfl.cu" -o "${BUILD_DIR}/cfl_gpu.o"

nvcc "${BUILD_DIR}/alpha_gpu_test.o" \
  "${BUILD_DIR}/alpha.o" \
  "${BUILD_DIR}/syntax.o" \
  "${BUILD_DIR}/cfl_cpu.o" \
  "${BUILD_DIR}/cfl_gpu.o" \
  "${BUILD_DIR}/av2_entropy.o" \
  -o "${BUILD_DIR}/cfl_alpha_cuda_test"

if [[ "${RUN}" -eq 1 ]]; then
  "${BUILD_DIR}/cfl_alpha_cuda_test"
fi
