#!/usr/bin/env bash
# Build/run the CPU-vs-GPU parity test for src/cuda/codec/inter/mv_stack.
#
#   ./build_gpu_test.sh                 # sm_86, build + run
#   ARCH=sm_120 ./build_gpu_test.sh
#   ./build_gpu_test.sh --no-run
set -euo pipefail

HERE="$(cd -- "$(dirname -- "$0")" && pwd)"
ROOT="$(cd -- "${HERE}" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
  ROOT="$(dirname -- "${ROOT}")"
done

ARCH="${ARCH:-sm_86}"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build-mv-stack-gpu}"
RUN=1
for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    --arch=*) ARCH="${arg#--arch=}" ;;
    -h|--help) sed -n '2,6p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found; use ./build_emulation_test.sh on this host." >&2
  exit 1
fi

mkdir -p "${BUILD_DIR}"
echo "nvcc: $(nvcc --version | tail -1)"
echo "arch: ${ARCH}"
echo "build: ${BUILD_DIR}/mv_stack_test"

CPU_SOURCES=(
  "${ROOT}/src/codec/inter/merge/merge.cpp"
  "${ROOT}/src/codec/inter/multiref/mvstack_multi.cpp"
  "${ROOT}/src/codec/inter/multiref/compound_predict.cpp"
  "${ROOT}/src/codec/inter/mc/mc.cpp"
  "${ROOT}/src/codec/inter/advanced/compound.cpp"
  "${ROOT}/src/codec/partition/tree/partition.cpp"
)

nvcc -std=c++20 -O3 -lineinfo -arch="${ARCH}" \
  -Xcompiler -Wall,-Wextra,-Wconversion,-Wshadow \
  -I"${ROOT}/src" -I"${HERE}" \
  "${HERE}/mv_stack.cu" "${HERE}/mv_stack_test.cu" \
  "${ROOT}/src/cuda/arena.cpp" \
  "${CPU_SOURCES[@]}" \
  -o "${BUILD_DIR}/mv_stack_test"

if [ "${RUN}" -eq 1 ]; then
  CUDA_DISABLE_PTX_JIT=1 "${BUILD_DIR}/mv_stack_test"
fi
