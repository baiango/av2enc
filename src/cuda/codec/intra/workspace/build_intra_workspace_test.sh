#!/usr/bin/env bash
# Build/run the arena-backed intra workspace CPU-vs-GPU test.
#
#   ./build_intra_workspace_test.sh                 # sm_86, build + run
#   ./build_intra_workspace_test.sh --no-run
#   ARCH=sm_120 ./build_intra_workspace_test.sh
#
# Self-contained: does not touch the root CMake build.  The CPU reference
# (src/codec/intra/modes/intra_modes.cpp) is compiled separately so this is a
# true CPU-vs-GPU comparison for the modes path, not two kernel copies.
set -euo pipefail
cd "$(dirname "$0")"

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ARCH="${ARCH:-sm_86}"
BUILD_DIR="${BUILD_DIR:-${HERE}/build-workspace-gpu}"
CXX="${CXX:-g++}"
RUN=1

for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    --arch=*) ARCH="${arg#--arch=}" ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

ROOT="${HERE}"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
  ROOT="$(dirname -- "${ROOT}")"
done

mkdir -p "${BUILD_DIR}"
echo "nvcc: $(nvcc --version | tail -1)"
echo "arch: ${ARCH}"
echo "build: ${BUILD_DIR}/intra_workspace_test"

HOST_FLAGS=(-std=c++20 -O2 -ffp-contract=off -I"${ROOT}/src")
if [ -d "${CUDA_HOME:-/usr/local/cuda}/include" ]; then
  HOST_FLAGS+=(-I"${CUDA_HOME:-/usr/local/cuda}/include")
fi

"${CXX}" "${HOST_FLAGS[@]}" -c "${ROOT}/src/cuda/arena.cpp" \
  -o "${BUILD_DIR}/arena.o"
"${CXX}" "${HOST_FLAGS[@]}" -c \
  "${ROOT}/src/codec/intra/modes/intra_modes.cpp" \
  -o "${BUILD_DIR}/intra_modes_cpu.o"
"${CXX}" "${HOST_FLAGS[@]}" -c \
  "${ROOT}/src/codec/intra/mrl/core/mrl.cpp" \
  -o "${BUILD_DIR}/mrl_cpu.o"
"${CXX}" "${HOST_FLAGS[@]}" -c \
  "${ROOT}/src/codec/intra/ibp/ibp.cpp" \
  -o "${BUILD_DIR}/ibp_cpu.o"
"${CXX}" "${HOST_FLAGS[@]}" -c \
  "${ROOT}/src/codec/intra/dip/dip.cpp" \
  -o "${BUILD_DIR}/dip_cpu.o"
"${CXX}" "${HOST_FLAGS[@]}" -c \
  "${ROOT}/src/codec/intra/cfl/cfl.cpp" \
  -o "${BUILD_DIR}/cfl_cpu.o"
"${CXX}" "${HOST_FLAGS[@]}" -c \
  "${ROOT}/src/codec/intra/palette/core/palette.cpp" \
  -o "${BUILD_DIR}/palette_cpu.o"
"${CXX}" "${HOST_FLAGS[@]}" -c \
  "${ROOT}/src/codec/entropy/av2/av2_entropy.cpp" \
  -o "${BUILD_DIR}/av2_entropy.o"
"${CXX}" "${HOST_FLAGS[@]}" -c \
  "${ROOT}/src/codec/intra/ibc/core/ibc.cpp" \
  -o "${BUILD_DIR}/ibc_cpu.o"
"${CXX}" "${HOST_FLAGS[@]}" -c \
  "${ROOT}/src/codec/partition/tree/partition.cpp" \
  -o "${BUILD_DIR}/partition.o"

nvcc -std=c++20 -O3 -lineinfo -arch="${ARCH}" \
  -I"${ROOT}/src" -I"${HERE}" \
  -Xcompiler -Wall,-Wextra,-Wconversion,-Wshadow \
  "${HERE}/intra_workspace.cu" \
  "${ROOT}/src/cuda/codec/intra/modes/intra_modes.cu" \
  "${ROOT}/src/cuda/codec/intra/mrl/mrl.cu" \
  "${ROOT}/src/cuda/codec/intra/ibp/ibp.cu" \
  "${ROOT}/src/cuda/codec/intra/dip/dip.cu" \
  "${ROOT}/src/cuda/codec/intra/cfl/cfl.cu" \
  "${ROOT}/src/cuda/codec/intra/palette/palette.cu" \
  "${ROOT}/src/cuda/codec/intra/ibc/ibc.cu" \
  "${HERE}/intra_workspace_test.cu" \
  "${BUILD_DIR}/arena.o" "${BUILD_DIR}/intra_modes_cpu.o" \
  "${BUILD_DIR}/mrl_cpu.o" "${BUILD_DIR}/ibp_cpu.o" \
  "${BUILD_DIR}/dip_cpu.o" "${BUILD_DIR}/cfl_cpu.o" \
  "${BUILD_DIR}/palette_cpu.o" "${BUILD_DIR}/av2_entropy.o" \
  "${BUILD_DIR}/ibc_cpu.o" "${BUILD_DIR}/partition.o" \
  -o "${BUILD_DIR}/intra_workspace_test"

if [ "${RUN}" -eq 1 ]; then
  "${BUILD_DIR}/intra_workspace_test"
fi
