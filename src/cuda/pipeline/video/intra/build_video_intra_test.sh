#!/usr/bin/env bash
# Build/run the M2 video intra candidate generator CPU-vs-GPU test.
#
#   ./build_video_intra_test.sh                 # sm_86, build + run
#   ./build_video_intra_test.sh --no-run        # build only
#   ./build_video_intra_test.sh --quick         # skip the TX sweep
#   ARCH=sm_120 ./build_video_intra_test.sh
#   SANITIZE=1 ./build_video_intra_test.sh      # see README for the sanitizer run
#
# Self-contained: does not touch the root CMake build.  The CPU references
# (`src/codec/intra/modes/intra_modes.cpp`, `src/codec/transform/variants/
# tx_variants.cpp`) are compiled by the host compiler with -ffp-contract=off;
# the CUDA side uses -fmad=false, matching the still TX stage.
set -euo pipefail
cd "$(dirname "$0")"

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${HERE}"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
  ROOT="$(dirname -- "${ROOT}")"
done
ARCH="${ARCH:-sm_86}"
CXX="${CXX:-g++}"
BUILD_DIR="${BUILD_DIR:-${HERE}/build-video-intra}"
RUN=1
QUICK=0
for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    --quick) QUICK=1 ;;
    --arch=*) ARCH="${arg#--arch=}" ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "${BUILD_DIR}"
echo "nvcc: $(nvcc --version | tail -1)"
echo "arch: ${ARCH}"
echo "build: ${BUILD_DIR}/video_intra_test"

HOST_FLAGS=(-std=c++20 -O3 -ffp-contract=off -I"${ROOT}/src")
if [ -d "${CUDA_HOME:-/usr/local/cuda}/include" ]; then
  HOST_FLAGS+=(-I"${CUDA_HOME:-/usr/local/cuda}/include")
fi

"${CXX}" "${HOST_FLAGS[@]}" -Wall -Wextra -Wconversion -Wshadow \
  -c "${ROOT}/src/cuda/arena.cpp" -o "${BUILD_DIR}/arena.o"
"${CXX}" "${HOST_FLAGS[@]}" -Wall -Wextra -Wconversion -Wshadow \
  -c "${ROOT}/src/codec/intra/modes/intra_modes.cpp" \
  -o "${BUILD_DIR}/intra_modes_cpu.o"
"${CXX}" "${HOST_FLAGS[@]}" -Wall -Wextra -Wconversion -Wshadow \
  -c "${ROOT}/src/codec/transform/variants/tx_variants.cpp" \
  -o "${BUILD_DIR}/tx_variants_cpu.o"

nvcc -std=c++20 -O3 -lineinfo -arch="${ARCH}" -fmad=false \
  -I"${ROOT}/src" -I"${HERE}" \
  -Xcompiler -Wall,-Wextra,-Wconversion,-Wshadow,-ffp-contract=off \
  --ptxas-options=-v \
  "${HERE}/video_intra.cu" \
  "${ROOT}/src/cuda/pipeline/still/tx/still_tx.cu" \
  "${HERE}/video_intra_test.cu" \
  "${BUILD_DIR}/arena.o" "${BUILD_DIR}/intra_modes_cpu.o" \
  "${BUILD_DIR}/tx_variants_cpu.o" \
  -o "${BUILD_DIR}/video_intra_test"

if [ "${RUN}" -eq 1 ]; then
  if [ "${QUICK}" -eq 1 ]; then
    "${BUILD_DIR}/video_intra_test" --quick
  else
    "${BUILD_DIR}/video_intra_test"
  fi
fi
