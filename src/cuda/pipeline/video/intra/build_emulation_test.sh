#!/usr/bin/env bash
# Build/run the host-emulation test for the M2 video intra candidate body.
#
#   ./build_emulation_test.sh             # build + run
#   ./build_emulation_test.sh --no-run    # build only
#   CXX=g++ SANITIZE=1 ./build_emulation_test.sh
#
# No CUDA compiler and no device: `video_intra_device.cuh` is compiled with the
# CUDA keywords stubbed and one emulated thread, then compared byte-for-byte with
# the CPU reference module.  The real sm_86 suite is build_video_intra_test.sh.
set -euo pipefail
cd "$(dirname "$0")"

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${HERE}"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
  ROOT="$(dirname -- "${ROOT}")"
done
CXX="${CXX:-g++}"
BUILD_DIR="${BUILD_DIR:-${HERE}/build-emulation}"
RUN=1
SAN_FLAGS=()
for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done
if [ "${SANITIZE:-0}" -eq 1 ]; then
  SAN_FLAGS=(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
fi

mkdir -p "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}/cuda_stub"
cat > "${BUILD_DIR}/cuda_stub/cuda_runtime.h" <<'EOF'
#pragma once
#include <cstddef>
typedef int cudaError_t;
typedef struct CUstream_st* cudaStream_t;
static const cudaError_t cudaSuccess = 0;
static const cudaError_t cudaErrorInvalidValue = 1;
static const cudaError_t cudaErrorNotSupported = 2;
static const cudaError_t cudaErrorMemoryAllocation = 3;
static const cudaError_t cudaErrorInvalidDevice = 4;
static const cudaError_t cudaErrorNoDevice = 5;
static const cudaError_t cudaErrorInsufficientDriver = 6;
static const cudaError_t cudaErrorInitializationError = 7;
static const cudaError_t cudaErrorInvalidConfiguration = 8;
static const cudaError_t cudaErrorInvalidResourceHandle = 9;
static const cudaError_t cudaErrorLaunchOutOfResources = 10;
static const cudaError_t cudaErrorNotReady = 11;
static const cudaError_t cudaErrorUnknown = 12;
inline const char* cudaGetErrorName(cudaError_t) { return "stub"; }
inline const char* cudaGetErrorString(cudaError_t) { return "stub"; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
EOF

echo "build: ${BUILD_DIR}/video_intra_emulation_test"

"${CXX}" -std=c++20 -O2 -ffp-contract=off \
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow \
  -I"${BUILD_DIR}/cuda_stub" -I"${ROOT}/src" \
  ${SAN_FLAGS[@]+"${SAN_FLAGS[@]}"} \
  "${HERE}/video_intra_emulation_test.cpp" \
  "${ROOT}/src/codec/intra/modes/intra_modes.cpp" \
  -o "${BUILD_DIR}/video_intra_emulation_test"

if [ "${RUN}" -eq 1 ]; then
  "${BUILD_DIR}/video_intra_emulation_test"
fi
