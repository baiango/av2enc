#!/usr/bin/env bash
# Real-frame CPU-vs-(emulated)GPU parity test for the M1 chroma module.
#
# Compiles the real still_chroma_test.cu as C++ with a CUDA-runtime stub and a
# shim that runs the exact device search body (still_chroma_device.cuh) with one
# emulated thread. No GPU or nvcc needed. This exercises the actual
# still_encoder.cpp block loop and the full 64x64 / 640x360 / 1920x1080 frame paths.
#
#   ./build_frame_emulation_test.sh [--y4m FILE ...]
#   ./build_frame_emulation_test.sh --no-run
#
# Without --y4m the test falls back to built-in 64x64 and 640x360 frames.
set -euo pipefail
cd "$(dirname "$0")"

ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
BUILD_DIR="${BUILD_DIR:-$ROOT/build-m1-chroma-frame-emulation}"
RUN=1
Y4M_ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --no-run) RUN=0; shift ;;
    --y4m) Y4M_ARGS+=(--y4m "$2"); shift 2 ;;
    --y4m=*) Y4M_ARGS+=(--y4m "${1#--y4m=}"); shift ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$BUILD_DIR/cuda_stub"
cat > "$BUILD_DIR/cuda_stub/cuda_runtime.h" <<'EOF'
#pragma once
#define CUDART_VERSION 12080
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
typedef int cudaError_t;
typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;
enum cudaMemcpyKind {
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3
};
static const cudaError_t cudaSuccess = 0;
static const cudaError_t cudaErrorInvalidValue = 1;
static const cudaError_t cudaErrorNotSupported = 2;
struct cudaDeviceProp { char name[256]; int major, minor, multiProcessorCount; };
static inline const char* cudaGetErrorString(cudaError_t) { return "emu"; }
static inline cudaError_t cudaGetLastError() { return cudaSuccess; }
template <class T>
static inline cudaError_t cudaMalloc(T** p, std::size_t n) {
  *p = static_cast<T*>(std::malloc(n ? n : 1));
  return *p ? cudaSuccess : cudaErrorInvalidValue;
}
static inline cudaError_t cudaFree(void* p) { std::free(p); return cudaSuccess; }
static inline cudaError_t cudaMemcpy(void* d, const void* s, std::size_t n,
                                     cudaMemcpyKind) {
  std::memcpy(d, s, n);
  return cudaSuccess;
}
static inline cudaError_t cudaMemcpy2D(void* d, std::size_t dp, const void* s,
                                      std::size_t sp, std::size_t w,
                                      std::size_t h, cudaMemcpyKind) {
  for (std::size_t r = 0; r < h; ++r)
    std::memcpy(static_cast<char*>(d) + r * dp,
                static_cast<const char*>(s) + r * sp, w);
  return cudaSuccess;
}
static inline cudaError_t cudaMemset(void* d, int v, std::size_t n) {
  std::memset(d, v, n);
  return cudaSuccess;
}
static inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
static inline cudaError_t cudaGetDevice(int* d) { *d = 0; return cudaSuccess; }
static inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p, int) {
  std::snprintf(p->name, sizeof(p->name), "emulated");
  p->major = 8; p->minor = 6; p->multiProcessorCount = 28;
  return cudaSuccess;
}
struct EmuEvent { float ms; };
static inline cudaError_t cudaEventCreate(cudaEvent_t* e) {
  *e = reinterpret_cast<cudaEvent_t>(new EmuEvent{0});
  return cudaSuccess;
}
static inline cudaError_t cudaEventDestroy(cudaEvent_t e) {
  delete reinterpret_cast<EmuEvent*>(e);
  return cudaSuccess;
}
static inline cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t = nullptr) {
  return cudaSuccess;
}
static inline cudaError_t cudaEventSynchronize(cudaEvent_t) { return cudaSuccess; }
static inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t,
                                               cudaEvent_t) {
  *ms = 0.1f;
  return cudaSuccess;
}
EOF

echo "build: $BUILD_DIR"
CXX="${CXX:-c++}"
CFLAGS=(-std=c++20 -O2 -ffp-contract=off -fno-fast-math
        -I"$BUILD_DIR/cuda_stub" -I"$ROOT/src" -I"$ROOT/src/cuda/pipeline/still/chroma")

# Compile the real GPU test TU as C++.
"$CXX" "${CFLAGS[@]}" -x c++ -c still_chroma_test.cu \
  -o "$BUILD_DIR/still_chroma_test.o"
"$CXX" "${CFLAGS[@]}" -x c++ -c frame_emulation_shim.cpp \
  -o "$BUILD_DIR/frame_emulation_shim.o"

# Shared CPU-reference source list (still_syntax + the integrated tool set);
# still_encoder.cpp is included textually by still_chroma_test.cu.
source "$ROOT/src/cuda/codec/filters/still_reference_sources.sh"
HOST_SOURCES=("${STILL_REFERENCE_SOURCES[@]}")

# Fill in the rest of the canonical CPU reference (`av2enc_core`).
source "$ROOT/src/cuda/pipeline/cpu_reference_sources.sh"
# still_encoder.cpp is #included by still_chroma_test.cu, so it must not be
# linked a second time.
av2enc_cpu_core_append HOST_SOURCES "" \
  "${HOST_SOURCES[@]+"${HOST_SOURCES[@]}"}" \
  src/pipeline/still/still_encoder.cpp

HOST_OBJECTS=()
for src in "${HOST_SOURCES[@]}"; do
  if [ ! -f "$ROOT/$src" ]; then
    echo "missing host source: $src" >&2
    exit 2
  fi
  obj="$BUILD_DIR/$(echo "$src" | tr '/' '_').o"
  "$CXX" -std=c++20 -O2 -ffp-contract=off -fno-fast-math \
    -I"$ROOT/src" -c "$ROOT/$src" -o "$obj"
  HOST_OBJECTS+=("$obj")
done

"$CXX" -std=c++20 -O2 \
  "$BUILD_DIR/still_chroma_test.o" "$BUILD_DIR/frame_emulation_shim.o" \
  "${HOST_OBJECTS[@]}" -o "$BUILD_DIR/still_chroma_frame_emulation"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/still_chroma_frame_emulation" "${Y4M_ARGS[@]+"${Y4M_ARGS[@]}"}"
fi
