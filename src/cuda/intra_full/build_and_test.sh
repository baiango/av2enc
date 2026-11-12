#!/usr/bin/env bash
# Host-emulation test for the shared intra-full device bodies.
#
# No nvcc and no GPU are needed.  The build:
#   * stubs cuda_runtime.h so the real Arena (src/cuda/arena.cpp) runs against
#     host memory (one posix_memalign per Arena::init, exactly like cudaMalloc),
#   * compiles intra_full_device.cuh with CUDA keywords stubbed and one emulated
#     thread,
#   * links the CPU reference components read-only:
#       src/encoder/m1/missing_bits/          (DC subsampled, filter corner)
#       src/encoder/tx_variants/              (transform reference)
#       src/encoder/transform/idtx_intra/     (chroma type derivation planner)
#       src/encoder/tx_type/ + encoder/intra_modes/
#
#   ./build_and_test.sh              # build + run (emulation on this host)
#   ./build_and_test.sh --no-run     # build only
#   SANITIZE=1 ./build_and_test.sh   # ASan + UBSan
set -euo pipefail
cd "$(dirname "$0")"

ROOT="$(cd ../../.. && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-intra-full-emulation}"
CXX="${CXX:-c++}"
SANITIZE="${SANITIZE:-0}"
RUN=1
for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$BUILD_DIR/cuda_stub"
cat > "$BUILD_DIR/cuda_stub/cuda_runtime.h" <<'EOF'
// Host emulation stub: just enough cuda_runtime.h for arena.h / error.h / our
// headers.  cudaMalloc is backed by 4096-byte-aligned host memory so Arena's
// alignment math matches a real device allocation (base_offset == 0).
#pragma once
#include <cstddef>
#include <cstdlib>

typedef int cudaError_t;
typedef struct CUstream_st* cudaStream_t;

static const cudaError_t cudaSuccess = 0;
static const cudaError_t cudaErrorInvalidValue = 1;
static const cudaError_t cudaErrorMemoryAllocation = 2;
static const cudaError_t cudaErrorInvalidDevice = 3;
static const cudaError_t cudaErrorNoDevice = 4;
static const cudaError_t cudaErrorInsufficientDriver = 5;
static const cudaError_t cudaErrorInitializationError = 6;
static const cudaError_t cudaErrorInvalidConfiguration = 7;
static const cudaError_t cudaErrorInvalidResourceHandle = 8;
static const cudaError_t cudaErrorLaunchOutOfResources = 9;
static const cudaError_t cudaErrorNotReady = 10;

static inline cudaError_t cudaMalloc(void** ptr, std::size_t bytes) {
  if (ptr == nullptr) {
    return cudaErrorInvalidValue;
  }
  void* p = nullptr;
  if (posix_memalign(&p, 4096, bytes == 0 ? 1 : bytes) != 0) {
    *ptr = nullptr;
    return cudaErrorMemoryAllocation;
  }
  *ptr = p;
  return cudaSuccess;
}

static inline cudaError_t cudaFree(void* ptr) {
  std::free(ptr);
  return cudaSuccess;
}

static inline cudaError_t cudaMemGetInfo(std::size_t* free_bytes,
                                         std::size_t* total_bytes) {
  if (free_bytes == nullptr || total_bytes == nullptr) {
    return cudaErrorInvalidValue;
  }
  *total_bytes = 8ull * 1024ull * 1024ull * 1024ull;
  *free_bytes = 7ull * 1024ull * 1024ull * 1024ull;
  return cudaSuccess;
}

static inline cudaError_t cudaGetDevice(int* device) {
  if (device == nullptr) {
    return cudaErrorInvalidValue;
  }
  *device = 0;
  return cudaSuccess;
}

static inline cudaError_t cudaSetDevice(int) { return cudaSuccess; }

static inline cudaError_t cudaGetLastError(void) { return cudaSuccess; }

static inline const char* cudaGetErrorName(cudaError_t) { return "stub"; }
static inline const char* cudaGetErrorString(cudaError_t) { return "stub"; }
EOF

REF_FLAGS=(-std=c++20 -O2 -ffp-contract=off -I"$BUILD_DIR/cuda_stub"
           -I"$ROOT/src" -w)
OWN_FLAGS=(-std=c++20 -O2 -ffp-contract=off -Wall -Wextra -Wpedantic
           -Wshadow -Wconversion -I"$BUILD_DIR/cuda_stub" -I"$ROOT/src"
           -I"$ROOT/src/cuda/intra_full")
if [ "$SANITIZE" = "1" ]; then
  REF_FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
  OWN_FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
fi

REF_SOURCES=(
  "$ROOT/src/cuda/arena.cpp"
  "$ROOT/src/encoder/m1/missing_bits/dc_intra_subsampled.cpp"
  "$ROOT/src/encoder/m1/missing_bits/intra_filter_corner.cpp"
  "$ROOT/src/encoder/tx_variants/tx_variants.cpp"
  "$ROOT/src/encoder/transform/idtx_intra/idtx_intra.cpp"
  "$ROOT/src/encoder/tx_type/tx_type.cpp"
  "$ROOT/src/encoder/intra_modes/intra_modes.cpp"
)

echo "compiler: $($CXX --version | head -1)"
echo "build:    $BUILD_DIR"

OBJECTS=()
index=0
for source in "${REF_SOURCES[@]}"; do
  object="$BUILD_DIR/ref_${index}.o"
  "$CXX" "${REF_FLAGS[@]}" -c "$source" -o "$object"
  OBJECTS+=("$object")
  index=$((index + 1))
done

"$CXX" "${OWN_FLAGS[@]}" -c intra_full_emulation_test.cpp \
  -o "$BUILD_DIR/intra_full_emulation_test.o"
"$CXX" "${OWN_FLAGS[@]}" "$BUILD_DIR/intra_full_emulation_test.o" \
  "${OBJECTS[@]}" -o "$BUILD_DIR/intra_full_emulation_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/intra_full_emulation_test"
fi
