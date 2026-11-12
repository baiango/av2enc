#!/usr/bin/env bash
# Build/run the AV2 §7.20.5 GDF CPU-vs-GPU parity test.
#
#   ./build_gpu_test.sh                 # sm_86, build + run
#   ./build_gpu_test.sh --no-run        # build only
#   ARCH=sm_120 ./build_gpu_test.sh     # secondary build target
#
# Needs nvcc and a CUDA device.  On hosts without nvcc use ./build_and_test.sh,
# which runs the same shared device header against the CPU reference on the
# host.
#
# All device buffers in the test come from one av2enc::cuda::Arena; the test
# itself calls no cudaMalloc/cudaFree/cudaMallocHost.
set -euo pipefail

cd "$(dirname "$0")"

HERE="$(pwd)"
ROOT="$(cd ../../.. && pwd)"
ARCH="${ARCH:-sm_86}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-gdf-gpu}"
CXX="${CXX:-c++}"
RUN=1

for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    --arch=*) ARCH="${arg#--arch=}" ;;
    -h|--help)
      sed -n '2,14p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found; use ./build_and_test.sh for host-emulation parity" >&2
  exit 2
fi

mkdir -p "$BUILD_DIR"

echo "nvcc: $(nvcc --version | tail -1)"
echo "host: $($CXX --version | head -1)"
echo "arch: $ARCH"
echo "build: $BUILD_DIR/gdf_test"

# CPU reference objects (the oracle), compiled by the host compiler.
"$CXX" -std=c++20 -O2 -g \
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow \
  -I "$ROOT/src" \
  -c "$ROOT/src/encoder/filters/spec_tables/gdf.cpp" \
  -o "$BUILD_DIR/gdf_cpu.o"
"$CXX" -std=c++20 -O2 -g \
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow \
  -I "$ROOT/src" \
  -c "$ROOT/src/encoder/filters/spec_tables/luma_source.cpp" \
  -o "$BUILD_DIR/luma_source_cpu.o"

# Kernel + test + the persistent arena runtime.
nvcc -std=c++20 -O3 -lineinfo -arch="$ARCH" --expt-relaxed-constexpr \
  -Xcompiler -Wall,-Wextra,-Wconversion,-Wshadow \
  --ptxas-options=-v \
  -I "$ROOT/src" -I "$HERE" \
  "$HERE/gdf.cu" "$HERE/gdf_test.cu" "$ROOT/src/cuda/arena.cpp" \
  "$BUILD_DIR/gdf_cpu.o" "$BUILD_DIR/luma_source_cpu.o" \
  -o "$BUILD_DIR/gdf_test"

if [ "$RUN" -eq 1 ]; then
  CUDA_DISABLE_PTX_JIT=1 "$BUILD_DIR/gdf_test"
fi
