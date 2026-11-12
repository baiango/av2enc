#!/usr/bin/env bash
# Build/run the real-GPU chroma-format parity test.
#
#   ./build_gpu_test.sh                 # sm_86, build + run
#   ./build_gpu_test.sh --no-run        # build only
#   ARCH=sm_120 ./build_gpu_test.sh     # secondary build target
#   GPU_ARCH=... is also accepted as an alias for ARCH.
#
# Needs nvcc and an NVIDIA GPU.  On hosts without nvcc use ./build_and_test.sh,
# which runs the same shared device header against the CPU reference on the
# host (no GPU).
#
# All device buffers in the test and in the component come from one
# av2enc::cuda::Arena; neither file calls cudaMalloc/cudaFree/cudaMallocHost.
set -euo pipefail

cd "$(dirname "$0")"

HERE="$(pwd)"
ROOT="$(cd ../../.. && pwd)"
ARCH="${ARCH:-${GPU_ARCH:-sm_86}}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-chroma-formats-gpu}"
CXX="${CXX:-c++}"
RUN=1

for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    --arch=*) ARCH="${arg#--arch=}" ;;
    -h|--help)
      sed -n '2,18p' "$0"
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
echo "build: $BUILD_DIR/chroma_formats_test"

# CPU references (the oracles), compiled by the host compiler.  -ffp-contract
# off matches the device kernels' explicit __dadd_rn/__dmul_rn arithmetic.
CUDA_INC="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)/include"
CPU_FLAGS=(-std=c++20 -O2 -g -ffp-contract=off -I "$ROOT/src" -I "$CUDA_INC")
"$CXX" "${CPU_FLAGS[@]}" -c "$ROOT/src/encoder/format/format.cpp" \
  -o "$BUILD_DIR/format_cpu.o"
"$CXX" "${CPU_FLAGS[@]}" -c \
  "$ROOT/src/encoder/format/chroma/chroma_format.cpp" \
  -o "$BUILD_DIR/chroma_format_cpu.o"
"$CXX" "${CPU_FLAGS[@]}" -c \
  "$ROOT/src/encoder/format/chroma/chroma_residual.cpp" \
  -o "$BUILD_DIR/chroma_residual_cpu.o"
"$CXX" "${CPU_FLAGS[@]}" -c "$ROOT/src/encoder/tx_variants/tx_variants.cpp" \
  -o "$BUILD_DIR/tx_variants_cpu.o"
"$CXX" "${CPU_FLAGS[@]}" -c "$ROOT/src/cuda/m1/hbd/hbd_cpu.cpp" \
  -o "$BUILD_DIR/hbd_cpu.o"

"$CXX" "${CPU_FLAGS[@]}" -c "$ROOT/src/cuda/chroma_formats/chroma_formats.cpp" \
  -o "$BUILD_DIR/chroma_formats_host.o"

# Kernels + GPU test + the persistent arena runtime.
nvcc -std=c++20 -O3 -lineinfo -arch="$ARCH" --expt-relaxed-constexpr \
  --fmad=false \
  -Xcompiler -Wall,-Wextra,-Wconversion,-Wshadow \
  --ptxas-options=-v \
  -I "$ROOT/src" -I "$HERE" \
  "$HERE/chroma_formats.cu" \
  "$HERE/chroma_formats_test.cu" \
  "$ROOT/src/cuda/arena.cpp" \
  "$BUILD_DIR/format_cpu.o" "$BUILD_DIR/chroma_format_cpu.o" \
  "$BUILD_DIR/chroma_residual_cpu.o" "$BUILD_DIR/tx_variants_cpu.o" \
  "$BUILD_DIR/hbd_cpu.o" "$BUILD_DIR/chroma_formats_host.o" \
  -o "$BUILD_DIR/chroma_formats_test"

if [ "$RUN" -eq 1 ]; then
  CUDA_DISABLE_PTX_JIT=1 "$BUILD_DIR/chroma_formats_test"
fi
