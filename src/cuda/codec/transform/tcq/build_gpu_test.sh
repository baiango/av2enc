#!/usr/bin/env bash
# Build and run the CPU-vs-GPU bit-exactness test for src/cuda/codec/transform/tcq.
#
#   ./build_gpu_test.sh                  # sm_86, build + run
#   ./build_gpu_test.sh --no-run
#   ./build_gpu_test.sh --arch=sm_120
#   ARCH=sm_120 ./build_gpu_test.sh
#
# Needs nvcc and a visible CUDA device.  The CPU references are compiled by the
# host compiler and linked in, so this is a true CPU-vs-GPU check.
#
# Device-visibility audit (run before compiling):
#   * every free function in tcq_device.cuh begins with TCQ_HD, i.e. is
#     __host__ __device__ under nvcc and plain inline for the host pass;
#   * the device header has no namespace-scope or constexpr array (nothing for
#     a kernel to odr-use from the host pass), and includes no encoder header.
set -euo pipefail
cd "$(dirname "$0")"

HERE="$(pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
ARCH="${ARCH:-sm_86}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-tcq-gpu}"
RUN=1
for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    --arch=*) ARCH="${arg#--arch=}" ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found; use ./build_and_test.sh for host-emulation parity" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# Device-visibility audit
# ---------------------------------------------------------------------------
DEVICE_HEADER="$HERE/tcq_device.cuh"
DEFINITIONS=$(grep -cE '^[A-Za-z_].*tcq_[a-z0-9_]+[[:space:]]*\(' "$DEVICE_HEADER" || true)
ANNOTATED=$(grep -cE '^TCQ_HD .*tcq_[a-z0-9_]+[[:space:]]*\(' "$DEVICE_HEADER" || true)
if [ "$DEFINITIONS" != "$ANNOTATED" ]; then
  echo "device audit: $DEFINITIONS tcq_* definitions but only $ANNOTATED TCQ_HD:" >&2
  grep -nE '^[A-Za-z_].*tcq_[a-z0-9_]+[[:space:]]*\(' "$DEVICE_HEADER" |
    grep -vE '^[0-9]+:TCQ_HD ' >&2 || true
  exit 3
fi
if grep -nE '^[[:space:]]*(constexpr|static|extern)[^;()]*\[' "$DEVICE_HEADER"; then
  echo "device audit: namespace-scope array in the device header" >&2
  exit 3
fi
if grep -nE '^[[:space:]]*#include[[:space:]]+.*(cuda_runtime|encoder/|codec/|core/|pipeline/|formats/|frame/)' "$DEVICE_HEADER"; then
  echo "device audit: device header pulls a host/CUDA-runtime header" >&2
  exit 3
fi
echo "device audit: $ANNOTATED TCQ_HD helpers, no storage arrays, no host includes"

CXX="${CXX:-c++}"
mkdir -p "$BUILD_DIR"

# Host-compiled CPU references.
CPU_FLAGS=(-std=c++20 -O3 -ffp-contract=off -Wall -Wextra -Wconversion
           -Wshadow -I"$ROOT/src" -I"$HERE")
CPU_OBJS=()
compile_cpu() {
  local src="$1"
  local obj="$BUILD_DIR/$(basename "${src%.*}")_cpu.o"
  "$CXX" "${CPU_FLAGS[@]}" -c "$src" -o "$obj"
  CPU_OBJS+=("$obj")
}
compile_cpu "$HERE/tcq_host.cpp"
compile_cpu "$ROOT/src/codec/transform/tcq/tcq.cpp"
compile_cpu "$ROOT/src/codec/transform/tcq/cctx.cpp"
compile_cpu "$ROOT/src/codec/transform/tcq/parity/parity_hiding.cpp"

echo "nvcc:     $(nvcc --version | tail -1)"
echo "arch:     $ARCH"
echo "build:    $BUILD_DIR/tcq_test"
nvcc -std=c++20 -arch="$ARCH" -O3 -lineinfo -fmad=false \
  --expt-relaxed-constexpr \
  -I"$ROOT/src" -I"$HERE" \
  -Xcompiler=-Wall,-Wextra,-Wconversion,-Wshadow,-ffp-contract=off \
  "$HERE/tcq.cu" "$HERE/tcq_test.cu" "$ROOT/src/cuda/arena.cpp" \
  "${CPU_OBJS[@]}" \
  -o "$BUILD_DIR/tcq_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/tcq_test"
fi
