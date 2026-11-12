#!/usr/bin/env bash
# Real CPU-vs-GPU bit-exactness test for the AV2 morphological prediction
# component. Needs nvcc and a CUDA device.
#
#   ./build_gpu_test.sh                 # sm_86, build + run
#   ./build_gpu_test.sh --no-run
#   ARCH=sm_120 ./build_gpu_test.sh
#
# Host-only, allocation-free component code is compiled with nvcc so it sees
# the same cuda_runtime.h. All device memory in the test comes from
# av2enc::cuda::Arena (one cudaMalloc per arena); there is no cudaMalloc in the
# component itself.

set -euo pipefail
cd "$(dirname "$0")"

HERE="$(pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
ARCH="${ARCH:-sm_86}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-morph-gpu}"
RUN=1
for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    --arch=*) ARCH="${arg#--arch=}" ;;
    -h|--help)
      sed -n '2,13p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found: use ./build_and_test.sh for host emulation parity" >&2
  exit 2
fi

mkdir -p "$BUILD_DIR"
echo "nvcc: $(nvcc --version | tail -1)"
echo "arch: $ARCH"
echo "build: $BUILD_DIR/morph_test"

nvcc -std=c++20 -O3 -lineinfo -arch="$ARCH" \
  -Xcompiler -Wall,-Wextra,-Wconversion,-Wshadow \
  --ptxas-options=-v \
  -I"$ROOT/src" -I"$HERE" \
  "$HERE/morph.cu" \
  "$HERE/morph_workspace.cpp" \
  "$HERE/morph_test.cu" \
  "$ROOT/src/cuda/arena.cpp" \
  -o "$BUILD_DIR/morph_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/morph_test"
fi
