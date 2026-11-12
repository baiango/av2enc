#!/usr/bin/env bash
# Build/run the AV2 TIP CPU-vs-CUDA parity test.
#
#   ./build_gpu_test.sh                  # sm_86, build + run
#   ./build_gpu_test.sh --no-run
#   ARCH=sm_120 ./build_gpu_test.sh
#
# Needs nvcc and a CUDA device.  The test allocates every device buffer from
# one `av2enc::cuda::Arena` sized from `required_device_bytes()`; there is no
# cudaMalloc/cudaFree/cudaMallocHost in the test.
#
# On hosts without nvcc use ./build_emulation_test.sh, which runs the same
# shared device header against the CPU reference.
set -euo pipefail
cd "$(dirname "$0")"

HERE="$(pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
ARCH="${ARCH:-sm_86}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-tip-gpu}"
RUN=1
for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    --arch=*) ARCH="${arg#--arch=}" ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found; use ./build_emulation_test.sh for host parity" >&2
  exit 2
fi

mkdir -p "$BUILD_DIR"
echo "nvcc: $(nvcc --version | tail -1)"
echo "arch: $ARCH"
echo "build: $BUILD_DIR/tip_test"

nvcc -std=c++20 -O3 -lineinfo -arch="$ARCH" --expt-relaxed-constexpr \
  -Xcompiler -Wall,-Wextra,-Wconversion,-Wshadow,-Wno-sign-conversion,-Wno-unused-function \
  -I"$ROOT/src" -I"$HERE" \
  "$HERE/tip.cu" "$HERE/tip_test.cu" "$ROOT/src/cuda/arena.cpp" \
  "$ROOT/src/codec/inter/temporal/motion_field.cpp" \
  "$ROOT/src/codec/inter/temporal/temporal_common.cpp" \
  "$ROOT/src/codec/inter/temporal/tip.cpp" \
  -o "$BUILD_DIR/tip_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/tip_test"
fi
