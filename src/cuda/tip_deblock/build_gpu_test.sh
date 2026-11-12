#!/usr/bin/env bash
# Build/run the AV2 §7.16 TIP deblocking CPU-vs-GPU parity test.
#
#   ./build_gpu_test.sh                 # sm_86, build + run
#   ./build_gpu_test.sh --no-run
#   ARCH=sm_120 ./build_gpu_test.sh
#
# Needs nvcc and a CUDA device.  On hosts without nvcc use build_and_test.sh or
# build_emulation_test.sh, which run the shared device header on the host.
#
# The test links the CPU reference (tip_deblock.cpp) and the runtime's
# persistent Arena implementation (arena.cpp); every device buffer comes from
# that arena, so the test also exercises required_device_bytes(), mark/rewind
# and the one-byte-short rejection.
set -euo pipefail
cd "$(dirname "$0")"

HERE="$(pwd)"
ROOT="$(cd ../../.. && pwd)"
ARCH="${ARCH:-sm_86}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-tip-deblock-gpu}"
RUN=1
for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    --arch=*) ARCH="${arg#--arch=}" ;;
    -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
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
echo "build: $BUILD_DIR/tip_deblock_test"

nvcc -std=c++20 -O3 -lineinfo -arch="$ARCH" \
  -Xcompiler -Wall,-Wextra,-Wconversion,-Wshadow \
  --ptxas-options=-v \
  -I"$ROOT/src" -I"$HERE" \
  "$HERE/tip_deblock.cu" \
  "$HERE/tip_deblock_test.cu" \
  "$ROOT/src/encoder/filters/tip_deblock/tip_deblock.cpp" \
  "$ROOT/src/cuda/arena.cpp" \
  -o "$BUILD_DIR/tip_deblock_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/tip_deblock_test" "$ROOT"
fi
