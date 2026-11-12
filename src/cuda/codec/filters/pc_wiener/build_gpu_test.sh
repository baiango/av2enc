#!/usr/bin/env bash
# Build/run the AV2 PC-Wiener CPU-vs-GPU parity test.
#
#   ./build_gpu_test.sh                 # sm_86, build + run
#   ./build_gpu_test.sh --no-run
#   ARCH=sm_120 ./build_gpu_test.sh
#
# Needs nvcc and a CUDA device.  On hosts without nvcc use
# ./build_emulation_test.sh, which runs the same shared device header against
# the CPU reference.
set -euo pipefail
cd "$(dirname "$0")"

HERE="$(pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
ARCH="${ARCH:-sm_86}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-pc-wiener-gpu}"
RUN=1
for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    --arch=*) ARCH="${arg#--arch=}" ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found; use ./build_emulation_test.sh for host parity" >&2
  exit 2
fi

mkdir -p "$BUILD_DIR"
nvcc -std=c++20 -O3 -lineinfo -arch="$ARCH" --expt-relaxed-constexpr \
  -Xcompiler -Wall,-Wextra,-Wconversion,-Wshadow \
  -I"$ROOT/src" -I"$HERE" \
  "$HERE/pc_wiener.cu" "$HERE/pc_wiener_test.cu" \
  "$ROOT/src/cuda/arena.cpp" \
  "$ROOT/src/codec/filters/tables/pc_wiener.cpp" \
  "$ROOT/src/codec/filters/tables/luma_source.cpp" \
  -o "$BUILD_DIR/pc_wiener_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/pc_wiener_test"
fi
