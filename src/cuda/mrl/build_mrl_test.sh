#!/usr/bin/env bash
# Build/run the AV2 MRL CPU-vs-GPU parity test (needs nvcc + a CUDA device).
#
#   ./build_mrl_test.sh
#   ./build_mrl_test.sh --no-run
#   ARCH=sm_120 ./build_mrl_test.sh
set -euo pipefail
cd "$(dirname "$0")"

ROOT="$(cd ../../.. && pwd)"
ARCH="${ARCH:-sm_86}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-mrl-gpu}"
RUN=1
for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    --arch=*) ARCH="${arg#--arch=}" ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$BUILD_DIR"
echo "nvcc: $(nvcc --version | tail -1)"
echo "arch: $ARCH"

nvcc -std=c++20 -O3 -lineinfo -arch="$ARCH" \
  -Xcompiler -Wall,-Wextra,-Wconversion,-Wshadow \
  -I"$ROOT/src" \
  mrl.cu mrl_test.cu \
  "$ROOT/src/encoder/mrl/mrl.cpp" \
  "$ROOT/src/encoder/intra_modes/intra_modes.cpp" \
  -o "$BUILD_DIR/mrl_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/mrl_test"
fi
