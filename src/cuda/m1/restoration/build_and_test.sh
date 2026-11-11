#!/usr/bin/env bash
# Build (and by default run) the M1 post-CDEF restoration CPU-vs-GPU test.
#
#   ./build_and_test.sh                       # sm_86, build + run synthetic cases
#   ./build_and_test.sh --no-run
#   ARCH=sm_120 ./build_and_test.sh
#   RUN_ARGS="--ivf /tmp/s64.ivf --ivf /tmp/s360.ivf" ./build_and_test.sh
#
# Real reconstruction inputs: `--ivf` for a single decode (curr = cdef), or
# `--ivf-pair PRE POST` for the real deblock -> CDEF -> CCSO -> Wiener chain.
# On the 3060 box:
#   build-cuda/av2enc tests/data/m0_64x64_1f.y4m  -o /tmp/pre64.ivf  --m1
#   build-cuda/av2enc tests/data/m0_64x64_1f.y4m  -o /tmp/post64.ivf --m1 --m1-cdef
#   RUN_ARGS="--ivf /tmp/pre64.ivf --ivf-pair /tmp/pre360.ivf /tmp/post360.ivf" \
#     ./build_and_test.sh
#
# Self-contained: does not touch the root CMake build or another component.
set -euo pipefail

cd "$(dirname "$0")"

ARCH="${ARCH:-sm_86}"
CXX="${CXX:-c++}"
BUILD_DIR="${BUILD_DIR:-build-cuda}"
RUN=1
RUN_ARGS="${RUN_ARGS:-}"

for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    --arch=*) ARCH="${arg#--arch=}" ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd ../../../.. && pwd)"
mkdir -p "$BUILD_DIR"

echo "nvcc: $(nvcc --version | tail -1)"
echo "host: $($CXX --version | head -1)"
echo "arch: $ARCH"

"$CXX" -std=c++20 -O2 -g \
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow \
  -I "$REPO_ROOT/src" \
  -c "$REPO_ROOT/src/encoder/filters/restoration/restoration.cpp" \
  -o "$BUILD_DIR/restoration_cpu.o"

nvcc -std=c++20 -O3 -lineinfo -arch="$ARCH" \
  -Xcompiler -Wall,-Wextra,-Wconversion,-Wshadow \
  -I "$REPO_ROOT/src" -I . \
  m1_restoration.cu m1_restoration_test.cu \
  ../../restoration/ccso.cu ../../restoration/wiener.cu \
  "$BUILD_DIR/restoration_cpu.o" \
  -o "$BUILD_DIR/m1_restoration_test"

if [ "$RUN" -eq 1 ]; then
  # shellcheck disable=SC2086
  "$BUILD_DIR/m1_restoration_test" $RUN_ARGS
fi
