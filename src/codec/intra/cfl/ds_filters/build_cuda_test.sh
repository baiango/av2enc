#!/usr/bin/env bash
# CUDA seam test for cfl_ds_filter_index.
#
# Compiles the *existing* kernel src/cuda/codec/intra/cfl/cfl.cu unchanged, the existing
# CPU reference src/codec/intra/cfl/cfl.cpp unchanged, and this component's
# ds_filters.cpp + ds_filters_cuda_test.cu.  Requires nvcc and a CUDA device.
#
# Usage:
#   ./src/codec/intra/cfl/ds_filters/build_cuda_test.sh
#   ARCH=sm_120 ./.../build_cuda_test.sh
#   ./.../build_cuda_test.sh --no-run
#
# The default CPU build (build_and_test.sh) skips this file when nvcc is not
# present; the macOS reference host has no nvcc.

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
ARCH="${ARCH:-sm_86}"
BUILD="$HERE/build-cuda"
RUN=1

for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    --arch=*) ARCH="${arg#--arch=}" ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if ! command -v nvcc >/dev/null 2>&1; then
  echo "build_cuda_test: nvcc not found; skipping (CPU tests cover this component)" >&2
  exit 0
fi

mkdir -p "$BUILD"
nvcc -std=c++20 -O3 -lineinfo -arch="$ARCH" \
  -Xcompiler -Wall,-Wextra,-Wconversion,-Wshadow \
  -I"$ROOT/src" \
  "$ROOT/src/cuda/codec/intra/cfl/cfl.cu" \
  "$HERE/ds_filters_cuda_test.cu" \
  "$HERE/ds_filters.cpp" \
  "$ROOT/src/codec/intra/cfl/cfl.cpp" \
  -o "$BUILD/ds_filters_cuda_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD/ds_filters_cuda_test"
fi
