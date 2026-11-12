#!/usr/bin/env bash
# Host-only GpuOptions plumbing test.  No CUDA toolkit or device required:
# still_gpu.h only declares PODs and inline mapping/validation helpers.
#
# Usage: src/cuda/pipeline/still/build_options_test.sh [--no-run]
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
OUT_DIR=${BUILD_DIR:-"$ROOT/build-m1-gpu-options"}
CXX=${CXX:-c++}
RUN=1
for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    -h|--help) sed -n '2,6p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$OUT_DIR"
"$CXX" -std=c++20 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow \
  -I"$ROOT/src" \
  "$HERE/still_gpu_options_test.cpp" \
  -o "$OUT_DIR/still_gpu_options_test"

if [ "$RUN" -eq 1 ]; then
  "$OUT_DIR/still_gpu_options_test"
fi
