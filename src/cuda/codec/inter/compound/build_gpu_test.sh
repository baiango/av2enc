#!/usr/bin/env bash
# Build/run the CPU-vs-GPU bit-exactness test for src/cuda/codec/inter/compound.
#
#   ./build_gpu_test.sh                 # sm_86, build + run
#   ./build_gpu_test.sh --no-run
#   ARCH=sm_120 ./build_gpu_test.sh
#
# Needs nvcc and a visible CUDA device. The CPU references are compiled by the
# host compiler and linked in, so this is a true CPU-vs-GPU check.
set -euo pipefail
cd "$(dirname "$0")"

HERE="$(pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
ARCH="${ARCH:-sm_86}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-compound-full-gpu}"
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

CXX="${CXX:-c++}"
mkdir -p "$BUILD_DIR"

# Host-compiled CPU references (avoids nvcc warnings from the STL headers).
for src in \
  "$HERE/compound_full_cpu_oracle.cpp" \
  "$ROOT/src/codec/inter/advanced/compound.cpp" \
  "$ROOT/src/codec/inter/temporal/opfl.cpp" \
  "$ROOT/src/codec/inter/temporal/refinemv.cpp" \
  "$ROOT/src/codec/inter/mc/mc.cpp" \
  "$ROOT/src/codec/inter/multiref/compound_predict.cpp"; do
  obj="$BUILD_DIR/$(basename "${src%.cpp}")_cpu.o"
  "$CXX" -std=c++20 -O2 -g -ffp-contract=off -I"$HERE" -I"$ROOT/src" \
    -c "$src" -o "$obj"
done

# arena.cpp includes <cuda_runtime.h>, so let nvcc compile it (it adds the CUDA
# include paths) instead of the plain host compiler.
nvcc -std=c++20 -O3 -lineinfo -arch="$ARCH" --expt-relaxed-constexpr \
  -Xcompiler -Wall,-Wextra,-Wconversion,-Wshadow \
  --ptxas-options=-v \
  -I"$ROOT/src" -I"$HERE" \
  "$HERE/compound_full.cu" "$HERE/compound_full_test.cu" \
  "$ROOT/src/cuda/arena.cpp" \
  "$BUILD_DIR/compound_full_cpu_oracle_cpu.o" \
  "$BUILD_DIR/compound_cpu.o" \
  "$BUILD_DIR/opfl_cpu.o" \
  "$BUILD_DIR/refinemv_cpu.o" \
  "$BUILD_DIR/mc_cpu.o" \
  "$BUILD_DIR/compound_predict_cpu.o" \
  -o "$BUILD_DIR/compound_full_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/compound_full_test"
fi
