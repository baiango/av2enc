#!/usr/bin/env bash
# Build and run the DDT host-emulation parity test.
#
# This is the no-GPU / no-nvcc half of the component: it compiles the shared
# device header (src/cuda/codec/transform/ddt/ddt_device.cuh) with the CUDA keywords stubbed and
# compares its integer arithmetic against the CPU reference
# (src/codec/transform/ddt/).  The real CUDA test is built by build_gpu_test.sh on
# the sm_86 box.
#
#   ./build_and_test.sh                 # build + run
#   ./build_and_test.sh --no-run        # build only
#   WERROR=1 ./build_and_test.sh
#   CXX=clang++ ./build_and_test.sh
#
# It also enforces the single-arena rule for this directory: no cudaMalloc or
# cudaFree may appear in the module, including comments-as-code.
set -euo pipefail
cd "$(dirname "$0")"

REPO="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${REPO}/CMakeLists.txt" ] && [ "${REPO}" != "/" ]; do
    REPO="$(dirname -- "${REPO}")"
done
CXX="${CXX:-c++}"
BUILD_DIR="${BUILD_DIR:-${TMPDIR:-/tmp}/av2enc_ddt_build}"
RUN=1
for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$BUILD_DIR"

# Single-arena guard (docs/cuda-interface.md).  Tests are exempt, but this
# module's test never allocates device memory directly either.
if grep -nE '(cudaMalloc|cudaFree|cudaMallocAsync|cudaFreeAsync)[[:space:]]*\(' \
    ddt_gpu.h ddt_gpu.cu ddt_device.cuh; then
  echo "error: direct CUDA allocation in src/cuda/codec/transform/ddt (use Arena)" >&2
  exit 1
fi

CXXFLAGS=(-std=c++20 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
          -Wno-missing-braces -I"$REPO/src")
if [[ "${WERROR:-0}" == "1" ]]; then
  CXXFLAGS+=(-Werror)
fi

echo "== DDT device emulation parity =="
"$CXX" "${CXXFLAGS[@]}" \
  ddt_device_emulation_test.cpp \
  "$REPO/src/codec/transform/ddt/ddt_transform.cpp" \
  "$REPO/src/codec/transform/variants/tx_variants.cpp" \
  -o "$BUILD_DIR/ddt_device_emulation_test"

if [[ "$RUN" == "1" ]]; then
  "$BUILD_DIR/ddt_device_emulation_test"
  echo "RESULT: PASS"
fi
