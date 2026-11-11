#!/usr/bin/env bash
# Build and run the AV2 secondary-transform CUDA component.
#
#   ./build_and_test.sh                    # host emulation + real GPU if nvcc
#   CUDA_ARCH=sm_120 ./build_and_test.sh
#   ./build_and_test.sh --no-run
#
# The emulation test compiles the exact device core (tx_secondary_device.cuh)
# with the host compiler and compares it bit-for-bit with the CPU reference.
# When nvcc is available it also builds the real one-thread-per-block kernel
# test and compares GPU output against the same CPU reference.
set -euo pipefail
cd "$(dirname "$0")"

CXX="${CXX:-c++}"
REPO="$(cd ../../.. && pwd)"
ARCH="${CUDA_ARCH:-sm_86}"
BUILD_DIR="${BUILD_DIR:-${TMPDIR:-/tmp}/av2enc_tx_secondary_cuda_build}"
WERROR="${WERROR:-0}"
RUN=1

for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$BUILD_DIR"
FLAGS=(-std=c++20 -O2 -ffp-contract=off -Wall -Wextra -Wpedantic
       -Wconversion -Wshadow -I"$REPO/src")
if [ "$WERROR" = "1" ]; then FLAGS+=(-Werror); fi

echo "compiler: $($CXX --version | head -1)"
echo "build:    $BUILD_DIR/tx_secondary_emulation_test"

"$CXX" "${FLAGS[@]}" \
  tx_secondary_emulation_test.cpp \
  "$REPO/src/encoder/tx_secondary/tx_secondary.cpp" \
  "$REPO/src/encoder/tx_variants/tx_variants.cpp" \
  "$REPO/src/bitstream/bitwriter.cpp" \
  -o "$BUILD_DIR/tx_secondary_emulation_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/tx_secondary_emulation_test"
fi

if command -v nvcc >/dev/null 2>&1; then
  CPU_FLAGS=(-std=c++20 -O3 -ffp-contract=off -Wall -Wextra -Wconversion
             -Wshadow -I"$REPO/src")
  CPU_OBJS=()
  compile_cpu() {
    local src="$1"
    local obj="$BUILD_DIR/$(basename "${src%.*}").o"
    "$CXX" "${CPU_FLAGS[@]}" -c "$src" -o "$obj"
    CPU_OBJS+=("$obj")
  }
  compile_cpu "$REPO/src/encoder/tx_secondary/tx_secondary.cpp"
  compile_cpu "$REPO/src/encoder/tx_variants/tx_variants.cpp"
  compile_cpu "$REPO/src/bitstream/bitwriter.cpp"

  echo "nvcc:     $(nvcc --version | tail -1)"
  echo "arch:     $ARCH"
  echo "build:    $BUILD_DIR/tx_secondary_gpu_test"
  nvcc -std=c++20 -arch="$ARCH" -O3 -lineinfo -fmad=false -I"$REPO/src" \
    -Xcompiler=-Wall,-Wextra,-Wconversion,-Wshadow,-ffp-contract=off \
    tx_secondary_gpu.cu tx_secondary_test.cu \
    "${CPU_OBJS[@]}" \
    -o "$BUILD_DIR/tx_secondary_gpu_test"
  if [ "$RUN" -eq 1 ]; then
    "$BUILD_DIR/tx_secondary_gpu_test"
  fi
else
  echo "nvcc not found: skipped the real GPU test (emulation core passed)."
fi
