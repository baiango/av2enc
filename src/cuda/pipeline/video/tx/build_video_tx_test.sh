#!/usr/bin/env bash
# Build/run the real-GPU parity test for the M2 video TX-selection module.
#
#   ./build_video_tx_test.sh              # sm_86, build + run
#   ./build_video_tx_test.sh --no-run
#   ./build_video_tx_test.sh --quick      # short qindex sweep while iterating
#   ARCH=sm_120 ./build_video_tx_test.sh
#
# The CPU reference and the device kernel are compiled from the same sources
# with `-ffp-contract=off` (host) / `-fmad=false` (nvcc) so the binary64 forward
# transform can be compared bit for bit.  Every device buffer belongs to one
# persistent av2enc::cuda::Arena; the component sources are checked for direct
# allocation calls below.
set -euo pipefail
cd "$(dirname "$0")"

ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
ARCH="${ARCH:-sm_86}"
BUILD_DIR="${BUILD_DIR:-${TMPDIR:-/tmp}/av2enc_video_tx}"
CXX="${CXX:-g++}"
RUN=1
TEST_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    --quick) TEST_ARGS+=(--quick) ;;
    --arch=*) ARCH="${arg#--arch=}" ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found; this suite needs a CUDA device" >&2
  exit 2
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  if ! nvidia-smi -L >/dev/null 2>&1; then
    echo "no CUDA device visible" >&2
    exit 2
  fi
fi

if grep -nE '\b(cudaMalloc|cudaFree|cudaMallocHost|cudaHostAlloc|cudaHostRegister)[[:space:]]*\(' \
    video_tx.h video_tx.cu; then
  echo "ERROR: direct device allocation in the component sources" >&2
  exit 1
fi

mkdir -p "$BUILD_DIR"

CPU_FLAGS=(-std=c++20 -O3 -ffp-contract=off -Wall -Wextra -Wconversion
           -Wshadow -I"$ROOT/src")
compile_cpu() {
  local src="$1"
  local obj="$BUILD_DIR/$(basename "${src%.*}").o"
  "$CXX" "${CPU_FLAGS[@]}" -c "$src" -o "$obj"
  echo "$obj"
}

TX_PARTITION="$(compile_cpu "$ROOT/src/codec/transform/tx_partition/tx_partition.cpp")"
TX_SCAN="$(compile_cpu "$ROOT/src/codec/transform/tx_partition/tx_scan.cpp")"
CPU_TXV="$(compile_cpu "$ROOT/src/codec/transform/variants/tx_variants.cpp")"
CPU_TXTYPE="$(compile_cpu "$ROOT/src/codec/transform/tx_type/tx_type.cpp")"
CPU_INTRA_MODES="$(compile_cpu "$ROOT/src/codec/intra/modes/intra_modes.cpp")"
CPU_INTER_TX_TYPE="$(compile_cpu "$ROOT/src/codec/transform/inter_residual/inter_tx_type.cpp")"


NVCC_FLAGS=(-std=c++20 -arch="$ARCH" -O3 -lineinfo -fmad=false
            --expt-relaxed-constexpr -I"$ROOT/src"
            -Xcompiler=-Wall,-Wextra,-Wconversion,-Wshadow,-ffp-contract=off
            -Xptxas=-v)

echo "nvcc:  $(nvcc --version | tail -1)"
echo "arch:  $ARCH"
echo "build: $BUILD_DIR/video_tx_test"

nvcc "${NVCC_FLAGS[@]}" -c video_tx.cu -o "$BUILD_DIR/video_tx.o"
nvcc "${NVCC_FLAGS[@]}" -c video_tx_test.cu -o "$BUILD_DIR/video_tx_test.o"
nvcc "${NVCC_FLAGS[@]}" -x cu -c "$ROOT/src/cuda/arena.cpp" \
  -o "$BUILD_DIR/arena.o"

nvcc "${NVCC_FLAGS[@]}" \
  "$BUILD_DIR/video_tx.o" \
  "$BUILD_DIR/video_tx_test.o" \
  "$BUILD_DIR/arena.o" \
  "$TX_PARTITION" "$TX_SCAN" "$CPU_TXV" "$CPU_TXTYPE" \
  "$CPU_INTRA_MODES" "$CPU_INTER_TX_TYPE" \
  -o "$BUILD_DIR/video_tx_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/video_tx_test" "${TEST_ARGS[@]}"
fi
