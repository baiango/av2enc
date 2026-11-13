#!/usr/bin/env bash
# Build/run the AV2 chroma loop-restoration CPU-vs-CUDA parity test.
#
#   ./build_gpu_test.sh                  # sm_86, build + run
#   ./build_gpu_test.sh --no-run
#   ARCH=sm_120 ./build_gpu_test.sh
#
# Needs nvcc and a CUDA device.  On hosts without nvcc use
# ./build_emulation_test.sh, which runs the same shared device header against
# the CPU reference.
#
# The test allocates every device buffer from one `av2enc::cuda::Arena`; there
# is no cudaMalloc/cudaFree/cudaMallocHost in this module or its test.
set -euo pipefail
cd "$(dirname "$0")"

HERE="$(pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
ARCH="${ARCH:-sm_86}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-lr-chroma-gpu}"
RUN=1
for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    --arch=*) ARCH="${arg#--arch=}" ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found; use ./build_emulation_test.sh for host parity" >&2
  exit 2
fi

mkdir -p "$BUILD_DIR"
CXX="${CXX:-c++}"

CORE=(
  "$ROOT/src/codec/filters/loop_restoration/chroma/chroma_lr.cpp"
  "$ROOT/src/codec/filters/loop_restoration/restoration.cpp"
  "$ROOT/src/codec/filters/loop_restoration/classes/wiener_fit.cpp"
  "$ROOT/src/codec/filters/loop_restoration/classes/wiener_syntax.cpp"
  "$ROOT/src/codec/filters/loop_restoration/classes/wiener_class.cpp"
  "$ROOT/src/codec/filters/stream/filters_stream.cpp"
  "$ROOT/src/codec/entropy/av2/av2_entropy.cpp"
  "$ROOT/src/core/bitstream/bitwriter.cpp"
)
OBJECTS=()
for src in "${CORE[@]}"; do
  obj="$BUILD_DIR/$(basename "${src%.cpp}").o"
  "$CXX" -std=c++20 -O2 -g -I"$ROOT/src" -c "$src" -o "$obj"
  OBJECTS+=("$obj")
done

echo "nvcc: $(nvcc --version | tail -1)"
echo "arch: $ARCH"
echo "build: $BUILD_DIR/lr_chroma_test"

nvcc -std=c++20 -O3 -lineinfo -arch="$ARCH" --expt-relaxed-constexpr \
  -Xcompiler -Wall,-Wextra,-Wconversion,-Wshadow,-Wno-sign-conversion,-Wno-unused-function \
  -I"$ROOT/src" -I"$HERE" \
  "$HERE/lr_chroma.cu" "$HERE/lr_chroma_test.cu" \
  "$ROOT/src/cuda/arena.cpp" "${OBJECTS[@]}" \
  -o "$BUILD_DIR/lr_chroma_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/lr_chroma_test"
fi
