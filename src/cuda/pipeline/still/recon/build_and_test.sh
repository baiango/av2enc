#!/usr/bin/env bash
# Build (and by default run) the M1 GPU reconstruction/deblock parity test.
#
#   ./build_and_test.sh                 # sm_86, build + run
#   ./build_and_test.sh --no-run        # build only
#   ARCH=sm_120 ./build_and_test.sh     # secondary build target
#   BUILD_DIR=/tmp/x ./build_and_test.sh
#
# Self-contained: does not touch the root CMake build or any other component.
# The CPU reference sources (including the M1 encoder) are compiled by the host
# compiler; the CUDA kernels and the test are compiled by nvcc, then linked.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${REPO_ROOT}/CMakeLists.txt" ] && [ "${REPO_ROOT}" != "/" ]; do
    REPO_ROOT="$(dirname -- "${REPO_ROOT}")"
done
ARCH="${ARCH:-sm_86}"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build-m1-recon}"
CXX="${CXX:-c++}"
RUN=1

for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    --arch=*) ARCH="${arg#--arch=}" ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$BUILD_DIR"
echo "nvcc: $(nvcc --version | tail -1)"
echo "host: $($CXX --version | head -1)"
echo "arch: $ARCH"
echo "build: $BUILD_DIR/recon_test"

CPU_SOURCES=(
  src/core/io/y4m.cpp
  src/core/bitstream/bitwriter.cpp
  src/core/bitstream/obu.cpp
  src/core/bitstream/basic_headers.cpp
  src/codec/partition/tree/partition.cpp
  src/codec/intra/modes/intra_modes.cpp
  src/codec/transform/variants/tx_variants.cpp
  src/codec/transform/variants/tx_select.cpp
  src/codec/intra/cfl/cfl.cpp
  src/codec/filters/deblock/deblock.cpp
  src/pipeline/still/still_encoder.cpp
  src/pipeline/still/still_syntax.cpp
  src/pipeline/still/still_headers.cpp
  src/codec/entropy/av2/av2_entropy.cpp
)

CPU_OBJECTS=()
for src in "${CPU_SOURCES[@]}"; do
  obj="$BUILD_DIR/$(basename "${src%.cpp}").o"
  "$CXX" -std=c++20 -O2 -g -ffp-contract=off \
    -I "$REPO_ROOT" -I "$REPO_ROOT/src" \
    -c "$REPO_ROOT/$src" -o "$obj"
  CPU_OBJECTS+=("$obj")
done

nvcc -std=c++20 -O3 -lineinfo -arch="$ARCH" -fmad=false \
  -Xcompiler -ffp-contract=off,-Wall,-Wextra,-Wconversion,-Wshadow \
  -I "$REPO_ROOT" -I "$REPO_ROOT/src" \
  -c "$HERE/recon.cu" -o "$BUILD_DIR/recon.o"

nvcc -std=c++20 -O3 -lineinfo -arch="$ARCH" \
  -I "$REPO_ROOT" -I "$REPO_ROOT/src" \
  -c "$REPO_ROOT/src/cuda/codec/filters/deblock/deblock.cu" -o "$BUILD_DIR/deblock_cuda.o"

nvcc -std=c++20 -O2 -lineinfo -arch="$ARCH" \
  -Xcompiler -ffp-contract=off,-Wall,-Wextra,-Wconversion,-Wshadow \
  -I "$REPO_ROOT" -I "$REPO_ROOT/src" \
  -c "$HERE/recon_test.cu" -o "$BUILD_DIR/recon_test.o"

nvcc -std=c++20 -arch="$ARCH" \
  "$BUILD_DIR/recon.o" "$BUILD_DIR/deblock_cuda.o" \
  "$BUILD_DIR/recon_test.o" "${CPU_OBJECTS[@]}" \
  -o "$BUILD_DIR/recon_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/recon_test" "$REPO_ROOT"
fi
