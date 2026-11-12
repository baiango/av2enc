#!/usr/bin/env bash
# Build (and by default run) the AV2 IBP CUDA parity test.
#
#   ./build_cuda_test.sh                 # sm_86, build + run
#   ./build_cuda_test.sh --no-run        # build only
#   ARCH=sm_120 ./build_cuda_test.sh     # secondary target
#
# Links the read-only intra reference/GPU component for the smooth/Paeth
# delegation path and the CPU oracle.  Does not touch the root CMake build.
set -euo pipefail
cd "$(dirname "$0")"

ARCH="${ARCH:-sm_86}"
BUILD_DIR="${BUILD_DIR:-build}"
RUN=1
for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    --arch=*) ARCH="${arg#--arch=}" ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
mkdir -p "$BUILD_DIR"

echo "nvcc: $(nvcc --version | tail -1)"
echo "arch: $ARCH"
echo "build: $BUILD_DIR/ibp_test_cuda"

nvcc -std=c++20 -O3 -lineinfo -arch="$ARCH" \
  -Xcompiler -Wall,-Wextra,-Wconversion,-Wshadow \
  --ptxas-options=-v \
  -I"$ROOT/src" -I"$ROOT/src/cuda/codec/intra/ibp" \
  -DAV2ENC_TESTDATA_DIR="\"$ROOT/tests/data\"" \
  ibp.cu ibp_test.cu "$ROOT/src/cuda/codec/intra/modes/intra_modes.cu" \
  "$ROOT/src/codec/intra/ibp/ibp.cpp" \
  "$ROOT/src/codec/intra/modes/intra_modes.cpp" \
  -o "$BUILD_DIR/ibp_test_cuda"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/ibp_test_cuda"
fi
