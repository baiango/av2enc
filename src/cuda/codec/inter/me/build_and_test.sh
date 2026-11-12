#!/usr/bin/env bash
# Build (and by default run) the AV2 inter motion-estimation CPU-vs-CUDA test.
#
#   ./build_and_test.sh                 # sm_86, build + run
#   ./build_and_test.sh --no-run        # build only
#   ARCH=sm_120 ./build_and_test.sh     # secondary build target
#
# Self-contained: does not touch the root CMake build or other components.
# The CPU reference object is compiled by the host compiler; the CUDA kernel
# and test are compiled by nvcc, then linked together.
set -euo pipefail

cd "$(dirname "$0")"

ARCH="${ARCH:-sm_86}"
CXX="${CXX:-c++}"
BUILD_DIR="${BUILD_DIR:-build-cuda}"
RUN=1

for arg in "$@"; do
  case "$arg" in
    --no-run)
      RUN=0
      ;;
    --arch=*)
      ARCH="${arg#--arch=}"
      ;;
    -h|--help)
      sed -n '2,11p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

REPO_ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${REPO_ROOT}/CMakeLists.txt" ] && [ "${REPO_ROOT}" != "/" ]; do
    REPO_ROOT="$(dirname -- "${REPO_ROOT}")"
done
mkdir -p "$BUILD_DIR"

echo "nvcc: $(nvcc --version | tail -1)"
echo "host: $($CXX --version | head -1)"
echo "arch: $ARCH"
echo "build: $BUILD_DIR/me_test"

"$CXX" -std=c++20 -O2 -g \
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow \
  -I "$REPO_ROOT/src" \
  -c "$REPO_ROOT/src/codec/inter/me/me.cpp" \
  -o "$BUILD_DIR/me_cpu.o"

nvcc -std=c++20 -O3 -lineinfo -arch="$ARCH" \
  -Xcompiler -Wall,-Wextra,-Wconversion,-Wshadow \
  --ptxas-options=-v \
  -I "$REPO_ROOT/src" -I . \
  me.cu me_test.cu "$BUILD_DIR/me_cpu.o" \
  -o "$BUILD_DIR/me_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/me_test" "$REPO_ROOT"
fi
