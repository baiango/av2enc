#!/usr/bin/env bash
# Build and run the me2 CPU-vs-CUDA bit-exact test.
#
#   ./build_gpu_and_test.sh                 # sm_86, build + run
#   ./build_gpu_and_test.sh --no-run        # build only
#   ARCH=sm_120 ./build_gpu_and_test.sh     # secondary build target
#
# Requires nvcc and an sm_86 (or newer) device.  When nvcc is not available
# (e.g. the macOS development host), this script runs the host-emulation
# parity test instead — it validates the same shared core and the exact
# warp-cooperative arithmetic without a GPU.
set -euo pipefail

cd "$(dirname "$0")"

ARCH="${ARCH:-sm_86}"
CXX="${CXX:-c++}"
BUILD_DIR="${BUILD_DIR:-build-cuda}"
RUN=1
for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    --arch=*) ARCH="${arg#--arch=}" ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found; running the host-emulation parity test instead."
  if [ "$RUN" -eq 1 ]; then
    ./build_and_test.sh --quick
  fi
  exit 0
fi

REPO_ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${REPO_ROOT}/CMakeLists.txt" ] && [ "${REPO_ROOT}" != "/" ]; do
    REPO_ROOT="$(dirname -- "${REPO_ROOT}")"
done
mkdir -p "$BUILD_DIR"

echo "nvcc: $(nvcc --version | tail -1)"
echo "host: $($CXX --version | head -1)"
echo "arch: $ARCH"
echo "build: $BUILD_DIR/me2_gpu_test"

"$CXX" -std=c++20 -O2 -g \
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow \
  -I "$REPO_ROOT/src" \
  -c "$REPO_ROOT/src/codec/inter/me_precision/me2.cpp" \
  -o "$BUILD_DIR/me2_cpu.o"
"$CXX" -std=c++20 -O2 -g \
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow \
  -I "$REPO_ROOT/src" \
  -c "$REPO_ROOT/src/codec/inter/me/me.cpp" \
  -o "$BUILD_DIR/me_cpu.o"

nvcc -std=c++20 -O3 -lineinfo -arch="$ARCH" \
  -Xcompiler -Wall,-Wextra,-Wconversion,-Wshadow \
  --ptxas-options=-v \
  -I "$REPO_ROOT/src" -I . \
  me2_gpu.cu me2_gpu_test.cu \
  "$BUILD_DIR/me2_cpu.o" "$BUILD_DIR/me_cpu.o" \
  -o "$BUILD_DIR/me2_gpu_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/me2_gpu_test" "$REPO_ROOT"
fi
