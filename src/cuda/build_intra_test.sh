#!/usr/bin/env bash
# Build (and by default run) the M0 intra component test.
#
#   ./build_intra_test.sh                 # sm_86, build + run
#   ./build_intra_test.sh --no-run        # build only
#   ARCH=sm_120 ./build_intra_test.sh     # secondary build target
#
# Self-contained: does not touch the root CMake build or any other component.
set -euo pipefail

cd "$(dirname "$0")"

ARCH="${ARCH:-sm_86}"
BUILD_DIR="${BUILD_DIR:-build-intra}"
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
      sed -n '2,9p' "$0"
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
echo "arch: $ARCH"
echo "build: $BUILD_DIR/intra_test"

nvcc -std=c++20 -O3 -lineinfo -arch="$ARCH" \
  -Xcompiler -Wall,-Wextra,-Wconversion,-Wshadow \
  --ptxas-options=-v \
  intra.cu intra_test.cu \
  -o "$BUILD_DIR/intra_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/intra_test"
fi
