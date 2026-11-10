#!/usr/bin/env bash
# Build (and by default run) the M1 full-intra CUDA component test.
#
#   ./build_intra_modes_test.sh                 # sm_86, build + run
#   ./build_intra_modes_test.sh --no-run        # build only
#   ./build_intra_modes_test.sh --cpu-parity    # also link the CPU module
#   ARCH=sm_120 ./build_intra_modes_test.sh     # secondary build target
#
# Self-contained: does not touch the root CMake build or any other component.
set -euo pipefail

cd "$(dirname "$0")"

ARCH="${ARCH:-sm_86}"
BUILD_DIR="${BUILD_DIR:-build-intra-modes}"
RUN=1
CPU_PARITY=0

for arg in "$@"; do
  case "$arg" in
    --no-run)
      RUN=0
      ;;
    --arch=*)
      ARCH="${arg#--arch=}"
      ;;
    --cpu-parity)
      CPU_PARITY=1
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

EXTRA_SRC=""
EXTRA_FLAGS=""
if [ "$CPU_PARITY" -eq 1 ]; then
  # Optional cross-check against the CPU reference module.  Read-only: this
  # script never edits src/encoder/intra_modes/.
  SRC_ROOT="$(cd ../.. && pwd)"
  EXTRA_SRC="$SRC_ROOT/encoder/intra_modes/intra_modes.cpp"
  EXTRA_FLAGS="-DAV2ENC_CPU_INTRA_PARITY -I$SRC_ROOT"
fi

mkdir -p "$BUILD_DIR"

echo "nvcc: $(nvcc --version | tail -1)"
echo "arch: $ARCH"
echo "build: $BUILD_DIR/intra_modes_test"

nvcc -std=c++20 -O3 -lineinfo -arch="$ARCH" \
  -Xcompiler -Wall,-Wextra,-Wconversion,-Wshadow \
  --ptxas-options=-v $EXTRA_FLAGS \
  intra_modes.cu intra_modes_test.cu $EXTRA_SRC \
  -o "$BUILD_DIR/intra_modes_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/intra_modes_test"
fi
