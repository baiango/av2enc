#!/usr/bin/env bash
# Build and run the M1 GPU CfL stage test.
#
#   ./build_still_cfl_test.sh              # sm_86, build + generate frames + run
#   ./build_still_cfl_test.sh --no-run     # build only
#   ARCH=sm_120 ./build_still_cfl_test.sh  # secondary target
#
# The build directory defaults to <repo>/build-m1-cfl (private to this
# workstream). The test links the CPU CfL reference, the CPU intra DC predictor
# and the partition geometry helpers, so CPU-vs-GPU is a real comparison.
set -euo pipefail

SELF=$(cd "$(dirname "$0")" && pwd)
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
ARCH="${ARCH:-sm_86}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-m1-cfl}"
RUN=1

for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    --arch=*) ARCH="${arg#--arch=}" ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$BUILD_DIR/frames"

FRAME64="$BUILD_DIR/frames/src_64x64.y4m"
FRAME640="$BUILD_DIR/frames/src_640x360.y4m"
if [ ! -s "$FRAME64" ] || [ ! -s "$FRAME640" ]; then
  if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "build_still_cfl_test: ffmpeg is required to generate test frames" >&2
    exit 2
  fi
  ffmpeg -y -v error -f lavfi -i "testsrc2=size=64x64:rate=1:duration=1" \
    -frames:v 1 -pix_fmt yuv420p -f yuv4mpegpipe "$FRAME64"
  ffmpeg -y -v error -f lavfi -i "testsrc2=size=640x360:rate=1:duration=1" \
    -frames:v 1 -pix_fmt yuv420p -f yuv4mpegpipe "$FRAME640"
fi

echo "nvcc: $(nvcc --version | tail -1)"
echo "arch: $ARCH"
echo "build: $BUILD_DIR/still_cfl_test"

nvcc -std=c++20 -O3 -lineinfo -arch="$ARCH" \
  -I"$ROOT/src" \
  -Xcompiler -Wall,-Wextra,-Wshadow \
  "$SELF/still_cfl.cu" \
  "$SELF/../../cfl/cfl.cu" \
  "$SELF/still_cfl_test.cu" \
  "$ROOT/src/codec/intra/cfl/cfl.cpp" \
  "$ROOT/src/codec/intra/modes/intra_modes.cpp" \
  "$ROOT/src/codec/partition/tree/partition.cpp" \
  -o "$BUILD_DIR/still_cfl_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/still_cfl_test" "$FRAME64" "$FRAME640"
fi
