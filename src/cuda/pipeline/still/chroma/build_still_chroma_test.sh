#!/usr/bin/env bash
# Build (and by default run) the M1 GPU chroma CPU-vs-GPU parity test.
#
#   ./build_still_chroma_test.sh                 # sm_86, build + run
#   ./build_still_chroma_test.sh --no-run        # build only
#   ARCH=sm_120 ./build_still_chroma_test.sh     # secondary target
#   BUILD_DIR=/path ./build_still_chroma_test.sh # private build dir override
#
# Self-contained: does not touch the root CMake build. The CPU reference is
# the real src/pipeline/still/still_encoder.cpp (included by the test TU) plus the real
# src/encoder component sources, so this is a CPU-vs-GPU parity check, not two
# transcriptions of the spec.
set -euo pipefail

cd "$(dirname "$0")"

ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
ARCH="${ARCH:-sm_86}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-m1-chroma}"
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
      sed -n '2,10p' "$0"
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
echo "build: $BUILD_DIR"

# Shared CPU-reference source list (still_syntax + the integrated tool set).
source "$ROOT/src/cuda/codec/filters/still_reference_sources.sh"
# still_chroma_test.cu includes still_encoder.cpp textually, so it must not be
# linked a second time.
HOST_SOURCES=("${STILL_REFERENCE_SOURCES[@]}")

# Fill in the rest of the canonical CPU reference (`av2enc_core`).
source "$ROOT/src/cuda/pipeline/cpu_reference_sources.sh"
# still_encoder.cpp is #included by still_chroma_test.cu, so it must not be
# linked a second time.
av2enc_cpu_core_append HOST_SOURCES "" \
  "${HOST_SOURCES[@]+"${HOST_SOURCES[@]}"}" \
  src/pipeline/still/still_encoder.cpp

HOST_OBJECTS=()
for src in "${HOST_SOURCES[@]}"; do
  if [ ! -f "$ROOT/$src" ]; then
    echo "warning: optional host source missing: $src" >&2
    continue
  fi
  obj="$BUILD_DIR/$(echo "$src" | tr '/' '_').o"
  g++ -std=c++20 -O2 -ffp-contract=off -fno-fast-math \
    -I"$ROOT/src" -c "$ROOT/$src" -o "$obj"
  HOST_OBJECTS+=("$obj")
done

nvcc -std=c++20 -O2 -lineinfo -arch="$ARCH" \
  -Xcompiler -Wall,-Wextra,-Wconversion,-Wshadow \
  -I"$ROOT/src" -c still_chroma.cu -o "$BUILD_DIR/still_chroma.o"

nvcc -std=c++20 -O2 -lineinfo -arch="$ARCH" \
  -Xcompiler -Wall,-Wextra,-Wconversion,-Wshadow \
  -I"$ROOT/src" -c still_chroma_test.cu -o "$BUILD_DIR/still_chroma_test.o"

nvcc -std=c++20 -O2 -arch="$ARCH" \
  "$BUILD_DIR/still_chroma_test.o" "$BUILD_DIR/still_chroma.o" \
  "${HOST_OBJECTS[@]}" -o "$BUILD_DIR/still_chroma_test"

if [ "$RUN" -eq 1 ]; then
  FRAME_DIR="$BUILD_DIR/frames"
  mkdir -p "$FRAME_DIR"
  Y4M_ARGS=()
  if command -v ffmpeg >/dev/null 2>&1; then
    for size in 64x64 640x360 1920x1080; do
      out="$FRAME_DIR/testsrc2_${size}.y4m"
      if [ ! -s "$out" ]; then
        ffmpeg -v error -f lavfi -i "testsrc2=size=${size}:rate=1" \
          -frames:v 1 -pix_fmt yuv420p -f yuv4mpegpipe "$out" -y
      fi
      Y4M_ARGS+=(--y4m "$out")
    done
    # Real-content frames when the (unsynced) test media is present.
    SHINE="$ROOT/testdata/shine_1280x720_60f.y4m"
    if [ -s "$SHINE" ]; then
      for size in 64x64 640x360 1920x1080; do
        out="$FRAME_DIR/shine_${size}.y4m"
        if [ ! -s "$out" ]; then
          w="${size%x*}"
          h="${size#*x}"
          ffmpeg -v error -i "$SHINE" -frames:v 1 \
            -vf "scale=${w}:${h}:flags=bicubic" -pix_fmt yuv420p \
            -f yuv4mpegpipe "$out" -y
        fi
        Y4M_ARGS+=(--y4m "$out")
      done
    fi
  else
    echo "ffmpeg not found: running the built-in synthetic frames only"
  fi
  "$BUILD_DIR/still_chroma_test" "${Y4M_ARGS[@]+"${Y4M_ARGS[@]}"}"
fi
