#!/usr/bin/env bash
# Build (and by default run) the M1 GPU chroma CPU-vs-GPU parity test.
#
#   ./build_m1_chroma_test.sh                 # sm_86, build + run
#   ./build_m1_chroma_test.sh --no-run        # build only
#   ARCH=sm_120 ./build_m1_chroma_test.sh     # secondary target
#   BUILD_DIR=/path ./build_m1_chroma_test.sh # private build dir override
#
# Self-contained: does not touch the root CMake build. The CPU reference is
# the real src/encoder/m1/m1_still.cpp (included by the test TU) plus the real
# src/encoder component sources, so this is a CPU-vs-GPU parity check, not two
# transcriptions of the spec.
set -euo pipefail

cd "$(dirname "$0")"

ROOT="$(cd ../../../.. && pwd)"
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

HOST_SOURCES=(
  "src/io/y4m.cpp"
  "src/bitstream/bitwriter.cpp"
  "src/bitstream/m0_headers.cpp"
  "src/bitstream/obu.cpp"
  "src/encoder/partition/partition.cpp"
  "src/encoder/intra_modes/intra_modes.cpp"
  "src/encoder/tx_variants/tx_variants.cpp"
  "src/encoder/tx_variants/tx_select.cpp"
  "src/encoder/cfl/cfl.cpp"
  "src/encoder/filters/deblock/deblock.cpp"
  "src/encoder/m1/m1_headers.cpp"
  "src/entropy/av2/av2_entropy.cpp"
)

# m1_still.cpp has been refactored to live next to the syntax writer. Depending
# on the tree, it either #includes src/encoder/m1/m1_syntax.cpp directly or
# relies on the build to list it; tolerate both.
if ! grep -q '#include "encoder/m1/m1_syntax.cpp"' \
    "$ROOT/src/encoder/m1/m1_still.cpp"; then
  HOST_SOURCES+=("src/encoder/m1/m1_syntax.cpp")
fi

HOST_OBJECTS=()
for src in "${HOST_SOURCES[@]}"; do
  obj="$BUILD_DIR/$(echo "$src" | tr '/' '_').o"
  g++ -std=c++20 -O2 -ffp-contract=off -fno-fast-math \
    -I"$ROOT/src" -c "$ROOT/$src" -o "$obj"
  HOST_OBJECTS+=("$obj")
done

nvcc -std=c++20 -O2 -lineinfo -arch="$ARCH" \
  -Xcompiler -Wall,-Wextra,-Wconversion,-Wshadow \
  -I"$ROOT/src" -c m1_chroma.cu -o "$BUILD_DIR/m1_chroma.o"

nvcc -std=c++20 -O2 -lineinfo -arch="$ARCH" \
  -Xcompiler -Wall,-Wextra,-Wconversion,-Wshadow \
  -I"$ROOT/src" -c m1_chroma_test.cu -o "$BUILD_DIR/m1_chroma_test.o"

nvcc -std=c++20 -O2 -arch="$ARCH" \
  "$BUILD_DIR/m1_chroma_test.o" "$BUILD_DIR/m1_chroma.o" \
  "${HOST_OBJECTS[@]}" -o "$BUILD_DIR/m1_chroma_test"

if [ "$RUN" -eq 1 ]; then
  FRAME_DIR="$BUILD_DIR/frames"
  mkdir -p "$FRAME_DIR"
  Y4M_ARGS=()
  if command -v ffmpeg >/dev/null 2>&1; then
    for size in 64x64 640x360; do
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
      for size in 64x64 640x360; do
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
  "$BUILD_DIR/m1_chroma_test" "${Y4M_ARGS[@]+"${Y4M_ARGS[@]}"}"
fi
