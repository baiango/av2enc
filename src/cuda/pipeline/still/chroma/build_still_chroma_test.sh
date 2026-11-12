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

HOST_SOURCES=(
  "src/core/io/y4m.cpp"
  "src/core/bitstream/bitwriter.cpp"
  "src/core/bitstream/basic_headers.cpp"
  "src/core/bitstream/obu.cpp"
  "src/codec/partition/tree/partition.cpp"
  "src/codec/intra/modes/intra_modes.cpp"
  "src/codec/rate/rdoq/rdoq.cpp"
  "src/codec/rate/rdoq/rdoq_av2_model.cpp"
  "src/codec/transform/tx_type/tx_type.cpp"
  "src/codec/transform/variants/tx_variants.cpp"
  "src/codec/transform/variants/tx_select.cpp"
  "src/codec/intra/cfl/cfl.cpp"
  "src/codec/filters/deblock/deblock.cpp"
  "src/codec/filters/cdef/cdef.cpp"
  "src/codec/rate/rdo/rdo.cpp"
  "src/codec/rate/rdo/intra_decision.cpp"
  "src/pipeline/still/still_headers.cpp"
  "src/cuda/pipeline/still/cdef/still_cdef.cpp"
  "src/codec/entropy/av2/av2_entropy.cpp"
)

# still_encoder.cpp has been refactored to live next to the syntax writer. Depending
# on the tree, it either #includes src/pipeline/still/still_syntax.cpp directly or
# relies on the build to list it; tolerate both.
if ! grep -q '#include "pipeline/still/still_syntax.cpp"' \
    "$ROOT/src/pipeline/still/still_encoder.cpp"; then
  HOST_SOURCES+=("src/pipeline/still/still_syntax.cpp")
fi

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
