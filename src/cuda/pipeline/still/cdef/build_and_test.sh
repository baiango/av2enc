#!/usr/bin/env bash
# Build and run the M1 GPU CDEF stage tests.
#
#   ./build_and_test.sh                       # host stage test (+ gate if dav2d)
#   ./build_and_test.sh --no-run              # build only
#   ./build_and_test.sh --gate <src.y4m>      # run the dav2d stream gate
#   ./build_and_test.sh --cuda [--arch=sm_86] # CPU-vs-GPU parity on a dump
#   ./build_and_test.sh --dump <planes.bin>   # CUDA test on an existing dump
#
# The host build links the CPU reference (src/codec/filters/cdef/cdef.cpp),
# the §8.2 coder (cdef_idx round-trip) and, for the gate, the M1 CPU encoder.
# Nothing here touches the root CMake build, src/main.cpp or any other agent's
# files.
#
# The stream gate needs the pinned decoder:
#   DAV2D=tools/bin/dav2d-macos ./build_and_test.sh --gate tests/data/basic_640x360_1f.y4m
set -euo pipefail

SELF=$(cd "$(dirname "$0")" && pwd)
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
BUILD_DIR="${BUILD_DIR:-$ROOT/build-m1-cdef}"
ARCH="${ARCH:-sm_86}"
CXX="${CXX:-c++}"
RUN=1
DO_CUDA=0
GATE_FRAME=""
DUMP=""

for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    --cuda) DO_CUDA=1 ;;
    --arch=*) ARCH="${arg#--arch=}" ;;
    --gate) GATE_FRAME="__next__" ;;
    --dump) DUMP="__next__" ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *)
      if [ "$GATE_FRAME" = "__next__" ]; then
        GATE_FRAME="$arg"
      elif [ "$DUMP" = "__next__" ]; then
        DUMP="$arg"
      else
        echo "unknown argument: $arg" >&2
        exit 2
      fi
      ;;
  esac
done
[ "$GATE_FRAME" = "__next__" ] && GATE_FRAME=""
[ "$DUMP" = "__next__" ] && DUMP=""

mkdir -p "$BUILD_DIR"
WARN="-Wall -Wextra -Wpedantic -Wconversion -Wshadow"
INC="-I$ROOT/src"

# Shared CPU-reference source list (still_syntax + the integrated tool set).
source "$ROOT/src/cuda/codec/filters/still_reference_sources.sh"
STILL_SOURCES=(
  "$ROOT/src/pipeline/still/still_encoder.cpp"
)
for src in "${STILL_REFERENCE_SOURCES[@]}"; do
  # The host stage half is compiled explicitly by each target below.
  case "$src" in
    src/cuda/pipeline/still/cdef/still_cdef.cpp) continue ;;
  esac
  STILL_SOURCES+=("$ROOT/$src")
done

# The still CPU reference grows with every integration wave; take the rest of
# the canonical `av2enc_core` source list instead of re-listing it here.
source "$ROOT/src/cuda/pipeline/cpu_reference_sources.sh"
# These two are passed to the binaries explicitly below; keep them out of the
# appended list so they are not linked twice.
av2enc_cpu_core_append STILL_SOURCES "$ROOT/" \
  "${STILL_SOURCES[@]+"${STILL_SOURCES[@]}"}" \
  src/cuda/pipeline/still/cdef/still_cdef.cpp \
  src/codec/filters/cdef/cdef.cpp

echo "CXX: $($CXX --version | head -1)"
echo "build: $BUILD_DIR"

if [ "$RUN" -eq 1 ] || [ "$DO_CUDA" -eq 0 ]; then
  "$CXX" -std=c++20 -O2 -g $WARN $INC \
    "$SELF/still_cdef_test.cpp" "$SELF/still_cdef.cpp" \
    "$ROOT/src/codec/filters/cdef/cdef.cpp" \
    "$ROOT/src/codec/filters/stream/filters_stream.cpp" \
    "$ROOT/src/core/bitstream/bitwriter.cpp" \
    "$ROOT/src/codec/entropy/av2/av2_entropy.cpp" \
    -o "$BUILD_DIR/still_cdef_test"
  "$CXX" -std=c++20 -O2 -g -w $INC \
    "$SELF/run_stream_gate.cpp" "$SELF/still_cdef.cpp" \
    "${STILL_SOURCES[@]}" \
    -o "$BUILD_DIR/run_stream_gate"
fi

if [ "$RUN" -eq 1 ]; then
  DUMP_ARG=""
  if [ -s "$BUILD_DIR/recon.bin" ]; then
    DUMP_ARG="$BUILD_DIR/recon.bin"
  fi
  "$BUILD_DIR/still_cdef_test" "$ROOT/decoded.y4m" "$DUMP_ARG"
  if [ -n "$GATE_FRAME" ]; then
    DAV2D="${DAV2D:-$ROOT/tools/bin/dav2d}"
    "$BUILD_DIR/run_stream_gate" "$GATE_FRAME" "$BUILD_DIR/gate" \
      --dav2d "$DAV2D" --dump-planes "$BUILD_DIR/recon.bin"
  fi
fi

if [ "$DO_CUDA" -eq 1 ]; then
  if ! command -v nvcc >/dev/null 2>&1; then
    echo "nvcc not found" >&2
    exit 127
  fi
  echo "nvcc: $(nvcc --version | tail -1), arch $ARCH"
  nvcc -std=c++20 -O3 -lineinfo -arch="$ARCH" $INC \
    -Xcompiler -Wall,-Wextra,-Wconversion,-Wshadow \
    "$SELF/still_cdef.cu" "$SELF/still_cdef_gpu_test.cu" \
    "$SELF/still_cdef.cpp" "$ROOT/src/codec/filters/cdef/cdef.cpp" \
    "$ROOT/src/codec/filters/stream/filters_stream.cpp" \
    "$ROOT/src/core/bitstream/bitwriter.cpp" \
    "$ROOT/src/codec/entropy/av2/av2_entropy.cpp" \
    -o "$BUILD_DIR/still_cdef_gpu_test"
  if [ "$RUN" -eq 1 ]; then
    if [ -z "$DUMP" ]; then
      DUMP="$BUILD_DIR/recon.bin"
    fi
    if [ ! -s "$DUMP" ]; then
      echo "no plane dump at $DUMP;" >&2
      echo "run: $BUILD_DIR/run_stream_gate <src.y4m> $BUILD_DIR/gate --dump-planes $DUMP" >&2
      exit 2
    fi
    "$BUILD_DIR/still_cdef_gpu_test" "$DUMP"
  fi
fi
