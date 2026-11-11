#!/usr/bin/env bash
# Build and run the standalone M1 GPU data-flow test.
#
#   src/cuda/m1/build_m1_gpu_test.sh                  # sm_86, build + run
#   src/cuda/m1/build_m1_gpu_test.sh --no-run         # build only
#   ARCH=sm_120 src/cuda/m1/build_m1_gpu_test.sh      # secondary target
#   src/cuda/m1/build_m1_gpu_test.sh -- --verbose      # extra test flags
#
# Checks plan -> host buffers -> stage hooks (or the CPU reference fallback) ->
# frame bytes.  With no stage components linked, the fallback frame must be
# byte-identical to m1::encode_m1_still_picture for every input.
#
# Self-contained: does not touch the root CMake build or another component's
# build directory.  Exit codes: 0 pass, 1 failure, 2 usage, 77 CUDA unavailable.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
BUILD_DIR="${BUILD_DIR:-${TMPDIR:-/tmp}/av2enc-m1-gpu-build}"
OUT_DIR="$BUILD_DIR/out"
CXX="${CXX:-g++}"
NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-sm_86}"
RUN=1
TEST_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --no-run)
            RUN=0
            ;;
        --arch=*)
            ARCH="${arg#--arch=}"
            ;;
        --)
            shift
            TEST_ARGS=("$@")
            break
            ;;
        -h|--help)
            sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            TEST_ARGS+=("$arg")
            ;;
    esac
done

if ! command -v "$NVCC" >/dev/null 2>&1; then
    echo "SKIP: nvcc not found; run this script on the CUDA host" >&2
    exit 77
fi
if ! command -v "$CXX" >/dev/null 2>&1; then
    echo "error: host compiler '$CXX' not found" >&2
    exit 2
fi

mkdir -p "$BUILD_DIR/obj" "$OUT_DIR"

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CUDA_INC=()
if [ -d "$CUDA_HOME/include" ]; then
    CUDA_INC=(-I"$CUDA_HOME/include")
fi

HOST_FLAGS=(-std=c++20 -O2 -I"$ROOT/src" "${CUDA_INC[@]}" -ffp-contract=off)
WARN_FLAGS=(-Wall -Wextra -Wpedantic -Wconversion -Wshadow)

# CPU reference dependencies of m1::encode_m1_still_picture.
CORE_SOURCES=(
    "$ROOT/src/encoder/m1/m1_still.cpp"
    "$ROOT/src/encoder/m1/m1_headers.cpp"
    "$ROOT/src/encoder/partition/partition.cpp"
    "$ROOT/src/encoder/intra_modes/intra_modes.cpp"
    "$ROOT/src/encoder/tx_variants/tx_variants.cpp"
    "$ROOT/src/encoder/cfl/cfl.cpp"
    "$ROOT/src/encoder/filters/deblock/deblock.cpp"
    "$ROOT/src/entropy/av2/av2_entropy.cpp"
    "$ROOT/src/bitstream/m0_headers.cpp"
    "$ROOT/src/bitstream/bitwriter.cpp"
    "$ROOT/src/io/y4m.cpp"
    "$ROOT/src/cuda/context.cpp"
    "$ROOT/src/cuda/arena.cpp"
)
# The CPU M1 encoder was split into a syntax writer while this wave was in
# flight; include the split source when the shared tree has it.
for optional in \
    "$ROOT/src/encoder/m1/m1_syntax.cpp" \
    "$ROOT/src/encoder/intra.cpp"; do
    if [ -f "$optional" ]; then
        CORE_SOURCES+=("$optional")
    fi
done
TEST_SOURCES=(
    "$ROOT/src/cuda/m1/m1_gpu_test.cc"
)
CUDA_SOURCES=(
    "$ROOT/src/cuda/m1/m1_gpu.cu"
)
# Shared CUDA components used by the stage-1 executor.  Added only when the
# corresponding workstream is present.  All are compiled with --fmad=false:
# m1_tx is the bit-exact double-precision transform.
CUDA_COMPONENT_SOURCES=()
for optional in \
    "$ROOT/src/cuda/intra_modes/intra_modes.cu" \
    "$ROOT/src/cuda/deblock/deblock.cu" \
    "$ROOT/src/cuda/cfl/cfl.cu" \
    "$ROOT/src/cuda/m1/tx/m1_tx.cu" \
    "$ROOT/src/cuda/m1/recon/recon.cu" \
    "$ROOT/src/cuda/m1/search/luma_search.cu" \
    "$ROOT/src/cuda/m1/chroma/m1_chroma.cu"; do
    if [ -f "$optional" ]; then
        CUDA_COMPONENT_SOURCES+=("$optional")
    fi
done
# Stage components that have an orchestrator adapter.  Added only when they
# exist, so this script keeps working on a tree where a workstream has not
# landed yet.  Compile without the strict warning set: they are not our files.
STAGE_SOURCES=()
if [ -f "$ROOT/src/cuda/m1/plan/plan_gpu_bridge.cpp" ]; then
    STAGE_SOURCES+=(
        "$ROOT/src/cuda/m1/plan/plan.cpp"
        "$ROOT/src/cuda/m1/plan/plan_gpu_bridge.cpp"
    )
fi
for optional in \
    "$ROOT/src/cuda/m1/sched/sched.cpp" \
    "$ROOT/src/cuda/m1/sched/sched_bridge.cpp" \
    "$ROOT/src/encoder/m1/stream/m1_stream_bridge.cpp"; do
    if [ -f "$optional" ]; then
        STAGE_SOURCES+=("$optional")
    fi
done

echo "== build =="
echo "  root:   $ROOT"
echo "  build:  $BUILD_DIR"
echo "  nvcc:   $NVCC ($ARCH)"
echo "  host:   $CXX"

# Object names must be unique: the tree has a CPU and a CUDA `intra_modes`
# and `deblock` translation unit with the same basename.
OBJS=()
OBJ_INDEX=0
for source in "${CORE_SOURCES[@]}" "${TEST_SOURCES[@]}"; do
    obj="$BUILD_DIR/obj/${OBJ_INDEX}_$(basename "${source%.*}").o"
    OBJ_INDEX=$((OBJ_INDEX + 1))
    "$CXX" "${HOST_FLAGS[@]}" "${WARN_FLAGS[@]}" -c "$source" -o "$obj"
    OBJS+=("$obj")
done
for source in "${STAGE_SOURCES[@]}"; do
    obj="$BUILD_DIR/obj/${OBJ_INDEX}_$(basename "${source%.*}").o"
    OBJ_INDEX=$((OBJ_INDEX + 1))
    "$CXX" "${HOST_FLAGS[@]}" -c "$source" -o "$obj"
    OBJS+=("$obj")
done
for source in "${CUDA_SOURCES[@]}" "${CUDA_COMPONENT_SOURCES[@]}"; do
    obj="$BUILD_DIR/obj/${OBJ_INDEX}_$(basename "${source%.*}").o"
    OBJ_INDEX=$((OBJ_INDEX + 1))
    "$NVCC" -std=c++20 -O2 -lineinfo --fmad=false -arch="$ARCH" \
        -I"$ROOT/src" "${CUDA_INC[@]}" \
        -Xcompiler -Wall,-Wextra,-Wconversion,-Wshadow \
        -c "$source" -o "$obj"
    OBJS+=("$obj")
done

BIN="$BUILD_DIR/m1_gpu_test"
"$NVCC" -std=c++20 -arch="$ARCH" "${OBJS[@]}" -o "$BIN" -lcudart
echo "  binary: $BIN"
echo

if [ "$RUN" -eq 0 ]; then
    exit 0
fi

INPUTS=()
if [ -f "$ROOT/tests/data/m0_64x64_1f.y4m" ]; then
    INPUTS+=(--input "$ROOT/tests/data/m0_64x64_1f.y4m")
fi
if [ -f "$ROOT/tests/data/m0_640x360_1f.y4m" ]; then
    INPUTS+=(--input "$ROOT/tests/data/m0_640x360_1f.y4m")
fi
if [ "${#INPUTS[@]}" -eq 0 ]; then
    echo "note: test media missing; falling back to the synthetic 64x64 frame" >&2
fi

"$BIN" "${INPUTS[@]}" "${TEST_ARGS[@]}"
