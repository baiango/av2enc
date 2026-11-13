#!/usr/bin/env bash
# Build and run the standalone M1 GPU data-flow test.
#
#   src/cuda/pipeline/still/build_still_gpu_test.sh                  # sm_86, build + run
#   src/cuda/pipeline/still/build_still_gpu_test.sh --no-run         # build only
#   ARCH=sm_120 src/cuda/pipeline/still/build_still_gpu_test.sh      # secondary target
#   src/cuda/pipeline/still/build_still_gpu_test.sh -- --verbose      # extra test flags
#
# Checks plan -> host buffers -> stage hooks (or the CPU reference fallback) ->
# frame bytes.  With no stage components linked, the fallback frame must be
# byte-identical to av2enc::still::encode_still_picture for every input.
#
# Self-contained: does not touch the root CMake build or another component's
# build directory.  Exit codes: 0 pass, 1 failure, 2 usage, 77 CUDA unavailable.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
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

# CPU reference dependencies of av2enc::still::encode_still_picture.
CORE_SOURCES=(
    "$ROOT/src/pipeline/still/still_encoder.cpp"
    "$ROOT/src/pipeline/still/still_headers.cpp"
    "$ROOT/src/codec/partition/tree/partition.cpp"
    "$ROOT/src/codec/intra/modes/intra_modes.cpp"
    "$ROOT/src/codec/transform/variants/tx_variants.cpp"
    "$ROOT/src/codec/intra/cfl/cfl.cpp"
    "$ROOT/src/codec/filters/deblock/deblock.cpp"
    "$ROOT/src/codec/entropy/av2/av2_entropy.cpp"
    "$ROOT/src/core/bitstream/basic_headers.cpp"
    "$ROOT/src/core/bitstream/bitwriter.cpp"
    "$ROOT/src/core/io/y4m.cpp"
    "$ROOT/src/cuda/context.cpp"
    "$ROOT/src/cuda/arena.cpp"
)
# The CPU M1 encoder was split into a syntax writer; keep adding it when the
# shared tree has it (it is also part of the canonical list below).
if [ -f "$ROOT/src/pipeline/still/still_syntax.cpp" ]; then
    CORE_SOURCES+=("$ROOT/src/pipeline/still/still_syntax.cpp")
fi
TEST_SOURCES=(
    "$ROOT/src/cuda/pipeline/still/still_gpu_test.cc"
)
CUDA_SOURCES=(
    "$ROOT/src/cuda/pipeline/still/still_gpu.cu"
)
# Shared CUDA components used by the stage-1 executor.  Added only when the
# corresponding workstream is present.  All are compiled with --fmad=false:
# still_tx is the bit-exact double-precision transform.
CUDA_COMPONENT_SOURCES=()
for optional in \
    "$ROOT/src/cuda/codec/intra/modes/intra_modes.cu" \
    "$ROOT/src/cuda/codec/filters/deblock/deblock.cu" \
    "$ROOT/src/cuda/codec/filters/cdef/cdef.cu" \
    "$ROOT/src/cuda/codec/filters/loop_restoration/ccso.cu" \
    "$ROOT/src/cuda/codec/filters/loop_restoration/wiener.cu" \
    "$ROOT/src/cuda/codec/filters/loop_restoration/chroma/lr_chroma.cu" \
    "$ROOT/src/cuda/codec/filters/pc_wiener/pc_wiener.cu" \
    "$ROOT/src/cuda/codec/filters/gdf/gdf.cu" \
    "$ROOT/src/cuda/codec/intra/cfl/cfl.cu" \
    "$ROOT/src/cuda/codec/intra/mrl/mrl.cu" \
    "$ROOT/src/cuda/codec/intra/dip/dip.cu" \
    "$ROOT/src/cuda/codec/intra/ibp/ibp.cu" \
    "$ROOT/src/cuda/pipeline/still/intra/intra_extra.cu" \
    "$ROOT/src/cuda/pipeline/still/tx/still_tx.cu" \
    "$ROOT/src/cuda/pipeline/still/recon/recon.cu" \
    "$ROOT/src/cuda/pipeline/still/search/luma_search.cu" \
    "$ROOT/src/cuda/pipeline/still/chroma/still_chroma.cu" \
    "$ROOT/src/cuda/pipeline/still/cdef/still_cdef.cu" \
    "$ROOT/src/cuda/pipeline/still/cfl/still_cfl.cu" \
    "$ROOT/src/cuda/pipeline/still/restoration/still_restoration.cu" \
    "$ROOT/src/cuda/pipeline/still/sched/sched_exec.cu" \
    "$ROOT/src/cuda/pipeline/still/hbd/hbd_frame.cu" \
    "$ROOT/src/cuda/pipeline/still/hbd/hbd_tx.cu" \
    "$ROOT/src/cuda/formats/high_bitdepth_gpu.cu" \
    "$ROOT/src/cuda/formats/chroma/chroma_formats.cu" \
    "$ROOT/src/cuda/codec/transform/variants/tx_variants.cu"; do
    if [ -f "$optional" ]; then
        CUDA_COMPONENT_SOURCES+=("$optional")
    fi
done
# Stage components that have an orchestrator adapter.  Added only when they
# exist, so this script keeps working on a tree where a workstream has not
# landed yet.  Compile without the strict warning set: they are not our files.
STAGE_SOURCES=()
if [ -f "$ROOT/src/cuda/pipeline/still/plan/plan_gpu_bridge.cpp" ]; then
    STAGE_SOURCES+=(
        "$ROOT/src/cuda/pipeline/still/plan/plan.cpp"
        "$ROOT/src/cuda/pipeline/still/plan/plan_gpu_bridge.cpp"
    )
fi
for optional in \
    "$ROOT/src/cuda/pipeline/still/hbd/hbd_cpu.cpp" \
    "$ROOT/src/cuda/formats/format_device.cpp" \
    "$ROOT/src/cuda/formats/chroma/chroma_formats.cpp" \
    "$ROOT/src/cuda/pipeline/still/sched/sched.cpp" \
    "$ROOT/src/cuda/pipeline/still/sched/sched_bridge.cpp" \
    "$ROOT/src/pipeline/still/stream/still_stream_bridge.cpp"; do
    if [ -f "$optional" ]; then
        STAGE_SOURCES+=("$optional")
    fi
done

# Fill in the rest of the canonical CPU reference (`av2enc_core`).  Pass the
# stage sources too so `still_stream_bridge.cpp` is not compiled twice.
source "$ROOT/src/cuda/pipeline/cpu_reference_sources.sh"
av2enc_cpu_core_append CORE_SOURCES "$ROOT/" \
    "${CORE_SOURCES[@]+"${CORE_SOURCES[@]}"}" \
    "${STAGE_SOURCES[@]+"${STAGE_SOURCES[@]}"}"

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

BIN="$BUILD_DIR/still_gpu_test"
"$NVCC" -std=c++20 -arch="$ARCH" "${OBJS[@]}" -o "$BIN" -lcudart
echo "  binary: $BIN"
echo

if [ "$RUN" -eq 0 ]; then
    exit 0
fi

INPUTS=()
if [ -f "$ROOT/tests/data/basic_64x64_1f.y4m" ]; then
    INPUTS+=(--input "$ROOT/tests/data/basic_64x64_1f.y4m")
fi
if [ -f "$ROOT/tests/data/basic_640x360_1f.y4m" ]; then
    INPUTS+=(--input "$ROOT/tests/data/basic_640x360_1f.y4m")
fi
if [ "${#INPUTS[@]}" -eq 0 ]; then
    echo "note: test media missing; falling back to the synthetic 64x64 frame" >&2
fi

"$BIN" "${INPUTS[@]}" "${TEST_ARGS[@]}"
