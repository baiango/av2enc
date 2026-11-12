#!/usr/bin/env bash
# Build and run the M1 scheduler adapter tests.
#
#   src/cuda/pipeline/still/sched/build_sched_test.sh [--plan-only] [--quick]
#                                        [--no-projection] [--full-intra]
#                                        [--iterations N] [--y4m FILE]...
#
# Environment:
#   CXX=g++            host compiler
#   NVCC=nvcc          CUDA compiler
#   CUDA_ARCH=sm_86    nvcc architecture
#   BUILD_DIR=...      default: ${TMPDIR:-/tmp}/av2enc-m1-sched-build
#
# The plan test is host-only and runs on macOS.  The GPU harness needs Linux +
# CUDA and links the CPU M1 encoder (for decisions/reference recon) with the
# shared CUDA components (intra_modes, tx_variants, cfl).
#
# Exit codes: 0 pass, 1 failure, 2 usage, 77 CUDA/inputs unavailable.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
SRC="$ROOT/src"
BUILD_DIR=${BUILD_DIR:-"${TMPDIR:-/tmp}/av2enc-m1-sched-build"}
CXX=${CXX:-g++}
NVCC=${NVCC:-nvcc}
CUDA_ARCH=${CUDA_ARCH:-sm_86}
PLAN_ONLY=0
QUICK=0
NO_PROJECTION=0
FULL_INTRA=0
ITERATIONS=5
Y4M_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --plan-only) PLAN_ONLY=1 ;;
        --quick) QUICK=1 ;;
        --no-projection) NO_PROJECTION=1 ;;
        --full-intra) FULL_INTRA=1 ;;
        --iterations)
            shift
            ITERATIONS=${1:-}
            ;;
        --y4m)
            shift
            Y4M_ARGS+=(--y4m "$1")
            ;;
        -h|--help)
            sed -n '2,26p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            exit 2
            ;;
    esac
    shift
done

CUDA_INCLUDE=${CUDA_INCLUDE:-/usr/local/cuda/include}
HOST_FLAGS=(-std=c++20 -O3 -I"$SRC" -I"$CUDA_INCLUDE" -Wall -Wextra -Wpedantic
            -Wconversion -Wshadow)
mkdir -p "$BUILD_DIR/obj"

echo "== M1 scheduler plan test (host only) =="
"$CXX" "${HOST_FLAGS[@]}" \
    "$SRC/cuda/pipeline/still/sched/sched.cpp" \
    "$SRC/cuda/pipeline/still/sched/sched_bridge.cpp" \
    "$SRC/cuda/pipeline/still/sched/sched_plan_test.cpp" \
    -o "$BUILD_DIR/sched_plan_test"
"$BUILD_DIR/sched_plan_test"

if [ "$PLAN_ONLY" -eq 1 ]; then
    echo
    echo "PASS: plan test only (GPU harness skipped)"
    exit 0
fi

if ! command -v "$NVCC" >/dev/null 2>&1; then
    echo "SKIP: nvcc not found; run the GPU harness on the CUDA host" >&2
    exit 77
fi

if [ -z "${Y4M_ARGS[*]:-}" ]; then
    Y4M_ARGS=(--y4m "$ROOT/tests/data/basic_64x64_1f.y4m"
              --y4m "$ROOT/tests/data/basic_640x360_1f.y4m")
fi
for ((i = 0; i < ${#Y4M_ARGS[@]}; i += 2)); do
    if [ ! -f "${Y4M_ARGS[$((i + 1))]}" ]; then
        echo "SKIP: missing input ${Y4M_ARGS[$((i + 1))]}" >&2
        exit 77
    fi
done

echo
echo "== build M1 scheduler GPU harness =="
echo "  build: $BUILD_DIR"
echo "  nvcc:  $NVCC ($CUDA_ARCH)"

# CPU reference half: the real M1 encoder (decisions + reference recon).
CORE_SOURCES=(
    "$SRC/pipeline/still/still_encoder.cpp"
    "$SRC/pipeline/still/still_headers.cpp"
    "$SRC/pipeline/still/still_syntax.cpp"
    "$SRC/codec/partition/tree/partition.cpp"
    "$SRC/codec/intra/modes/intra_modes.cpp"
    "$SRC/codec/transform/variants/tx_variants.cpp"
    "$SRC/codec/transform/variants/tx_select.cpp"
    "$SRC/codec/intra/cfl/cfl.cpp"
    "$SRC/codec/filters/deblock/deblock.cpp"
    "$SRC/codec/rate/rdo/rdo.cpp"
    "$SRC/codec/entropy/av2/av2_entropy.cpp"
    "$SRC/core/bitstream/bitwriter.cpp"
    "$SRC/core/bitstream/obu.cpp"
    "$SRC/core/bitstream/basic_headers.cpp"
    "$SRC/core/io/y4m.cpp"
    "$SRC/core/io/ivf.cpp"
    "$SRC/cuda/context.cpp"
    "$SRC/cuda/arena.cpp"
    "$SRC/cuda/pipeline/still/sched/sched.cpp"
    "$SRC/cuda/pipeline/still/sched/sched_bridge.cpp"
)
# Scheduler device half + the verification executor.
CUDA_SOURCES=(
    "$SRC/cuda/pipeline/still/sched/sched_exec.cu"
    "$SRC/cuda/pipeline/still/sched/sched_ref_bench.cu"
    "$SRC/cuda/pipeline/still/sched/sched_bench.cpp"
    "$SRC/cuda/codec/intra/modes/intra_modes.cu"
    "$SRC/cuda/codec/transform/variants/tx_variants.cu"
    "$SRC/cuda/codec/intra/cfl/cfl.cu"
)

OBJS=()
obj_path() {
    local source=$1
    local rel=${source#"$SRC"/}
    rel=${rel//\//_}
    printf '%s/obj/%s.o' "$BUILD_DIR" "$rel"
}
for source in "${CORE_SOURCES[@]}"; do
    obj=$(obj_path "$source")
    "$CXX" "${HOST_FLAGS[@]}" -c "$source" -o "$obj"
    OBJS+=("$obj")
done

NVCC_FLAGS=(-std=c++20 -O3 -lineinfo --fmad=false -arch="$CUDA_ARCH" -I"$SRC"
            -Xcompiler "-Wall -Wextra -Wconversion -Wshadow")
for source in "${CUDA_SOURCES[@]}"; do
    obj=$(obj_path "$source")
    "$NVCC" "${NVCC_FLAGS[@]}" -c "$source" -o "$obj"
    OBJS+=("$obj")
done

BIN="$BUILD_DIR/sched_bench"
"$NVCC" -std=c++20 -arch="$CUDA_ARCH" "${OBJS[@]}" -o "$BIN" -lcudart
echo "  binary: $BIN"

RUN_ARGS=(--iterations "$ITERATIONS")
if [ "$NO_PROJECTION" -eq 1 ]; then
    RUN_ARGS+=(--no-projection)
fi
if [ "$QUICK" -eq 1 ]; then
    RUN_ARGS+=(--quick)
fi
if [ "$FULL_INTRA" -eq 1 ]; then
    RUN_ARGS+=(--full-intra)
fi

echo
"$BIN" "${Y4M_ARGS[@]}" "${RUN_ARGS[@]}"
