#!/usr/bin/env bash
# Build and run the wavefront scheduler module tests.
#
# Usage:
#   src/cuda/pipeline/dispatch/scheduler/build_scheduler_test.sh [--plan-only] [--no-validate]
#                                             [--quick]
#
# Environment:
#   CUDA_ARCH=sm_86        nvcc architecture (component default; contract)
#   CXX=g++                host C++ compiler
#   NVCC=nvcc              CUDA compiler
#   BUILD_DIR=...          default: $TMPDIR/av2enc-scheduler-build
#   OUT_DIR=...            IVF output directory (default: $BUILD_DIR/out)
#   DAV2D=/path/to/dav2d   decoder override (else tools/bin/dav2d[-macos])
#
# What it does:
#   1. builds and runs the host-only plan test (no CUDA needed);
#   2. builds the GPU harness: per-block M0 path vs wavefront eager vs CUDA
#      graph, requiring bit-exact reconstruction, coefficients and predictions;
#   3. benchmarks 64x64, 640x360 and generated 720p/1080p, printing host calls,
#      kernel launches, kernel ms, submit ms and wall ms per frame;
#   4. if a pinned dav2d binary is available, validates the IVF written by the
#      production M0 path against the reference MD5s.
#
# Exit codes: 0 pass, 1 failure, 2 usage, 77 CUDA/inputs unavailable.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
BUILD_DIR=${BUILD_DIR:-"${TMPDIR:-/tmp}/av2enc-scheduler-build"}
OUT_DIR=${OUT_DIR:-"$BUILD_DIR/out"}
CXX=${CXX:-g++}
NVCC=${NVCC:-nvcc}
CUDA_ARCH=${CUDA_ARCH:-sm_86}
PLAN_ONLY=0
VALIDATE=1
QUICK=0

for arg in "$@"; do
    case "$arg" in
        --plan-only) PLAN_ONLY=1 ;;
        --no-validate) VALIDATE=0 ;;
        --quick) QUICK=1 ;;
        -h|--help)
            sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "error: unknown argument: $arg" >&2
            exit 2
            ;;
    esac
done

if ! command -v "$CXX" >/dev/null 2>&1; then
    echo "error: host compiler '$CXX' not found" >&2
    exit 2
fi

mkdir -p "$BUILD_DIR/obj" "$OUT_DIR"

SRC="$ROOT/src"
WARN_FLAGS=(-Wall -Wextra -Wpedantic -Wconversion -Wshadow)
HOST_FLAGS=(-std=c++20 -O3 -I"$SRC" "${WARN_FLAGS[@]}")

echo "== plan test (host only) =="
"$CXX" "${HOST_FLAGS[@]}" \
    "$SRC/cuda/pipeline/dispatch/scheduler/scheduler.cpp" \
    "$SRC/cuda/pipeline/dispatch/scheduler/scheduler_plan_test.cpp" \
    -o "$BUILD_DIR/scheduler_plan_test"
"$BUILD_DIR/scheduler_plan_test"

if [ "$PLAN_ONLY" -eq 1 ]; then
    echo
    echo "PASS: plan test only (GPU harness skipped)"
    exit 0
fi

if ! command -v "$NVCC" >/dev/null 2>&1; then
    echo "SKIP: nvcc not found; run the GPU harness on the CUDA host" >&2
    exit 77
fi

echo
echo "== build GPU harness =="
echo "  root:   $ROOT"
echo "  build:  $BUILD_DIR"
echo "  nvcc:   $NVCC ($CUDA_ARCH)"
echo "  host:   $CXX"

CORE_SOURCES=(
    "$SRC/pipeline/still/basic_encoder.cpp"
    "$SRC/codec/entropy/av2/av2_entropy.cpp"
    "$SRC/core/bitstream/basic_headers.cpp"
    "$SRC/core/bitstream/bitwriter.cpp"
    "$SRC/core/io/y4m.cpp"
    "$SRC/core/io/ivf.cpp"
)
HOST_SOURCES=(
    "$SRC/cuda/basic_pipeline.cpp"
    "$SRC/cuda/pipeline/dispatch/scheduler/scheduler.cpp"
    "$SRC/cuda/pipeline/dispatch/scheduler/scheduler_harness.cpp"
)
# Shared components and runtime (docs/cuda-interface.md ownership).
CUDA_SOURCES=(
    "$SRC/cuda/basic_pipeline.cu"
    "$SRC/cuda/pipeline/dispatch/scheduler/scheduler.cu"
    "$SRC/cuda/transform.cu"
    "$SRC/cuda/intra.cu"
    "$SRC/cuda/context.cpp"
    "$SRC/cuda/arena.cpp"
)

OBJS=()
obj_path() {
    local source=$1
    local rel=${source#"$SRC"/}
    rel=${rel//\//_}
    printf '%s/obj/%s.o' "$BUILD_DIR" "$rel"
}
for source in "${CORE_SOURCES[@]}" "${HOST_SOURCES[@]}"; do
    obj=$(obj_path "$source")
    "$CXX" "${HOST_FLAGS[@]}" -c "$source" -o "$obj"
    OBJS+=("$obj")
done

NVCC_FLAGS=(-std=c++20 -O3 -lineinfo --fmad=false -arch="$CUDA_ARCH"
            -I"$SRC" -Xcompiler "-Wall -Wextra -Wconversion -Wshadow")
for source in "${CUDA_SOURCES[@]}"; do
    obj=$(obj_path "$source")
    "$NVCC" "${NVCC_FLAGS[@]}" -c "$source" -o "$obj"
    OBJS+=("$obj")
done

BIN="$BUILD_DIR/scheduler_harness"
"$NVCC" -std=c++20 -arch="$CUDA_ARCH" "${OBJS[@]}" -o "$BIN" -lcudart
echo "  binary: $BIN"

validate_ivf() {
    local ivf=$1
    local input=$2
    local expected=$3
    if [ "$VALIDATE" -eq 0 ]; then
        return 0
    fi
    local decoder=""
    if [ -n "${DAV2D:-}" ] && [ -x "${DAV2D}" ]; then
        decoder="$DAV2D"
    elif [ -x "$ROOT/tools/bin/dav2d" ]; then
        decoder="$ROOT/tools/bin/dav2d"
    elif [ -x "$ROOT/tools/bin/dav2d-macos" ]; then
        decoder="$ROOT/tools/bin/dav2d-macos"
    else
        echo "SKIP: dav2d binary not found; decoded MD5 not checked" >&2
        return 0
    fi
    EXPECTED_MD5="$expected" DAV2D="$decoder" \
        "$ROOT/tools/validate_dav2d.sh" "$ivf" "$input"
}

run_case() {
    local name=$1
    local input=$2
    local expected=$3
    local iterations=$4
    shift 4
    if [ ! -f "$input" ]; then
        echo "SKIP: missing input $input" >&2
        return 77
    fi
    echo
    echo "== case: $name =="
    if ! "$BIN" --y4m "$input" --out-dir "$OUT_DIR" --name "$name" \
            --iterations "$iterations"; then
        echo "error: scheduler harness failed for $name" >&2
        return 1
    fi
    validate_ivf "$OUT_DIR/gpu_$name.ivf" "$input" "$expected"
    return 0
}

run_generated() {
    local name=$1
    local width=$2
    local height=$3
    local iterations=$4
    echo
    echo "== case: $name (generated pattern) =="
    if ! "$BIN" --width "$width" --height "$height" --out-dir "$OUT_DIR" \
            --name "$name" --iterations "$iterations"; then
        echo "error: scheduler harness failed for $name" >&2
        return 1
    fi
    return 0
}

set +e
run_case 64x64 "$ROOT/tests/data/basic_64x64_1f.y4m" \
    7efc3162bc0a4785685eba291a8096fd 5
RC64=$?
run_case 640x360 "$ROOT/tests/data/basic_640x360_1f.y4m" \
    56f11d4a63ff07f2356ef5064b84fe0e 5
RC360=$?
run_generated 1280x720 1280 720 5
RC720=$?
if [ "$QUICK" -eq 1 ]; then
    RC1080=0
else
    run_generated 1920x1080 1920 1080 5
    RC1080=$?
fi
set -e

if [ "$RC64" -eq 77 ] || [ "$RC360" -eq 77 ]; then
    echo "SKIP: test inputs unavailable" >&2
    exit 77
fi
if [ "$RC64" -ne 0 ] || [ "$RC360" -ne 0 ] || [ "$RC720" -ne 0 ] ||
   [ "$RC1080" -ne 0 ]; then
    echo "FAIL: scheduler harness failed (rc 64=$RC64 360=$RC360 720=$RC720 1080=$RC1080)" >&2
    exit 1
fi

echo
echo "PASS: wavefront scheduler matches the per-block M0 path and the dav2d gate"
