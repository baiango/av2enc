#!/usr/bin/env bash
# Build and run the CUDA M0 pipeline integration test.
#
# Usage:
#   src/cuda/build_pipeline_test.sh [--no-validate]
#
# Environment:
#   CUDA_ARCH=sm_86        nvcc architecture (component default; contract)
#   CXX=g++                host C++ compiler
#   NVCC=nvcc              CUDA compiler
#   BUILD_DIR=...          default: $TMPDIR/av2enc-m0-pipeline-build
#   DAV2D=/path/to/dav2d   decoder override (else tools/bin/dav2d[-macos])
#   OUT_DIR=...            IVF output directory (default: $BUILD_DIR/out)
#
# What it does:
#   1. compiles the host orchestrator + test with g++ and the kernels with nvcc;
#   2. runs the CPU-vs-GPU harness on 64x64 and 640x360, requiring byte-identical
#      frame payloads and reconstructed planes;
#   3. if a pinned dav2d binary is available, validates the GPU IVF files against
#      the expected decoded MD5s (7efc... and 56f1...).
#
# Exit codes: 0 pass, 1 failure, 2 usage, 77 CUDA/inputs unavailable.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BUILD_DIR=${BUILD_DIR:-"${TMPDIR:-/tmp}/av2enc-m0-pipeline-build"}
OUT_DIR=${OUT_DIR:-"$BUILD_DIR/out"}
CXX=${CXX:-g++}
NVCC=${NVCC:-nvcc}
CUDA_ARCH=${CUDA_ARCH:-sm_86}
VALIDATE=1

for arg in "$@"; do
    case "$arg" in
        --no-validate) VALIDATE=0 ;;
        -h|--help)
            sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "error: unknown argument: $arg" >&2
            exit 2
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

HOST_FLAGS=(-std=c++20 -O3 -I"$ROOT/src")
CORE_SOURCES=(
    "$ROOT/src/pipeline/still/basic_encoder.cpp"
    "$ROOT/src/codec/entropy/av2/av2_entropy.cpp"
    "$ROOT/src/core/bitstream/basic_headers.cpp"
    "$ROOT/src/core/bitstream/bitwriter.cpp"
    "$ROOT/src/core/io/y4m.cpp"
    "$ROOT/src/core/io/ivf.cpp"
)
HOST_SOURCES=(
    "$ROOT/src/cuda/basic_pipeline.cpp"
    "$ROOT/src/cuda/basic_pipeline_test.cpp"
)
# Shared component layer (docs/cuda-interface.md ownership):
# runtime -> context.*/arena.*, transform -> transform.*, intra -> intra.*.
CUDA_COMPONENT_SOURCES=(
    "$ROOT/src/cuda/context.cpp"
    "$ROOT/src/cuda/arena.cpp"
    "$ROOT/src/cuda/transform.cu"
    "$ROOT/src/cuda/intra.cu"
)

echo "== build =="
echo "  root:   $ROOT"
echo "  build:  $BUILD_DIR"
echo "  nvcc:   $NVCC ($CUDA_ARCH)"
echo "  host:   $CXX"

OBJS=()
for source in "${CORE_SOURCES[@]}" "${HOST_SOURCES[@]}"; do
    obj="$BUILD_DIR/obj/$(basename "${source%.cpp}").o"
    "$CXX" "${HOST_FLAGS[@]}" -c "$source" -o "$obj"
    OBJS+=("$obj")
done

CUDA_OBJ="$BUILD_DIR/obj/basic_pipeline_cuda.o"
"$NVCC" -std=c++20 -O3 -lineinfo --fmad=false -arch="$CUDA_ARCH" \
    -I"$ROOT/src" -c "$ROOT/src/cuda/basic_pipeline.cu" -o "$CUDA_OBJ"
OBJS+=("$CUDA_OBJ")

for source in "${CUDA_COMPONENT_SOURCES[@]}"; do
    obj="$BUILD_DIR/obj/$(basename "${source%.*}").o"
    "$NVCC" -std=c++20 -O3 -lineinfo --fmad=false -arch="$CUDA_ARCH" \
        -I"$ROOT/src" -c "$source" -o "$obj"
    OBJS+=("$obj")
done

BIN="$BUILD_DIR/basic_pipeline_test"
"$NVCC" -std=c++20 -arch="$CUDA_ARCH" "${OBJS[@]}" -o "$BIN" -lcudart
echo "  binary: $BIN"

run_case() {
    local name=$1
    local input=$2
    local expected=$3
    if [ ! -f "$input" ]; then
        echo "SKIP: missing input $input" >&2
        return 77
    fi
    echo
    echo "== $name =="
    if ! "$BIN" --y4m "$input" --out-dir "$OUT_DIR" --name "$name" --verbose; then
        echo "error: pipeline harness failed for $name" >&2
        return 1
    fi
    if [ "$VALIDATE" -eq 1 ]; then
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
            "$ROOT/tools/validate_dav2d.sh" "$OUT_DIR/gpu_$name.ivf" "$input"
    fi
    return 0
}

set +e
run_case 64x64 "$ROOT/tests/data/basic_64x64_1f.y4m" \
    7efc3162bc0a4785685eba291a8096fd
RC64=$?
run_case 640x360 "$ROOT/tests/data/basic_640x360_1f.y4m" \
    56f11d4a63ff07f2356ef5064b84fe0e
RC360=$?
set -e

if [ "$RC64" -eq 77 ] || [ "$RC360" -eq 77 ]; then
    echo "SKIP: test inputs unavailable" >&2
    exit 77
fi
if [ "$RC64" -ne 0 ] || [ "$RC360" -ne 0 ]; then
    echo "FAIL: pipeline gate failed (64x64 rc=$RC64, 640x360 rc=$RC360)" >&2
    exit 1
fi

echo
echo "PASS: CUDA M0 pipeline matches the CPU reference and decoded MD5 gates"
