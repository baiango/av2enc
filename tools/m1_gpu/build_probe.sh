#!/usr/bin/env bash
# Build the optional M1 GPU recon/stats probe against an existing CUDA build.
#
#   tools/m1_gpu/build_probe.sh
#
# The probe is NOT part of the encoder build. It links the same core library as
# build-cuda/av2enc and calls av2enc::cuda::m1gpu::encode_m1_gpu directly, so it
# only works after the orchestrator entry point is in the library:
#
#   cmake --build build-cuda --target av2enc
#
# (Building the encoder itself is workstream 9's job; this script only compiles
# tools/m1_gpu/m1_gpu_probe.cpp and does not touch CMake or src/.)
#
# Env: CXX (default g++), LIB, OUT, CUDART_LIB_DIR (default /usr/local/cuda/lib64)
set -euo pipefail

SELF=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SELF/../.." && pwd)
CXX=${CXX:-g++}
LIB=${LIB:-$ROOT/build-cuda/libav2enc_core.a}
OUT=${OUT:-$ROOT/tools/m1_gpu/m1_gpu_probe}
CUDART_LIB_DIR=${CUDART_LIB_DIR:-/usr/local/cuda/lib64}

if [ ! -f "$LIB" ]; then
    echo "build_probe: error: no core library at $LIB" >&2
    echo "build_probe: build build-cuda/av2enc first (cmake --build build-cuda)" >&2
    exit 2
fi

# The orchestrator may live in the core archive or in the separate
# av2enc_m1_gpu object target (CMake links it into the executable only). Use
# the archive when it defines encode_m1_gpu, otherwise link the object files.
# No grep -q in the pipeline: with pipefail an early exit SIGPIPEs nm/c++filt.
EXTRA_OBJS=()
if nm --defined-only "$LIB" 2>/dev/null | c++filt |
        grep "av2enc::cuda::m1gpu::encode_m1_gpu" >/dev/null; then
    :
else
    BUILD_DIR=$(cd "$(dirname "$LIB")" && pwd)
    OBJ_LIST=$(find "$BUILD_DIR/CMakeFiles/av2enc_m1_gpu.dir" -name '*.o' \
        2>/dev/null | sort | tr '\n' ' ')
    if [ -z "$OBJ_LIST" ]; then
        echo "build_probe: error: $LIB has no encode_m1_gpu symbol and no" >&2
        echo "build_probe: av2enc_m1_gpu object files were found next to it." >&2
        echo "build_probe: the M1 GPU orchestrator is not in this build; the" >&2
        echo "build_probe: gate will use the CLI path only." >&2
        exit 3
    fi
    # shellcheck disable=SC2206
    EXTRA_OBJS=($OBJ_LIST)
    echo "build_probe: linking ${#EXTRA_OBJS[@]} av2enc_m1_gpu object(s)"
fi

echo "build_probe: $LIB -> $OUT"
if [ ${#EXTRA_OBJS[@]} -gt 0 ]; then
    "$CXX" -O2 -std=c++20 -I"$ROOT/src" \
        "$SELF/m1_gpu_probe.cpp" -o "$OUT" \
        "${EXTRA_OBJS[@]}" "$LIB" -L"$CUDART_LIB_DIR" -lcudart
else
    "$CXX" -O2 -std=c++20 -I"$ROOT/src" \
        "$SELF/m1_gpu_probe.cpp" -o "$OUT" \
        "$LIB" -L"$CUDART_LIB_DIR" -lcudart
fi

echo "build_probe: ok: $OUT"
"$OUT" --help >/dev/null
