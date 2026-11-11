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

# No grep -q here: with pipefail, an early grep exit SIGPIPEs nm/c++filt and
# the pipeline looks failed. Redirect the full match list instead.
if ! nm --defined-only "$LIB" 2>/dev/null | c++filt |
        grep "av2enc::cuda::m1gpu::encode_m1_gpu" >/dev/null; then
    echo "build_probe: error: $LIB has no encode_m1_gpu symbol" >&2
    echo "build_probe: the M1 GPU orchestrator is not in this build; --m1-gpu" >&2
    echo "build_probe: cannot be probed. The gate will use the CLI path only." >&2
    exit 3
fi

echo "build_probe: $LIB -> $OUT"
"$CXX" -O2 -std=c++20 -I"$ROOT/src" \
    "$SELF/m1_gpu_probe.cpp" -o "$OUT" \
    "$LIB" -L"$CUDART_LIB_DIR" -lcudart

echo "build_probe: ok: $OUT"
"$OUT" --help >/dev/null
