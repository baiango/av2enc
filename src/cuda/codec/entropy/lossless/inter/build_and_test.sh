#!/usr/bin/env bash
# Build/run the AV2 lossless inter CUDA component tests.
#
# With nvcc: CPU-vs-GPU parity and arena accounting
# (lossless_inter_test.cu).  Without nvcc: the same shared device header is
# compiled with the CUDA keywords stubbed and compared against the CPU
# reference (lossless_inter_emulation_test.cpp).
set -euo pipefail
cd "$(dirname "$0")"

if command -v nvcc >/dev/null 2>&1; then
  exec ./build_gpu_test.sh "$@"
fi
echo "nvcc not found: running host emulation parity only" >&2
exec ./build_emulation_test.sh "$@"
