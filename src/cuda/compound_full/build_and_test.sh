#!/usr/bin/env bash
# Build/run the full compound-path tests.
#
# With nvcc + a CUDA device: CPU-vs-GPU bit-exact parity, arena contract and
# batch launches (compound_full_test.cu).
# Without nvcc: the same shared device header compiled as host code and
# compared against the CPU references (compound_full_emulation_test.cpp).
set -euo pipefail
cd "$(dirname "$0")"

if command -v nvcc >/dev/null 2>&1; then
  exec ./build_gpu_test.sh "$@"
fi
echo "nvcc not found: running host-emulation parity only" >&2
exec ./build_emulation_test.sh "$@"
