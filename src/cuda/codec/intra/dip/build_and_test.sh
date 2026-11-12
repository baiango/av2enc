#!/usr/bin/env bash
# Build/run AV2 DIP tests.
#
# With nvcc + a CUDA device: CPU-vs-GPU bit-exact parity (dip_test.cu).
# Without nvcc: the same shared device header is compiled with CUDA keywords
# stubbed and compared against the CPU reference (dip_emulation_test.cpp).
set -euo pipefail
cd "$(dirname "$0")"

if command -v nvcc >/dev/null 2>&1; then
  exec ./build_gpu_test.sh "$@"
fi
echo "nvcc not found: running host emulation parity only" >&2
exec ./build_emulation_test.sh "$@"
