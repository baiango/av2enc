#!/usr/bin/env bash
# Build/run the AV2 §7.16 TIP deblocking tests.
#
# With nvcc + a CUDA device: CPU-vs-GPU bit-exact parity (tip_deblock_test.cu).
# Without nvcc: the same shared device header is compiled with CUDA keywords
# absent and compared against the CPU reference / an independent 10-bit
# transcription (tip_deblock_emulation_test.cpp).
#
#   ./build_and_test.sh                 # dispatches to GPU or emulation
#   ./build_and_test.sh --no-run
#   ARCH=sm_120 ./build_and_test.sh
set -euo pipefail
cd "$(dirname "$0")"

if command -v nvcc >/dev/null 2>&1; then
  exec ./build_gpu_test.sh "$@"
fi
echo "nvcc not found: running host emulation parity only" >&2
exec ./build_emulation_test.sh "$@"
