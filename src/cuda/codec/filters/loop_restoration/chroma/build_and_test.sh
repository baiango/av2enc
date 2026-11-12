#!/usr/bin/env bash
# Build/run the AV2 chroma loop-restoration tests.
#
# With nvcc: CPU-vs-GPU bit-exact parity through the kernels
# (lr_chroma_test.cu).  Without nvcc: the same shared device header compiled by
# the host compiler and compared against the CPU reference
# (lr_chroma_emulation_test.cpp).
#
#   ./build_and_test.sh
#   ./build_and_test.sh --no-run
#   ARCH=sm_120 ./build_and_test.sh
set -euo pipefail
cd "$(dirname "$0")"

./audit_device_visibility.sh

if command -v nvcc >/dev/null 2>&1; then
  exec ./build_gpu_test.sh "$@"
fi

ARGS=()
for arg in "$@"; do
  case "$arg" in
    --arch=*) ;;  # emulation has no architecture
    *) ARGS+=("$arg") ;;
  esac
done
echo "nvcc not found: running host-emulation parity only" >&2
exec ./build_emulation_test.sh ${ARGS[@]+"${ARGS[@]}"}
