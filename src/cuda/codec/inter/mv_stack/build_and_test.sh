#!/usr/bin/env bash
# Build/run the CUDA MV-stack component test.
#
#   ./build_and_test.sh          # GPU test when nvcc is present, else emulation
#   ./build_and_test.sh --emulation
#   ARCH=sm_120 ./build_and_test.sh
set -euo pipefail

HERE="$(cd -- "$(dirname -- "$0")" && pwd)"
MODE="auto"
for arg in "$@"; do
  case "$arg" in
    --emulation) MODE="emulation" ;;
    -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if [ "${MODE}" = "emulation" ] || ! command -v nvcc >/dev/null 2>&1; then
  exec "${HERE}/build_emulation_test.sh"
fi
exec "${HERE}/build_gpu_test.sh"
