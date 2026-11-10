#!/bin/sh
# Build and run the standalone DCT benchmark on the RTX 3060 box.
#
# From the local repo:
#   rsync -a --delete bench/cuda_dct_perf/ <user>@<host>:/workspace/av2enc/bench/cuda_dct_perf/
#   ssh -p <port> <user>@<host> 'sh /workspace/av2enc/bench/cuda_dct_perf/run_remote.sh'
#
# The script is intentionally plain sh; it does not touch the root project.
set -eu

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO=$(CDPATH= cd -- "$HERE/../.." && pwd)
BUILD="$HERE/build"

cd "$HERE"

cmake -G Ninja -S . -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build "$BUILD"

echo "=== nvidia-smi ==="
nvidia-smi
echo "=== compute capability ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,clocks.max.sm,clocks.max.mem,power.limit --format=csv,noheader
echo "=== nvcc ==="
nvcc --version

# Prove the sm_86 cubin executes; do not silently JIT the embedded PTX.
export CUDA_DISABLE_PTX_JIT=1

Y4M="$REPO/testdata/shine_1920x1080_10f.y4m"
if [ -f "$Y4M" ]; then
  exec "$BUILD/cuda_dct_perf" --y4m "$Y4M" "$@"
else
  exec "$BUILD/cuda_dct_perf" "$@"
fi
