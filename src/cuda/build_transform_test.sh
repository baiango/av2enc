#!/usr/bin/env bash
# Build the CUDA transform component test.
#
#   ./build_transform_test.sh            # sm_86 (RTX 3060 target)
#   CUDA_ARCH=sm_120 ./build_transform_test.sh
#
# -fmad=false and -ffp-contract=off keep both the device kernels and the host
# CPU reference free of FMA contraction, so the comparison is meaningful.
# -lineinfo is on for Nsight Systems/Compute.  -Wpedantic is omitted because
# nvcc's generated stub files trip it with "style of line directive" noise.
set -euo pipefail
cd "$(dirname "$0")"

ARCH="${CUDA_ARCH:-sm_86}"

nvcc -std=c++17 \
  -arch="${ARCH}" \
  -O3 \
  -lineinfo \
  -fmad=false \
  -I.. \
  -Xcompiler=-Wall,-Wextra,-Wconversion,-Wshadow,-ffp-contract=off \
  -Xptxas=-v \
  transform.cu transform_test.cu \
  -o transform_test

echo "built $(pwd)/transform_test (arch ${ARCH})"
