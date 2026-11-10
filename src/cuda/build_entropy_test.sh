#!/usr/bin/env bash
# Build and run the CUDA AV2 §8.2 entropy port test.
#
# The default target is the primary 8 GB Ampere box: nvcc 12.8, sm_86, with
# -lineinfo.  No Blackwell-only features are used; ARCH=sm_120 is accepted for
# the secondary build/smoke target.
#
# Usage:
#   ./src/cuda/build_entropy_test.sh
#   ARCH=sm_120 ./src/cuda/build_entropy_test.sh
#   OUT=build/av2_cuda_entropy_test ./src/cuda/build_entropy_test.sh --no-bench
#   WERROR=1 ./src/cuda/build_entropy_test.sh
#
# The test links the CPU reference implementation (av2_entropy.cpp) as the
# byte-exactness oracle.  It does not modify or rebuild the rest of the tree.

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${HERE}/../.." && pwd)"
NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-sm_86}"
OUT="${OUT:-${TMPDIR:-/tmp}/av2_cuda_entropy_test}"
WERROR="${WERROR:-0}"

FLAGS=(-std=c++20 -O2 -lineinfo "-arch=${ARCH}")
# nvcc generates GNU-style #line markers in its host stub, so -Wpedantic is
# omitted here; the CPU reference build keeps it.  All module code is still
# compiled with -Wall -Wextra -Wconversion -Wshadow.
HOST_FLAGS=(-Wall -Wextra -Wconversion -Wshadow)

if [[ "${WERROR}" == "1" ]]; then
  HOST_FLAGS+=(-Werror)
fi

# nvcc forwards host warnings through -Xcompiler; bare -Wall is rejected.
JOINED_HOST_FLAGS="$(IFS=,; echo "${HOST_FLAGS[*]}")"
FLAGS+=("-Xcompiler=${JOINED_HOST_FLAGS}")

"${NVCC}" "${FLAGS[@]}" \
  -I"${ROOT}/src" \
  "${HERE}/entropy.cu" \
  "${HERE}/entropy_test.cu" \
  "${ROOT}/src/entropy/av2/av2_entropy.cpp" \
  -o "${OUT}"

"${OUT}" "$@"
