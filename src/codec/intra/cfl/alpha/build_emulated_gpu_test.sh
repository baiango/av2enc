#!/usr/bin/env bash
# Local emulation of the CUDA API test (no nvcc, no GPU).
#
# This runs alpha_gpu_test.cu as ordinary C++ against cuda_stub/cuda_runtime.h
# and cuda_emulation.cpp, which forwards to the shared CPU core. It validates
# the test logic and the alpha plumbing; it is NOT a GPU parity result. The
# real gate is build_cuda_test.sh on the sm_86 box.
#
#   ./build_emulated_gpu_test.sh
#   SANITIZE=1 WERROR=1 ./build_emulated_gpu_test.sh
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CXX="${CXX:-c++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
OUT="${OUT:-${TMPDIR:-/tmp}/cfl_alpha_emu_test}"

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
if [[ "${SANITIZE}" == "1" ]]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2)
fi
if [[ "${WERROR}" == "1" ]]; then
  FLAGS+=(-Werror)
fi

"${CXX}" "${FLAGS[@]}" \
  -I"${HERE}" -I"${HERE}/cuda_stub" -DAV2ENC_CUDA_EMULATION=1 \
  -x c++ "${HERE}/alpha_gpu_test.cu" -x none \
  "${HERE}/cuda_emulation.cpp" \
  "${HERE}/alpha.cpp" \
  "${HERE}/syntax.cpp" \
  "${HERE}/../cfl.cpp" \
  "${HERE}/../../../entropy/av2/av2_entropy.cpp" \
  -o "${OUT}"

"${OUT}"
