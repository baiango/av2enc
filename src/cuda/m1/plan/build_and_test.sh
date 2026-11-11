#!/usr/bin/env bash
# Build and run the CPU tests for the M1 GPU FramePlan builder.
#
#   ./src/cuda/m1/plan/build_and_test.sh
#   CXX=clang++ ./src/cuda/m1/plan/build_and_test.sh
#   SANITIZE=1 ./src/cuda/m1/plan/build_and_test.sh
#   WERROR=1 ./src/cuda/m1/plan/build_and_test.sh
#   OUT=/tmp/plan_test ./src/cuda/m1/plan/build_and_test.sh
#
# Self-contained: does not touch the root CMake build or another component.
# The target is Linux + CUDA; this CPU binary is only a correctness oracle.
#
# -ffp-contract=off keeps the double variance accumulation from fusing into
# FMA on hosts that have it (e.g. macOS arm64), so plan decisions are stable
# against the x86 CPU M1 reference.

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${HERE}/../../../.." && pwd)"
CXX="${CXX:-c++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
OUT="${OUT:-${TMPDIR:-/tmp}/m1_plan_test}"

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
       -ffp-contract=off)

if [[ "${SANITIZE}" == "1" ]]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2)
fi

if [[ "${WERROR}" == "1" ]]; then
  FLAGS+=(-Werror)
fi

"${CXX}" "${FLAGS[@]}" -I"${ROOT}/src" \
  -DPLAN_REPO_ROOT="\"${ROOT}\"" \
  "${HERE}/plan.cpp" \
  "${HERE}/plan_gpu_bridge.cpp" \
  "${HERE}/plan_test.cpp" \
  "${ROOT}/src/io/y4m.cpp" \
  "${ROOT}/src/encoder/partition/partition.cpp" \
  "${ROOT}/src/encoder/tx_variants/tx_variants.cpp" \
  -o "${OUT}"

"${OUT}"
