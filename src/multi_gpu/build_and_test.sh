#!/usr/bin/env bash
#
# Build and run the standalone multi-GPU planner tests.
#
#   ./build_and_test.sh
#   CXX=g++-13 ./build_and_test.sh
#   SANITIZE=1 ./build_and_test.sh
#   WERROR=1 ./build_and_test.sh
#
# No CUDA, no devices, no CMake. The tile geometry comes from
# ../encoder/tiles/, which is compiled here directly.
set -euo pipefail
cd "$(dirname "$0")"

CXX="${CXX:-c++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow -ffp-contract=off)
if [[ "${SANITIZE}" == "1" ]]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2)
fi
if [[ "${WERROR}" == "1" ]]; then
  FLAGS+=(-Werror)
fi

build_dir="$(mktemp -d "${TMPDIR:-/tmp}/av2enc-multigpu.XXXXXX")"
trap 'rm -rf "${build_dir}"' EXIT
binary="${build_dir}/multi_gpu_planner_test"

echo "compiler: ${CXX}"
"${CXX}" --version 2>/dev/null | head -n 1 || true
echo "flags:    ${FLAGS[*]}"
echo "building: planner.cpp planner_test.cpp ../encoder/tiles/tile_layout.cpp"
echo

"${CXX}" "${FLAGS[@]}" -I. -I../encoder/tiles \
  planner.cpp planner_test.cpp ../encoder/tiles/tile_layout.cpp \
  -o "${binary}"

echo "running: ${binary}"
echo
"${binary}"
