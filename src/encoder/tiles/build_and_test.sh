#!/usr/bin/env bash
#
# Build and run the standalone tile/SVC component tests.
#
#   ./build_and_test.sh
#   CXX=g++-13 ./build_and_test.sh
#   SANITIZE=1 ./build_and_test.sh
#   WERROR=1 ./build_and_test.sh
#
# No CMake, no CUDA, no external dependencies. This directory is intentionally
# not wired into the root build while the component is developed in isolation.
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

build_dir="$(mktemp -d "${TMPDIR:-/tmp}/av2enc-tiles.XXXXXX")"
trap 'rm -rf "${build_dir}"' EXIT
binary="${build_dir}/tiles_test"

echo "compiler: ${CXX}"
"${CXX}" --version 2>/dev/null | head -n 1 || true
echo "flags:    ${FLAGS[*]}"
echo "building: tile_layout.cpp tile_group.cpp svc_layers.cpp tiles_test.cpp"
echo

"${CXX}" "${FLAGS[@]}" -I. \
  tile_layout.cpp tile_group.cpp svc_layers.cpp tiles_test.cpp \
  -o "${binary}"

echo "running: ${binary}"
echo
"${binary}"
