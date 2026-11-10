#!/usr/bin/env bash
#
# Build and run the standalone CPU RDO/RDOQ tests.
#
#   ./build_and_test.sh
#   CXX=g++-16 ./build_and_test.sh
#   RDO_SANITIZE=1 ./build_and_test.sh   # ASan + UBSan
#
# No CMake and no external dependencies. This directory is intentionally not
# wired into the root build while the component is developed in isolation.
set -euo pipefail
cd "$(dirname "$0")"

CXX="${CXX:-c++}"
CXXFLAGS=(
  -std=c++20
  -O2
  -Wall
  -Wextra
  -Wpedantic
  -Wconversion
  -Wshadow
  -ffp-contract=off
)

if [[ "${RDO_SANITIZE:-0}" == "1" ]]; then
  CXXFLAGS+=(-fsanitize=address,undefined -fno-omit-frame-pointer)
fi

build_dir="$(mktemp -d "${TMPDIR:-/tmp}/av2enc-rdo.XXXXXX")"
trap 'rm -rf "${build_dir}"' EXIT

binary="${build_dir}/rdo_test"

echo "compiler: ${CXX}"
"${CXX}" --version 2>/dev/null | head -n 1 || true
echo "flags:    ${CXXFLAGS[*]}"
echo "building: src/encoder/rdo/rdo.cpp src/encoder/rdo/rdo_test.cpp"
echo

"${CXX}" "${CXXFLAGS[@]}" -I. rdo.cpp rdo_test.cpp -o "${binary}"

echo "running: ${binary}"
echo
"${binary}"
