#!/usr/bin/env bash
#
# Build and run the standalone CPU rate-control tests.
#
#   ./build_and_test.sh
#   CXX=g++-13 ./build_and_test.sh
#   RC_SANITIZE=1 ./build_and_test.sh   # ASan + UBSan
#   RC_WERROR=1 ./build_and_test.sh     # warnings are errors
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

if [[ "${RC_SANITIZE:-0}" == "1" ]]; then
  CXXFLAGS+=(-fsanitize=address,undefined -fno-omit-frame-pointer)
fi

if [[ "${RC_WERROR:-0}" == "1" ]]; then
  CXXFLAGS+=(-Werror)
fi

build_dir="$(mktemp -d "${TMPDIR:-/tmp}/av2enc-rc.XXXXXX")"
trap 'rm -rf "${build_dir}"' EXIT

binary="${build_dir}/rc_test"

echo "compiler: ${CXX}"
"${CXX}" --version 2>/dev/null | head -n 1 || true
echo "flags:    ${CXXFLAGS[*]}"
echo "building: src/codec/rate/rc/rc.cpp src/codec/rate/rc/rc_test.cpp"
echo

"${CXX}" "${CXXFLAGS[@]}" -I. rc.cpp rc_test.cpp -o "${binary}"

echo "running: ${binary}"
echo
"${binary}"
