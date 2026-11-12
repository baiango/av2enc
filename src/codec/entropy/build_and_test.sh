#!/usr/bin/env bash
#
# Build and run the standalone range-coder tests.
#
# No CMake, no external dependencies.  Uses g++ with the flags required by the
# project; override the compiler with CXX=... (e.g. CXX=g++-16 on macOS where
# /usr/bin/g++ is Apple clang).
set -euo pipefail

cd "$(dirname "$0")"

CXX="${CXX:-g++}"
CXXFLAGS=(-std=c++20 -O2 -Wall -Wextra -Wpedantic)

build_dir="$(mktemp -d "${TMPDIR:-/tmp}/av2enc-entropy.XXXXXX")"
trap 'rm -rf "${build_dir}"' EXIT

binary="${build_dir}/entropy_test"

echo "compiler: ${CXX}"
"${CXX}" --version 2>/dev/null | head -n 1 || true
echo "flags:    ${CXXFLAGS[*]}"
echo "building: src/codec/entropy/range_coder.cpp src/codec/entropy/entropy_test.cpp"
echo

"${CXX}" "${CXXFLAGS[@]}" -I. range_coder.cpp entropy_test.cpp -o "${binary}"

echo "running: ${binary}"
echo
"${binary}"
