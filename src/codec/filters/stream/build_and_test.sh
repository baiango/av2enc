#!/usr/bin/env bash
# Standalone build/run for the M1 filter/restoration syntax adapters.
#
# No CUDA and no CMake changes are needed: the module is host-only and links
# the existing §8.2 entropy coder and BitWriter sources.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${root}/CMakeLists.txt" ] && [ "${root}" != "/" ]; do
    root="$(dirname -- "${root}")"
done
cxx="${CXX:-c++}"
out="$here/build"
mkdir -p "$out"

flags=(
  -std=c++20
  -O2
  -Wall
  -Wextra
  -Wpedantic
  -Wconversion
  -Wshadow
  -I "$root/src"
  -I "$root/tests"
)

if [[ "${ASAN:-0}" == "1" ]]; then
  flags+=(-g -fsanitize=address,undefined -fno-omit-frame-pointer)
fi

"$cxx" "${flags[@]}" \
  "$here/filters_stream_test.cpp" \
  "$here/filters_stream.cpp" \
  "$root/src/codec/entropy/av2/av2_entropy.cpp" \
  "$root/src/core/bitstream/bitwriter.cpp" \
  -o "$out/filters_stream_test"

"$out/filters_stream_test"
