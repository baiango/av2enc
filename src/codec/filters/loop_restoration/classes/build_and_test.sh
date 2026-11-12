#!/usr/bin/env bash
# Build and run the encoder-side Wiener search/syntax tests.
#
#   ./build_and_test.sh              # build + run both binaries
#   ./build_and_test.sh --no-run     # build only
#   ASAN=1 ./build_and_test.sh       # ASan/UBSan
#
# CPU-only, self-contained: it does not touch the root CMake build, main.cpp,
# or any other component.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${root}/CMakeLists.txt" ] && [ "${root}" != "/" ]; do
    root="$(dirname -- "${root}")"
done
cxx="${CXX:-c++}"
out="$here/build-cpu"
run=1

for arg in "$@"; do
  case "$arg" in
    --no-run) run=0 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$out"

flags=(
  -std=c++20 -O2 -g -Wall -Wextra -Wpedantic -Wconversion -Wshadow
  -I "$root/src"
)
if [[ "${ASAN:-0}" == "1" ]]; then
  flags+=(-fsanitize=address,undefined -fno-omit-frame-pointer)
fi

common=(
  "$here/wiener_fit.cpp"
  "$here/wiener_class.cpp"
  "$here/wiener_syntax.cpp"
  "$root/src/codec/filters/loop_restoration/restoration.cpp"
  "$root/src/codec/entropy/av2/av2_entropy.cpp"
  "$root/src/core/bitstream/bitwriter.cpp"
)

echo "CXX: $("$cxx" --version | head -1)"
echo "build: $out/wiener_syntax_test"
"$cxx" "${flags[@]}" \
  "$here/wiener_syntax_test.cpp" \
  "$here/wiener_syntax.cpp" \
  "$root/src/codec/entropy/av2/av2_entropy.cpp" \
  "$root/src/core/bitstream/bitwriter.cpp" \
  -o "$out/wiener_syntax_test"

echo "build: $out/wiener_classes_test"
"$cxx" "${flags[@]}" \
  "$here/wiener_classes_test.cpp" \
  "$here/wiener_decision.cpp" \
  "${common[@]}" \
  -o "$out/wiener_classes_test"

if [[ "$run" -eq 1 ]]; then
  cd "$root"
  "$out/wiener_syntax_test"
  "$out/wiener_classes_test"
fi
