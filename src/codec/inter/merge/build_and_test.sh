#!/usr/bin/env bash
# Build and run the AV2 M2 skip / merge / DRL tests.
#
#   ./build_and_test.sh                 # build + run
#   ./build_and_test.sh --no-run        # build only
#   SANITIZE=1 ./build_and_test.sh      # ASan + UBSan
#   CXX=clang++ ./build_and_test.sh     # alternate compiler
#
# Self-contained: does not touch the root CMake build, src/main.cpp, the M2
# core, or src/codec/inter/mc/.  It compiles the shared partition and MC
# translation units because the audit wrapper calls inter::mc::build_mv_stack.
set -euo pipefail

cd "$(dirname "$0")"

CXX="${CXX:-c++}"
BUILD_DIR="${BUILD_DIR:-build-cpu}"
SANITIZE="${SANITIZE:-0}"
RUN=1

for arg in "$@"; do
  case "$arg" in
    --no-run)
      RUN=0
      ;;
    -h|--help)
      sed -n '2,14p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
mkdir -p "$BUILD_DIR"

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
       -I "$ROOT/src")
if [[ "$SANITIZE" == "1" ]]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2 -g)
fi

echo "CXX: $($CXX --version | head -1)"
echo "build: $BUILD_DIR/merge_test"

"$CXX" "${FLAGS[@]}" \
  merge.cpp merge_test.cpp \
  "$ROOT/src/codec/partition/tree/partition.cpp" \
  "$ROOT/src/codec/inter/mc/mc.cpp" \
  -o "$BUILD_DIR/merge_test"

if [[ "$RUN" -eq 1 ]]; then
  "$BUILD_DIR/merge_test"
fi
