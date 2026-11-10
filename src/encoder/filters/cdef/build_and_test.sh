#!/usr/bin/env bash
# Build (and by default run) the AV2 CDEF CPU reference test.
#
#   ./build_and_test.sh                 # build + run
#   ./build_and_test.sh --no-run        # build only
#   CXX=clang++ ./build_and_test.sh     # alternate compiler
#   SANITIZE=1 ./build_and_test.sh      # ASan + UBSan
#
# Self-contained: does not touch the root CMake build, src/main.cpp, or any
# other component.  CPU-only; the CUDA test lives in src/cuda/cdef/.
set -euo pipefail

cd "$(dirname "$0")"

CXX="${CXX:-c++}"
BUILD_DIR="${BUILD_DIR:-build-cpu}"
RUN=1
SANITIZE="${SANITIZE:-0}"

for arg in "$@"; do
  case "$arg" in
    --no-run)
      RUN=0
      ;;
    -h|--help)
      sed -n '2,10p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

REPO_ROOT="$(cd ../../../.. && pwd)"
mkdir -p "$BUILD_DIR"

EXTRA=()
if [ "$SANITIZE" -eq 1 ]; then
  EXTRA+=(-fsanitize=address,undefined -fno-omit-frame-pointer)
fi

echo "CXX: $($CXX --version | head -1)"
echo "build: $BUILD_DIR/cdef_test"

"$CXX" -std=c++20 -O2 -g \
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow \
  ${EXTRA[@]+"${EXTRA[@]}"} \
  -I "$REPO_ROOT/src" \
  cdef.cpp cdef_test.cpp \
  -o "$BUILD_DIR/cdef_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/cdef_test" "$REPO_ROOT"
fi
