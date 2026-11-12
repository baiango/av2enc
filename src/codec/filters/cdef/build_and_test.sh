#!/usr/bin/env bash
# Build (and by default run) the AV2 CDEF CPU reference tests.
#
#   ./build_and_test.sh                 # build + run both test binaries
#   ./build_and_test.sh --no-run        # build only
#   CXX=clang++ ./build_and_test.sh     # alternate compiler
#   SANITIZE=1 ./build_and_test.sh      # ASan + UBSan
#
# Self-contained: does not touch the root CMake build, src/main.cpp, or any
# other component.  CPU-only; the CUDA test lives in src/cuda/codec/filters/cdef/.
#
# Binaries:
#   cdef_test         §7.18 filter/decision reference (242 checks)
#   cdef_search_test  encoder-side strength search (cdef_search.h)
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

REPO_ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${REPO_ROOT}/CMakeLists.txt" ] && [ "${REPO_ROOT}" != "/" ]; do
    REPO_ROOT="$(dirname -- "${REPO_ROOT}")"
done
mkdir -p "$BUILD_DIR"

EXTRA=()
if [ "$SANITIZE" -eq 1 ]; then
  EXTRA+=(-fsanitize=address,undefined -fno-omit-frame-pointer)
fi

echo "CXX: $($CXX --version | head -1)"
echo "build: $BUILD_DIR/cdef_test, $BUILD_DIR/cdef_search_test"

"$CXX" -std=c++20 -O2 -g \
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow \
  ${EXTRA[@]+"${EXTRA[@]}"} \
  -I "$REPO_ROOT/src" \
  cdef.cpp cdef_test.cpp \
  -o "$BUILD_DIR/cdef_test"

"$CXX" -std=c++20 -O2 -g \
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow \
  ${EXTRA[@]+"${EXTRA[@]}"} \
  -I "$REPO_ROOT/src" \
  cdef.cpp cdef_search_test.cpp \
  -o "$BUILD_DIR/cdef_search_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/cdef_test" "$REPO_ROOT"
  "$BUILD_DIR/cdef_search_test" "$REPO_ROOT/decoded.y4m"
fi
