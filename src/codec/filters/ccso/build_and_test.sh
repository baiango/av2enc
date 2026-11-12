#!/usr/bin/env bash
# Build and run the temporal CCSO reuse component tests.
#
#   ./build_and_test.sh                 # build + run ccso_reuse_test
#   ./build_and_test.sh --no-run        # build only
#   SANITIZE=1 ./build_and_test.sh      # ASan + UBSan
#   CXX=clang++ ./build_and_test.sh
#
# Self-contained: does not touch the root CMake build, src/main.cpp, or any
# other component.
set -euo pipefail

cd "$(dirname "$0")"

CXX="${CXX:-c++}"
BUILD_DIR="${BUILD_DIR:-build-cpu}"
RUN=1
SANITIZE="${SANITIZE:-0}"

for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
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
echo "build: $BUILD_DIR/ccso_reuse_test"

"$CXX" -std=c++20 -O2 -g \
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow \
  ${EXTRA[@]+"${EXTRA[@]}"} \
  -I "$REPO_ROOT/src" \
  ccso_reuse.cpp ccso_reuse_test.cpp \
  "$REPO_ROOT/src/codec/filters/stream/filters_stream.cpp" \
  "$REPO_ROOT/src/codec/filters/loop_restoration/restoration.cpp" \
  "$REPO_ROOT/src/codec/entropy/av2/av2_entropy.cpp" \
  "$REPO_ROOT/src/core/bitstream/bitwriter.cpp" \
  -o "$BUILD_DIR/ccso_reuse_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/ccso_reuse_test"
fi
