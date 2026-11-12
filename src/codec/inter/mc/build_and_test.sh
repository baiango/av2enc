#!/usr/bin/env bash
# Build (and by default run) the AV2 inter motion-compensation CPU reference test.
#
#   ./build_and_test.sh                  # build + run
#   ./build_and_test.sh --no-run         # build only
#   ./build_and_test.sh --check-tables   # verify the generated subpel table
#   CXX=clang++ ./build_and_test.sh      # alternate compiler
#
# Self-contained: does not touch the root CMake build, src/main.cpp, or any
# other component.  The CUDA test lives in src/cuda/codec/inter/mc/.
set -euo pipefail

cd "$(dirname "$0")"

CXX="${CXX:-c++}"
BUILD_DIR="${BUILD_DIR:-build-cpu}"
RUN=1
CHECK_TABLES=0

for arg in "$@"; do
  case "$arg" in
    --no-run)
      RUN=0
      ;;
    --check-tables)
      CHECK_TABLES=1
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

if [ "$CHECK_TABLES" -eq 1 ]; then
  python3 gen_subpel_tables.py --check
fi

echo "CXX: $($CXX --version | head -1)"
echo "build: $BUILD_DIR/mc_test"

"$CXX" -std=c++20 -O2 -g \
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow \
  -I "$REPO_ROOT/src" \
  mc.cpp mc_test.cpp \
  -o "$BUILD_DIR/mc_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/mc_test" "$REPO_ROOT"
fi
