#!/usr/bin/env bash
# Build and run the recovered-table checks.
#
#   ./build_and_test.sh
#   ./build_and_test.sh --no-run
#   SANITIZE=1 ./build_and_test.sh
#   CXX=clang++ ./build_and_test.sh
#
# The first step is generate_missing_tables.py --check: pinned PDF hash,
# per-table extents/value digests, local-copy cross-checks, and generated-header
# freshness. It does not touch the root CMake build or any other component.
set -euo pipefail

cd "$(dirname "$0")"

CXX="${CXX:-c++}"
BUILD_DIR="${BUILD_DIR:-build}"
SANITIZE="${SANITIZE:-0}"
RUN=1

for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$BUILD_DIR"

echo "check: generate_missing_tables.py --check"
python3 generate_missing_tables.py --check

FLAGS=(-std=c++20 -O2 -g -Wall -Wextra -Wpedantic -Wconversion -Wshadow
       -I ../../..)
if [ "$SANITIZE" -eq 1 ]; then
  FLAGS=(-std=c++20 -O1 -g -Wall -Wextra -Wpedantic -Wconversion -Wshadow
         -fsanitize=address,undefined -fno-omit-frame-pointer
         -I ../../..)
fi

echo "build: $BUILD_DIR/missing_tables_test"
"$CXX" "${FLAGS[@]}" missing_tables_test.cpp \
  ../../pipeline/tiles/tile_layout.cpp -o "$BUILD_DIR/missing_tables_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/missing_tables_test"
fi
