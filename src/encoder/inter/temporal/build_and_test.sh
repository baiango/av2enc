#!/usr/bin/env bash
# Build (and by default run) the AV2 temporal inter CPU reference tests.
#
#   ./build_and_test.sh            # build + run
#   ./build_and_test.sh --no-run   # build only
#   ./build_and_test.sh --asan     # ASan/UBSan build + run
#   CXX=clang++ ./build_and_test.sh
#
# Self-contained: no root CMake, src/main.cpp, M2 or CUDA build changes.
set -euo pipefail

cd "$(dirname "$0")"

CXX="${CXX:-c++}"
BUILD_DIR="${BUILD_DIR:-build-cpu}"
RUN=1
SAN=0

for arg in "$@"; do
  case "$arg" in
    --no-run)
      RUN=0
      ;;
    --asan)
      SAN=1
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

WARN="-Wall -Wextra -Wpedantic -Wconversion -Wshadow"
if [ "$SAN" -eq 1 ]; then
  OPT="-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer"
  echo "sanitizers: address,undefined"
else
  OPT="-O2 -g"
fi

echo "CXX: $($CXX --version | head -1)"
echo "build: $BUILD_DIR/temporal_test"

"$CXX" -std=c++20 $OPT $WARN -ffp-contract=off -I "$REPO_ROOT/src" \
  temporal_common.cpp motion_field.cpp opfl.cpp ref_mv_bank.cpp refinemv.cpp \
  temporal_syntax.cpp tip.cpp temporal_test.cpp \
  "$REPO_ROOT/src/bitstream/bitwriter.cpp" \
  -o "$BUILD_DIR/temporal_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/temporal_test"
fi
