#!/usr/bin/env bash
# Build (and by default run) the AV2 inter motion-estimation CPU reference
# test.  Self-contained: does not touch the root CMake build or other
# components.
#
#   ./build_and_test.sh                 # build + run
#   ./build_and_test.sh --no-run        # build only
#   ./build_and_test.sh --asan          # ASan/UBSan build + run
#   CXX=clang++ ./build_and_test.sh     # alternate compiler
#
# The CUDA CPU-vs-GPU test lives in src/cuda/inter/me/.
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
      sed -n '2,11p' "$0"
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
echo "build: $BUILD_DIR/me_test"

# -ffp-contract=off matches the project's no-contraction rule; ME itself is
# integer-only, so this is belt-and-braces for the test harness.
"$CXX" -std=c++20 $OPT $WARN -ffp-contract=off -I "$REPO_ROOT/src" \
  me.cpp me_test.cpp -o "$BUILD_DIR/me_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/me_test" "$REPO_ROOT"
fi
