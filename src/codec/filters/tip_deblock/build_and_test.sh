#!/usr/bin/env bash
# Build and run the AV2 §7.16 TIP deblocking CPU reference tests and gate.
#
#   ./build_and_test.sh                 # build + unit tests + synthetic gate
#   ./build_and_test.sh --no-run        # build only
#   ./build_and_test.sh --asan          # ASan + UBSan
#   CXX=clang++ ./build_and_test.sh
#
# Self-contained: does not touch the root CMake build, src/main.cpp, the M1/M2
# encoders, or CUDA.  The real `dav2d --strict 1 --threads 1` gate needs a TIP
# stream; the gate binary documents the seam and accepts --tip-ivf when one is
# produced by the TIP stream agent (see gate/tip_deblock_gate.cpp).
set -euo pipefail

cd "$(dirname "$0")"

CXX="${CXX:-c++}"
BUILD_DIR="${BUILD_DIR:-build-cpu}"
RUN=1
SAN=0
if [ "${SANITIZE:-0}" -eq 1 ]; then
  SAN=1
fi

for arg in "$@"; do
  case "$arg" in
    --no-run)
      RUN=0
      ;;
    --asan)
      SAN=1
      ;;
    -h|--help)
      sed -n '2,12p' "$0"
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

WARN="-Wall -Wextra -Wpedantic -Wconversion -Wshadow"
if [ "$SAN" -eq 1 ]; then
  OPT="-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer"
  echo "sanitizers: address,undefined"
else
  OPT="-O2 -g"
fi

echo "CXX: $($CXX --version | head -1)"
echo "build: $BUILD_DIR/tip_deblock_test, $BUILD_DIR/tip_deblock_gate"

"$CXX" -std=c++20 $OPT $WARN -I "$REPO_ROOT/src" \
  tip_deblock.cpp tip_deblock_test.cpp \
  -o "$BUILD_DIR/tip_deblock_test"

"$CXX" -std=c++20 $OPT $WARN -I "$REPO_ROOT/src" \
  tip_deblock.cpp gate/tip_deblock_gate.cpp \
  -o "$BUILD_DIR/tip_deblock_gate"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/tip_deblock_test" "$REPO_ROOT"
  "$BUILD_DIR/tip_deblock_gate"
fi
