#!/usr/bin/env bash
# Build and run the AV2 extended motion-estimation (me2) tests.
#
#   ./build_and_test.sh              # CPU tests + GPU host-emulation parity
#   ./build_and_test.sh --asan       # ASan/UBSan CPU + emulation
#   ./build_and_test.sh --no-run     # build only
#   ./build_and_test.sh --quick      # skip the real-720p report
#   CXX=clang++ ./build_and_test.sh
#
# The CUDA kernel (me2_gpu.cu) is built by build_gpu_and_test.sh when nvcc is
# available.  The host-emulation test always runs: it validates the exact
# warp-cooperative cost arithmetic and the shared search core without a GPU.
set -euo pipefail

cd "$(dirname "$0")"

CXX="${CXX:-c++}"
BUILD_DIR="${BUILD_DIR:-build-cpu}"
RUN=1
SAN=0
QUICK=0
for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    --asan) SAN=1 ;;
    --quick) QUICK=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
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

if [ "$QUICK" -eq 1 ]; then
  export ME2_SKIP_720P=1
fi

echo "CXX: $($CXX --version | head -1)"
echo "build: $BUILD_DIR/me2_test"

"$CXX" -std=c++20 $OPT $WARN -ffp-contract=off -I "$REPO_ROOT/src" \
  me2.cpp me2_test.cpp "$REPO_ROOT/src/codec/inter/me/me.cpp" \
  -o "$BUILD_DIR/me2_test" -lm

"$CXX" -std=c++20 $OPT $WARN -ffp-contract=off -I "$REPO_ROOT/src" \
  me2.cpp me2_gpu_emulation_test.cpp "$REPO_ROOT/src/codec/inter/me/me.cpp" \
  -o "$BUILD_DIR/me2_gpu_emulation_test" -lm

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/me2_test" "$REPO_ROOT"
  "$BUILD_DIR/me2_gpu_emulation_test" "$REPO_ROOT"
fi
