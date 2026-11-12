#!/usr/bin/env bash
# Build and run the transform-mode / transform-partition component test.
#
#   ./build_and_test.sh                 # -O2 build + run
#   ./build_and_test.sh --no-run        # build only
#   ./build_and_test.sh --print-goldens # print pinned golden inputs
#   SANITIZE=1 ./build_and_test.sh      # ASan + UBSan
#   WERROR=1 ./build_and_test.sh        # promote warnings to errors
#   CXX=clang++ ./build_and_test.sh     # alternate compiler
#
# Links only what the component needs: the §8.2 entropy coder, the TX-size
# tables, and the committed m0/tx_type scan tables used as a cross-check.
# Self-contained: no root CMake edits, no build artifacts in the repository.
set -euo pipefail

cd "$(dirname "$0")"

CXX="${CXX:-c++}"
REPO="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${REPO}/CMakeLists.txt" ] && [ "${REPO}" != "/" ]; do
    REPO="$(dirname -- "${REPO}")"
done
BUILD_DIR="${BUILD_DIR:-${TMPDIR:-/tmp}/av2enc_tx_partition_build}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
RUN=1
ARGS=()

for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    --print-goldens) ARGS+=("$arg") ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$BUILD_DIR"

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
       -Wno-missing-braces -I"$REPO/src")
if [ "$WERROR" = "1" ]; then
  FLAGS+=(-Werror)
fi
if [ "$SANITIZE" = "1" ]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2)
fi

echo "compiler: $($CXX --version | head -1)"
echo "build:    $BUILD_DIR/tx_partition_test"

"$CXX" "${FLAGS[@]}" \
  tx_partition.cpp tx_scan.cpp tx_partition_test.cpp \
  "$REPO/src/codec/entropy/av2/av2_entropy.cpp" \
  -o "$BUILD_DIR/tx_partition_test"

if [ "$RUN" -eq 1 ]; then
  "${BUILD_DIR}/tx_partition_test" ${ARGS[@]+"${ARGS[@]}"}
fi
