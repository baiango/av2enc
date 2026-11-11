#!/usr/bin/env bash
# Build/run the DC/AC delta-Q unit tests.
#
#   ./build_and_test.sh
#   WERROR=1 ./build_and_test.sh
#   SANITIZE=1 ./build_and_test.sh
#   BUILD_DIR=/tmp/dcdq ./build_and_test.sh --no-run
set -euo pipefail
cd "$(dirname "$0")"

ROOT="$(cd ../../../.. && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-dcdeltaq}"
RUN=1
for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$BUILD_DIR"
CXX="${CXX:-c++}"
FLAGS=(-std=c++20 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
       -I"$ROOT/src")
if [ "${WERROR:-0}" -ne 0 ]; then FLAGS+=(-Werror); fi
if [ "${SANITIZE:-0}" -ne 0 ]; then
  FLAGS+=(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
fi

"$CXX" "${FLAGS[@]}" \
  dcdeltaq_test.cpp syntax.cpp quant.cpp search.cpp \
  "$ROOT/src/bitstream/bitwriter.cpp" \
  "$ROOT/src/encoder/qm/qm.cpp" \
  "$ROOT/src/bitstream/m0_headers.cpp" \
  "$ROOT/src/encoder/tx_variants/tx_variants.cpp" \
  -o "$BUILD_DIR/dcdeltaq_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/dcdeltaq_test"
fi
