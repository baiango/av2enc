#!/usr/bin/env bash
#
# Standalone build + self-test for the unified M1 RDO loop policy layer.
#
#   ./build_and_test.sh
#   SANITIZE=1 ./build_and_test.sh
#   WERROR=1 ./build_and_test.sh
#   CXX=clang++ ./build_and_test.sh
#
# No CMake and no CUDA: this is host-only reference code.

set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../../.." && pwd)
CXX=${CXX:-c++}
BUILD=${BUILD_DIR:-"$ROOT/build-rdo-loop-tests"}
mkdir -p "$BUILD"

COMMON=(-std=c++20 -O2 -I"$ROOT/src" -ffp-contract=off)
WARN=(-Wall -Wextra -Wpedantic -Wconversion -Wshadow)
if [ "${WERROR:-0}" = "1" ]; then
  WARN+=(-Werror)
fi
SAN=()
if [ "${SANITIZE:-0}" = "1" ]; then
  SAN=(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
fi

SOURCES=(
  "$HERE/config.cpp"
  "$HERE/cost.cpp"
  "$HERE/search.cpp"
  "$HERE/batch.cpp"
  "$HERE/activity.cpp"
  "$HERE/rdo_loop_test.cpp"
  "$ROOT/src/encoder/partition/partition.cpp"
  "$ROOT/src/encoder/partition_ext/partition_ext.cpp"
  "$ROOT/src/encoder/rdo/rdo.cpp"
)

OBJECTS=()
for source in "${SOURCES[@]}"; do
  object="$BUILD/$(basename "${source%.cpp}").o"
  "$CXX" "${COMMON[@]}" "${WARN[@]}" ${SAN[@]+"${SAN[@]}"} -c "$source" -o "$object"
  OBJECTS+=("$object")
done

"$CXX" ${SAN[@]+"${SAN[@]}"} "${OBJECTS[@]}" -o "$BUILD/rdo_loop_test"
"$BUILD/rdo_loop_test"
