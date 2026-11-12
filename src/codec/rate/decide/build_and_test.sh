#!/usr/bin/env bash
#
# Standalone build + self-test for the M1 RDO decision layer.
#
#   ./build_and_test.sh
#   SANITIZE=1 ./build_and_test.sh
#   WERROR=1 ./build_and_test.sh
#   CXX=clang++ ./build_and_test.sh
#
# No CMake and no CUDA: the layer is host-only reference code.

set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
CXX=${CXX:-c++}
BUILD=${BUILD_DIR:-"$ROOT/build-rdo-decide-tests"}
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
  "$HERE/transform_stream.cpp"
  "$HERE/intra_rdo.cpp"
  "$HERE/chroma_rdo.cpp"
  "$HERE/partition_rdo.cpp"
  "$HERE/tx_rdo.cpp"
  "$HERE/rdo_decide_test.cpp"
  "$ROOT/src/codec/rate/rdo/rdo.cpp"
  "$ROOT/src/codec/rate/rdo/intra_decision.cpp"
  "$ROOT/src/codec/partition/tree/partition.cpp"
  "$ROOT/src/codec/transform/variants/tx_variants.cpp"
  "$ROOT/src/codec/transform/tx_type/tx_type.cpp"
  "$ROOT/src/codec/intra/modes/intra_modes.cpp"
)

OBJECTS=()
for source in "${SOURCES[@]}"; do
  object="$BUILD/$(basename "${source%.cpp}").o"
  "$CXX" "${COMMON[@]}" "${WARN[@]}" ${SAN[@]+"${SAN[@]}"} -c "$source" -o "$object"
  OBJECTS+=("$object")
done

"$CXX" ${SAN[@]+"${SAN[@]}"} "${OBJECTS[@]}" -o "$BUILD/rdo_decide_test"
"$BUILD/rdo_decide_test"
