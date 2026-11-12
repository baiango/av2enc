#!/usr/bin/env bash
# Standalone build + tests for src/codec/intra/cfl/ds_filters/.
#
#   ./src/codec/intra/cfl/ds_filters/build_and_test.sh
#   SANITIZE=1 WERROR=1 ./.../build_and_test.sh
#   GATE=1 ./.../build_and_test.sh          # also run the dav2d gate
#   QUALITY=1 ./.../build_and_test.sh       # also run 640x360/720p deltas
#   CUDA=1 ./.../build_and_test.sh          # also build/run the CUDA seam test
#   TESTDATA=/path/to/frame.y4m ./.../build_and_test.sh
#
# The CPU reference binary is the product-independent oracle; the CUDA test is
# skipped on hosts without nvcc.

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
OUT="${OUT:-$HERE/build}"
CXX="${CXX:-c++}"
TESTDATA="${TESTDATA:-$ROOT/testdata/shine_1280x720_60f.y4m}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
GATE="${GATE:-0}"
QUALITY="${QUALITY:-0}"
CUDA="${CUDA:-0}"

mkdir -p "$OUT"

FLAGS=(-std=c++20 -I"$ROOT/src" -DAV2ENC_TESTDATA="\"$TESTDATA\"")
WARN=(-Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wno-missing-braces)
if [ "$SANITIZE" = "1" ]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2)
fi
if [ "$WERROR" = "1" ]; then
  WARN+=(-Werror)
fi

echo "== building ds_filters_test"
"$CXX" "${FLAGS[@]}" "${WARN[@]}" \
  "$HERE/ds_filters.cpp" "$HERE/ds_filters_test.cpp" \
  "$ROOT/src/codec/intra/cfl/cfl.cpp" -o "$OUT/ds_filters_test"
echo "== running ds_filters_test"
"$OUT/ds_filters_test"

if [ "$CUDA" = "1" ]; then
  echo "== CUDA seam test"
  "$HERE/build_cuda_test.sh"
fi

if [ "$GATE" = "1" ]; then
  echo "== decoder gate (non-zero cfl_ds_filter_index)"
  "$HERE/gate/check_gate.sh"
fi

if [ "$QUALITY" = "1" ]; then
  echo "== quality/rate per filter"
  "$HERE/quality/quality_per_filter.sh"
fi
