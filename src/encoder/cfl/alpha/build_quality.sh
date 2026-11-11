#!/usr/bin/env bash
# Build the prediction-level derived-vs-explicit quality tool.
#
#   ./build_quality.sh
#   ./build_quality.sh testdata/shine_256x256_1f.y4m 8
#
# The second form builds and runs. Without arguments it only builds; the tool
# then prints its own usage.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CXX="${CXX:-c++}"
OUT="${OUT:-${TMPDIR:-/tmp}/cfl_alpha_quality}"

"${CXX}" -std=c++20 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow \
  -I"${HERE}" \
  "${HERE}/alpha_quality.cpp" \
  "${HERE}/alpha.cpp" \
  "${HERE}/syntax.cpp" \
  "${HERE}/../cfl.cpp" \
  "${HERE}/../../../entropy/av2/av2_entropy.cpp" \
  -o "${OUT}"

if [[ "$#" -gt 0 ]]; then
  "${OUT}" "$@"
fi
