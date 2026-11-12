#!/usr/bin/env bash
# Build the standalone plan dump tool (JSON / TSV).
#
#   ./src/cuda/pipeline/still/plan/build_plan_dump.sh
#   OUT=/tmp/plan_dump ./src/cuda/pipeline/still/plan/build_plan_dump.sh
#
# The dump tool reads the first frame of a Y4M or generates a deterministic
# synthetic plane.  It is the bridge between this host plan and the CPU
# AV2ENC_STILL_TRACE=1 block order; compare_trace.py parses both.

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
CXX="${CXX:-c++}"
OUT="${OUT:-${TMPDIR:-/tmp}/still_plan_dump}"

"${CXX}" -std=c++20 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow \
  -ffp-contract=off -I"${ROOT}/src" \
  "${HERE}/plan.cpp" \
  "${HERE}/plan_dump.cpp" \
  "${ROOT}/src/core/io/y4m.cpp" \
  "${ROOT}/src/codec/partition/tree/partition.cpp" \
  "${ROOT}/src/codec/transform/variants/tx_variants.cpp" \
  -o "${OUT}"

echo "built ${OUT}"
