#!/usr/bin/env bash
# Live CPU-trace parity check for the M1 frame plan (leaves + preorder tree).
#
#   AV2ENC_BIN=/workspace/av2enc/build-plan-m1/av2enc \
#     ./src/cuda/pipeline/still/plan/verify_trace.sh
#
# Builds the dump tool if needed, then for tests/data/basic_64x64_1f.y4m and
# tests/data/basic_640x360_1f.y4m runs the CPU M1 reference with
# AV2ENC_STILL_TRACE=1 and compares:
#   * plan leaf order        vs "m1 block" lines
#   * plan preorder tree     vs both "m1 node" passes
#
# The unit test `plan_test.cpp` embeds the same CPU trace signatures as
# tests/trace_order_*.txt so this can also run without the CPU binary.

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
AV2ENC_BIN="${AV2ENC_BIN:-${ROOT}/build-m1/av2enc}"
PLAN_DUMP="${PLAN_DUMP:-${TMPDIR:-/tmp}/still_plan_dump}"
WORK="${WORK:-$(mktemp -d "${TMPDIR:-/tmp}/still_plan_trace.XXXXXX")}"

if [[ ! -x "${AV2ENC_BIN}" ]]; then
  echo "verify_trace: AV2ENC_BIN is not executable: ${AV2ENC_BIN}" >&2
  echo "  set AV2ENC_BIN to a CPU M1 build (e.g. build-plan-m1/av2enc)" >&2
  exit 2
fi
if [[ ! -x "${PLAN_DUMP}" ]]; then
  OUT="${PLAN_DUMP}" "${HERE}/build_plan_dump.sh" >/dev/null
fi

for name in basic_64x64_1f basic_640x360_1f; do
  src="${ROOT}/tests/data/${name}.y4m"
  if [[ ! -f "${src}" ]]; then
    echo "verify_trace: missing ${src}" >&2
    exit 2
  fi
  echo "=== ${name}"
  "${PLAN_DUMP}" --print-options
  AV2ENC_STILL_TRACE=1 "${AV2ENC_BIN}" "${src}" \
    -o "${WORK}/${name}.ivf" --still --frames 1 \
    2> "${WORK}/${name}.trace" >/dev/null
  "${PLAN_DUMP}" -i "${src}" --tsv > "${WORK}/${name}.blocks.tsv"
  "${PLAN_DUMP}" -i "${src}" --tree --tsv > "${WORK}/${name}.tree.tsv"
  python3 "${HERE}/compare_trace.py" "${WORK}/${name}.blocks.tsv" \
    "${WORK}/${name}.trace"
  python3 "${HERE}/compare_trace.py" --nodes "${WORK}/${name}.tree.tsv" \
    "${WORK}/${name}.trace"
done

echo "verify_trace: all order checks passed (work dir ${WORK})"
