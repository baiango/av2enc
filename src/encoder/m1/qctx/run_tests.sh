#!/usr/bin/env bash
# Standalone qctx test driver.  Deliberately does not touch CMakeLists.txt:
# this module owns only src/encoder/m1/qctx/.
#
# Runs:
#   1. generator --check (header freshness + slice-0 byte compatibility)
#   2. qctx_test (selection, every slice/context, slice-0 vs shipping tables)
#   3. qctx_decode_gate (normative §8.2 encode -> decode round trip)
#
# Usage: src/encoder/m1/qctx/run_tests.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../../.." && pwd)"
CXX="${CXX:-c++}"
CXXFLAGS=(-std=c++20 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
INCLUDES=(-I"${ROOT}/src")

TMP="$(mktemp -d "${TMPDIR:-/tmp}/qctx-tests.XXXXXX")"
trap 'rm -rf "${TMP}"' EXIT

echo "== generator --check =="
python3 "${HERE}/generate_qctx_tables.py" --check

echo "== qctx_test =="
"${CXX}" "${CXXFLAGS[@]}" "${INCLUDES[@]}" \
    "${HERE}/qctx_test.cpp" \
    "${ROOT}/src/entropy/av2/av2_entropy.cpp" \
    -o "${TMP}/qctx_test"
"${TMP}/qctx_test"

echo "== qctx_decode_gate =="
"${CXX}" "${CXXFLAGS[@]}" "${INCLUDES[@]}" \
    "${HERE}/qctx_decode_gate.cpp" \
    "${ROOT}/src/entropy/av2/av2_entropy.cpp" \
    -o "${TMP}/qctx_decode_gate"
"${TMP}/qctx_decode_gate"

if [[ "${QCTX_SKIP_END_TO_END:-0}" != "1" ]]; then
  echo "== end-to-end dav2d decode gate =="
  set +e
  bash "${HERE}/decode_gate.sh"
  status=$?
  set -e
  if [[ ${status} -eq 77 ]]; then
    echo "SKIP: end-to-end gate (no encoder/decoder/test media)"
  elif [[ ${status} -ne 0 ]]; then
    exit ${status}
  fi
fi
