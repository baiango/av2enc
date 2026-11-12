#!/usr/bin/env bash
# Standalone build + test + decoder gate for the AV2 lossless inter module.
#
# The module owns only src/codec/entropy/lossless/inter/.  It is deliberately not
# wired into CMakeLists.txt or the CLI by this workstream.
#
# Usage:
#   ./build_and_test.sh
#   SANITIZE=1 WERROR=1 ./build_and_test.sh
#   GATE_REQUIRED=1 ./build_and_test.sh
#
# Environment:
#   DAV2D    path to a dav2d decoder (default: tools/bin/dav2d[-macos])
#   BUILD_DIR directory for objects (default: fresh mktemp dir, removed)
#   KEEP=1   keep the build dir and gate media
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
CXX="${CXX:-c++}"
WARN="-Wall -Wextra -Wpedantic -Wshadow -Wno-sign-conversion -Wno-missing-braces"
COMMON=(-std=c++20 -isystem "${ROOT}/src" -ffp-contract=off ${WARN})

if [[ "${WERROR:-0}" == "1" ]]; then
  COMMON+=(-Werror)
fi
if [[ "${SANITIZE:-0}" == "1" ]]; then
  COMMON+=(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
fi

SOURCES=(
  "${HERE}/lossless_inter.cpp"
  "${ROOT}/src/core/bitstream/bitwriter.cpp"
  "${ROOT}/src/codec/entropy/av2/av2_entropy.cpp"
  "${ROOT}/src/codec/intra/modes/intra_modes.cpp"
  "${ROOT}/src/codec/partition/tree/partition.cpp"
)

if [[ -z "${BUILD_DIR:-}" ]]; then
  BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/av2enc-lossless-inter.XXXXXX")"
  CLEAN_BUILD=1
else
  CLEAN_BUILD=0
  mkdir -p "${BUILD_DIR}"
fi

cleanup() {
  if [[ "${CLEAN_BUILD}" == "1" && "${KEEP:-0}" != "1" ]]; then
    rm -rf "${BUILD_DIR}"
  fi
}
trap cleanup EXIT

echo "== building lossless_inter_test =="
"${CXX}" "${COMMON[@]}" -O1 \
  "${HERE}/lossless_inter_test.cpp" "${SOURCES[@]}" \
  -o "${BUILD_DIR}/lossless_inter_test"

echo "== running lossless_inter_test =="
"${BUILD_DIR}/lossless_inter_test"

echo "== building lossless_inter_gate =="
"${CXX}" "${COMMON[@]}" -O2 \
  "${HERE}/lossless_inter_gate.cpp" "${SOURCES[@]}" "${ROOT}/src/core/io/ivf.cpp" \
  -o "${BUILD_DIR}/lossless_inter_gate"

DAV2D="${DAV2D:-}"
if [[ -z "${DAV2D}" ]]; then
  for candidate in "${ROOT}/tools/bin/dav2d" "${ROOT}/tools/bin/dav2d-macos"; do
    if [[ -x "${candidate}" ]]; then DAV2D="${candidate}"; break; fi
  done
fi

GATE_ARGS=()
[[ -n "${DAV2D}" ]] && GATE_ARGS+=(--dav2d "${DAV2D}")
[[ "${KEEP:-0}" == "1" ]] && GATE_ARGS+=(--keep)

if [[ ${#GATE_ARGS[@]} -eq 0 ]]; then
  echo "SKIP: no dav2d decoder found (set DAV2D)"
  if [[ "${GATE_REQUIRED:-0}" == "1" ]]; then
    exit 1
  fi
  exit 0
fi

echo "== running dav2d --strict 1 --threads 1 gate =="
set +e
"${BUILD_DIR}/lossless_inter_gate" "${GATE_ARGS[@]}"
status=$?
set -e
if [[ ${status} -eq 77 ]]; then
  echo "SKIP: decoder gate (no usable decoder)"
  if [[ "${GATE_REQUIRED:-0}" == "1" ]]; then
    exit 1
  fi
  exit 0
fi
if [[ ${status} -ne 0 ]]; then
  exit "${status}"
fi

echo "PASS: lossless inter CPU tests + dav2d strict gate"
