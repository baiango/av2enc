#!/usr/bin/env bash
# Standalone CPU build + test for the AV2 lossless / DPCM / FSC module.
#
# The module is intentionally not wired into CMakeLists.txt by this workstream
# (another agent owns the core build).  This script compiles the same sources
# the core library would compile and runs:
#   1. the CPU reference tests (flags, reversible transforms, full round trips)
#   2. the external decoder gate when a pinned dav2d or avmdec is available
#
# Usage:
#   ./build_and_test.sh
#   SANITIZE=1 WERROR=1 ./build_and_test.sh
#   CXX=g++-13 ./build_and_test.sh
#   GATE_REQUIRED=1 ./build_and_test.sh
#
# Environment:
#   DAV2D    path to a dav2d decoder (default: tools/bin/dav2d[-macos])
#   AVMDEC   path to an avmdec decoder (default: tools/bin/avm-macos/avmdec)
#   BUILD_DIR directory for objects (default: a fresh mktemp dir, removed)
#   KEEP=1   keep BUILD_DIR and gate media
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"
CXX="${CXX:-c++}"
WARN="-Wall -Wextra -Wpedantic -Wshadow"
# The module is not yet in the core CMake target.  Keep the same warning set
# but do not fail on sign-conversion from the generated table headers / vector
# indexing.  The two suppressions below are noise, not semantics.
WARN="${WARN} -Wno-sign-conversion -Wno-missing-braces"
COMMON=(-std=c++20 -isystem "${ROOT}/src" -ffp-contract=off ${WARN})

if [[ "${WERROR:-0}" == "1" ]]; then
  COMMON+=(-Werror)
fi
if [[ "${SANITIZE:-0}" == "1" ]]; then
  COMMON+=(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
fi

SOURCES=(
  "${ROOT}/src/encoder/lossless/lossless.cpp"
  "${ROOT}/src/bitstream/bitwriter.cpp"
  "${ROOT}/src/entropy/av2/av2_entropy.cpp"
  "${ROOT}/src/encoder/intra_modes/intra_modes.cpp"
  "${ROOT}/src/encoder/partition/partition.cpp"
)

if [[ -z "${BUILD_DIR:-}" ]]; then
  BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/av2enc-lossless.XXXXXX")"
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

echo "== building lossless_test =="
"${CXX}" "${COMMON[@]}" -O1 \
  "${HERE}/lossless_test.cpp" "${SOURCES[@]}" \
  -o "${BUILD_DIR}/lossless_test"

echo "== running lossless_test =="
"${BUILD_DIR}/lossless_test"

echo "== building lossless_gate =="
"${CXX}" "${COMMON[@]}" -O2 \
  "${HERE}/lossless_gate.cpp" "${SOURCES[@]}" "${ROOT}/src/io/ivf.cpp" \
  "${ROOT}/src/io/y4m.cpp" -o "${BUILD_DIR}/lossless_gate"

DAV2D="${DAV2D:-}"
if [[ -z "${DAV2D}" ]]; then
  for candidate in "${ROOT}/tools/bin/dav2d" "${ROOT}/tools/bin/dav2d-macos"; do
    if [[ -x "${candidate}" ]]; then DAV2D="${candidate}"; break; fi
  done
fi
AVMDEC="${AVMDEC:-}"
if [[ -z "${AVMDEC}" ]]; then
  if [[ -x "${ROOT}/tools/bin/avm-macos/avmdec" ]]; then
    AVMDEC="${ROOT}/tools/bin/avm-macos/avmdec"
  fi
fi

GATE_ARGS=()
[[ -n "${DAV2D}" ]] && GATE_ARGS+=(--dav2d "${DAV2D}")
[[ -n "${AVMDEC}" ]] && GATE_ARGS+=(--avmdec "${AVMDEC}")
Y4M="${Y4M:-${ROOT}/tests/data/m0_64x64_1f.y4m}"
[[ -f "${Y4M}" ]] && GATE_ARGS+=(--y4m "${Y4M}")
[[ "${KEEP:-0}" == "1" ]] && GATE_ARGS+=(--keep)

if [[ ${#GATE_ARGS[@]} -eq 0 ]]; then
  echo "SKIP: no external decoder found (set DAV2D / AVMDEC)"
  if [[ "${GATE_REQUIRED:-0}" == "1" ]]; then
    exit 1
  fi
  exit 0
fi

echo "== running external decoder gate =="
set +e
"${BUILD_DIR}/lossless_gate" "${GATE_ARGS[@]}"
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

echo "PASS: lossless CPU tests + external decoder gate"
