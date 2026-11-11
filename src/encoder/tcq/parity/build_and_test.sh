#!/usr/bin/env bash
#
# Standalone build + test for the AV2 parity-hiding component.
#
#   ./build_and_test.sh
#   CXX=g++-13 ./build_and_test.sh
#   PARITY_SANITIZE=1 ./build_and_test.sh   # ASan + UBSan, core tests + gate
#   PARITY_SKIP_RD=1 ./build_and_test.sh    # tests + gate only
#   PARITY_QPS="60 90 120" ./build_and_test.sh
#
# No CMake and no writes into the repository.  The existing tx_variants, rdoq,
# rdo and y4m components are compiled read-only as the RD reference.
set -euo pipefail
cd "$(dirname "$0")"

ROOT="$(cd ../../../.. && pwd)"
CXX="${CXX:-c++}"
BUILD_DIR="${BUILD_DIR:-${TMPDIR:-/tmp}/av2enc_parity_build}"
mkdir -p "${BUILD_DIR}"

CORE_FLAGS=(
  -std=c++20 -O2
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow
  -ffp-contract=off
  -I"${ROOT}/src" -I.
)
REF_FLAGS=(
  -std=c++20 -O2 -ffp-contract=off -w
  -I"${ROOT}/src"
)

if [[ "${PARITY_SANITIZE:-0}" == "1" ]]; then
  CORE_FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
fi

echo "compiler: ${CXX}"
"${CXX}" --version 2>/dev/null | head -n 1 || true
echo "flags:    ${CORE_FLAGS[*]}"
echo

echo "== parity-hiding core / syntax / entropy / quantiser tests =="
"${CXX}" "${CORE_FLAGS[@]}" \
  parity_hiding.cpp parity_entropy.cpp ../tcq_flags.cpp \
  "${ROOT}/src/entropy/av2/av2_entropy.cpp" \
  parity_test.cpp \
  -o "${BUILD_DIR}/parity_test"
"${BUILD_DIR}/parity_test"
echo

echo "== coefficient-payload decoder gate (separate writer/reader) =="
GATE_FILE="${BUILD_DIR}/parity_gate.ph"
"${BUILD_DIR}/parity_test" --write-gate "${GATE_FILE}"
"${BUILD_DIR}/parity_test" --read-gate "${GATE_FILE}"
echo

if [[ "${PARITY_SKIP_RD:-0}" == "1" ]]; then
  echo "PARITY_SKIP_RD=1: skipping the real-coefficient RD harness"
  echo "RESULT: PASS"
  exit 0
fi

echo "== real-coefficient RD harness (scalar vs RDOQ vs parity hiding) =="
reference_sources=(
  "${ROOT}/src/encoder/tx_variants/tx_variants.cpp"
  "${ROOT}/src/encoder/rdoq/rdoq.cpp"
  "${ROOT}/src/encoder/rdoq/rdoq_av2_model.cpp"
  "${ROOT}/src/encoder/rdo/rdo.cpp"
  "${ROOT}/src/io/y4m.cpp"
)
objects=()
index=0
for source in "${reference_sources[@]}"; do
  object="${BUILD_DIR}/ref_${index}.o"
  "${CXX}" "${REF_FLAGS[@]}" -c "${source}" -o "${object}"
  objects+=("${object}")
  index=$((index + 1))
done
"${CXX}" "${CORE_FLAGS[@]}" -c parity_hiding.cpp -o "${BUILD_DIR}/parity_hiding.o"
"${CXX}" "${CORE_FLAGS[@]}" -w -c parity_rd_compare.cpp \
  -o "${BUILD_DIR}/parity_rd_compare.o"
"${CXX}" "${CORE_FLAGS[@]}" "${objects[@]}" \
  "${BUILD_DIR}/parity_hiding.o" "${BUILD_DIR}/parity_rd_compare.o" \
  -o "${BUILD_DIR}/parity_rd_compare"

QPS="${PARITY_QPS:-90}"
for y4m in "${ROOT}/tests/data/m0_64x64_1f.y4m" \
           "${ROOT}/tests/data/m0_640x360_1f.y4m"; do
  if [[ ! -f "${y4m}" ]]; then
    echo "missing test input: ${y4m}" >&2
    exit 1
  fi
  for qp in ${QPS}; do
    echo "-- $(basename "${y4m}") qp=${qp} block=8 --"
    "${BUILD_DIR}/parity_rd_compare" -i "${y4m}" --qp "${qp}" --block 8
    echo
  done
done

echo "note: the RD numbers are static-CDF model bits on real coefficients."
echo "note: a real dav2d parity-hiding stream is blocked by core files that"
echo "      hardcode enable_parity_hiding = 0 and by the coefficient writer;"
echo "      see README 'Decoder gate'."
echo "RESULT: PASS"
