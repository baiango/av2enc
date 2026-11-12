#!/usr/bin/env bash
#
# Standalone build + test for the AV2 TCQ / CCTX component.
#
#   ./build_and_test.sh
#   CXX=g++-13 ./build_and_test.sh
#   TCQ_SANITIZE=1 ./build_and_test.sh      # ASan + UBSan on the core tests
#   TCQ_QPS="60 90 120" ./build_and_test.sh # RD harness QP sweep
#   TCQ_SKIP_RD=1 ./build_and_test.sh       # unit/model tests only
#
# No CMake and no writes into the repository.  Existing components (tx_variants,
# rdoq, rdo, y4m) are compiled read-only as the RD reference.
set -euo pipefail
cd "$(dirname "$0")"

ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
CXX="${CXX:-c++}"
BUILD_DIR="${BUILD_DIR:-${TMPDIR:-/tmp}/av2enc_tcq_build}"
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

if [[ "${TCQ_SANITIZE:-0}" == "1" ]]; then
  CORE_FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
fi

echo "compiler: ${CXX}"
"${CXX}" --version 2>/dev/null | head -n 1 || true
echo "flags:    ${CORE_FLAGS[*]}"
echo

echo "== core TCQ / CCTX / header tests =="
"${CXX}" "${CORE_FLAGS[@]}" \
  tcq.cpp cctx.cpp tcq_flags.cpp tcq_test.cpp \
  -o "${BUILD_DIR}/tcq_test"
"${BUILD_DIR}/tcq_test"
echo

echo "== TCQ rate-model tests =="
"${CXX}" "${CORE_FLAGS[@]}" \
  tcq.cpp tcq_av2_model.cpp tcq_model_test.cpp \
  "${ROOT}/src/codec/transform/variants/tx_variants.cpp" \
  -o "${BUILD_DIR}/tcq_model_test"
"${BUILD_DIR}/tcq_model_test"
echo

if [[ "${TCQ_SKIP_RD:-0}" == "1" ]]; then
  echo "TCQ_SKIP_RD=1: skipping the real-coefficient RD harness"
  echo "RESULT: PASS"
  exit 0
fi

echo "== real-coefficient RD harness =="
reference_sources=(
  "${ROOT}/src/codec/transform/variants/tx_variants.cpp"
  "${ROOT}/src/codec/rate/rdoq/rdoq.cpp"
  "${ROOT}/src/codec/rate/rdoq/rdoq_av2_model.cpp"
  "${ROOT}/src/codec/rate/rdo/rdo.cpp"
  "${ROOT}/src/core/io/y4m.cpp"
)
objects=()
index=0
for source in "${reference_sources[@]}"; do
  object="${BUILD_DIR}/ref_${index}.o"
  "${CXX}" "${REF_FLAGS[@]}" -c "${source}" -o "${object}"
  objects+=("${object}")
  index=$((index + 1))
done

"${CXX}" "${CORE_FLAGS[@]}" -c tcq.cpp -o "${BUILD_DIR}/tcq.o"
"${CXX}" "${CORE_FLAGS[@]}" -c cctx.cpp -o "${BUILD_DIR}/cctx.o"
"${CXX}" "${CORE_FLAGS[@]}" -c tcq_flags.cpp -o "${BUILD_DIR}/tcq_flags.o"
"${CXX}" "${CORE_FLAGS[@]}" -c tcq_av2_model.cpp -o "${BUILD_DIR}/tcq_model.o"
"${CXX}" "${CORE_FLAGS[@]}" -w -c tcq_rd_compare.cpp -o "${BUILD_DIR}/tcq_rd.o"
"${CXX}" "${CORE_FLAGS[@]}" "${objects[@]}" \
  "${BUILD_DIR}/tcq.o" "${BUILD_DIR}/cctx.o" "${BUILD_DIR}/tcq_flags.o" \
  "${BUILD_DIR}/tcq_model.o" "${BUILD_DIR}/tcq_rd.o" \
  -o "${BUILD_DIR}/tcq_rd_compare"

QPS="${TCQ_QPS:-90}"
for y4m in "${ROOT}/tests/data/basic_64x64_1f.y4m" \
           "${ROOT}/tests/data/basic_640x360_1f.y4m"; do
  if [[ ! -f "${y4m}" ]]; then
    echo "missing test input: ${y4m}" >&2
    exit 1
  fi
  for qp in ${QPS}; do
    echo "-- $(basename "${y4m}") qp=${qp} --"
    "${BUILD_DIR}/tcq_rd_compare" -i "${y4m}" --qp "${qp}" --block 8
    echo
  done
done

echo "note: the RD numbers are static-CDF model bits on real coefficients."
echo "note: a real dav2d TCQ/CCTX stream gate is blocked by core files that"
echo "      hardcode enable_tcq/enable_cctx to 0; see README 'Integration'."
echo "RESULT: PASS"
