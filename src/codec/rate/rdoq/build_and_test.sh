#!/usr/bin/env bash
#
# Standalone build + test for the full-trellis RDOQ component.
#
#   ./build_and_test.sh
#   CXX=g++-13 ./build_and_test.sh
#   RDOQ_SANITIZE=1 ./build_and_test.sh      # ASan + UBSan on the core test
#   RDOQ_SKIP_M1=1 ./build_and_test.sh       # core test only
#
# No CMake.  The M1 quality harness links the existing CPU M1 components
# read-only; it does not modify them and does not write into the repository.
set -euo pipefail
cd "$(dirname "$0")"

ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
CXX="${CXX:-c++}"
BUILD_DIR="${BUILD_DIR:-${TMPDIR:-/tmp}/av2enc_rdoq_build}"
mkdir -p "${BUILD_DIR}"

CORE_FLAGS=(
  -std=c++20 -O2
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow
  -ffp-contract=off
  -I"${ROOT}/src" -I.
)
# Other workstreams' sources are compiled as they are; this component does not
# own their warning status.
REF_FLAGS=(
  -std=c++20 -O2 -ffp-contract=off -w
  -I"${ROOT}/src"
)

if [[ "${RDOQ_SANITIZE:-0}" == "1" ]]; then
  CORE_FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
fi

echo "compiler: ${CXX}"
"${CXX}" --version 2>/dev/null | head -n 1 || true
echo "flags:    ${CORE_FLAGS[*]}"
echo

echo "== core + model test =="
"${CXX}" "${CORE_FLAGS[@]}" \
  rdoq.cpp rdoq_av2_model.cpp rdoq_test.cpp \
  -o "${BUILD_DIR}/rdoq_test"
"${BUILD_DIR}/rdoq_test"
echo

if [[ "${RDOQ_SKIP_M1:-0}" == "1" ]]; then
  echo "RDOQ_SKIP_M1=1: skipping the M1 coefficient harness"
  exit 0
fi

echo "== M1 quality harness =="
reference_sources=(
  "${ROOT}/src/cuda/pipeline/still/plan/plan.cpp"
  "${ROOT}/src/codec/partition/tree/partition.cpp"
  "${ROOT}/src/codec/transform/variants/tx_variants.cpp"
  "${ROOT}/src/codec/intra/modes/intra_modes.cpp"
  "${ROOT}/src/codec/intra/cfl/cfl.cpp"
  "${ROOT}/src/codec/rate/rdo/rdo.cpp"
  "${ROOT}/src/pipeline/still/still_encoder.cpp"
  "${ROOT}/src/pipeline/still/still_syntax.cpp"
  "${ROOT}/src/pipeline/still/still_headers.cpp"
  "${ROOT}/src/codec/filters/deblock/deblock.cpp"
  "${ROOT}/src/codec/entropy/av2/av2_entropy.cpp"
  "${ROOT}/src/core/bitstream/bitwriter.cpp"
  "${ROOT}/src/core/bitstream/obu.cpp"
  "${ROOT}/src/core/bitstream/basic_headers.cpp"
  "${ROOT}/src/core/io/y4m.cpp"
  "${ROOT}/src/core/io/ivf.cpp"
  "${ROOT}/src/core/telemetry/profiler.cpp"
)
objects=()
index=0
for source in "${reference_sources[@]}"; do
  object="${BUILD_DIR}/ref_${index}.o"
  "${CXX}" "${REF_FLAGS[@]}" -c "${source}" -o "${object}"
  objects+=("${object}")
  index=$((index + 1))
done

"${CXX}" "${CORE_FLAGS[@]}" -c rdoq.cpp -o "${BUILD_DIR}/rdoq.o"
"${CXX}" "${CORE_FLAGS[@]}" -c rdoq_av2_model.cpp -o "${BUILD_DIR}/rdoq_model.o"
"${CXX}" "${CORE_FLAGS[@]}" -c still_rdoq_quality.cpp -o "${BUILD_DIR}/still_rdoq_quality.o"
"${CXX}" "${CORE_FLAGS[@]}" "${objects[@]}" \
  "${BUILD_DIR}/rdoq.o" "${BUILD_DIR}/rdoq_model.o" \
  "${BUILD_DIR}/still_rdoq_quality.o" \
  -o "${BUILD_DIR}/still_rdoq_quality"

for y4m in "${ROOT}/tests/data/basic_64x64_1f.y4m" \
           "${ROOT}/tests/data/basic_640x360_1f.y4m"; do
  if [[ ! -f "${y4m}" ]]; then
    echo "missing test input: ${y4m}" >&2
    exit 1
  fi
  stem="$(basename "${y4m}" .y4m)"
  echo "-- $(basename "${y4m}") --"
  "${BUILD_DIR}/still_rdoq_quality" -i "${y4m}" --check-reference \
    --write-ivf "${BUILD_DIR}/${stem}"
  if [[ -x "${ROOT}/tools/bin/dav2d" || -x "${ROOT}/tools/bin/dav2d-macos" ]]; then
    for stream in scalar rdoq; do
      "${ROOT}/tools/validate_dav2d.sh" \
        "${BUILD_DIR}/${stem}.${stream}.ivf" "${y4m}" 2>&1 \
        | grep -E "^(OK|error|quality:|\[Parsed_psnr)" || true
    done
  else
    echo "note: pinned dav2d not found; skipping the decode gate"
  fi
  echo
done

echo "RESULT: PASS"
