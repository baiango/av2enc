#!/usr/bin/env bash
# Build and run the M1 MRL syntax / interaction / minimal-stream tests.
#
#   ./build_and_test.sh
#   CXX=g++-13 ./build_and_test.sh
#   SANITIZE=1 WERROR=1 ./build_and_test.sh
#   SKIP_GATE=1 ./build_and_test.sh        # unit tests only
#   OUT=/tmp/mrl_stream_test ./build_and_test.sh
#   ./build_and_test.sh --dump-goldens     # print pinned bit goldens, no run
#
# Host-only CPU correctness oracle.  No CUDA, no CMake edits.  Our own files
# are compiled with -Werror when WERROR=1; the shared core is compiled without
# it because src/codec/intra/mrl/core/mrl.cpp has pre-existing clang sign-conversion
# warnings.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
CXX="${CXX:-c++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
OUT="${OUT:-${TMPDIR:-/tmp}/mrl_stream_test}"

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
       -I"${ROOT}/src")
# still_cdf_tables.h / still_tables.h use flat 2-D initialisers; silence that one
# third-party warning for our translation units.
FLAGS+=(-Wno-missing-braces)

if [[ "${SANITIZE}" == "1" ]]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2)
fi

OUR_FLAGS=("${FLAGS[@]}")
if [[ "${WERROR}" == "1" ]]; then
  OUR_FLAGS+=(-Werror)
fi

OBJ="${OUT}.obj"
rm -rf "${OBJ}"
mkdir -p "${OBJ}"

CORE=(
  "${ROOT}/src/core/bitstream/bitwriter.cpp"
  "${ROOT}/src/core/bitstream/obu.cpp"
  "${ROOT}/src/core/bitstream/basic_headers.cpp"
  "${ROOT}/src/codec/partition/tree/partition.cpp"
  "${ROOT}/src/codec/partition/ext/partition_ext.cpp"
  "${ROOT}/src/codec/partition/ext/ext_syntax.cpp"
  "${ROOT}/src/codec/intra/modes/intra_modes.cpp"
  "${ROOT}/src/codec/transform/variants/tx_variants.cpp"
  "${ROOT}/src/codec/transform/tx_type/tx_type.cpp"
  "${ROOT}/src/codec/intra/mrl/core/mrl.cpp"
  "${ROOT}/src/pipeline/still/still_headers.cpp"
  "${ROOT}/src/codec/partition/stream/partition_stream.cpp"
  "${ROOT}/src/codec/entropy/av2/av2_entropy.cpp"
)

MRL=(
  "${HERE}/mrl_syntax.cpp"
  "${HERE}/mrl_interaction.cpp"
  "${HERE}/mrl_stream.cpp"
)

OBJECTS=()
for src in "${MRL[@]}"; do
  obj="${OBJ}/$(basename "${src%.cpp}").o"
  "${CXX}" "${OUR_FLAGS[@]}" -c "${src}" -o "${obj}"
  OBJECTS+=("${obj}")
done
for src in "${CORE[@]}"; do
  obj="${OBJ}/$(basename "${src%.cpp}").o"
  "${CXX}" "${FLAGS[@]}" -c "${src}" -o "${obj}"
  OBJECTS+=("${obj}")
done

TEST_OBJ="${OBJ}/mrl_stream_test.o"
"${CXX}" "${OUR_FLAGS[@]}" -c "${HERE}/mrl_stream_test.cpp" -o "${TEST_OBJ}"
"${CXX}" "${FLAGS[@]}" "${OBJECTS[@]}" "${TEST_OBJ}" -o "${OUT}"

if [[ "${1:-}" == "--dump-goldens" ]]; then
  "${OUT}" --dump-goldens
  exit 0
fi

"${OUT}"

if [[ "${SKIP_GATE:-0}" != "0" ]]; then
  exit 0
fi

echo "== minimal M1 + MRL dav2d gate =="
GATE_DIR="${OUT}_gate_out"
GATE_OBJ="${OBJ}/mrl_gate.o"
IVF_OBJ="${OBJ}/ivf.o"
"${CXX}" "${FLAGS[@]}" -c "${ROOT}/src/core/io/ivf.cpp" -o "${IVF_OBJ}"
"${CXX}" "${OUR_FLAGS[@]}" -c "${HERE}/mrl_gate.cpp" -o "${GATE_OBJ}"
"${CXX}" "${FLAGS[@]}" "${OBJECTS[@]}" "${GATE_OBJ}" "${IVF_OBJ}" \
  -o "${OUT}_gate"
"${OUT}_gate" --out "${GATE_DIR}"
python3 "${HERE}/gate_dav2d_mrl.py" --dir "${GATE_DIR}"
