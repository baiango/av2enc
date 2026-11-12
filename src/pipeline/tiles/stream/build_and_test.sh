#!/usr/bin/env bash
#
# Build and run the M1 tiled still-writer tests.
#
#   ./build_and_test.sh
#   CXX=g++-13 ./build_and_test.sh
#   SANITIZE=1 ./build_and_test.sh
#   WERROR=1 ./build_and_test.sh
#
# No CMake and no CUDA. The module links the public CPU M1 reference plus the
# geometry/tile-group planners; this script compiles those core translation
# units with -w (they carry pre-existing conversion warnings) and this
# directory's own sources with the project warning set.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
CXX="${CXX:-c++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"

BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/av2enc-m1-tiles-stream.XXXXXX")"
trap 'rm -rf "${BUILD_DIR}"' EXIT

COMMON=(-std=c++20 -ffp-contract=off -I"$ROOT/src")
STRICT=(-Wall -Wextra -Wpedantic -Wconversion -Wshadow)
if [[ "${SANITIZE}" == "1" ]]; then
  COMMON+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  COMMON+=(-O2)
fi
if [[ "${WERROR}" == "1" ]]; then
  STRICT+=(-Werror)
fi

# CPU reference pieces used by encode_still_picture(). Keep this list in
# sync with src/pipeline/planner/gate_dav2d_tiled.sh plus the restoration
# sources the current M1 core links.
CORE=(
  "$ROOT/src/core/bitstream/bitwriter.cpp"
  "$ROOT/src/core/bitstream/obu.cpp"
  "$ROOT/src/core/bitstream/basic_headers.cpp"
  "$ROOT/src/codec/partition/tree/partition.cpp"
  "$ROOT/src/codec/intra/modes/intra_modes.cpp"
  "$ROOT/src/codec/transform/variants/tx_variants.cpp"
  "$ROOT/src/codec/transform/variants/tx_select.cpp"
  "$ROOT/src/codec/intra/cfl/cfl.cpp"
  "$ROOT/src/codec/filters/deblock/deblock.cpp"
  "$ROOT/src/codec/filters/cdef/cdef.cpp"
  "$ROOT/src/codec/filters/loop_restoration/restoration.cpp"
  "$ROOT/src/pipeline/still/still_headers.cpp"
  "$ROOT/src/pipeline/still/still_syntax.cpp"
  "$ROOT/src/pipeline/still/still_encoder.cpp"
  "$ROOT/src/cuda/pipeline/still/cdef/still_cdef.cpp"
  "$ROOT/src/codec/rate/rdo/rdo.cpp"
  "$ROOT/src/codec/rate/rdo/intra_decision.cpp"
  "$ROOT/src/codec/transform/tx_type/tx_type.cpp"
  "$ROOT/src/codec/rate/rdoq/rdoq.cpp"
  "$ROOT/src/codec/rate/rdoq/rdoq_av2_model.cpp"
  "$ROOT/src/pipeline/tiles/tile_layout.cpp"
  "$ROOT/src/pipeline/tiles/tile_group.cpp"
  "$ROOT/src/codec/entropy/av2/av2_entropy.cpp"
)

echo "compiler: ${CXX}"
"${CXX}" --version 2>/dev/null | head -n 1 || true
echo "flags:    ${COMMON[*]} ${STRICT[*]}"
echo

OBJECTS=()
index=0
for source in "${CORE[@]}"; do
  object="${BUILD_DIR}/core_${index}_$(basename "${source}" .cpp).o"
  "${CXX}" "${COMMON[@]}" -w -c "${source}" -o "${object}"
  OBJECTS+=("${object}")
  index=$((index + 1))
done

for source in tiles_stream.cpp tiles_stream_test.cpp; do
  object="${BUILD_DIR}/$(basename "${source}" .cpp).o"
  "${CXX}" "${COMMON[@]}" "${STRICT[@]}" -c "$HERE/${source}" -o "${object}"
  OBJECTS+=("${object}")
done

BIN="${BUILD_DIR}/tiles_stream_test"
"${CXX}" "${COMMON[@]}" "${OBJECTS[@]}" -o "${BIN}"

echo "running: ${BIN}"
echo
"${BIN}"
