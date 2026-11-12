#!/usr/bin/env bash
#
# Build and run the AV2 tile/frame CDF continuity tests.
#
#   ./build_and_test.sh
#   CXX=g++-13 ./build_and_test.sh
#   SANITIZE=1 ./build_and_test.sh
#   WERROR=1 ./build_and_test.sh
#   GATE=0 ./build_and_test.sh          # unit/lockstep tests only
#   DAV2D=/path/to/dav2d ./build_and_test.sh
#
# Host-only CPU build.  No CMake and no CUDA.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
CXX="${CXX:-c++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
GATE="${GATE:-1}"
BUILD="${BUILD:-${TMPDIR:-/tmp}/av2enc-cdf-continuity}"

mkdir -p "$BUILD"

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

MODULE=(
  "$HERE/cdf_continuity.cpp"
  "$ROOT/src/codec/entropy/avg_cdf/avg_cdf_bank.cpp"
)
ENTROPY=(
  "$ROOT/src/codec/entropy/av2/av2_entropy.cpp"
  "$ROOT/src/core/bitstream/bitwriter.cpp"
)

echo "compiler: ${CXX}"
"${CXX}" --version 2>/dev/null | head -n 1 || true
echo "flags:    ${COMMON[*]} ${STRICT[*]}"
echo

echo "building: cdf_continuity_test"
"${CXX}" "${COMMON[@]}" "${STRICT[@]}" \
  "${MODULE[@]}" "$HERE/cdf_continuity_test.cpp" "${ENTROPY[@]}" \
  -o "$BUILD/cdf_continuity_test"
echo "running: $BUILD/cdf_continuity_test"
"$BUILD/cdf_continuity_test"
echo

echo "building: lockstep_stream_test"
"${CXX}" "${COMMON[@]}" "${STRICT[@]}" \
  "${MODULE[@]}" "$HERE/lockstep_stream_test.cpp" "${ENTROPY[@]}" \
  -o "$BUILD/lockstep_stream_test"
echo "running: $BUILD/lockstep_stream_test"
"$BUILD/lockstep_stream_test" --out "$BUILD/continuity_stream.bin"
echo

if [[ "${GATE}" != "1" ]]; then
  echo "GATE=0: dav2d gate skipped"
  exit 0
fi

# CPU reference pieces used by the M2 tiled writer.  Mirrors
# src/codec/entropy/avg_cdf/build_and_test.sh and
# src/pipeline/video/post/build_and_test.sh; keep this list in sync with them.
CORE=(
  "$ROOT/src/pipeline/video/post/rc_adapter.cpp"
  "$ROOT/src/pipeline/video/post/bit_depth_checklist.cpp"
  "$ROOT/src/pipeline/video/post/loop_filters.cpp"
  "$ROOT/src/pipeline/video/post/tiled_frame.cpp"
  "$ROOT/src/codec/rate/rc/rc.cpp"
  "$ROOT/src/pipeline/tiles/tile_layout.cpp"
  "$ROOT/src/pipeline/tiles/tile_group.cpp"
  "$ROOT/src/codec/filters/deblock/deblock.cpp"
  "$ROOT/src/codec/filters/cdef/cdef.cpp"
  "$ROOT/src/codec/filters/loop_restoration/restoration.cpp"
  "$ROOT/src/codec/filters/stream/filters_stream.cpp"
  "$ROOT/src/pipeline/video/video_headers.cpp"
  "$ROOT/src/pipeline/video/video_syntax.cpp"
  "$ROOT/src/codec/inter/merge/merge.cpp"
  "$ROOT/src/pipeline/planner/pipeline.cpp"
  "$ROOT/src/pipeline/dispatch/planner.cpp"
  "$ROOT/src/cuda/pipeline/planner/pipeline_gpu.cpp"
  "$ROOT/src/codec/partition/tree/partition.cpp"
  "$ROOT/src/codec/intra/modes/intra_modes.cpp"
  "$ROOT/src/codec/transform/variants/tx_variants.cpp"
  "$ROOT/src/codec/transform/variants/tx_select.cpp"
  "$ROOT/src/codec/inter/me/me.cpp"
  "$ROOT/src/codec/inter/mc/mc.cpp"
  "$ROOT/src/formats/format.cpp"
  "$ROOT/src/codec/entropy/av2/av2_entropy.cpp"
  "$ROOT/src/core/bitstream/bitwriter.cpp"
  "$ROOT/src/core/bitstream/obu.cpp"
  "$ROOT/src/core/bitstream/basic_headers.cpp"
)

echo "building: gate_dav2d_continuity"
# The M2 core carries pre-existing conversion warnings; compile it with -w and
# keep the strict warning set for this directory's own sources.
OBJECTS=()
index=0
for source in "${CORE[@]}"; do
  object="$BUILD/core_${index}_$(basename "$source" .cpp).o"
  "${CXX}" "${COMMON[@]}" -w -c "$source" -o "$object"
  OBJECTS+=("$object")
  index=$((index + 1))
done
for source in cdf_continuity.cpp gate_dav2d_continuity.cpp; do
  object="$BUILD/$(basename "$source" .cpp).o"
  "${CXX}" "${COMMON[@]}" "${STRICT[@]}" -c "$HERE/$source" -o "$object"
  OBJECTS+=("$object")
done
"${CXX}" "${COMMON[@]}" "${OBJECTS[@]}" \
  "$ROOT/src/codec/entropy/avg_cdf/avg_cdf_bank.cpp" \
  "$ROOT/src/core/io/y4m.cpp" "$ROOT/src/core/io/ivf.cpp" \
  -o "$BUILD/gate_dav2d_continuity"

echo "running: $BUILD/gate_dav2d_continuity"
if [[ -z "${DAV2D:-}" ]]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [[ -x "$candidate" ]]; then
      DAV2D="$candidate"
      export DAV2D
      break
    fi
  done
fi
set +e
"$BUILD/gate_dav2d_continuity" --out "$BUILD/continuity_gate.ivf"
status=$?
set -e
if [[ $status -eq 77 ]]; then
  echo "dav2d gate skipped (no decoder)"
  exit 0
fi
exit $status
