#!/usr/bin/env bash
#
# Build and run the AV2 average-CDF bank tests.
#
#   ./build_and_test.sh
#   CXX=g++-13 ./build_and_test.sh
#   SANITIZE=1 ./build_and_test.sh
#   WERROR=1 ./build_and_test.sh
#   GATE=0 ./build_and_test.sh          # unit tests only
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
BUILD="${BUILD:-${TMPDIR:-/tmp}/av2enc-avg-cdf}"

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

echo "compiler: ${CXX}"
"${CXX}" --version 2>/dev/null | head -n 1 || true
echo "flags:    ${COMMON[*]} ${STRICT[*]}"
echo

echo "building: avg_cdf_test"
"${CXX}" "${COMMON[@]}" "${STRICT[@]}" \
  "$HERE/avg_cdf_bank.cpp" \
  "$HERE/avg_cdf_test.cpp" \
  "$ROOT/src/codec/entropy/av2/av2_entropy.cpp" \
  "$ROOT/src/core/bitstream/bitwriter.cpp" \
  -o "$BUILD/avg_cdf_test"
echo "running: $BUILD/avg_cdf_test"
"$BUILD/avg_cdf_test"
echo

if [[ "${GATE}" != "1" ]]; then
  echo "GATE=0: dav2d gate skipped"
  exit 0
fi

# CPU reference pieces used by the M2 tiled writer.  Mirrors
# src/pipeline/video/post/build_and_test.sh; keep this list in sync with it.
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

echo "building: gate_dav2d_avg_cdf"
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
for source in avg_cdf_bank.cpp gate_dav2d_avg_cdf.cpp; do
  object="$BUILD/$(basename "$source" .cpp).o"
  "${CXX}" "${COMMON[@]}" "${STRICT[@]}" -c "$HERE/$source" -o "$object"
  OBJECTS+=("$object")
done
"${CXX}" "${COMMON[@]}" "${OBJECTS[@]}" \
  "$ROOT/src/core/io/y4m.cpp" "$ROOT/src/core/io/ivf.cpp" \
  -o "$BUILD/gate_dav2d_avg_cdf"

echo "running: $BUILD/gate_dav2d_avg_cdf"
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
"$BUILD/gate_dav2d_avg_cdf" --out "$BUILD/avg_cdf_gate.ivf"
status=$?
set -e
if [[ $status -eq 77 ]]; then
  echo "dav2d gate skipped (no decoder)"
  exit 0
fi
exit $status
