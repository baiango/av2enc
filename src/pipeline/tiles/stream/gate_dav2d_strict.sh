#!/usr/bin/env bash
#
# Build and run the M1 tiled still dav2d --strict 1 gate.
#
#   ./gate_dav2d_strict.sh
#   DAV2D=/path/to/dav2d ./gate_dav2d_strict.sh
#   ./gate_dav2d_strict.sh tests/data/basic_640x360_1f.y4m --layout 2x2
#
# Default matrix per source: 1x1, 2x1, 2x2, 4x2, 8x2, 16x2. That covers
# 1/2/4/8/16 tiles at 640x360 and 1920x1080; 16x2 (32 tiles) runs where the
# planner can represent it. Every case must decode with `dav2d --strict 1` to
# byte-identical Y/U/V planes. A 2x2 deblock-on diagnostic then re-runs each
# source and reports the cross-tile deblock finding without making it part of
# the exactness pass/fail matrix.
#
# A missing pinned decoder is exit 127, never a pass.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
CXX="${CXX:-c++}"
WERROR="${WERROR:-0}"
BUILD_DIR="${BUILD_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/av2enc-m1-tiles-gate.XXXXXX")}"
trap 'rm -rf "${BUILD_DIR}"' EXIT
BIN="${BUILD_DIR}/gate_dav2d_strict"

COMMON=(-std=c++20 -O2 -g -ffp-contract=off -I"$ROOT/src")
STRICT=(-Wall -Wextra -Wpedantic -Wconversion -Wshadow)
if [[ "${WERROR}" == "1" ]]; then
  STRICT+=(-Werror)
fi

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
  "$ROOT/src/core/io/y4m.cpp"
  "$ROOT/src/core/io/ivf.cpp"
)

echo "compiler: ${CXX}"
"${CXX}" --version 2>/dev/null | head -n 1 || true
echo "building tiled dav2d --strict 1 gate (no CMake, no core edits)"

OBJECTS=()
index=0
for source in "${CORE[@]}"; do
  object="${BUILD_DIR}/core_${index}_$(basename "${source}" .cpp).o"
  "${CXX}" "${COMMON[@]}" -w -c "${source}" -o "${object}"
  OBJECTS+=("${object}")
  index=$((index + 1))
done

for source in tiles_stream.cpp gate_dav2d_strict.cpp; do
  object="${BUILD_DIR}/$(basename "${source}" .cpp).o"
  "${CXX}" "${COMMON[@]}" "${STRICT[@]}" -c "$HERE/${source}" -o "${object}"
  OBJECTS+=("${object}")
done

"${CXX}" "${COMMON[@]}" "${OBJECTS[@]}" -o "${BIN}"
echo

# Default sources cover the two required resolutions. Do not append them when
# the caller supplied a positional source.
have_source=0
skip_next=0
for arg in "$@"; do
  if [[ "${skip_next}" == "1" ]]; then
    skip_next=0
    continue
  fi
  case "${arg}" in
    --dav2d|--out|--base-q|--layout) skip_next=1 ;;
    --*) ;;
    *) have_source=1 ;;
  esac
done
defaults=()
if [[ "${have_source}" == "0" ]]; then
  defaults+=("$ROOT/tests/data/basic_640x360_1f.y4m")
  defaults+=("$ROOT/testdata/shine_1920x1080_10f.y4m")
fi

decoder_args=()
if [[ -n "${DAV2D:-}" ]]; then
  decoder_args+=(--dav2d "${DAV2D}")
else
  decoder="$ROOT/tools/bin/dav2d"
  if [[ ! -x "${decoder}" && -x "${ROOT}/tools/bin/dav2d-macos" ]]; then
    decoder="$ROOT/tools/bin/dav2d-macos"
  fi
  decoder_args+=(--dav2d "${decoder}")
fi

out_args=()
if [[ -n "${OUT_DIR:-}" ]]; then
  out_args+=(--out "${OUT_DIR}")
else
  out_args+=(--out "${BUILD_DIR}/gate")
fi

exec "${BIN}" ${defaults[@]+"${defaults[@]}"} "$@" \
  ${decoder_args[@]+"${decoder_args[@]}"} "${out_args[@]}"
