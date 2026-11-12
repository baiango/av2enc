#!/usr/bin/env bash
#
# Build and run the tiled still-picture dav2d gate.
#
#   ./gate_dav2d_tiled.sh [source.y4m] [layout] [--out DIR] [--dav2d PATH]
#
# Examples:
#   ./gate_dav2d_tiled.sh ../../../testdata/shine_256x256_1f.y4m 2x2
#   ./gate_dav2d_tiled.sh ../../../testdata/shine_1920x1080_10f.y4m 4x2
#   ./gate_dav2d_tiled.sh ../../../testdata/shine_1920x1080_10f.y4m 16x2
#
# No CMake and no core edits. The harness builds only from public M1 encoder
# entry points plus its own tile_info()/tile-group writer. A missing pinned
# decoder is exit 127, not a pass.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
CXX="${CXX:-c++}"
BUILD_DIR="${BUILD_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/av2enc-tiled-gate.XXXXXX")}"
trap 'rm -rf "${BUILD_DIR}"' EXIT
BIN="${BUILD_DIR}/tiled_still_gate"

# The gate links the M1 reference, which currently emits unrelated warnings;
# use -w for the link line (the harness itself is checked separately with
# -fsyntax-only -Werror during development).
FLAGS=(-std=c++20 -O2 -g -ffp-contract=off -w -I"$ROOT/src")

SOURCES=(
  "$HERE/tiled_still_gate.cpp"
  "$ROOT/src/pipeline/still/still_encoder.cpp"
  "$ROOT/src/pipeline/still/still_syntax.cpp"
  "$ROOT/src/pipeline/still/still_headers.cpp"
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
echo "building tiled gate (no CMake, no core edits)"
"${CXX}" "${FLAGS[@]}" "${SOURCES[@]}" -o "${BIN}"
echo

# Keep artefacts out of the source tree unless the caller asks for a path.
OUT_DIR="${OUT_DIR:-$BUILD_DIR/gate}"
has_out=0
for arg in "$@"; do
  if [[ "$arg" == "--out" ]]; then
    has_out=1
  fi
done
if [[ "$has_out" == "0" ]]; then
  exec "${BIN}" "$@" --out "$OUT_DIR"
fi
exec "${BIN}" "$@"
