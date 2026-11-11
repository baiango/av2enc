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
ROOT="$(cd -- "$HERE/../../.." && pwd)"
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
  "$ROOT/src/encoder/m1/m1_still.cpp"
  "$ROOT/src/encoder/m1/m1_syntax.cpp"
  "$ROOT/src/encoder/m1/m1_headers.cpp"
  "$ROOT/src/bitstream/bitwriter.cpp"
  "$ROOT/src/bitstream/obu.cpp"
  "$ROOT/src/bitstream/m0_headers.cpp"
  "$ROOT/src/encoder/partition/partition.cpp"
  "$ROOT/src/encoder/intra_modes/intra_modes.cpp"
  "$ROOT/src/encoder/tx_variants/tx_variants.cpp"
  "$ROOT/src/encoder/tx_variants/tx_select.cpp"
  "$ROOT/src/encoder/cfl/cfl.cpp"
  "$ROOT/src/encoder/filters/deblock/deblock.cpp"
  "$ROOT/src/encoder/filters/cdef/cdef.cpp"
  "$ROOT/src/cuda/m1/cdef/m1_cdef.cpp"
  "$ROOT/src/encoder/rdo/rdo.cpp"
  "$ROOT/src/encoder/rdo/intra_decision.cpp"
  "$ROOT/src/encoder/tx_type/tx_type.cpp"
  "$ROOT/src/encoder/rdoq/rdoq.cpp"
  "$ROOT/src/encoder/rdoq/rdoq_av2_model.cpp"
  "$ROOT/src/encoder/tiles/tile_layout.cpp"
  "$ROOT/src/encoder/tiles/tile_group.cpp"
  "$ROOT/src/entropy/av2/av2_entropy.cpp"
  "$ROOT/src/io/y4m.cpp"
  "$ROOT/src/io/ivf.cpp"
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
