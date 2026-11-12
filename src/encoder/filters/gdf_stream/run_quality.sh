#!/usr/bin/env bash
# GDF quality measurement at 640x360 and 720p.
#
# Encodes the first frame with the shipped M1 CPU still encoder, runs the GDF
# encoder decision on the real reconstruction, and measures:
#   base        M1 reconstruction (no GDF)
#   gdf_frame   frame-wide GDF (gdf_per_block = 0)
#   gdf_block   per-block GDF (gdf_per_block = 1)
# for PSNR-Y (ffmpeg) and CVVDP-JOD (cvvdp, when installed), plus the exact
# §5.4.10/§5.18.7.9/§5.20.10.3 signalling bit cost from gdf_quality.
#
# Nothing here edits the encoder or the M1/M2 paths.
#
#   ./run_quality.sh
#   QP=120 LAMBDA=0 ./run_quality.sh
#   DAV2D=... CVVDP=... INPUT_720=... INPUT_640=... ./run_quality.sh
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../../../.." && pwd)"
CXX="${CXX:-c++}"
BUILD="${BUILD:-${TMPDIR:-/tmp}/av2enc-gdf-quality}"
OUT="${OUT:-$BUILD/out}"
QP="${QP:-90}"
LAMBDA="${LAMBDA:-0}"
FFMPEG="${FFMPEG:-ffmpeg}"
CVVDP="${CVVDP:-cvvdp}"
INPUT_720="${INPUT_720:-$ROOT/testdata/shine_1280x720_60f.y4m}"
INPUT_640="${INPUT_640:-}"

mkdir -p "$BUILD" "$OUT"
FLAGS=(-std=c++20 -O2 -w -ffp-contract=off -I"$ROOT/src")

SOURCES=(
  "$HERE/gdf_syntax.cpp"
  "$HERE/gdf_decision.cpp"
  "$ROOT/src/encoder/filters/spec_tables/gdf.cpp"
  "$ROOT/src/encoder/filters/spec_tables/luma_source.cpp"
  "$ROOT/src/io/y4m.cpp"
  "$ROOT/src/io/ivf.cpp"
  "$ROOT/src/bitstream/bitwriter.cpp"
  "$ROOT/src/bitstream/obu.cpp"
  "$ROOT/src/bitstream/m0_headers.cpp"
  "$ROOT/src/encoder/m0_still.cpp"
  "$ROOT/src/encoder/partition/partition.cpp"
  "$ROOT/src/encoder/intra_modes/intra_modes.cpp"
  "$ROOT/src/encoder/rdoq/rdoq.cpp"
  "$ROOT/src/encoder/rdoq/rdoq_av2_model.cpp"
  "$ROOT/src/encoder/tx_type/tx_type.cpp"
  "$ROOT/src/encoder/tx_variants/tx_variants.cpp"
  "$ROOT/src/encoder/tx_variants/tx_select.cpp"
  "$ROOT/src/encoder/cfl/cfl.cpp"
  "$ROOT/src/encoder/filters/deblock/deblock.cpp"
  "$ROOT/src/encoder/filters/cdef/cdef.cpp"
  "$ROOT/src/encoder/filters/restoration/restoration.cpp"
  "$ROOT/src/encoder/rdo/rdo.cpp"
  "$ROOT/src/encoder/rdo/intra_decision.cpp"
  "$ROOT/src/encoder/m1/m1_headers.cpp"
  "$ROOT/src/encoder/m1/m1_syntax.cpp"
  "$ROOT/src/encoder/m1/m1_still.cpp"
  "$ROOT/src/encoder/m1/stream/m1_stream_bridge.cpp"
  "$ROOT/src/entropy/av2/av2_entropy.cpp"
  "$ROOT/src/profile/profiler.cpp"
  "$ROOT/src/cuda/m1/cdef/m1_cdef.cpp"
)

"${CXX}" "${FLAGS[@]}" "$HERE/gdf_quality.cpp" "${SOURCES[@]}" \
  -o "$BUILD/gdf_quality"

if [[ -z "$INPUT_640" ]]; then
  INPUT_640="$OUT/640x360_input.y4m"
  "$FFMPEG" -y -loglevel error -i "$INPUT_720" -frames:v 1 -r 30 \
    -vf "crop=640:360:0:0" -f yuv4mpegpipe "$INPUT_640"
fi

run_mode() {
  local name="$1" dir="$2" mode="$3" file="$4"
  if [[ ! -f "$file" ]]; then
    return
  fi
  local ref="$dir/ref.y4m"
  local psnr
  psnr="$("$FFMPEG" -i "$ref" -i "$file" -lavfi psnr -f null - 2>&1 |
    grep -o 'y:[0-9.]*' | head -1 | cut -d: -f2)"
  local jod="n/a"
  if command -v "$CVVDP" >/dev/null 2>&1; then
    local out
    out="$("$CVVDP" -t "$file" -r "$ref" -d standard_fhd --device mps 2>&1 || true)"
    local parsed
    parsed="$(printf '%s\n' "$out" | grep -o 'cvvdp=[-0-9.]*' | head -1 |
      cut -d= -f2)"
    if [[ -n "$parsed" ]]; then jod="$parsed"; fi
  fi
  local bits
  bits="$(grep "\"mode\":\"$mode\"" "$dir/quality.json" |
    grep -o '"header_bits":[0-9]*,"symbol_bits":[0-9]*' | head -1)"
  local units
  units="$(grep "\"mode\":\"$mode\"" "$dir/quality.json" |
    grep -o '"units":[0-9]*' | head -1)"
  printf '| %-8s | %-10s | %-8s | %-8s | %s %s |\n' \
    "$name" "$mode" "$psnr" "$jod" "$bits" "$units"
}

printf '| clip | mode | PSNR-Y | CVVDP | signalling bits |\n'
printf '|---|---|---:|---:|---|\n'
for clip in 640x360 720p; do
  if [[ "$clip" == "640x360" ]]; then
    input="$INPUT_640"
  else
    input="$INPUT_720"
  fi
  dir="$OUT/$clip"
  mkdir -p "$dir"
  "$FFMPEG" -y -loglevel error -i "$input" -frames:v 1 -r 30 \
    -f yuv4mpegpipe "$dir/ref.y4m"
  "$BUILD/gdf_quality" --input "$input" --out "$dir" --qp "$QP" \
    --lambda "$LAMBDA" | tee "$dir/quality.json"
  run_mode "$clip" "$dir" base "$dir/base.y4m"
  run_mode "$clip" "$dir" gdf_frame "$dir/gdf_frame.y4m"
  run_mode "$clip" "$dir" gdf_block "$dir/gdf_block.y4m"
done

printf '\nArtifacts: %s\n' "$OUT"
