#!/usr/bin/env bash
# Chroma loop-restoration quality measurement (640x360 and 720p).
#
# Encodes the fixture with the shipped M1 still encoder, decodes with the
# pinned dav2d, then measures the chroma restoration decision on the real
# reconstruction:
#
#   baseline PSNR(U/V) -> restored PSNR(U/V), exact header+tile bits,
#   filtered Y4M for CVVDP.
#
# Requirements: build-m1/av2enc (CPU reference build), tools/bin/dav2d*, ffmpeg
# and (optionally) cvvdp.  Nothing here edits the encoder or the M1/M2 paths.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
CXX="${CXX:-c++}"
BUILD="${BUILD:-${TMPDIR:-/tmp}/av2enc-lr-chroma-quality}"
OUT="${OUT:-$BUILD/out}"
DAV2D="${DAV2D:-}"
ENCODER="${ENCODER:-$ROOT/build-m1/av2enc}"
FFMPEG="${FFMPEG:-ffmpeg}"
CVVDP="${CVVDP:-cvvdp}"

mkdir -p "$BUILD" "$OUT"
FLAGS=(-std=c++20 -O2 -w -I"$ROOT/src")

SOURCES=(
  "$ROOT/src/codec/filters/loop_restoration/chroma/chroma_lr.cpp"
  "$ROOT/src/codec/filters/loop_restoration/restoration.cpp"
  "$ROOT/src/codec/filters/loop_restoration/classes/wiener_fit.cpp"
  "$ROOT/src/codec/filters/loop_restoration/classes/wiener_syntax.cpp"
  "$ROOT/src/codec/filters/loop_restoration/classes/wiener_class.cpp"
  "$ROOT/src/codec/filters/stream/filters_stream.cpp"
  "$ROOT/src/codec/entropy/av2/av2_entropy.cpp"
  "$ROOT/src/core/bitstream/bitwriter.cpp"
  "$ROOT/src/core/io/y4m.cpp"
)
"${CXX}" "${FLAGS[@]}" "$HERE/lr_chroma_quality.cpp" "${SOURCES[@]}" \
  -o "$BUILD/lr_chroma_quality"

if [[ -z "$DAV2D" ]]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [[ -x "$candidate" ]]; then DAV2D="$candidate"; break; fi
  done
fi
if [[ -z "$DAV2D" || ! -x "$DAV2D" ]]; then
  echo "error: dav2d not found (set DAV2D=)" >&2
  exit 127
fi
if [[ ! -x "$ENCODER" ]]; then
  echo "error: $ENCODER not found; build the CPU reference first:" >&2
  echo "  cmake -S $ROOT -B $ROOT/build-m1 -G Ninja -DCMAKE_BUILD_TYPE=Release \\" >&2
  echo "    -DAV2ENC_ENABLE_CUDA=OFF -DAV2ENC_ALLOW_CPU_ONLY=ON" >&2
  echo "  cmake --build $ROOT/build-m1 -j" >&2
  exit 127
fi

FIXTURE_360="$ROOT/tests/data/basic_640x360_1f.y4m"
FIXTURE_720="$BUILD/basic_1280x720_1f.y4m"
if [[ ! -f "$FIXTURE_360" ]]; then
  echo "error: missing $FIXTURE_360" >&2
  exit 2
fi
if [[ ! -f "$FIXTURE_720" ]]; then
  "$FFMPEG" -hide_banner -v error -y -i "$FIXTURE_360" \
    -vf scale=1280:720 -frames:v 1 "$FIXTURE_720"
fi

run_case() {
  local name=$1
  local source=$2
  local ivf="$OUT/$name.ivf"
  local recon="$OUT/$name.recon.y4m"
  echo
  echo "=================================================================="
  echo "case: $name ($(basename "$source"))"
  echo "=================================================================="
  "$ENCODER" "$source" -o "$ivf" --frames 1 --still >/dev/null
  "$DAV2D" -q -i "$ivf" -o "$recon" --threads 1 --strict 1
  echo "ivf: $(wc -c < "$ivf") bytes"
  "$BUILD/lr_chroma_quality" --source "$source" --recon "$recon" \
    --out-prefix "$OUT/$name"
  local filtered="$OUT/$name.filtered.y4m"
  if command -v "$CVVDP" >/dev/null 2>&1; then
    echo
    echo "CVVDP (filtered vs source):"
    "$CVVDP" -t "$filtered" -r "$source" --device cpu 2>/dev/null |
      grep -E "JOD|Jod|jod|Test" | tail -4 || true
    echo "CVVDP (recon vs source):"
    "$CVVDP" -t "$recon" -r "$source" --device cpu 2>/dev/null |
      grep -E "JOD|Jod|jod|Test" | tail -4 || true
  else
    echo "CVVDP: not installed; filtered Y4M left at $filtered"
  fi
}

run_case "q640x360" "$FIXTURE_360"
run_case "q1280x720" "$FIXTURE_720"

echo
echo "outputs: $OUT"
