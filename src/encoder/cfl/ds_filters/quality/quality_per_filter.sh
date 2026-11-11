#!/usr/bin/env bash
# Quality/rate delta per cfl_ds_filter_index on real 4:2:0 frames.
#
# Inputs:
#   640x360 : first frame of testdata/shine_1280x720_60f.y4m, scaled with
#             lanczos (ffmpeg);
#   720p    : first frame of testdata/shine_1280x720_60f.y4m.
#
# For each index 0/1/2 the script encodes with the instrumented M1 build from
# gate/build_gate.sh, decodes with the pinned dav2d, and reports file bytes,
# Y/U/V PSNR and weighted PSNR.  Delta rows are relative to index 0.
#
# Usage:
#   ./src/encoder/cfl/ds_filters/quality/quality_per_filter.sh
#   SKIP_BUILD=1 ./.../quality_per_filter.sh
#
# Exit: 0 if every stream decodes; 2 on setup error.

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../../../../.." && pwd)"
GATE="$HERE/../gate"
OUT="$HERE/build"
SRC720="$ROOT/testdata/shine_1280x720_60f.y4m"

if [ "${SKIP_BUILD:-0}" != "1" ]; then
  "$GATE/build_gate.sh"
fi
[ -x "$GATE/build/av2enc_ds" ] || { echo "quality: run gate/build_gate.sh" >&2; exit 2; }
[ -f "$SRC720" ] || { echo "quality: missing $SRC720" >&2; exit 2; }

DAV2D="${DAV2D:-}"
if [ -z "$DAV2D" ]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$candidate" ]; then DAV2D="$candidate"; break; fi
  done
fi
[ -n "$DAV2D" ] || { echo "quality: dav2d not found (set DAV2D)" >&2; exit 2; }

mkdir -p "$OUT"
SRC640="$OUT/src_640x360.y4m"
ffmpeg -y -v error -i "$SRC720" -frames:v 1 -vf scale=640:360:flags=lanczos \
  -pix_fmt yuv420p "$SRC640"
ffmpeg -y -v error -i "$SRC720" -frames:v 1 -pix_fmt yuv420p "$OUT/src_720p.y4m"

printf 'size      filter  bytes   Y_psnr  U_psnr  V_psnr  W_psnr   d_bytes  dW_psnr\n'
printf '%s\n' '---------------------------------------------------------------------------'
for size in 640x360 720p; do
  case "$size" in
    640x360) src="$SRC640" ;;
    720p) src="$OUT/src_720p.y4m" ;;
  esac
  base_bytes=""
  base_psnr=""
  for index in 0 1 2; do
    ivf="$OUT/${size}_f${index}.ivf"
    dec="$OUT/${size}_f${index}_dec.y4m"
    AV2ENC_CFL_DS_FILTER="$index" "$GATE/build/av2enc_ds" "$src" \
      -o "$ivf" --m1 >/dev/null
    "$DAV2D" -q -i "$ivf" -o "$dec" --strict 1 --threads 1
    bytes=$(stat -f%z "$ivf" 2>/dev/null || stat -c%s "$ivf")
    metrics=$(python3 "$HERE/y4m_psnr.py" "$src" "$dec")
    read -r y u v w <<<"$(python3 -c \
      'import json,sys; m=json.loads(sys.argv[1]); print("%.4f %.4f %.4f %.4f" % (m["y"]["psnr"], m["u"]["psnr"], m["v"]["psnr"], m["weighted_psnr"]))' \
      "$metrics")"
    if [ "$index" = "0" ]; then
      base_bytes=$bytes
      base_psnr=$w
      dbytes=0
      dpsnr=0.0000
    else
      dbytes=$((bytes - base_bytes))
      dpsnr=$(python3 -c "print(f'{$w - $base_psnr:+.4f}')")
    fi
    printf '%-9s %-7s %-7s %-7s %-7s %-7s %-8s %+8s %+8s\n' \
      "$size" "$index" "$bytes" "$y" "$u" "$v" "$w" "$dbytes" "$dpsnr"
  done
done
echo
echo "quality: artifacts in $OUT"
