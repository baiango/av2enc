#!/usr/bin/env bash
# Lambda-scale sweep for the M2 P-frame rate-aware layer.
#
# Reuses the RDO gate built by run_m2_rdo_experiment.sh and runs the 8-frame
# full-tree case at several AV2ENC_M2_RDO_SCALE values.  Every stream is
# decoded through dav2d_strict.sh (--strict 1) and checked against the encoder
# reconstruction before tools/quality.py scores it.  No speed claim.
#
# Usage:
#   ./run_m2_rdo_scale_sweep.sh --work /tmp/m2_rdo_experiment
#   ./run_m2_rdo_scale_sweep.sh --scales 0.05,0.1,0.25,0.5 --out /tmp/sweep
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../../../../.." && pwd)"
WORK="${WORK:-${TMPDIR:-/tmp}/m2_rdo_experiment}"
OUT="${OUT:-$WORK/sweep}"
INPUT="${INPUT:-$ROOT/testdata/shine_1280x720_60f.y4m}"
SCALES="${SCALES:-0.05,0.10,0.20,0.25,0.50}"

while [ $# -gt 0 ]; do
  case "$1" in
    --work) WORK="$2"; shift ;;
    --out) OUT="$2"; shift ;;
    --input|-i) INPUT="$2"; shift ;;
    --scales) SCALES="$2"; shift ;;
    -h|--help)
      sed -n '2,11p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *) echo "run_m2_rdo_scale_sweep: unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

GATE="$WORK/m2_gate_rdo"
[ -x "$GATE" ] || {
  echo "missing $GATE; run run_m2_rdo_experiment.sh first" >&2
  exit 2
}
[ -f "$INPUT" ] || { echo "missing input $INPUT" >&2; exit 2; }

if [ -z "${REAL_DAV2D:-}" ]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos" \
                   "$(command -v dav2d || true)"; do
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then
      REAL_DAV2D="$candidate"
      break
    fi
  done
fi
[ -n "${REAL_DAV2D:-}" ] || { echo "dav2d not found" >&2; exit 2; }
REAL_DAV2D="$(cd -- "$(dirname -- "$REAL_DAV2D")" && pwd)/$(basename -- "$REAL_DAV2D")"
WRAPPER="$HERE/dav2d_strict.sh"
chmod +x "$WRAPPER"

mkdir -p "$OUT"
TSV="$OUT/sweep.tsv"
printf 'scale\tbytes\tpsnr_y\tpsnr_avg\tcvvdp\tivf\n' > "$TSV"

IFS=',' read -r -a SCALE_LIST <<< "$SCALES"
for scale in "${SCALE_LIST[@]}"; do
  out_ivf="$OUT/8f_tree.scale_$scale.ivf"
  log="$OUT/8f_tree.scale_$scale.log"
  echo "== 8f_tree / scale $scale =="
  env AV2ENC_M2_RDO=1 AV2ENC_M2_RDO_SCALE="$scale" \
    DAV2D="$WRAPPER" REAL_DAV2D="$REAL_DAV2D" \
    "$GATE" --input "$INPUT" --frames 8 --out "$out_ivf" > "$log" 2>&1
  grep -q "M2 GATE PASS" "$log" || { cat "$log" >&2; exit 1; }
  grep -q "recon exact" "$log" || { cat "$log" >&2; exit 1; }
  bytes=$(grep -o 'total_bytes=[0-9]*' "$log" | tail -1 | cut -d= -f2)
  json="$OUT/8f_tree.scale_$scale.json"
  env DAV2D="$WRAPPER" REAL_DAV2D="$REAL_DAV2D" \
    python3 "$ROOT/tools/quality.py" --ref "$INPUT" \
    --ivf "$out_ivf.clip.ivf" --json "$json" \
    > "$OUT/8f_tree.scale_$scale.quality.log" 2>&1
  read -r psnr_y psnr_avg cvvdp < <(
    python3 - "$json" <<'PY'
import json
import sys
with open(sys.argv[1]) as handle:
    report = json.load(handle)
print(f"{report['psnr']['psnr_y']:.4f} {report['psnr']['psnr_avg']:.4f} "
      f"{report['cvvdp_jod']:.4f}")
PY
  )
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$scale" "$bytes" "$psnr_y" "$psnr_avg" "$cvvdp" "$out_ivf" >> "$TSV"
done

MARKDOWN="$OUT/sweep.md"
{
  echo "# M2 RDO lambda-scale sweep (8-frame full tree, 1280x720)"
  echo
  echo "All streams decoded with \`dav2d --strict 1\`; recon comparison is"
  echo "sample-exact.  Quality from \`tools/quality.py\`."
  echo
  echo "| scale | bytes | PSNR-Y | PSNR-avg | CVVDP |"
  echo "|---:|---:|---:|---:|---:|"
  tail -n +2 "$TSV" | while IFS=$'\t' read -r scale bytes psnr_y psnr_avg cvvdp ivf; do
    printf '| %s | %s | %s | %s | %s |\n' \
      "$scale" "$bytes" "$psnr_y" "$psnr_avg" "$cvvdp"
  done
} > "$MARKDOWN"

cat "$MARKDOWN"
echo
echo "results: $MARKDOWN"
