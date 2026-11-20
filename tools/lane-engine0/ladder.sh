#!/usr/bin/env bash
# AV2ENC_RD_DRIVER sequential key-frame driver ladder (docs/rd-engine0.md).
#
# Two arms (base = the merged 86.013 recipe, drv = +AV2ENC_RD_DRIVER), 5
# interleaved QPs, arm order rotated per QP.  Each point is strict-decoded,
# validated, per-frame PSNR'd, and its --video-recon payload compared with the
# decoder output.  Matched K is tools/oracle.py --target-kbps 100.
#
#   tools/lane-engine0/ladder.sh /tmp/rd-engine0 "116 117 118 119 120"
set -u
ROOT=$(cd -- "$(dirname -- "$0")/../.." && pwd)
BIN=${BIN:-$ROOT/build-release/av2enc}
REF=${REF:-$ROOT/testdata/nasa_480x240_30fps_60f.y4m}
OUT=${1:-/tmp/rd-engine0-ladder}
QPS=${2:-"116 117 118 119 120"}
DAV2D=${DAV2D:-$ROOT/tools/bin/dav2d-macos}
BASE_ENV=(AV2ENC_EFF_PERC=1 AV2ENC_RD_UVFULL=1
          AV2ENC_TX_DETAIL_SEARCH_TAIL=1 AV2ENC_TX_DETAIL_MIN=0.50
          AV2ENC_RD_KEY_TX=1 AV2ENC_RD_UVLEVEL=1)
COMMON=(--video --frames 60 --cpu-threads 8 --nightly
        --video-drift-refresh-threshold 1.5 --video-drift-refresh-budget 3250)
mkdir -p "$OUT"
fail=0
arm_env() {
  case "$1" in
    base) : ;;
    drv) echo "AV2ENC_RD_DRIVER=1 ${DRV_EXTRA:-}" ;;
  esac
}
for qp in $QPS; do
  arms=(base drv)
  off=$(( qp % 2 ))
  rotated=()
  for i in 0 1; do rotated+=("${arms[$(( (i + off) % 2 ))]}"); done
  for arm in "${rotated[@]}"; do
    tag="${arm}_q${qp}"
    ivf="$OUT/$tag.ivf"
    if [ "${REUSE:-0}" = 1 ] && [ -f "$ivf" ] && [ -s "$OUT/$tag.psnr" ]; then
      echo "cached $tag"
      continue
    fi
    extra=$(arm_env "$arm")
    diag=""
    if [ "$qp" = 119 ]; then
      diag="AV2ENC_BIT_CENSUS=$OUT/$tag.census.jsonl AV2ENC_DECISION_DUMP=$OUT/$tag.dec"
    fi
    # shellcheck disable=SC2086
    if ! env "${BASE_ENV[@]}" $extra $diag "$BIN" "$REF" -o "$ivf" \
         --qp "$qp" "${COMMON[@]}" --video-recon "$OUT/$tag.recon.y4m" \
         > "$OUT/$tag.log" 2>&1; then
      echo "encode failed: $tag" >&2
      fail=1
      continue
    fi
    "$DAV2D" --strict 1 -i "$ivf" -o "$OUT/$tag.dec.y4m" \
        > "$OUT/$tag.dec.log" 2>&1 ||
        { echo "strict decode failed: $tag" >&2; fail=1; }
    "$ROOT/tools/validate_dav2d.sh" "$ivf" "$REF" > "$OUT/$tag.val.log" 2>&1 ||
        { echo "validate failed: $tag" >&2; fail=1; }
    python3 "$ROOT/tools/perframe_psnr.py" "$REF" "$OUT/$tag.dec.y4m" "$ivf" \
        --frames 60 > "$OUT/$tag.psnr" 2>&1 ||
        { echo "psnr failed: $tag" >&2; fail=1; }
    python3 - "$OUT/$tag.recon.y4m" "$OUT/$tag.dec.y4m" "$OUT/$tag.recon.ok" <<'PY'
import sys
def payload(p):
    data = open(p, 'rb').read()
    return data[data.find(b'FRAME'):]
ok = payload(sys.argv[1]) == payload(sys.argv[2])
open(sys.argv[3], 'w').write('ok\n' if ok else 'MISMATCH\n')
sys.exit(0 if ok else 1)
PY
    [ -s "$OUT/$tag.recon.ok" ] || { echo "recon mismatch: $tag" >&2; fail=1; }
    echo "done $tag $(stat -f%z "$ivf") bytes"
  done
done
ivfs=()
for qp in $QPS; do
  for arm in base drv; do
    [ -f "$OUT/${arm}_q${qp}.ivf" ] && ivfs+=(--ivf "${arm}@${qp}=$OUT/${arm}_q${qp}.ivf")
  done
done
python3 "$ROOT/tools/oracle.py" --ref "$REF" "${ivfs[@]}" \
    --target-kbps 100 --no-segments | tee "$OUT/oracle.md"
exit "$fail"
