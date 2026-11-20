#!/usr/bin/env bash
# AV2ENC_RD_TAIL_FLOOR ladder (docs/rd-tail-floor.md).
#
# Two arms: base = the current best recipe (driver + tail exact-rate), cand =
# base + the measured tail coding-block floor.  QPs are interleaved and the arm
# order rotates per QP.  Each point is strict-decoded, validated, per-frame
# PSNR'd, and its --video-recon payload compared with the decoder output.
# Matched K and the source-replacement segment ceilings come from
# tools/oracle.py --target-kbps 100 (run separately on the streams this writes).
#
#   tools/lane-tail-floor/ladder.sh /tmp/rd-tail-floor "113 114 115 116 117"
set -u
ROOT=$(cd -- "$(dirname -- "$0")/../.." && pwd)
BIN=${BIN:-$ROOT/build-release/av2enc}
REF=${REF:-$ROOT/testdata/nasa_480x240_30fps_60f.y4m}
OUT=${1:-/tmp/rd-tail-floor-ladder}
QPS=${2:-"113 114 115 116 117"}
DAV2D=${DAV2D:-$ROOT/tools/bin/dav2d-macos}
BASE_ENV=(AV2ENC_EFF_PERC=1 AV2ENC_RD_UVFULL=1
          AV2ENC_TX_DETAIL_SEARCH_TAIL=1 AV2ENC_TX_DETAIL_MIN=0.50
          AV2ENC_RD_KEY_TX=1 AV2ENC_RD_UVLEVEL=1 AV2ENC_RD_DRIVER=1
          AV2ENC_RD_TAIL_TX=1)
COMMON=(--video --frames 60 --cpu-threads 8 --nightly
        --video-drift-refresh-threshold 1.5 --video-drift-refresh-budget 3250)
DIAG_QP=${DIAG_QP:-115}
ARMS=${ARMS:-"base cand"}
mkdir -p "$OUT"
fail=0
arm_env() {
  case "$1" in
    base) : ;;
    cand) echo "AV2ENC_RD_TAIL_FLOOR=1 ${CAND_EXTRA:-}" ;;
  esac
}
for qp in $QPS; do
  arms=($ARMS)
  n=${#arms[@]}
  off=$(( qp % n ))
  rotated=()
  for i in $(seq 0 $((n - 1))); do
    rotated+=("${arms[$(( (i + off) % n ))]}")
  done
  for arm in "${rotated[@]}"; do
    tag="${arm}_q${qp}"
    ivf="$OUT/$tag.ivf"
    if [ "${REUSE:-0}" = 1 ] && [ -f "$ivf" ] && [ -s "$OUT/$tag.psnr" ]; then
      echo "cached $tag"
      continue
    fi
    extra=$(arm_env "$arm")
    diag=""
    if [ "$qp" = "$DIAG_QP" ]; then
      diag="AV2ENC_BIT_CENSUS=$OUT/$tag.census.jsonl AV2ENC_DECISION_DUMP=$OUT/$tag.dec"
      [ "$arm" = cand ] && diag="$diag AV2ENC_RD_TAIL_FLOOR_STATS=1"
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
echo "ladder done fail=$fail"
exit "$fail"
