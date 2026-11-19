#!/usr/bin/env bash
# Four-arm interleaved RD ladder for the L3 x L5 composition (docs/rd-merge.md).
#
# Encodes base / +AV2ENC_RD_KEY_TX / +AV2ENC_RD_UVLEVEL / both, rotating the arm
# order at every QP so no arm is always first.  Each stream is strict-decoded,
# checked with tools/validate_dav2d.sh, scored for per-frame PSNR, and compared
# with the encoder's own --video-recon payload.  The QP119 streams also get the
# AV2ENC_BIT_CENSUS + AV2ENC_DECISION_DUMP diagnostics.
#
#   tools/lane-merge/ladder.sh /tmp/rdm/ladder "116 117 118 119 120 121 122"
#
# Matched-rate K comes from tools/oracle.py on the resulting IVFs:
#   python3 tools/oracle.py --ref testdata/nasa_480x240_30fps_60f.y4m \
#     --ivf base@116=... --ivf base@117=... --target-kbps 100 --no-segments
#
# Env: BIN, REF, DAV2D, REUSE=1 (skip points whose IVF and PSNR exist).
set -u
ROOT=$(cd -- "$(dirname -- "$0")/../.." && pwd)
BIN=${BIN:-$ROOT/build-release/av2enc}
REF=${REF:-$ROOT/testdata/nasa_480x240_30fps_60f.y4m}
OUT=${1:-/tmp/rd-merge-ladder}
QPS=${2:-"116 117 118 119 120 121 122"}
DAV2D=${DAV2D:-$ROOT/tools/bin/dav2d-macos}
BASE_ENV=(AV2ENC_EFF_PERC=1 AV2ENC_RD_UVFULL=1
          AV2ENC_TX_DETAIL_SEARCH_TAIL=1 AV2ENC_TX_DETAIL_MIN=0.50)
COMMON=(--video --frames 60 --cpu-threads 8 --nightly
        --video-drift-refresh-threshold 1.5 --video-drift-refresh-budget 3250)
mkdir -p "$OUT"
fail=0
arm_env() {
  case "$1" in
    base) ;;
    l3) echo "AV2ENC_RD_KEY_TX=1" ;;
    l5) echo "AV2ENC_RD_UVLEVEL=1" ;;
    l3l5) echo "AV2ENC_RD_KEY_TX=1 AV2ENC_RD_UVLEVEL=1" ;;
  esac
}
for qp in $QPS; do
  arms=(base l3 l5 l3l5)
  off=$(( qp % 4 ))
  rotated=()
  for i in 0 1 2 3; do rotated+=("${arms[$(( (i + off) % 4 ))]}"); done
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
    # `extra`/`diag` are intentionally unquoted: they are space-separated
    # VAR=value assignments (this script runs under bash, which splits them).
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
    [ $? -eq 0 ] || { echo "recon mismatch: $tag" >&2; fail=1; }
    echo "$tag $(stat -f%z "$ivf") $(md5 -q "$ivf") $(cat "$OUT/$tag.recon.ok")" \
        >> "$OUT/sizes.txt"
  done
done
echo "fail=$fail"
exit $fail
