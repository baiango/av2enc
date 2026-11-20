#!/usr/bin/env bash
# Band-D ladder: does a high-pass error weight (the 1-px band) buy K?
#
# The week-1 census found that within a tail frame, local SS2 tracks the 1-px
# error band (within-frame r = -0.82 across 120x120 tiles), while the
# frame-to-frame variance is content.  The encoder already has the band term
# as AV2ENC_RD_TAIL_HF on the exact-rate TX objective.  This ladder measures
# it on the merged opt-in base.
#
#   tools/lane-band/ladder.sh /tmp/rd-band "114 116 118" "opt hf5 hf20 hf50"
set -u
ROOT=$(cd -- "$(dirname -- "$0")/../.." && pwd)
BIN=${BIN:-$ROOT/build-release/av2enc}
REF=${REF:-$ROOT/testdata/nasa_480x240_30fps_60f.y4m}
DAV2D=${DAV2D:-$ROOT/tools/bin/dav2d-macos}
OUT=${1:-/tmp/rd-band}
QPS=${2:-"114 116 118"}
ARMS=${3:-"opt hf20"}
BASE_ENV=(
  AV2ENC_EFF_PERC=1
  AV2ENC_RD_UVFULL=1
  AV2ENC_TX_DETAIL_SEARCH_TAIL=1
  AV2ENC_TX_DETAIL_MIN=0.50
  AV2ENC_RD_KEY_TX=1
  AV2ENC_RD_UVLEVEL=1
  AV2ENC_RD_DRIVER=1
  AV2ENC_RD_HEAD_TX=1
  AV2ENC_RD_TAIL_TX=1
)
COMMON=(
  --video --frames 60 --cpu-threads 8 --nightly
  --video-drift-refresh-threshold 1.5
  --video-drift-refresh-budget 3250
)
mkdir -p "$OUT"
fail=0
arm_env() {
  case "$1" in
    opt) : ;;
    hf5) echo "AV2ENC_RD_TAIL_HF=5" ;;
    hf20) echo "AV2ENC_RD_TAIL_HF=20" ;;
    hf50) echo "AV2ENC_RD_TAIL_HF=50" ;;
    *) echo "" ;;
  esac
}
read -r -a arm_list <<< "$ARMS"
for qp in $QPS; do
  for arm in "${arm_list[@]}"; do
    tag="${arm}_q${qp}"
    ivf="$OUT/$tag.ivf"
    extra=$(arm_env "$arm")
    # shellcheck disable=SC2086
    if ! env "${BASE_ENV[@]}" $extra "$BIN" "$REF" -o "$ivf" \
         --qp "$qp" "${COMMON[@]}" --video-recon "$OUT/$tag.recon.y4m" \
         > "$OUT/$tag.log" 2>&1; then
      echo "encode failed: $tag" >&2
      fail=1
      continue
    fi
    "$DAV2D" --strict 1 -i "$ivf" -o "$OUT/$tag.dec.y4m" \
        > "$OUT/$tag.dec.log" 2>&1 ||
        { echo "decode failed: $tag" >&2; fail=1; }
    python3 "$ROOT/tools/perframe_psnr.py" "$REF" "$OUT/$tag.dec.y4m" "$ivf" \
        --frames 60 > "$OUT/$tag.psnr" 2>&1 || fail=1
    python3 - "$OUT/$tag.recon.y4m" "$OUT/$tag.dec.y4m" <<'PY'
import sys
a = open(sys.argv[1], 'rb').read(); b = open(sys.argv[2], 'rb').read()
print('recon', 'ok' if a[a.find(b'FRAME'):] == b[b.find(b'FRAME'):] else 'MISMATCH')
PY
    echo "done $tag $(stat -f%z "$ivf") bytes"
  done
done
ivfs=()
for qp in $QPS; do
  for arm in "${arm_list[@]}"; do
    [ -f "$OUT/${arm}_q${qp}.ivf" ] && ivfs+=(--ivf "${arm}@${qp}=$OUT/${arm}_q${qp}.ivf")
  done
done
python3 "$ROOT/tools/oracle.py" --ref "$REF" "${ivfs[@]}" \
    --target-kbps 100 > "$OUT/oracle.md" 2>&1
echo "ladder done fail=$fail"
exit "$fail"
