#!/usr/bin/env bash
# One gate-on ablation point at QP115 (docs/rd-tail-floor.md §6).
#   tools/lane-tail-floor/ablation.sh /tmp/tf-abl "depth1" "AV2ENC_RD_TAIL_FLOOR_DEPTH=1"
#   tools/lane-tail-floor/ablation.sh /tmp/tf-abl "lambda2" "AV2ENC_RD_TAIL_FLOOR_LAMBDA=2.0"
set -u
ROOT=$(cd -- "$(dirname -- "$0")/../.." && pwd)
DIR=${1:-/tmp/tf-abl}
TAG=${2:?tag}
EXTRA=${3:-}
BIN=${BIN:-$ROOT/build-release/av2enc}
REF=${REF:-$ROOT/testdata/nasa_480x240_30fps_60f.y4m}
DAV2D=${DAV2D:-$ROOT/tools/bin/dav2d-macos}
mkdir -p "$DIR"
ivf="$DIR/${TAG}_q115.ivf"
# shellcheck disable=SC2086
env AV2ENC_EFF_PERC=1 AV2ENC_RD_UVFULL=1 AV2ENC_TX_DETAIL_SEARCH_TAIL=1 \
    AV2ENC_TX_DETAIL_MIN=0.50 AV2ENC_RD_KEY_TX=1 AV2ENC_RD_UVLEVEL=1 \
    AV2ENC_RD_DRIVER=1 AV2ENC_RD_TAIL_TX=1 AV2ENC_RD_TAIL_FLOOR=1 \
    AV2ENC_RD_TAIL_FLOOR_STATS=1 $EXTRA \
    "$BIN" "$REF" -o "$ivf" --qp 115 --video --frames 60 --cpu-threads 8 \
    --nightly --video-drift-refresh-threshold 1.5 \
    --video-drift-refresh-budget 3250 --video-recon "$DIR/${TAG}_q115.recon.y4m" \
    > "$DIR/${TAG}_q115.log" 2>&1 || { echo "encode failed: $TAG"; exit 1; }
"$DAV2D" --strict 1 -i "$ivf" -o "$DIR/${TAG}_q115.dec.y4m" \
    > "$DIR/${TAG}_q115.dec.log" 2>&1 || { echo "decode failed: $TAG"; exit 1; }
"$ROOT/tools/validate_dav2d.sh" "$ivf" "$REF" > "$DIR/${TAG}_q115.val.log" 2>&1 \
    || { echo "validate failed: $TAG"; exit 1; }
python3 "$ROOT/tools/lane-tail/perframe_metrics.py" --ref "$REF" \
    --test "$DIR/${TAG}_q115.dec.y4m" --frames 60 \
    --json "$DIR/${TAG}_q115.frames.json" > "$DIR/${TAG}_q115.frames.txt" 2>&1 \
    || { echo "metrics failed: $TAG"; exit 1; }
python3 "$ROOT/tools/oracle.py" --ref "$REF" \
    --ivf "$TAG@115=$ivf" --target-kbps 100 \
    --json "$DIR/oracle_$TAG.json" > "$DIR/oracle_$TAG.md" 2> "$DIR/oracle_$TAG.log" \
    || { echo "oracle failed: $TAG"; exit 1; }
echo "done $TAG $(stat -f%z "$ivf") bytes"
grep "rd-tail-floor" "$DIR/${TAG}_q115.log"
cat "$DIR/${TAG}_q115.frames.txt"
grep -A 3 "Matched rate" "$DIR/oracle_$TAG.md"
