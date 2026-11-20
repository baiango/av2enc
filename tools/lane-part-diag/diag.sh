#!/usr/bin/env bash
# P-frame partition diagnostic (docs/rd-part-diag.md).
#
# Measures both sides of every emitted P leaf on the production recipe and
# commits the reference, so the stream is unchanged.  Prints one summary line
# per (window, QP).
#
#   tools/lane-part-diag/diag.sh 114 116
#
# Environment must be an array (not a quoted string): zsh does not word-split
# `env $BASE`, and the first run of this lane silently dropped the recipe.
set -u
ROOT=$(cd -- "$(dirname -- "$0")/../.." && pwd)
BIN=${BIN:-$ROOT/build-release/av2enc}
REF=${REF:-$ROOT/testdata/nasa_480x240_30fps_60f.y4m}
OUT=${1:-/tmp/rd-part-diag}
shift || true
QPS=${*:-"114 116"}

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
  AV2ENC_PART_DIAG=1
)
COMMON=(
  --video --frames 60 --cpu-threads 8 --nightly
  --video-drift-refresh-threshold 1.5
  --video-drift-refresh-budget 3250
)

mkdir -p "$OUT"
for qp in $QPS; do
  for window in "1 32 head" "33 59 tail"; do
    set -- $window
    first=$1
    last=$2
    name=$3
    ivf="$OUT/${name}_q${qp}.ivf"
    echo "== $name QP$qp"
    env "${BASE_ENV[@]}" \
        AV2ENC_PART_DIAG_FIRST="$first" \
        AV2ENC_PART_DIAG_LAST="$last" \
        "$BIN" "$REF" -o "$ivf" "${COMMON[@]}" --qp "$qp" 2>&1 |
      grep "^rd-part-diag" || true
    if [ -s "$ivf" ]; then
      echo "   stream $(stat -f%z "$ivf") B $(md5 -q "$ivf")"
    fi
  done
done
