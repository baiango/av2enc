#!/usr/bin/env bash
# Interleaved 5-QP ladder for the chroma-RD lane (docs/rd-lane5.md).
#
# Encodes the base recipe and the candidate arm alternately at each QP, so the
# two ladders see the same machine state.  Nothing here touches the gate or the
# pins; it only writes streams and prints their sizes.
#
#   tools/lane-5/ladder.sh /tmp/l5/final
#
# Env overrides:
#   BIN=build-release/av2enc
#   REF=testdata/nasa_480x240_30fps_60f.y4m
#   QPS="118 119 120 121 122"
#   CAND_ENV="AV2ENC_RD_UVLEVEL=1 ..."   (extra candidate-arm assignments)
set -u
OUT=${1:-/tmp/rd-lane5-ladder}
BIN=${BIN:-build-release/av2enc}
REF=${REF:-testdata/nasa_480x240_30fps_60f.y4m}
QPS=${QPS:-"118 119 120 121 122"}
CAND_ENV=${CAND_ENV:-"AV2ENC_RD_UVLEVEL=1 AV2ENC_RD_UVLAMBDA=1.0 AV2ENC_RD_UVLEVEL_LAMBDA=0.125"}
BASE_ENV="AV2ENC_EFF_PERC=1 AV2ENC_RD_UVFULL=1 AV2ENC_TX_DETAIL_SEARCH_TAIL=1 AV2ENC_TX_DETAIL_MIN=0.50"
COMMON="--video --frames 60 --cpu-threads 8 --nightly --video-drift-refresh-threshold 1.5 --video-drift-refresh-budget 3250"
mkdir -p "$OUT"
for qp in $QPS; do
  for arm in base cand; do
    extra=$BASE_ENV
    [ "$arm" = cand ] && extra="$BASE_ENV $CAND_ENV"
    env $extra AV2ENC_RD_UVFULL_STATS=1 "$BIN" "$REF" \
      -o "$OUT/${arm}_q${qp}.ivf" $COMMON --qp "$qp" \
      >"$OUT/${arm}_q${qp}.out" 2>"$OUT/${arm}_q${qp}.log"
    printf '%-5s q%-3s %8s B\n' "$arm" "$qp" "$(stat -f%z "$OUT/${arm}_q${qp}.ivf" 2>/dev/null || stat -c%s "$OUT/${arm}_q${qp}.ivf")"
    grep -h 'uv-full' "$OUT/${arm}_q${qp}.log" || true
  done
done
