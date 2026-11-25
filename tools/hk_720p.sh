#!/bin/zsh
# Reproduce every number in docs/k-720p-levers.md on the 720p still.
#
# The docs' base recipe is `--nightly-b` at MIN=16, a lone key: the chained
# inter tools do not run, and this is the arm the 720p docs measure.
# `tools/hk_eval.py` walks QP to bracket --target in BYTES, `tools/hk_match.py`
# interpolates two arms at equal bytes.  Usage:
#
#   tools/hk_720p.sh base      # base + the whole filter stack off
#   tools/hk_720p.sh winner    # gdf + uv -23, then uv -16 and uv -23 alone
#   tools/hk_720p.sh inert     # every arm that measured inert or negative
#   tools/hk_720p.sh match     # equal-byte comparison of base vs winner
#
# Streams, decodes and scores go to $OUT (default ~/Downloads/av2enc-src720/hk).
set -u
cd /Users/arin/av2enc-publish
Y4M=${SRC:-$HOME/Downloads/av2enc-src720/src720_1f.y4m}
OUT=${OUT:-$HOME/Downloads/av2enc-src720/hk}
export AV2ENC_RD_DRIVER_MIN=${MIN:-16}
WIN=(--video-gdf --video-uv-qp-offset -23)

ev () {  # $1 = label, $2 = start QP, $3 = target, rest = arm
  local label=$1 qp=$2 tgt=$3; shift 3
  python3 tools/hk_eval.py --in $Y4M --target $tgt --out $OUT \
      --label $label --qp $qp --max-steps 6 -- "$@"
}

case ${1:-winner} in
base)
  ev base 185 25010 --nightly-b
  ev base_nofilters 175 25010 --nightly-b --video-loop-filters-off
  ;;
winner)
  ev uv16 186 25010 --nightly-b --video-gdf --video-uv-qp-offset -16
  ev uv23 187 25010 --nightly-b --video-gdf --video-uv-qp-offset -23
  # The lead at a second, higher rate: 33..36 kB, where it narrows.
  ev base_hi 179 35000 --nightly-b
  ev uv23_hi 179 35000 --nightly-b "${WIN[@]}"
  ;;
inert)
  # These must be byte-identical to `base` (same size, same K).
  ev inert_ctxrate 185 25010 --nightly-b --video-tx-context-rate
  ev inert_cfl 185 25010 --nightly-b "${WIN[@]}" --video-cfl
  ev inert_cdef4 185 25010 --nightly-b --video-cdef-strength 4
  # These must be worse.
  ev neg_uv0 185 25010 --nightly-b --video-uv-qp-offset 0
  ev neg_lr 190 25010 --nightly-b --video-wiener --video-chroma-lr
  ;;
match)
  python3 tools/hk_match.py --dir $OUT --base base --arm uv23 \
      --targets 25010 25500 26000 26500
  python3 tools/hk_match.py --dir $OUT --base base_hi --arm uv23_hi \
      --targets 34000 35000 36000
  ;;
*)
  echo "usage: $0 base|winner|inert|match" >&2
  exit 2
  ;;
esac
echo HK_720P_DONE
