#!/usr/bin/env bash
# Oracle runs for docs/rd-tail-floor.md.
#
#   1. ladder:  five points per arm -> matched K (bracket + method) and the
#               nearest-point segment ceilings.
#   2. single:  one oracle run per stream, so each QP is its own nearest point
#               and reports that stream's source-replacement segment rows
#               (the per-QP ceiling curve).
#
#   tools/lane-tail-floor/oracle.sh /tmp/tf-ladder "113 114 115 116 117" ladder
#   tools/lane-tail-floor/oracle.sh /tmp/tf-ladder "113 114 115 116 117" single
set -u
ROOT=$(cd -- "$(dirname -- "$0")/../.." && pwd)
DIR=${1:-/tmp/rd-tail-floor-ladder}
QPS=${2:-"113 114 115 116 117"}
MODE=${3:-ladder}
PREFIX=${PREFIX:-f}
REF=${REF:-$ROOT/testdata/nasa_480x240_30fps_60f.y4m}
mkdir -p "$DIR/oracle"
case "$MODE" in
  ladder)
    args=()
    ARMS=${ARMS:-"base cand"}
    for qp in $QPS; do
      for arm in $ARMS; do
        args+=(--ivf "${PREFIX}${arm}@$qp=$DIR/${arm}_q$qp.ivf")
      done
    done
    python3 "$ROOT/tools/oracle.py" --ref "$REF" "${args[@]}" \
        --target-kbps 100 --json "$DIR/oracle/ladder.json" \
        > "$DIR/oracle/ladder.md" 2> "$DIR/oracle/ladder.log"
    echo "ladder oracle rc=$?"
    ;;
  single)
    ARMS=${ARMS:-"base cand"}
    for qp in $QPS; do
      for arm in $ARMS; do
        tag="${PREFIX}${arm}_q$qp"
        python3 "$ROOT/tools/oracle.py" --ref "$REF" \
            --ivf "$tag=$DIR/${arm}_q$qp.ivf" --target-kbps 100 \
            --json "$DIR/oracle/$tag.json" \
            > "$DIR/oracle/$tag.md" 2> "$DIR/oracle/$tag.log"
        echo "single $tag rc=$?"
      done
    done
    ;;
  *)
    echo "mode must be ladder or single" >&2
    exit 2
    ;;
esac
