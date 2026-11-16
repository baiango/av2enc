#!/usr/bin/env bash
# Interleaved A/B wall-clock: medians of N runs per binary, alternating.
#   ab_time.sh --a BIN --b BIN --reps 3 --fixture X --args "..."
set -uo pipefail
A=""; B=""; REPS=3; FIX=""; ARGS=""; LABEL=""
while [ $# -gt 0 ]; do
    case "$1" in
        --a) A=$2; shift 2;;
        --b) B=$2; shift 2;;
        --reps) REPS=$2; shift 2;;
        --fixture) FIX=$2; shift 2;;
        --args) ARGS=$2; shift 2;;
        --label) LABEL=$2; shift 2;;
        *) echo "bad arg $1" >&2; exit 2;;
    esac
done
TMP=$(mktemp -d "${TMPDIR:-/tmp}/abtime.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
ta=(); tb=(); ma=(); mb=()
for r in $(seq 1 "$REPS"); do
  for side in a b; do
    bin=$A; [ $side = b ] && bin=$B
    out="$TMP/$side-$r.ivf"
    s=$(python3 -c 'import time;print(f"{time.time():.4f}")')
    $bin "$FIX" -o "$out" $ARGS >"$TMP/$side-$r.log" 2>&1
    e=$(python3 -c 'import time;print(f"{time.time():.4f}")')
    t=$(python3 -c "print(f'{float($e)-float($s):.3f}')")
    m=$(md5 -q "$out")
    if [ $side = a ]; then ta+=("$t"); ma+=("$m"); else tb+=("$t"); mb+=("$m"); fi
    echo "  $LABEL $side rep$r ${t}s $m"
  done
done
python3 - "$LABEL" "${ta[*]}" "${tb[*]}" "${ma[*]}" "${mb[*]}" <<'PY'
import sys, statistics
label, a, b, ma, mb = sys.argv[1:6]
ta=[float(x) for x in a.split()]; tb=[float(x) for x in b.split()]
print(f"{label}: A median {statistics.median(ta):.3f}s {sorted(ta)}  B median {statistics.median(tb):.3f}s {sorted(tb)}  B/A {statistics.median(tb)/statistics.median(ta):.3f}")
print(f"  md5 A {set(ma.split())} B {set(mb.split())}")
PY
