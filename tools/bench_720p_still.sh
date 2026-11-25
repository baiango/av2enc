#!/usr/bin/env bash
# Interleaved A/B for the 720p still CPU-speed work (docs/cpu-speed-720p-still.md).
#
# Times the --nightly-b key on src720_1f.y4m at --cpu-threads 1 and 8, swapping
# between two encoder binaries on every run so host drift hits both arms, and
# prints the median per arm, its speedup and the md5 of every output (a speedup
# with a different md5 is a bug, not a result).
#
#   tools/bench_720p_still.sh --a build-base/av2enc --b build-release/av2enc
#   tools/bench_720p_still.sh --a old --b new --runs 5 --qp 177 --threads 1,8
#
# --src defaults to ~/Downloads/av2enc-src720/src720_1f.y4m.  That file is made
# from ~/Downloads/src720.png:
#
#   ffmpeg -i src720.png -pix_fmt yuv420p -color_range tv \
#          -colorspace bt709 -color_primaries bt709 -color_trc bt709 \
#          src720_1f.y4m                       # md5 80708336148219e2d20c502a43675f94
set -u
A=""
B=""
SRC=${SRC:-$HOME/Downloads/av2enc-src720/src720_1f.y4m}
RUNS=3
QP=177
THREADS=1,8
MIN=${MIN:-16}
OUTDIR=${OUTDIR:-${TMPDIR:-/tmp}/av2enc-720p-ab}

while [ $# -gt 0 ]; do
  case "$1" in
    --a) A=$2; shift 2 ;;
    --b) B=$2; shift 2 ;;
    --src) SRC=$2; shift 2 ;;
    --runs) RUNS=$2; shift 2 ;;
    --qp) QP=$2; shift 2 ;;
    --threads) THREADS=$2; shift 2 ;;
    --out) OUTDIR=$2; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "bench_720p_still: unknown argument: $1" >&2; exit 2 ;;
  esac
done

[ -n "$A" ] && [ -n "$B" ] || {
  echo "bench_720p_still: --a and --b are required (try --help)" >&2; exit 2; }
[ -f "$SRC" ] || {
  echo "bench_720p_still: missing input: $SRC (see --help for the ffmpeg line)" >&2
  exit 2; }
mkdir -p "$OUTDIR"

md5_of() { md5 -q "$1" 2>/dev/null || md5sum "$1" | cut -d' ' -f1; }

now() { python3 -c 'import time; print(time.time())'; }

run_one() {  # $1 = arm, $2 = binary, $3 = threads, $4 = rep
  local arm=$1 bin=$2 t=$3 rep=$4
  local out="$OUTDIR/${arm}_t${t}_r${rep}.ivf"
  local t0 t1
  t0=$(now)
  AV2ENC_RD_DRIVER_MIN=$MIN "$bin" "$SRC" -o "$out" --video --frames 1 \
    --qp "$QP" --cpu-threads "$t" --nightly-b --video-gdf \
    --video-uv-qp-offset -23 >/dev/null 2>&1
  local rc=$?
  t1=$(now)
  if [ $rc -ne 0 ] || [ ! -s "$out" ]; then
    echo "  $arm t=$t run $rep: FAILED rc=$rc" >&2
    return 1
  fi
  printf '%s %s %s\n' "$arm" "$(python3 -c "print(f'{$t1 - $t0:.2f}')")" "$(md5_of "$out")"
}

status=0
for t in ${THREADS//,/ }; do
  echo "== --cpu-threads $t, QP $QP, MIN $MIN, $RUNS reps =="
  rows_a=(); rows_b=(); md5_a=""; md5_b=""
  for r in $(seq 1 "$RUNS"); do
    for arm_bin in "a:$A" "b:$B"; do
      arm=${arm_bin%%:*}; bin=${arm_bin#*:}
      line=$(run_one "$arm" "$bin" "$t" "$r") || { status=1; continue; }
      tval=$(echo "$line" | awk '{print $2}')
      hval=$(echo "$line" | awk '{print $3}')
      echo "  $arm run $r: ${tval} s  ${hval}"
      if [ "$arm" = a ]; then rows_a+=("$tval"); md5_a+="$hval "; else rows_b+=("$tval"); md5_b+="$hval "; fi
    done
  done
  med_a=$(printf '%s\n' "${rows_a[@]}" | sort -n | awk '{v[NR]=$1} END {print (NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2}')
  med_b=$(printf '%s\n' "${rows_b[@]}" | sort -n | awk '{v[NR]=$1} END {print (NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2}')
  uniq_a=$(echo $md5_a | tr ' ' '\n' | sort -u | tr '\n' ' ')
  uniq_b=$(echo $md5_b | tr ' ' '\n' | sort -u | tr '\n' ' ')
  echo "  a median ${med_a} s   md5 ${uniq_a}"
  echo "  b median ${med_b} s   md5 ${uniq_b}"
  python3 -c "print(f'  speedup b vs a: {$med_a / $med_b:.3f}x')"
  if [ "$uniq_a" != "$uniq_b" ] || [ "$(echo $uniq_a | wc -w)" -ne 1 ]; then
    echo "  STREAMS DIFFER — that is a bug, not a result" >&2
    status=1
  else
    echo "  streams identical"
  fi
done
exit $status
