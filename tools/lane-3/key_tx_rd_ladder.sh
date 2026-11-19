#!/usr/bin/env bash
# Key-intra TX/level RD ladder (AV2ENC_RD_KEY_TX, docs/rd-lane3.md).
#
# Encodes a base and/or a gated arm over an interleaved QP ladder, decodes every
# stream with the pinned dav2d, scores CVVDP / Butteraugli / SSIMULACRA2, records
# the key-frame bytes/PSNR and interpolates K at 100 kbps linearly in rate (the
# docs/k-oracle.md method).
#
# Usage:
#   tools/lane-3/key_tx_rd_ladder.sh [--bin build-release/av2enc] [--out DIR]
#                                    [--qps 117,118,119,120,121]
#                                    [--arms both|base|arm] [--base-dir DIR]
#                                    [--arm-scale 0.25] [--reuse]
#
# `--arms base` and `--arms arm` split a ladder so several TX weights can share
# one base run; `--base-dir` points the arm run at that base directory.
#
# Exit: 0 every stream decoded and has metrics; 1 otherwise.
set -uo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/../.." && pwd)
BIN=$ROOT/build-release/av2enc
OUT=${TMPDIR:-/tmp}/rd_lane3_ladder
QPS=117,118,119,120,121
REF=$ROOT/testdata/nasa_480x240_30fps_60f.y4m
ARMS=both
BASE_DIR=""
ARM_SCALE=0.25
REUSE=0

while [ $# -gt 0 ]; do
    case "$1" in
        --bin) BIN=${2:-}; shift 2 ;;
        --out) OUT=${2:-}; shift 2 ;;
        --qps) QPS=${2:-}; shift 2 ;;
        --arms) ARMS=${2:-}; shift 2 ;;
        --base-dir) BASE_DIR=${2:-}; shift 2 ;;
        --arm-scale) ARM_SCALE=${2:-}; shift 2 ;;
        --reuse) REUSE=1; shift ;;
        -h|--help) sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "key_tx_rd_ladder: unknown argument $1" >&2; exit 2 ;;
    esac
done

case "$ARMS" in both|base|arm) ;; *) echo "key_tx_rd_ladder: --arms must be both|base|arm" >&2; exit 2 ;; esac
[ -x "$BIN" ] || { echo "key_tx_rd_ladder: no binary at $BIN" >&2; exit 2; }
[ -f "$REF" ] || { echo "key_tx_rd_ladder: missing $REF" >&2; exit 2; }
DAV2D=${DAV2D:-$ROOT/tools/bin/dav2d-macos}
[ -x "$DAV2D" ] || { echo "key_tx_rd_ladder: no dav2d at $DAV2D" >&2; exit 2; }
mkdir -p "$OUT"
[ -n "$BASE_DIR" ] || BASE_DIR=$OUT

RECIPE=(--video --frames 60 --cpu-threads 8 --nightly
        --video-drift-refresh-threshold 1.5 --video-drift-refresh-budget 3250)
BASE_ENV=(AV2ENC_EFF_PERC=1 AV2ENC_RD_UVFULL=1
          AV2ENC_TX_DETAIL_SEARCH_TAIL=1 AV2ENC_TX_DETAIL_MIN=0.50)

fail=0
case "$ARMS" in both) ARM_LIST="base arm" ;; base) ARM_LIST="base" ;; arm) ARM_LIST="arm" ;; esac
for qp in ${QPS//,/ }; do
    for arm in $ARM_LIST; do
        tag=${arm}${qp}
        ivf=$OUT/$tag.ivf
        if [ "$REUSE" = 1 ] && [ -f "$OUT/$tag.k.json" ] && [ -f "$OUT/$tag.psnr" ]; then
            echo "cached $tag"
            continue
        fi
        if [ "$arm" = arm ]; then
            env "${BASE_ENV[@]}" AV2ENC_RD_KEY_TX=1 \
                AV2ENC_RD_KEY_TX_SCALE="$ARM_SCALE" \
                "$BIN" "$REF" -o "$ivf" --qp "$qp" "${RECIPE[@]}" \
                > "$OUT/$tag.log" 2>&1 || fail=1
        else
            env "${BASE_ENV[@]}" \
                "$BIN" "$REF" -o "$ivf" --qp "$qp" "${RECIPE[@]}" \
                > "$OUT/$tag.log" 2>&1 || fail=1
        fi
        "$DAV2D" --strict 1 -i "$ivf" -o "$OUT/$tag.dec.y4m" \
            > "$OUT/$tag.dec.log" 2>&1 || { echo "decode failed: $tag" >&2; fail=1; }
        DAV2D=$DAV2D python3 "$ROOT/tools/quality.py" --ref "$REF" --ivf "$ivf" \
            --json "$OUT/$tag.q.json" > "$OUT/$tag.q.log" 2>&1 || fail=1
        python3 "$ROOT/tools/extra_metrics.py" --ref "$REF" \
            --test "$OUT/$tag.dec.y4m" --json "$OUT/$tag.e.json" \
            > "$OUT/$tag.e.log" 2>&1 || fail=1
        python3 "$ROOT/tools/kpi.py" --quality-json "$OUT/$tag.q.json" \
            --extra-json "$OUT/$tag.e.json" --json "$OUT/$tag.k.json" \
            > "$OUT/$tag.k.log" 2>&1 || fail=1
        python3 "$ROOT/tools/perframe_psnr.py" "$OUT/$tag.dec.y4m" "$REF" "$ivf" \
            --frames 60 > "$OUT/$tag.psnr" 2>&1 || fail=1
        echo "done $tag"
    done
done

python3 - "$OUT" "$BASE_DIR" "$ARMS" "$QPS" <<'PY'
import json, sys
out, base_dir, arms, qps_s = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
qps = [int(x) for x in qps_s.split(',')]
def load(dir_, tag):
    q = json.load(open(f'{dir_}/{tag}.q.json'))
    k = json.load(open(f'{dir_}/{tag}.k.json'))
    keyb = keyp = None
    try:
        for line in open(f'{dir_}/{tag}.psnr'):
            t = line.split()
            if len(t) >= 3 and t[0] == '0':
                keyp, keyb = float(t[1]), int(t[2])
                break
    except FileNotFoundError:
        pass
    return dict(kbps=q['bitrate_kbps'], jod=q['cvvdp_jod'], ba=k['butteraugli'],
                ss2=k['ssimulacra2'], kpi=k['kpi'], keyb=keyb, keyp=keyp)
rows = {}
print(f"{'tag':10} {'kbps':>7} {'JOD':>7} {'BA':>6} {'SS2':>7} {'KPI':>7} {'keyB':>6} {'keyP':>7}")
for arm in ('base', 'arm'):
    src = base_dir if arm == 'base' else out
    for qp in qps:
        tag = f'{arm}{qp}'
        try:
            r = load(src, tag)
        except FileNotFoundError as exc:
            if arms == 'both' or arms == arm:
                print(f"missing {tag}: {exc}")
            continue
        rows[tag] = r
        if arms == 'both' or arms == arm:
            print(f"{tag:10} {r['kbps']:7.2f} {r['jod']:7.4f} {r['ba']:6.3f} "
                  f"{r['ss2']:7.2f} {r['kpi']:7.3f} {str(r['keyb']):>6} {r['keyp']:7.3f}")
def interp(prefix):
    if not all(f'{prefix}{qp}' in rows for qp in qps):
        return None
    pts = sorted((rows[f'{prefix}{qp}']['kbps'], rows[f'{prefix}{qp}']['kpi']) for qp in qps)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= 100.0 <= x1:
            return y0 + (y1 - y0) * (100.0 - x0) / (x1 - x0)
    return None
b = interp('base')
a = interp('arm')
if b is not None:
    print(f"base K@100 = {b:.3f}")
if a is not None:
    print(f"arm  K@100 = {a:.3f}" + (f"   delta = {a - b:+.3f}" if b is not None else ""))
PY
exit $fail
