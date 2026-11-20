#!/usr/bin/env bash
# Key-picture score for the sequential key tree (docs/rd-seq.md).
#
# Encodes one frame (the key picture), strict-decodes it, prints the key payload,
# PSNR-Y, Butteraugli and SSIMULACRA2 of that frame, and the rd-seqtree counters.
# The key frame is independent of the rest of the sequence (verified by the
# 60-frame frame-0 bits), so this is the cheap loop for the D-model sweep.
#
#   tools/lane-seq/keyscore.sh OUTDIR QP [extra env assignments...]
#
# Env: BIN, REF, DAV2D.  The base recipe env is always applied.
set -u
ROOT=$(cd -- "$(dirname -- "$0")/../.." && pwd)
BIN=${BIN:-$ROOT/build-release/av2enc}
REF=${REF:-$ROOT/testdata/nasa_480x240_30fps_60f.y4m}
DAV2D=${DAV2D:-$ROOT/tools/bin/dav2d-macos}
OUT=${1:?usage: keyscore.sh OUTDIR QP [env ...]}
QP=${2:?usage: keyscore.sh OUTDIR QP [env ...]}
shift 2
mkdir -p "$OUT"
REF1=$OUT/ref1.y4m
if [ ! -f "$REF1" ]; then
  python3 - "$REF" "$REF1" <<'PY'
import sys
data = open(sys.argv[1], 'rb').read()
hdr_end = data.index(b'\n') + 1
hdr = data[:hdr_end]
# 480x240 C420 = 172800 bytes per frame
open(sys.argv[2], 'wb').write(hdr + data[hdr_end:hdr_end + 6 + 172800])
PY
fi
tag="q${QP}"
for kv in "$@"; do tag="$tag.${kv%%=*}${kv#*=}"; done
ivf="$OUT/$tag.ivf"
env AV2ENC_EFF_PERC=1 AV2ENC_RD_UVFULL=1 AV2ENC_TX_DETAIL_SEARCH_TAIL=1 \
    AV2ENC_TX_DETAIL_MIN=0.50 AV2ENC_RD_KEY_TX=1 AV2ENC_RD_UVLEVEL=1 \
    "$@" "$BIN" --video --frames 1 --cpu-threads 8 --nightly --qp "$QP" \
    "$REF" -o "$ivf" > "$OUT/$tag.log" 2>&1 || { echo "$tag encode FAILED"; exit 1; }
"$DAV2D" --strict 1 -i "$ivf" -o "$OUT/$tag.dec.y4m" > "$OUT/$tag.dec.log" 2>&1 ||
  { echo "$tag decode FAILED"; exit 1; }
python3 "$ROOT/tools/perframe_psnr.py" "$REF" "$OUT/$tag.dec.y4m" "$ivf" --frames 1 \
  > "$OUT/$tag.psnr" 2>&1
python3 "$ROOT/tools/extra_metrics.py" --ref "$REF1" --test "$OUT/$tag.dec.y4m" \
  --json "$OUT/$tag.e.json" > "$OUT/$tag.e.log" 2>&1
python3 - "$OUT/$tag" "$ivf" "$@" <<'PY'
import json, os, re, sys
stem, ivf = sys.argv[1], sys.argv[2]
extra = ' '.join(sys.argv[3:])
try:
    e = json.load(open(stem + '.e.json'))
    ba = e.get('butteraugli_mean', float('nan'))
    ss2 = e.get('ssimulacra2_mean', float('nan'))
except Exception:
    ba = ss2 = float('nan')
psnr = bytes_b = None
for line in open(stem + '.psnr'):
    t = line.split()
    if len(t) >= 3 and t[0] == '0':
        psnr, bytes_ = float(t[1]), int(t[2])
stats = ''
log = open(stem + '.log', errors='replace').read()
m = re.search(r'^rd-driver: (.*)$', log, re.M)
if m:
    keep = {k: v for k, v in re.findall(r'(\w+)=(\S+)', m.group(1))
            if k in ('leaves', 'plans', 'rect', 'ms', 'Dsum', 'D_ratio',
                     'changed', 'nodes', 'rejected', 'D', 'R', 'J')}
    stats = ' '.join(f'{k}={v}' for k, v in keep.items())
print(f'{os.path.basename(stem):28} keyB={bytes_} psnr={psnr:.3f} '
      f'BA={ba:.4f} SS2={ss2:.3f} {stats} {extra}')
PY
