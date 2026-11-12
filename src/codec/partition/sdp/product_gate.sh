#!/usr/bin/env bash
# Product gate for the integrated SDP path (--still --still-sdp).
#
# Drives the product CLI with enable_sdp signalled in the sequence header.  The
# current gated configuration forces PARTITION_NONE on both the luma and the
# chroma tree of every intra 64x64 SDP split, which exercises the two-tree walk,
# PlaneStart-selected partition CDFs, the luma/chroma block split and chroma
# residual ownership.  The gate decodes with the pinned dav2d and requires the
# encoder reconstruction to be sample-exact on all three planes.
#
# Usage:
#   src/codec/partition/sdp/product_gate.sh [build-dir]
#
# Env: AV2ENC, DAV2D, SKIP_GATE=1
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../../.." && pwd)
BUILD=${1:-$ROOT/build-int}
AV2ENC=${AV2ENC:-$BUILD/av2enc}

if [ ! -x "$AV2ENC" ]; then
  echo "error: no encoder at $AV2ENC" >&2
  exit 2
fi
if [ -z "${DAV2D:-}" ]; then
  for c in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$c" ]; then DAV2D=$c; break; fi
  done
fi
if [ -z "${DAV2D:-}" ] || [ ! -x "$DAV2D" ]; then
  echo "error: dav2d not found; set DAV2D=/path/to/dav2d" >&2
  exit 127
fi
if [ "${SKIP_GATE:-0}" = "1" ]; then
  echo "sdp product gate: SKIP_GATE=1"
  exit 0
fi

TMP=$(mktemp -d "${TMPDIR:-/tmp}/sdp-product.XXXXXX")
trap 'rm -rf "$TMP"' EXIT

python3 - "$TMP" <<'PY'
import sys, os
out = sys.argv[1]
def write(w, h):
    p = os.path.join(out, f"src_{w}x{h}.y4m")
    with open(p, "wb") as f:
        f.write(b"YUV4MPEG2 W%d H%d F25:1 Ip A0:0 C420\n" % (w, h))
        f.write(b"FRAME\n")
        y = bytearray(w * h)
        for r in range(h):
            for c in range(w):
                y[r * w + c] = 220 if (((r >> 2) + (c >> 2)) & 1) else 40
        f.write(y)
        f.write(bytes([128]) * ((w // 2) * (h // 2) * 2))
# All sizes keep the 64x64 root's PARTITION_NONE legal (no forced sub-tree
# split), so the first gated NONE/NONE configuration applies.
for w, h in [(64, 64), (128, 128), (256, 192), (640, 384), (640, 360)]:
    write(w, h)
PY

cat > "$TMP/cmp.py" <<'PY'
import sys
def read(p):
    f = open(p, "rb")
    hdr = f.readline().strip().split()
    w = h = None; c = b"420"
    for x in hdr[1:]:
        if x[:1] == b"W": w = int(x[1:])
        elif x[:1] == b"H": h = int(x[1:])
        elif x[:1] == b"C": c = x[1:]
    cs = [(w, h), (w // 2, h // 2), (w // 2, h // 2)] if b"420" in c else [(w, h)] * 3
    out = []
    while True:
        line = f.readline()
        if not line or not line.startswith(b"FRAME"): break
        planes = []
        for pw, ph in cs:
            n = pw * ph; b = f.read(n)
            if len(b) != n: raise SystemExit("short read")
            planes.append(b)
        out.append(planes)
    return out
a = read(sys.argv[1]); b = read(sys.argv[2])
if len(a) != len(b): raise SystemExit("frame count differs")
for fi, (pa, pb) in enumerate(zip(a, b)):
    for pi, (x, y) in enumerate(zip(pa, pb)):
        if x != y:
            n = sum(1 for i in range(len(x)) if x[i] != y[i])
            m = max(abs(x[i] - y[i]) for i in range(len(x)))
            raise SystemExit(f"frame {fi} plane {pi}: {n} diffs max {m}")
print("recon exact")
PY

fail=0
for src in "$TMP"/src_*.y4m; do
  name=$(basename "$src" .y4m)
  for variant in default dc nocfl nodeblock; do
    cmd=("$AV2ENC" "$src" -o "$TMP/o.ivf" --still --still-sdp)
    case "$variant" in
      dc) cmd+=(--still-dc) ;;
      nocfl) cmd+=(--no-cfl) ;;
      nodeblock) cmd+=(--no-deblock) ;;
    esac
    cmd+=(--recon "$TMP/rec.y4m")
    if ! "${cmd[@]}" >/dev/null; then
      echo "FAIL $name ($variant): encode"; fail=1; continue
    fi
    if ! "$DAV2D" -q -i "$TMP/o.ivf" -o "$TMP/dec.y4m" --strict 1 --threads 1 >/dev/null 2>&1; then
      echo "FAIL $name ($variant): decode"; fail=1; continue
    fi
    if ! python3 "$TMP/cmp.py" "$TMP/rec.y4m" "$TMP/dec.y4m" >/dev/null; then
      echo "FAIL $name ($variant): recon mismatch"; fail=1; continue
    fi
    echo "PASS $name ($variant): decode + sample-exact recon"
  done
done

if [ "$fail" -ne 0 ]; then
  echo "sdp product gate: FAIL"
  exit 1
fi
echo "sdp product gate: PASS"
