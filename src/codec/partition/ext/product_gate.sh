#!/usr/bin/env bash
# Product gate for the integrated extended-partition path
# (--still --still-ext-partitions), including the shared-chroma cases.
#
# The gate needs a tile that actually selects the shared-chroma rows:
#   HORZ_3@8x32  (middle 4x16 + 4x16 share one 8x16 chroma block)
#   VERT_3@32x8  (16x4 + 16x4 share one 16x8 chroma block)
#   HORZ_4A@32x32 (first three 32x4 children code no chroma; the last owns the
#                  32x32 chroma block)
# `testdata/shine_640x360_1f.y4m` does all three with the current SSE decision.
# The gate checks the partition trace for each case, decodes with the pinned
# dav2d and requires sample-exact Y/U/V reconstruction (deblock on and off).
#
# Usage: src/codec/partition/ext/product_gate.sh [build-dir]
# Env: AV2ENC, DAV2D, SKIP_GATE=1
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../../.." && pwd)
BUILD=${1:-$ROOT/build-int}
AV2ENC=${AV2ENC:-$BUILD/av2enc}
SRC=${SRC:-$ROOT/testdata/shine_640x360_1f.y4m}

if [ ! -x "$AV2ENC" ]; then echo "error: no encoder at $AV2ENC" >&2; exit 2; fi
if [ ! -f "$SRC" ]; then echo "SKIP: missing $SRC" >&2; exit 77; fi
if [ -z "${DAV2D:-}" ]; then
  for c in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$c" ]; then DAV2D=$c; break; fi
  done
fi
if [ -z "${DAV2D:-}" ] || [ ! -x "$DAV2D" ]; then
  echo "error: dav2d not found; set DAV2D=/path/to/dav2d" >&2; exit 127
fi
if [ "${SKIP_GATE:-0}" = "1" ]; then echo "ext product gate: SKIP_GATE=1"; exit 0; fi

TMP=$(mktemp -d "${TMPDIR:-/tmp}/ext-product.XXXXXX")
trap 'rm -rf "$TMP"' EXIT

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
for variant in deblock nodeblock; do
  cmd=("$AV2ENC" "$SRC" -o "$TMP/o.ivf" --still --still-ext-partitions)
  [ "$variant" = "nodeblock" ] && cmd+=(--no-deblock)
  cmd+=(--recon "$TMP/rec.y4m")
  trace=$("${cmd[@]}" 2>&1) || { echo "FAIL ($variant): encode"; fail=1; continue; }
  if ! grep -q "m1 ext node" <<<"$trace"; then
    trace=$(AV2ENC_STILL_TRACE=1 "${cmd[@]}" 2>&1) || true
  fi
  if ! "$DAV2D" -q -i "$TMP/o.ivf" -o "$TMP/dec.y4m" --strict 1 --threads 1 >/dev/null 2>&1; then
    echo "FAIL ($variant): decode"; fail=1; continue
  fi
  if ! python3 "$TMP/cmp.py" "$TMP/rec.y4m" "$TMP/dec.y4m" >/dev/null; then
    echo "FAIL ($variant): recon mismatch"; fail=1; continue
  fi
  echo "PASS ($variant): decode + sample-exact recon"
done

# Coverage check: the decision must select each shared-chroma row at least
# once, otherwise the recon gate above proves nothing about them.
TRACE=$(AV2ENC_STILL_TRACE=1 "$AV2ENC" "$SRC" -o "$TMP/t.ivf" --still \
        --still-ext-partitions --no-deblock 2>&1 || true)
for pat in "8x32 part=3" "32x8 part=4" "32x32 part=5"; do
  if grep -q "$pat" <<<"$TRACE"; then
    echo "PASS cover $pat"
  else
    echo "FAIL cover $pat (shared-chroma row not selected)"; fail=1
  fi
done

if [ "$fail" -ne 0 ]; then echo "ext product gate: FAIL"; exit 1; fi
echo "ext product gate: PASS"
