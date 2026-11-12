#!/usr/bin/env bash
# Product gate for the 128x128 / 256x256 superblock paths combined with the
# superblock-size-dependent filter syntax (--still-sb128/--still-sb256 with
# --still-cdef / --still-ccso).
#
# What this proves:
#   * CDEF is signalled with cdef_sb_width_log2 = 5 on a 128x128 frame
#     superblock (`cdef_index0_context` above-superblock-row test).
#   * CCSO uses CcsoLumaSizeLog2 = 7 (ccso_unit_matches_sb_size = 1) for 128
#     units, i.e. one ccso_blk symbol per 128x128 luma unit.
#   * Both filters reconstruct exactly in the encoder and the pinned dav2d
#     match on all three planes.
#
# Usage: src/codec/partition/sb128/product_gate.sh [build-dir]
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
if [ "${SKIP_GATE:-0}" = "1" ]; then echo "sb-filter product gate: SKIP_GATE=1"; exit 0; fi

TMP=$(mktemp -d "${TMPDIR:-/tmp}/sbfilter-product.XXXXXX")
trap 'rm -rf "$TMP"' EXIT

python3 - "$TMP" "$SRC" <<'PY'
import sys, os
out, src = sys.argv[1], sys.argv[2]
f = open(src, "rb")
hdr = f.readline().strip().split()
W = H = None
for t in hdr[1:]:
    if t[:1] == b"W": W = int(t[1:])
    elif t[:1] == b"H": H = int(t[1:])
f.readline()
Y = f.read(W * H); U = f.read((W // 2) * (H // 2)); V = f.read((W // 2) * (H // 2))
def crop(x0, y0, w, h, name):
    y = bytearray()
    for r in range(y0, y0 + h):
        y += Y[r * W + x0:r * W + x0 + w]
    u = bytearray(); v = bytearray()
    for r in range(y0 // 2, (y0 + h) // 2):
        u += U[r * (W // 2) + x0 // 2:r * (W // 2) + x0 // 2 + w // 2]
        v += V[r * (W // 2) + x0 // 2:r * (W // 2) + x0 // 2 + w // 2]
    with open(os.path.join(out, name), "wb") as g:
        g.write(b"YUV4MPEG2 W%d H%d F25:1 Ip A0:0 C420\nFRAME\n" % (w, h))
        g.write(y); g.write(u); g.write(v)
crop(0, 0, W, H, "src_edge.y4m")       # 640x360: bottom/right edges
crop(0, 0, 256, 256, "src_256x256.y4m")  # 2x2 128-SB grid
crop(0, 0, 100, 70, "src_100x70.y4m")   # partial 128-SB in both axes
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
run() {
  name=$1; src=$2; shift 2
  if ! "$AV2ENC" "$src" -o "$TMP/o.ivf" --still "$@" --recon "$TMP/rec.y4m" >/dev/null; then
    echo "FAIL $name: encode"; fail=1; return
  fi
  if ! "$DAV2D" -q -i "$TMP/o.ivf" -o "$TMP/dec.y4m" --strict 1 --threads 1 >/dev/null 2>&1; then
    echo "FAIL $name: decode"; fail=1; return
  fi
  if ! python3 "$TMP/cmp.py" "$TMP/rec.y4m" "$TMP/dec.y4m" >/dev/null; then
    echo "FAIL $name: recon mismatch"; fail=1; return
  fi
  echo "PASS $name"
}

for sb in --still-sb128 --still-sb256; do
  for src in "$TMP/src_edge.y4m" "$TMP/src_256x256.y4m" "$TMP/src_100x70.y4m"; do
    base=$(basename "$src" .y4m)${sb#--still-}
    run "${base}_cdef" "$src" $sb --still-cdef
    run "${base}_ccso" "$src" $sb --still-ccso
    run "${base}_cdef_ccso" "$src" $sb --still-cdef --still-ccso
    run "${base}_cdef8" "$src" $sb --still-cdef --still-cdef-strengths 8
    run "${base}_ccso_edge" "$src" $sb --still-ccso --still-ccso-edge
    # §7.13.2.1 MRL uses sbHeight = Block_Height[SbSize]; a hardcoded 64
    # desynchronised every band below y=64 on a 128-SB frame.
    run "${base}_mrl" "$src" $sb --still-mrl
    run "${base}_mrl_cdef" "$src" $sb --still-mrl --still-cdef
    # §5.18.7.9 GdfBlkSize follows the frame SbSize (128) and the sequence
    # gdf_unit_matches_sb_size bit is inferred 0 outside a 64x64 sequence SB.
    run "${base}_gdf" "$src" $sb --still-gdf
    run "${base}_gdf_cdef" "$src" $sb --still-gdf --still-cdef
    # §5.18.7.11 limits a 128x128 SbSize to luma shift 2 / chroma shift 2,
    # i.e. one restoration unit per frame superblock.  The header size flags
    # and the §7.20 SFU/unit grid must both follow that SbSize.
    run "${base}_wiener" "$src" $sb --still-wiener
    run "${base}_chroma_lr" "$src" $sb --still-chroma-lr
    run "${base}_pc_wiener" "$src" $sb --still-pc-wiener
    run "${base}_wiener_chroma" "$src" $sb --still-wiener --still-chroma-lr
    run "${base}_cdef_wiener" "$src" $sb --still-cdef --still-wiener
    run "${base}_cdef_chroma_lr" "$src" $sb --still-cdef --still-chroma-lr
    run "${base}_gdf_wiener_chroma" "$src" $sb --still-gdf --still-wiener --still-chroma-lr
  done
done

if [ "$fail" -ne 0 ]; then echo "sb-filter product gate: FAIL"; exit 1; fi
echo "sb-filter product gate: PASS"
