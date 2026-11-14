#!/usr/bin/env bash
# 4:2:2 16-pixel leaves (--video-max-depth 3/4) on the format video core.
#
# The partition module now models 4:2:2 chroma (Subsampled_Size[.][1][0]), so
# the tree is not luma-only and 16-pixel leaves (16x32 / 32x16) decode exactly.
# Every case is encoded with --video-recon, decoded with the pinned dav2d at
# --strict 1, and the decoded Y4M must equal the encoder reconstruction.
#
# Usage: gate_video_422_depth.sh --bin build/av2enc [--dav2d path]
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/../../.." && pwd)
BIN=""
DAV2D="${DAV2D:-}"
while [ $# -gt 0 ]; do
  case "$1" in
    --bin) BIN=$2; shift 2 ;;
    --dav2d) DAV2D=$2; shift 2 ;;
    -h|--help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "gate_video_422_depth: unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -n "$BIN" ] && [ -x "$BIN" ] || { echo "gate_video_422_depth: --bin required" >&2; exit 2; }
if [ -z "$DAV2D" ]; then
  for c in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    [ -x "$c" ] && DAV2D=$c && break
  done
fi
[ -n "$DAV2D" ] && [ -x "$DAV2D" ] || { echo "gate_video_422_depth: SKIP (no dav2d)"; exit 77; }
command -v python3 >/dev/null || { echo "gate_video_422_depth: SKIP (no python3)"; exit 77; }

TMP=$(mktemp -d "${TMPDIR:-/tmp}/gate_video_422_depth.XXXXXX")
trap 'rm -rf "$TMP"' EXIT

python3 - "$ROOT" "$TMP" <<'PY'
import sys, pathlib
root, tmp = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])

def load420(src):
    d = src.read_bytes()
    end = d.index(b'\n') + 1
    toks = d[:end].decode('latin1').split()
    w = int(next(t[1:] for t in toks if t.startswith('W')))
    h = int(next(t[1:] for t in toks if t.startswith('H')))
    pos = d.find(b'FRAME', end); pos = d.find(b'\n', pos) + 1
    cw, ch = (w + 1) // 2, (h + 1) // 2
    size = w * h + 2 * cw * ch
    body = d[pos:pos + size]
    y = body[:w * h]
    u = body[w * h:w * h + cw * ch]
    v = body[w * h + cw * ch:]
    return w, h, cw, ch, y, u, v

def write422(src, out, frames):
    w, h, cw, ch, y, u, v = load420(src)
    dw = (w + 1) // 2
    du = bytearray(dw * h); dv = bytearray(dw * h)
    for yy in range(h):
        for xx in range(dw):
            du[yy * dw + xx] = u[min(yy, ch - 1) * cw + min(xx, cw - 1)]
            dv[yy * dw + xx] = v[min(yy, ch - 1) * cw + min(xx, cw - 1)]
    with open(out, 'wb') as f:
        f.write(f'YUV4MPEG2 W{w} H{h} F30:1 Ip A0:0 C422\n'.encode())
        for _ in range(frames):
            f.write(b'FRAME\n'); f.write(y); f.write(bytes(du)); f.write(bytes(dv))
    return w, h, du, dv, y

def write422p10(src, out):
    d = src.read_bytes()
    end = d.index(b'\n') + 1
    toks = d[:end].decode('latin1').split()
    w = int(next(t[1:] for t in toks if t.startswith('W')))
    h = int(next(t[1:] for t in toks if t.startswith('H')))
    fs = w * h + 2 * ((w + 1) // 2) * h
    pos = d.find(b'FRAME', end); pos = d.find(b'\n', pos) + 1
    with open(out, 'wb') as f:
        f.write(f'YUV4MPEG2 W{w} H{h} F30:1 Ip A0:0 C422p10\n'.encode())
        i = pos
        while i + fs <= len(d):
            f.write(b'FRAME\n')
            for s in d[i:i + fs]:
                f.write(bytes([(s << 2) & 0xff, (s << 2) >> 8]))
            i += fs + 6

write422(root / 'tests/data/basic_64x64_1f.y4m', tmp / 'a64.y4m', 4)
write422(root / 'tests/data/basic_640x360_1f.y4m', tmp / 'a360.y4m', 4)
write422p10(tmp / 'a64.y4m', tmp / 'a64p10.y4m')
PY

FAIL=0
run_case() {
  local label=$1 frames=$2 clip=$3 depth=$4
  shift 4
  local ivf="$TMP/$label.ivf" rec="$TMP/$label.rec.y4m" dec="$TMP/$label.dec.y4m"
  if ! "$BIN" "$clip" -o "$ivf" --video --frames "$frames" \
      --chroma 422 --video-max-depth "$depth" --video-inter-residual \
      --video-deblock --video-recon "$rec" "$@" >"$TMP/$label.log" 2>&1; then
    echo "FAIL $label: encode failed"; tail -4 "$TMP/$label.log" >&2; FAIL=1; return
  fi
  if ! "$DAV2D" --threads 1 --strict 1 -q -i "$ivf" -o "$dec" >/dev/null 2>&1; then
    echo "FAIL $label: dav2d --strict 1 rejected the stream"; FAIL=1; return
  fi
  local pics
  pics=$(LC_ALL=C grep -a -o 'FRAME' "$dec" | wc -l | tr -d ' ')
  if [ "$pics" -ne "$frames" ]; then
    echo "FAIL $label: decoded $pics of $frames pictures"; FAIL=1; return
  fi
  if ! cmp -s "$rec" "$dec"; then
    echo "FAIL $label: recon != decoded"; FAIL=1; return
  fi
  local md5
  md5=$(md5 -q "$ivf" 2>/dev/null || md5sum "$ivf" | cut -d' ' -f1)
  echo "PASS $label: $pics picture(s), depth $depth, $md5"
}

# Default 4:2:2 remains 32x32-root (depth 2); explicit deeper trees now work.
run_case d2_64 4 "$TMP/a64.y4m" 2
run_case d3_64 4 "$TMP/a64.y4m" 3
run_case d4_64 4 "$TMP/a64.y4m" 4
run_case d2_360 4 "$TMP/a360.y4m" 2
run_case d3_360 4 "$TMP/a360.y4m" 3
run_case d4_360 4 "$TMP/a360.y4m" 4
run_case d3_64_p10 4 "$TMP/a64p10.y4m" 3 --bit-depth 10
run_case d4_64_p10 4 "$TMP/a64p10.y4m" 4 --bit-depth 10

# The deeper tree must actually change the stream (otherwise the gate is inert).
if cmp -s "$TMP/d2_64.ivf" "$TMP/d3_64.ivf"; then
  echo "FAIL: depth 3 stream is identical to depth 2"; FAIL=1
fi
if [ "$FAIL" -ne 0 ]; then
  echo "gate_video_422_depth: FAIL"; exit 1
fi
echo "gate_video_422_depth: PASS"
