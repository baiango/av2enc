#!/usr/bin/env bash
# dav2d gate for the standalone 10-bit M10 still picture.
#
#   ./gate_10bit_still.sh out.ivf [source.y4m] [recon.y4m]
#
# Checks:
#   1. dav2d decodes the IVF (exit 0);
#   2. the decoded Y4M is C420p10 (10-bit 4:2:0), with the IVF's dimensions;
#   3. the body is a whole number of 10-bit 4:2:0 frames;
#   4. when recon.y4m is given, every decoded sample equals the encoder
#      reconstruction (0 LSB mismatch, byte-for-byte body comparison);
#   5. when source.y4m is given, the project validate_dav2d.sh gate runs too.
#
# Exit codes: 0 ok, 1 decode/format/recon failure, 2 usage, 127 no decoder.
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 out.ivf [source.y4m] [recon.y4m]" >&2
  exit 2
fi

IVF=$1
SRC=${2:-}
RECON=${3:-}
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"

if [ ! -f "$IVF" ]; then
  echo "error: no such file: $IVF" >&2
  exit 2
fi

DAV2D="${DAV2D:-}"
if [ -z "$DAV2D" ]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
  done
fi
if [ -z "$DAV2D" ] || [ ! -x "$DAV2D" ]; then
  echo "error: dav2d not found; set DAV2D=/path/to/dav2d" >&2
  exit 127
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
DECODED="$TMP/decoded.y4m"

echo "decoder: $("$DAV2D" --version 2>&1 | tail -1)"
echo "input:   $IVF"
if [ -n "${EXPECTED_IVF_MD5:-}" ]; then
  if command -v md5sum >/dev/null 2>&1; then
    GOT=$(md5sum "$IVF" | awk '{print $1}')
  else
    GOT=$(md5 -q "$IVF")
  fi
  if [ "$GOT" != "$EXPECTED_IVF_MD5" ]; then
    echo "error: IVF MD5 $GOT != expected $EXPECTED_IVF_MD5" >&2
    exit 1
  fi
  echo "md5:     $GOT (ok)"
fi
if ! "$DAV2D" -q -i "$IVF" -o "$DECODED" --threads "${THREADS:-0}"; then
  echo "error: dav2d failed to decode $IVF" >&2
  exit 1
fi
if [ ! -s "$DECODED" ]; then
  echo "error: dav2d produced no output" >&2
  exit 1
fi

python3 - "$IVF" "$DECODED" "$RECON" <<'PY'
import re
import struct
import sys

ivf, decoded, recon = sys.argv[1], sys.argv[2], sys.argv[3]

def read_one_y4m(path):
    data = open(path, "rb").read()
    header_end = data.find(b"\n")
    if header_end < 0:
        raise SystemExit(f"error: {path}: no Y4M header line")
    header = data[:header_end].decode("latin1")
    if not header.startswith("YUV4MPEG2 "):
        raise SystemExit(f"error: {path}: not a Y4M file")
    m_w = re.search(r"\bW(\d+)", header)
    m_h = re.search(r"\bH(\d+)", header)
    m_c = re.search(r"\bC([^\s]+)", header)
    if not (m_w and m_h and m_c):
        raise SystemExit(f"error: {path}: header lacks W/H/C")
    width = int(m_w.group(1))
    height = int(m_h.group(1))
    chroma = m_c.group(1)
    cw, ch = (width + 1) // 2, (height + 1) // 2
    frame_bytes = 2 * (width * height + 2 * cw * ch)
    pos = header_end + 1
    if data[pos:pos + 6] != b"FRAME\n":
        raise SystemExit(f"error: {path}: missing FRAME tag")
    pos += 6
    body = data[pos:pos + frame_bytes]
    if len(body) != frame_bytes:
        raise SystemExit(f"error: {path}: truncated frame body")
    if pos + frame_bytes != len(data):
        raise SystemExit(
            f"error: {path}: expected exactly one frame; "
            f"{len(data) - (pos + frame_bytes)} trailing bytes")
    return width, height, chroma, body

width, height, chroma, body = read_one_y4m(decoded)
if chroma != "420p10":
    raise SystemExit(f"error: decoded chroma is {chroma}, expected 420p10")
frame_bytes = len(body)
print(f"format:  {width}x{height} {chroma}, 1 frame, {frame_bytes} bytes")

# Cross-check the IVF dimensions.
raw = open(ivf, "rb").read(32)
if len(raw) >= 32 and raw[:4] == b"DKIF":
    ivf_w, ivf_h = struct.unpack_from("<HH", raw, 12)
    if (ivf_w, ivf_h) != (width, height):
        raise SystemExit(
            f"error: IVF {ivf_w}x{ivf_h} != decoded {width}x{height}")
    print(f"ivf:     {ivf_w}x{ivf_h}, 1 frame")

if recon:
    rw, rh, rc, rbody = read_one_y4m(recon)
    if (rw, rh) != (width, height) or rc != "420p10":
        raise SystemExit(
            f"error: recon geometry {rw}x{rh} {rc} != {width}x{height} 420p10")
    if rbody != body:
        n = min(len(body), len(rbody))
        first = next((i for i in range(n) if body[i] != rbody[i]), n)
        raise SystemExit(
            f"error: decoded/recon mismatch at byte {first} "
            f"(sample {first // 2}); lengths {len(body)} vs {len(rbody)}")
    print("recon:   0-LSB mismatch (decoded plane == encoder recon, all planes)")
PY

if [ -n "$SRC" ]; then
  echo "quality:"
  "$ROOT/tools/validate_dav2d.sh" "$IVF" "$SRC"
fi

echo "OK: 10-bit 4:2:0 M10 still passed the dav2d gate"
