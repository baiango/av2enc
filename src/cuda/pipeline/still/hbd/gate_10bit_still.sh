#!/usr/bin/env bash
# Gate a 10-bit 4:2:0 still against the pinned dav2d decoder.
#
#   ./gate_10bit_still.sh out.ivf [source.y4m]
#
# This script cannot produce a 10-bit stream by itself: the M1 header writer
# still hard-codes bit_depth_idc=1 and an 8-bit base_q_idx.  Run it after the
# integration steps in README.md are applied.  It checks:
#   1. dav2d decodes the IVF (via tools/validate_dav2d.sh, the project gate);
#   2. the decoded Y4M is C420p10 (not C420, not C420p12);
#   3. the Y4M file size matches the 10-bit 4:2:0 frame geometry exactly.
#
# Exit codes: 0 ok, 1 format/decode failure, 2 usage, 127 no decoder.
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 out.ivf [source.y4m]" >&2
  exit 2
fi

IVF=$1
SRC=${2:-}
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done

if [ ! -f "$IVF" ]; then
  echo "error: no such file: $IVF" >&2
  exit 2
fi

DAV2D="${DAV2D:-}"
if [ -z "$DAV2D" ]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$candidate" ]; then
      DAV2D=$candidate
      break
    fi
  done
fi
if [ -z "$DAV2D" ] || [ ! -x "$DAV2D" ]; then
  echo "error: dav2d not found; set DAV2D=/path/to/dav2d" >&2
  exit 127
fi

TMPDIR_GATE="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_GATE"' EXIT
DECODED="$TMPDIR_GATE/decoded.y4m"

"$DAV2D" -q -i "$IVF" -o "$DECODED" --threads "${THREADS:-0}"

python3 - "$DECODED" <<'PY'
import os
import re
import sys

path = sys.argv[1]
with open(path, "rb") as f:
    header = f.readline()
    if not header.startswith(b"YUV4MPEG2 "):
        raise SystemExit("error: decoder output is not Y4M")
    text = header.decode("latin1")
    match_w = re.search(r"\bW(\d+)", text)
    match_h = re.search(r"\bH(\d+)", text)
    match_c = re.search(r"\bC([^\s]+)", text)
    if not (match_w and match_h and match_c):
        raise SystemExit("error: Y4M header is missing W/H/C")
    width = int(match_w.group(1))
    height = int(match_h.group(1))
    chroma = match_c.group(1)
    if chroma != "420p10":
        raise SystemExit(
            f"error: decoded chroma is {chroma}, expected 420p10 "
            "(10-bit 4:2:0)")
    frame_bytes = 6 + width * height * 2
    frame_bytes += 2 * ((width + 1) // 2) * ((height + 1) // 2) * 2
    body = os.path.getsize(path) - len(header)
    if body <= 0 or body % frame_bytes != 0:
        raise SystemExit(
            f"error: body {body} is not a whole number of {frame_bytes}-byte "
            f"{width}x{height} 420p10 frames")
    frames = body // frame_bytes
    print(f"10-bit check: {width}x{height} {chroma}, {frames} frame(s), "
          f"{frame_bytes} bytes/frame")
PY

# Project-standard decode/PSNR gate.
"$ROOT/tools/validate_dav2d.sh" "$IVF" ${SRC:+"$SRC"}

echo "OK: 10-bit 4:2:0 still passed the dav2d gate"
