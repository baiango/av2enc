#!/usr/bin/env bash
# CLI gate for the opt-in multi-frame inter path.
#
# Encodes N frames through the product CLI (`--video`), decodes the resulting
# IVF with the pinned dav2d through tools/validate_dav2d.sh, then compares every
# decoded sample against the encoder reconstruction dumped by --video-recon.
#
# Usage:
#   src/pipeline/video/video_cli_gate.sh [BIN] [INPUT] [FRAMES] [extra args...]
#
# Defaults: BIN=build-int/av2enc, INPUT=testdata/shine_1280x720_60f.y4m,
# FRAMES=4.  Set OUT=/tmp/prefix to keep the streams.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/../../.." && pwd)
BIN=${1:-"$ROOT/build-int/av2enc"}
INPUT=${2:-"$ROOT/testdata/shine_1280x720_60f.y4m"}
FRAMES=${3:-4}
if [ "$#" -gt 0 ]; then shift; fi
if [ "$#" -gt 0 ]; then shift; fi
if [ "$#" -gt 0 ]; then shift; fi
EXTRA=("$@")

OUT=${OUT:-"${TMPDIR:-/tmp}/video_cli_gate"}
OUT="${OUT%*/}/video_cli"

mkdir -p "$(dirname -- "$OUT")"

echo "== video CLI gate: $INPUT ($FRAMES frame(s)) =="
"$BIN" "$INPUT" --video --frames "$FRAMES" \
  --video-recon "${OUT}.recon.y4m" -o "${OUT}.ivf" "${EXTRA[@]+"${EXTRA[@]}"}"

"$ROOT/tools/validate_dav2d.sh" "${OUT}.ivf"

DAV2D=${DAV2D:-}
if [ -z "$DAV2D" ]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
  done
fi
if [ -z "$DAV2D" ]; then
  echo "error: dav2d not found" >&2
  exit 127
fi

"$DAV2D" -q -i "${OUT}.ivf" -o "${OUT}.decoded.y4m" --strict 1 --threads 1

python3 - "${OUT}.recon.y4m" "${OUT}.decoded.y4m" <<'PY'
import sys


def read_y4m(path):
    frames = []
    with open(path, "rb") as f:
        header = f.readline()
        if not header.startswith(b"YUV4MPEG2 "):
            raise SystemExit(f"{path}: not a Y4M file")
        fields = header.split()
        width = height = None
        for field in fields[1:]:
            if field.startswith(b"W"):
                width = int(field[1:])
            elif field.startswith(b"H"):
                height = int(field[1:])
        if width is None or height is None:
            raise SystemExit(f"{path}: missing dimensions")
        cw = (width + 1) // 2
        ch = (height + 1) // 2
        size = width * height + 2 * cw * ch
        while True:
            line = f.readline()
            if not line:
                break
            if not line.startswith(b"FRAME"):
                raise SystemExit(f"{path}: bad frame marker")
            data = f.read(size)
            if len(data) != size:
                raise SystemExit(f"{path}: truncated frame")
            frames.append(data)
    return frames


expected = read_y4m(sys.argv[1])
decoded = read_y4m(sys.argv[2])
if len(expected) != len(decoded):
    raise SystemExit(
        f"frame count mismatch: encoder recon {len(expected)} vs decoder {len(decoded)}")
diffs = 0
first = None
for index, (a, b) in enumerate(zip(expected, decoded)):
    if a == b:
        continue
    for pos, (x, y) in enumerate(zip(a, b)):
        if x != y:
            diffs += 1
            if first is None:
                first = (index, pos, x, y)
print(f"  recon compare: {len(expected)} frame(s), {diffs} sample diff(s)")
if diffs:
    print(f"  first diff: frame={first[0]} offset={first[1]} "
          f"dec={first[2]} enc={first[3]}")
    raise SystemExit(1)
print("PASS: decoder output matches encoder reconstruction sample for sample")
PY
