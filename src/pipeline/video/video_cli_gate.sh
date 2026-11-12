#!/usr/bin/env bash
# CLI gate for the opt-in multi-frame inter path.
#
# Encodes N frames through the product CLI (`--video`), decodes the resulting
# IVF with the pinned dav2d through tools/validate_dav2d.sh, then compares every
# decoded sample against the encoder reconstruction dumped by --video-recon.
#
# Usage:
#   src/pipeline/video/video_cli_gate.sh --bin build-int/av2enc \
#       [--input testdata/shine_1280x720_60f.y4m] [--frames 4] [--dav2d PATH]
#       [-- extra encoder args...]
#
# The legacy positional form is still accepted:
#   src/pipeline/video/video_cli_gate.sh [BIN] [INPUT] [FRAMES] [extra args...]
#
# Defaults: BIN=build-int/av2enc, INPUT=testdata/shine_1280x720_60f.y4m,
# FRAMES=4.  Set OUT=/tmp/prefix to keep the streams.
#
# Exit codes: 0 pass, 1 failure, 2 usage, 77 missing fixture/decoder.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/../../.." && pwd)
BIN=""
INPUT=""
FRAMES=""
DAV2D="${DAV2D:-}"
TESTDATA="${AV2ENC_TESTDATA:-$ROOT/testdata}"
EXTRA=()
POSITIONAL=()

while [ $# -gt 0 ]; do
  case "$1" in
    --bin) BIN=$2; shift 2 ;;
    --input) INPUT=$2; shift 2 ;;
    --frames) FRAMES=$2; shift 2 ;;
    --dav2d) DAV2D=$2; shift 2 ;;
    --) shift; EXTRA+=("$@"); break ;;
    -h|--help)
      sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    -*) echo "video_cli_gate: unknown argument $1" >&2; exit 2 ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done

# Legacy positional form: BIN [INPUT] [FRAMES] [extra...].
if [ "${#POSITIONAL[@]}" -gt 0 ]; then
  [ -n "$BIN" ] || BIN=${POSITIONAL[0]}
  if [ "${#POSITIONAL[@]}" -gt 1 ] && [ -z "$INPUT" ]; then
    INPUT=${POSITIONAL[1]}
  fi
  if [ "${#POSITIONAL[@]}" -gt 2 ] && [ -z "$FRAMES" ]; then
    FRAMES=${POSITIONAL[2]}
  fi
  if [ "${#POSITIONAL[@]}" -gt 3 ]; then
    EXTRA+=("${POSITIONAL[@]:3}")
  fi
fi

BIN=${BIN:-"$ROOT/build-int/av2enc"}
INPUT=${INPUT:-"$TESTDATA/shine_1280x720_60f.y4m"}
FRAMES=${FRAMES:-4}

if [ ! -x "$BIN" ]; then
  echo "video_cli_gate: --bin must point at an av2enc executable" >&2
  exit 2
fi
if [ ! -f "$INPUT" ]; then
  echo "SKIP: missing input $INPUT" >&2
  exit 77
fi
if [ -z "$DAV2D" ]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
  done
fi
if [ -z "$DAV2D" ] || [ ! -x "$DAV2D" ]; then
  echo "video_cli_gate: no dav2d binary (set DAV2D=...)" >&2
  exit 77
fi
export DAV2D
if ! command -v python3 >/dev/null 2>&1; then
  echo "video_cli_gate: python3 is required for the recon comparison" >&2
  exit 77
fi

if [ -n "${OUT:-}" ]; then
  OUT="${OUT%*/}/video_cli"
else
  OUT="$(mktemp -d "${TMPDIR:-/tmp}/av2enc-video-gate.XXXXXX")/video_cli"
fi
mkdir -p "$(dirname -- "$OUT")"

echo "== video CLI gate: $INPUT ($FRAMES frame(s)) =="
"$BIN" "$INPUT" --video --frames "$FRAMES" \
  --video-recon "${OUT}.recon.y4m" -o "${OUT}.ivf" "${EXTRA[@]+"${EXTRA[@]}"}"

"$ROOT/tools/validate_dav2d.sh" "${OUT}.ivf"

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
