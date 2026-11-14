#!/usr/bin/env bash
# CPU gate for multi-reference hierarchical-P temporal layers (--video-tlayers).
#
# The plan now carries two references per interior picture (the first inter
# picture keeps the key as its only reference).  Every reference is chosen from
# already-coded pictures with tlayer <= own, so the default §5.4.1
# TLayerDependencyMap (`ref_tlayer <= obu_tlayer_id`) admits both.  The §7.22
# motion fields therefore exist and §7.9 must project real candidates.
#
# Every case decodes with the pinned dav2d at --strict 1 --threads 1 and is
# compared sample for sample against the encoder reconstruction.  The
# --video-ref-mvs cases additionally require a non-zero temporal-candidate
# count, otherwise the gate would pass on an inert flag.
#
# Usage:
#   src/pipeline/video/gate_video_tlayer_refs.sh --bin build-int/av2enc \
#       [--dav2d /path/to/dav2d]
#
# Exit codes: 0 pass, 1 failure, 2 usage, 77 missing fixture/decoder.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/../../.." && pwd)
BIN=""
DAV2D="${DAV2D:-}"

while [ $# -gt 0 ]; do
  case "$1" in
    --bin) BIN=$2; shift 2 ;;
    --dav2d) DAV2D=$2; shift 2 ;;
    -h|--help) sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "gate_video_tlayer_refs: unknown argument $1" >&2; exit 2 ;;
  esac
done

[ -n "$BIN" ] && [ -x "$BIN" ] || {
  echo "gate_video_tlayer_refs: --bin is required" >&2; exit 2; }
if [ -z "$DAV2D" ]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
  done
fi
[ -n "$DAV2D" ] && [ -x "$DAV2D" ] || {
  echo "gate_video_tlayer_refs: dav2d not found (set DAV2D)" >&2; exit 77; }
command -v python3 >/dev/null 2>&1 || {
  echo "gate_video_tlayer_refs: python3 is required" >&2; exit 77; }

OUT="${OUT:-${TMPDIR:-/tmp}/video_tlayer_refs_gate}"
mkdir -p "$OUT"
FAIL=0
PASS=0
ok() { PASS=$((PASS + 1)); echo "  PASS: $*"; }
bad() { FAIL=$((FAIL + 1)); echo "  FAIL: $*"; }

FIX64="$OUT/fixture_trefs64.y4m"
FIX360="$OUT/fixture_trefs360.y4m"
if [ ! -s "$FIX64" ] || [ ! -s "$FIX360" ]; then
  python3 - "$FIX64" "$FIX360" <<'PY'
import math
import sys

def write(path, w, h, frames):
    cw, ch = w // 2, h // 2
    with open(path, "wb") as out:
        out.write(b"YUV4MPEG2 W%d H%d F30:1 Ip A1:1 C420\n" % (w, h))
        for f in range(frames):
            out.write(b"FRAME\n")
            for y in range(h):
                out.write(bytes(int(128 + 60 * math.sin((x + 3 * f) * 0.35) +
                                    40 * math.cos((y + 2 * f) * 0.22)) & 255
                                for x in range(w)))
            for p in (1, 2):
                for y in range(ch):
                    out.write(bytes(int(128 + 30 *
                                        math.cos((2 * x + f) * 0.25) +
                                        (20 if p == 2 else 0)) & 255
                                    for x in range(cw)))

write(sys.argv[1], 64, 64, 8)
write(sys.argv[2], 640, 360, 8)
PY
fi

compare_recon() {
  python3 - "$1" "$2" "$3" <<'PY'
import sys

def read_y4m(path):
    frames = []
    with open(path, "rb") as f:
        header = f.readline()
        if not header.startswith(b"YUV4MPEG2 "):
            raise SystemExit(f"{path}: not a Y4M file")
        width = height = None
        for field in header.split()[1:]:
            if field.startswith(b"W"):
                width = int(field[1:])
            elif field.startswith(b"H"):
                height = int(field[1:])
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

want = int(sys.argv[3])
a = read_y4m(sys.argv[1])
b = read_y4m(sys.argv[2])
if len(b) != want:
    raise SystemExit(f"decoded {len(b)} of {want} picture(s)")
if len(a) != want:
    raise SystemExit(f"encoder recon has {len(a)} of {want} picture(s)")
if a != b:
    diffs = sum(1 for fa, fb in zip(a, b) for x, y in zip(fa, fb) if x != y)
    raise SystemExit(f"{diffs} sample difference(s)")
PY
}

# run_case <label> <fixture> <frames> <pictures> <want-temporal> -- flags
run_case() {
  local label=$1 fixture=$2 frames=$3 pictures=$4 want_temporal=$5
  shift 5
  local ivf="$OUT/$label.ivf" recon="$OUT/$label.recon.y4m"
  local dec="$OUT/$label.dec.y4m" log="$OUT/$label.log"
  echo "== $label: $*"
  if ! "$BIN" "$fixture" -o "$ivf" --video --frames "$frames" \
      --video-recon "$recon" "$@" >"$log" 2>&1; then
    bad "$label: encode failed"; tail -8 "$log"; return
  fi
  if ! "$DAV2D" -q -i "$ivf" -o "$dec" --strict 1 --threads 1 \
      >"$OUT/$label.dec.log" 2>&1; then
    bad "$label: dav2d --strict 1 rejected the stream"
    tail -8 "$OUT/$label.dec.log"; return
  fi
  if ! compare_recon "$recon" "$dec" "$pictures"; then
    bad "$label: decoded pictures differ from the encoder reconstruction"
    return
  fi
  local got_temporal
  got_temporal=$(sed -n \
    's/.*temporal MV candidates \([0-9][0-9]*\).*/\1/p' "$log" | tail -1)
  got_temporal=${got_temporal:-0}
  if [ "$want_temporal" = "yes" ] && [ "$got_temporal" -le 0 ]; then
    bad "$label: §7.9 inert (0 temporal MV candidates)"; return
  fi
  ok "$label: $pictures picture(s) strict, temporal=$got_temporal, recon exact"
}

run_case tl4_64 "$FIX64" 4 4 no --video-tlayers 4
run_case tl8_64 "$FIX64" 8 8 no --video-tlayers 8
run_case tl4_360_refmvs "$FIX360" 4 4 yes \
  --video-tlayers 4 --video-ref-mvs
run_case tl8_360_refmvs "$FIX360" 8 8 yes \
  --video-tlayers 8 --video-ref-mvs
run_case tl4_drop0 "$FIX64" 4 2 no --video-tlayers 4 --video-tlayer-drop 0
run_case tl8_drop1 "$FIX64" 8 4 no --video-tlayers 8 --video-tlayer-drop 1

echo
if [ "$FAIL" -ne 0 ]; then
  echo "gate_video_tlayer_refs: $FAIL failure(s), $PASS passed" >&2
  exit 1
fi
echo "gate_video_tlayer_refs: $PASS/$PASS cases passed"
