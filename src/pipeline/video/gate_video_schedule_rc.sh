#!/usr/bin/env bash
# CPU gate for per-frame rate control on the reordering schedules:
#   --rc crf / --rc vbr with --b-frames (multi-GOP), --video-tlayers and
#   --svc-layers (temporal GOP).
#
# The rate-control planner walks the schedule's actual coding order and uses
# each picture's primary reference as its inter proxy; the writers apply
# frame_base_q_idx[display] to the encoder, the frame header and the SVC
# per-mlayer ladder.  Every case is decoded with the pinned dav2d at
# --strict 1 --threads 1 and compared sample for sample against the encoder's
# --video-recon dump.  The report must carry one QP per coded picture, VBR must
# stay within 10% of its target, and the default stream must stay byte-
# identical to --rc cqp.
#
# Usage:
#   src/pipeline/video/gate_video_schedule_rc.sh --bin build-int/av2enc \
#       [--dav2d /path/to/dav2d] [--input clip.y4m]
#
# Exit codes: 0 pass, 1 failure, 2 usage, 77 missing fixture/decoder.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/../../.." && pwd)
BIN=""
DAV2D="${DAV2D:-}"
INPUT="${AV2ENC_TESTDATA:-$ROOT/testdata}/shine_1280x720_60f.y4m"

while [ $# -gt 0 ]; do
  case "$1" in
    --bin) BIN=$2; shift 2 ;;
    --dav2d) DAV2D=$2; shift 2 ;;
    --input) INPUT=$2; shift 2 ;;
    -h|--help) sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "gate_video_schedule_rc: unknown argument $1" >&2; exit 2 ;;
  esac
done

[ -n "$BIN" ] && [ -x "$BIN" ] || {
  echo "gate_video_schedule_rc: --bin is required" >&2; exit 2; }
if [ -z "$DAV2D" ]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
  done
fi
[ -n "$DAV2D" ] && [ -x "$DAV2D" ] || {
  echo "gate_video_schedule_rc: dav2d not found (set DAV2D)" >&2; exit 77; }
[ -f "$INPUT" ] || {
  echo "gate_video_schedule_rc: missing fixture $INPUT" >&2; exit 77; }
command -v python3 >/dev/null 2>&1 || {
  echo "gate_video_schedule_rc: python3 is required" >&2; exit 77; }

OUT="${OUT:-${TMPDIR:-/tmp}/video_schedule_rc_gate}"
mkdir -p "$OUT"
FAIL=0
PASS=0
ok() { PASS=$((PASS + 1)); echo "  PASS: $*"; }
bad() { FAIL=$((FAIL + 1)); echo "  FAIL: $*"; }

FIX360="$OUT/fixture_rc360.y4m"
if [ ! -s "$FIX360" ]; then
  python3 - "$FIX360" <<'PY'
import math
import sys

w, h, frames = 640, 360, 8
cw, ch = w // 2, h // 2
with open(sys.argv[1], "wb") as out:
    out.write(b"YUV4MPEG2 W%d H%d F30:1 Ip A1:1 C420\n" % (w, h))
    for f in range(frames):
        out.write(b"FRAME\n")
        for y in range(h):
            out.write(bytes(int(128 + 60 * math.sin((x + 3 * f) * 0.35) +
                                40 * math.cos((y + 2 * f) * 0.22)) & 255
                            for x in range(w)))
        for p in (1, 2):
            for y in range(ch):
                out.write(bytes(int(128 + 30 * math.cos((2 * x + f) * 0.25) +
                                    (20 if p == 2 else 0)) & 255
                                for x in range(cw)))
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
if a != b:
    diffs = sum(1 for fa, fb in zip(a, b) for x, y in zip(fa, fb) if x != y)
    raise SystemExit(f"{diffs} sample difference(s)")
PY
}

# run_case <label> <fixture> <frames> -- flags
run_case() {
  local label=$1 fixture=$2 frames=$3
  shift 3
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
  if ! compare_recon "$recon" "$dec" "$frames"; then
    bad "$label: decoded pictures differ from the encoder reconstruction"
    return
  fi
  local qp_count
  qp_count=$(sed -n 's/^  frame QP: *//p' "$log" | tail -1 |
    wc -w | tr -d ' ')
  if [ "$qp_count" -ne "$frames" ]; then
    bad "$label: frame QP plan has $qp_count of $frames entries"; return
  fi
  ok "$label: $frames picture(s) strict, $qp_count QP entries, recon exact"
}

# check_vbr <label> <target-kbps>
check_vbr() {
  local label=$1 target=$2
  local log="$OUT/$label.log"
  if ! grep -q "closed-loop" "$log"; then
    bad "$label: report does not say closed-loop"; return
  fi
  local achieved
  achieved=$(grep -o 'achieved [0-9.]* kbps' "$log" | head -1 |
    awk '{print $2}')
  if [ -z "$achieved" ]; then
    bad "$label: no achieved-kbps report"; return
  fi
  if ! python3 - "$achieved" "$target" <<'PY'
import sys
achieved = float(sys.argv[1])
target = float(sys.argv[2])
raise SystemExit(0 if abs(achieved - target) / target <= 0.10 else 1)
PY
  then
    bad "$label: achieved $achieved kbps is more than 10% off $target kbps"
    return
  fi
  ok "$label: closed-loop achieved $achieved kbps for $target kbps"
}

run_case b_crf "$FIX360" 4 --b-frames 2 --rc crf --crf 30
run_case b_vbr "$FIX360" 8 --b-frames 2 --video-gops 2 --rc vbr --bitrate 800
check_vbr b_vbr 800
run_case tl_crf "$FIX360" 8 --video-tlayers 8 --rc crf --crf 30
run_case tl_vbr "$FIX360" 8 --video-tlayers 8 --rc vbr --bitrate 600
check_vbr tl_vbr 600
run_case svc_crf "$INPUT" 4 \
  --svc-layers 0=160x144 --svc-temporal 2 --rc crf --crf 30
run_case svc_vbr "$INPUT" 4 \
  --svc-layers 0=160x144 --svc-temporal 2 --rc vbr --bitrate 300
check_vbr svc_vbr 300

# --- defaults unchanged ---------------------------------------------------
"$BIN" "$FIX360" -o "$OUT/default.ivf" --video --frames 8 >/dev/null 2>&1
"$BIN" "$FIX360" -o "$OUT/cqp.ivf" --video --frames 8 --rc cqp \
  >/dev/null 2>&1
if cmp -s "$OUT/default.ivf" "$OUT/cqp.ivf"; then
  ok "default --video is byte-identical to --rc cqp"
else
  bad "--rc cqp is not byte-identical to the default --video stream"
fi

echo
if [ "$FAIL" -ne 0 ]; then
  echo "gate_video_schedule_rc: $FAIL failure(s), $PASS passed" >&2
  exit 1
fi
echo "gate_video_schedule_rc: $PASS/$PASS cases passed"
