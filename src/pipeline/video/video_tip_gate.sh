#!/usr/bin/env bash
# CPU gate for the --video-tip §7.10.3/.4 hole fill on a coded reference field.
#
# The product TIP stream signals §5.4.6 enable_tip_hole_fill = 1 and the
# per-frame §5.18.2 allow_tip_hole_fill bit, so the pinned decoder runs the
# same hole fill / block-average filter as the encoder.  The gate proves that
# claim end to end: every TIP case is encoded with the product CLI, decoded
# with dav2d --strict 1 --threads 1, and the decoder output must be
# sample-for-sample equal (Y4M MD5) to the encoder's --video-recon dump.
#
# It also covers the two ways the stream would look "exact" for the wrong
# reason:
#   * a hole-fill-off control (--video-tip-no-hole-fill) must also be exact,
#     and its stream/recon must DIFFER from the hole-fill-on case.  If the
#     two reconstructions were equal the fill would be inert and the gate
#     would be vacuous; if the on case did not decode exactly, the frame-level
#     allow_tip_hole_fill bit (or the published field) is wrong.
#   * 4 and 8 pictures exercise one and six chained TIP frames, so a TIP frame
#     stored as a reference for the next TIP is covered too.
#
# Fixtures are the gitignored testdata/ Y4M clips.  The 640x360 4-frame clip
# is the one with a sparse coded anchor field (many §7.10.3 holes); the 720p
# clip covers 8 pictures.  Missing fixtures or decoder => exit 77 (skip).
#
# Usage:
#   src/pipeline/video/video_tip_gate.sh --bin build/av2enc \
#       [--input clip8.y4m] [--input-360 clip4.y4m] [--dav2d PATH]
#
# Exit codes: 0 pass, 1 failure, 2 usage, 77 missing fixture/decoder.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/../../.." && pwd)
TESTDATA="${AV2ENC_TESTDATA:-$ROOT/testdata}"
BIN=""
INPUT="$TESTDATA/shine_1280x720_60f.y4m"
INPUT_360="$TESTDATA/shine_640x360_4f.y4m"
DAV2D="${DAV2D:-}"

while [ $# -gt 0 ]; do
  case "$1" in
    --bin) BIN=$2; shift 2 ;;
    --input) INPUT=$2; shift 2 ;;
    --input-360) INPUT_360=$2; shift 2 ;;
    --dav2d) DAV2D=$2; shift 2 ;;
    -h|--help) sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "video_tip_gate: unknown argument $1" >&2; exit 2 ;;
  esac
done

[ -n "$BIN" ] || { echo "video_tip_gate: --bin is required" >&2; exit 2; }
[ -x "$BIN" ] || { echo "video_tip_gate: no such binary $BIN" >&2; exit 2; }
if [ -z "$DAV2D" ]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
  done
fi
if [ -z "$DAV2D" ] || [ ! -x "$DAV2D" ]; then
  echo "video_tip_gate: dav2d not found (set DAV2D)" >&2
  exit 77
fi
[ -f "$INPUT" ] || { echo "video_tip_gate: missing fixture $INPUT" >&2; exit 77; }

# The 8-picture case is the six-TIP chain; a shorter clip would silently code
# fewer pictures and fail the count assertion for the wrong reason.  Y4M frame
# payloads carry no trailing newline, so step over each frame instead of
# grepping for FRAME.
Y4M_FRAMES_PY=$(cat <<'PY'
import sys
data = open(sys.argv[1], 'rb').read()
lines = data.split(b'\n', 1)
if len(lines) != 2 or not lines[0].startswith(b'YUV4MPEG2'):
    print(0)
    raise SystemExit
w = h = 0
chroma = '420'
for tag in lines[0].split():
    if tag.startswith(b'W'):
        w = int(tag[1:])
    elif tag.startswith(b'H'):
        h = int(tag[1:])
    elif tag.startswith(b'C'):
        chroma = tag[1:].decode()
if chroma.startswith('444'):
    frame = w * h * 3
elif chroma.startswith('422'):
    frame = w * h * 2
elif chroma.startswith('mono'):
    frame = w * h
else:
    frame = w * h + 2 * ((w + 1) // 2) * ((h + 1) // 2)
body = lines[1]
count = 0
while body.startswith(b'FRAME'):
    newline = body.find(b'\n')
    if newline < 0:
        break
    body = body[newline + 1 + frame:]
    count += 1
print(count)
PY
)
y4m_frames() {
  command -v python3 >/dev/null 2>&1 || { echo 8; return; }
  python3 -c "$Y4M_FRAMES_PY" "$1" 2>/dev/null || echo 0
}
INPUT_FRAMES=$(y4m_frames "$INPUT")
INPUT_FRAMES=${INPUT_FRAMES:-0}
if [ "$INPUT_FRAMES" -lt 8 ]; then
  echo "video_tip_gate: $INPUT has $INPUT_FRAMES frame(s), need 8" >&2
  exit 77
fi
if [ -f "$INPUT_360" ] && [ "$(y4m_frames "$INPUT_360")" -lt 4 ]; then
  echo "video_tip_gate: $INPUT_360 has fewer than 4 frames; using $INPUT"
  INPUT_360=""
fi

OUT="${OUT:-${TMPDIR:-/tmp}/video_tip_gate}"
mkdir -p "$OUT"
FAIL=0
CASES=0
LAST_STREAM_MD5=""
LAST_RECON_MD5=""

md5() { md5sum "$1" 2>/dev/null | cut -d' ' -f1 || md5 -q "$1"; }

# run_case <label> <input> <frames> <extra encoder args...>
run_case() {
  local label=$1 input=$2 frames=$3
  shift 3
  local ivf="$OUT/$label.ivf" recon="$OUT/$label.recon.y4m"
  local dec="$OUT/$label.dec.y4m" log="$OUT/$label.log"
  echo "== $label: $frames picture(s)${*:+ ($*)}"
  if ! "$BIN" "$input" -o "$ivf" --video --frames "$frames" --video-tip \
      --video-recon "$recon" "$@" >"$log" 2>&1; then
    echo "FAIL($label): encode failed"; tail -5 "$log"; FAIL=1; return
  fi
  if ! "$DAV2D" -q -i "$ivf" -o "$dec" --strict 1 --threads 1; then
    echo "FAIL($label): dav2d --strict 1 rejected the stream"; FAIL=1; return
  fi
  local recon_md5 dec_md5
  recon_md5=$(md5 "$recon"); dec_md5=$(md5 "$dec")
  if [ "$recon_md5" != "$dec_md5" ]; then
    echo "FAIL($label): recon mismatch (encoder $recon_md5 decoder $dec_md5)"
    FAIL=1; return
  fi
  # Count decoded pictures without depending on the muxer's summary message.
  local dec_frames
  dec_frames=$(LC_ALL=C grep -a -o 'FRAME' "$dec" | wc -l | tr -d ' ')
  if [ "$dec_frames" -ne "$frames" ]; then
    echo "FAIL($label): decoded $dec_frames of $frames picture(s)"
    FAIL=1; return
  fi
  LAST_STREAM_MD5=$(md5 "$ivf")
  LAST_RECON_MD5=$recon_md5
  CASES=$((CASES + 1))
  echo "  PASS: $frames picture(s), stream $LAST_STREAM_MD5, recon $recon_md5"
}

# Runs the on/off pair, then requires that the two modes do not collapse into
# the same stream or the same reconstruction (that would make the gate vacuous
# or hide a frame-level allow_tip_hole_fill regression).
# run_pair <tag> <input> <frames>
run_pair() {
  local tag=$1 input=$2 frames=$3
  local on_stream="" on_recon="" off_stream="" off_recon=""
  run_case "${tag}_holefill_on" "$input" "$frames"
  if [ -n "$LAST_STREAM_MD5" ]; then
    on_stream=$LAST_STREAM_MD5; on_recon=$LAST_RECON_MD5
  fi
  run_case "${tag}_holefill_off" "$input" "$frames" --video-tip-no-hole-fill
  if [ -n "$LAST_STREAM_MD5" ]; then
    off_stream=$LAST_STREAM_MD5; off_recon=$LAST_RECON_MD5
  fi
  [ -n "$on_stream" ] && [ -n "$off_stream" ] || return 0
  if [ "$on_stream" = "$off_stream" ]; then
    echo "FAIL(${tag}): hole fill left the stream byte-identical to the control"
    FAIL=1
  fi
  if [ "$on_recon" = "$off_recon" ]; then
    echo "FAIL(${tag}): hole fill left the recon identical (inert fill)"
    FAIL=1
  fi
}

# --- 4 pictures on the sparse coded field (the original repro) ------------
if [ -f "$INPUT_360" ]; then
  run_pair p4 "$INPUT_360" 4
else
  echo "== p4: no 360p fixture ($INPUT_360); using $INPUT"
  run_pair p4 "$INPUT" 4
fi

# --- 8 pictures (six chained TIP frames) ----------------------------------
run_pair p8 "$INPUT" 8

# --- TIP-as-reference (§7.9 over a TIP display frame) ---------------------
#
# With --video-ref-mvs the last input picture is coded as a tile-coded P whose
# reference list is [final TIP display frame, anchor].  The anchor P has a
# single reference and the TIP frames never build the temporal stack, so any
# §7.12.2.7/.8 temporal candidate in the log comes from that tail P.
check_tail_temporal() {
  local label=$1
  local candidates
  candidates=$(sed -n 's/.*temporal MV candidates \([0-9]*\).*/\1/p' \
      "$OUT/$label.log" | tail -1)
  candidates=${candidates:-0}
  if [ "$candidates" -le 0 ]; then
    echo "FAIL($label): no §7.9 candidates from the TIP-referencing P"
    FAIL=1
  else
    echo "  PASS($label): $candidates temporal candidates from the TIP slot"
  fi
}
if [ -f "$INPUT_360" ]; then
  run_case tipref_p4 "$INPUT_360" 4 --video-ref-mvs
  check_tail_temporal tipref_p4
fi
run_case tipref_p8 "$INPUT" 8 --video-ref-mvs
check_tail_temporal tipref_p8

# The schedule needs a TIP display frame to reference: 3 pictures must be
# rejected by name rather than silently coding no P.
if "$BIN" "$INPUT_360" -o "$OUT/reject3.ivf" --video --frames 3 \
    --video-tip --video-ref-mvs >"$OUT/reject3.log" 2>&1; then
  echo "FAIL: --video-tip --video-ref-mvs accepted only 3 pictures"
  FAIL=1
else
  grep -q "at least 4 input" "$OUT/reject3.log" || {
    echo "FAIL: 3-picture TIP+ref-mvs rejection does not name the limit"
    FAIL=1
  }
fi

# The A/B control is not a product surface: using it without --video-tip must
# be rejected by name, not silently change a default stream.
if "$BIN" "$INPUT" -o "$OUT/reject.ivf" --video --frames 3 \
    --video-tip-no-hole-fill >"$OUT/reject.log" 2>&1; then
  echo "FAIL: --video-tip-no-hole-fill was accepted without --video-tip"
  FAIL=1
fi

if [ "$FAIL" -ne 0 ]; then
  echo "video_tip_gate: FAIL"
  exit 1
fi
echo "video_tip_gate: PASS ($CASES TIP stream(s), hole fill on and off)"
