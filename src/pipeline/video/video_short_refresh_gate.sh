#!/usr/bin/env bash
# CPU gate for --video-short-refresh (§5.4.6
# enable_short_refresh_frame_flags, §5.18.2 has_refresh_frame_flags /
# frame_to_refresh).
#
# Every case encodes through the product CLI, decodes with the pinned dav2d at
# --strict 1 --threads 1 and compares the decoder output sample-for-sample
# (Y4M MD5) with the encoder --video-recon dump.  The summary must report the
# expected number of compact refresh headers, otherwise the gate would pass on
# an inert sequence bit.  The default (flag off) invocations must be
# byte-identical to a pre-change binary supplied with --base-bin.
#
# Usage:
#   src/pipeline/video/video_short_refresh_gate.sh --bin build/av2enc \
#       [--dav2d /path/to/dav2d] [--base-bin old/av2enc]
#
# Exit codes: 0 pass, 1 failure, 2 usage, 77 missing fixture/decoder.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/../../.." && pwd)
BIN=""
DAV2D="${DAV2D:-}"
BASE_BIN=""
TESTDATA="${AV2ENC_TESTDATA:-$ROOT/testdata}"

while [ $# -gt 0 ]; do
  case "$1" in
    --bin) BIN=$2; shift 2 ;;
    --dav2d) DAV2D=$2; shift 2 ;;
    --base-bin) BASE_BIN=$2; shift 2 ;;
    -h|--help) sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "video_short_refresh_gate: unknown argument $1" >&2; exit 2 ;;
  esac
done

[ -n "$BIN" ] || { echo "video_short_refresh_gate: --bin is required" >&2; exit 2; }
[ -x "$BIN" ] || { echo "video_short_refresh_gate: no such binary $BIN" >&2; exit 2; }
if [ -z "$DAV2D" ]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
  done
fi
[ -n "$DAV2D" ] && [ -x "$DAV2D" ] || {
  echo "video_short_refresh_gate: dav2d not found (set DAV2D)" >&2; exit 77; }

OUT="${OUT:-${TMPDIR:-/tmp}/video_short_refresh_gate}"
mkdir -p "$OUT"
FAIL=0
CASES=0

md5() { md5sum "$1" 2>/dev/null | cut -d' ' -f1 || md5 -q "$1"; }

make_fixture() {
  local src=$1
  [ -f "$src" ] || {
    echo "video_short_refresh_gate: missing fixture $src" >&2; exit 77; }
  python3 "$ROOT/src/pipeline/video/crop_y4m_fixture.py" \
    --input "$src" --width 64 --height 64 --frames 8 \
    --out "$OUT/clip64.y4m"
}

FIX64="$OUT/clip64.y4m"
make_fixture "$TESTDATA/shine_1280x720_60f.y4m"
FIX360="$TESTDATA/shine_640x360_4f.y4m"
[ -f "$FIX360" ] || {
  echo "video_short_refresh_gate: missing fixture $FIX360" >&2; exit 77; }

run_case() {
  local label=$1 input=$2 frames=$3 expect_refresh=$4
  shift 4
  local ivf="$OUT/$label.ivf" recon="$OUT/$label.recon.y4m"
  local dec="$OUT/$label.dec.y4m" log="$OUT/$label.log"
  echo "== $label: $*"
  if ! "$BIN" "$input" -o "$ivf" --video --frames "$frames" \
      --video-recon "$recon" "$@" >"$log" 2>&1; then
    echo "FAIL($label): encode failed"; tail -5 "$log"; FAIL=1; return
  fi
  local got
  got=$(sed -n 's/.*short refresh frames \([0-9]*\).*/\1/p' "$log" | tail -1)
  got=${got:-0}
  if [ "$got" -ne "$expect_refresh" ]; then
    echo "FAIL($label): $got compact refresh frame(s), want $expect_refresh"
    FAIL=1; return
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
  local dec_frames
  dec_frames=$(LC_ALL=C grep -a -o 'FRAME' "$dec" | wc -l | tr -d ' ')
  if [ "$dec_frames" -ne "$frames" ]; then
    echo "FAIL($label): decoded $dec_frames of $frames picture(s)"; FAIL=1; return
  fi
  echo "  PASS: $frames picture(s), compact refresh $got, md5=$recon_md5"
  CASES=$((CASES + 1))
}

run_case p64_r1 "$FIX64" 8 7 --video-short-refresh
run_case p64_r2 "$FIX64" 8 7 --video-refs 2 --video-short-refresh
run_case p360_r2 "$FIX360" 4 3 --video-refs 2 --video-short-refresh

# The compact header must change the bitstream (not just the summary).
"$BIN" "$FIX360" -o "$OUT/flag_on.ivf" --video --frames 4 --video-refs 2 \
  --video-short-refresh >/dev/null 2>&1
"$BIN" "$FIX360" -o "$OUT/flag_off.ivf" --video --frames 4 --video-refs 2 \
  >/dev/null 2>&1
if cmp -s "$OUT/flag_on.ivf" "$OUT/flag_off.ivf"; then
  echo "FAIL: --video-short-refresh left the stream byte-identical"; FAIL=1
else
  echo "  PASS(marker): --video-short-refresh changes the stream"
fi

if [ -n "$BASE_BIN" ] && [ -x "$BASE_BIN" ]; then
  for spec in "default:4:" "refs2:4:--video-refs 2"; do
    label=${spec%%:*}; rest=${spec#*:}; frames=${rest%%:*}; extra=${rest#*:}
    # shellcheck disable=SC2086
    "$BASE_BIN" "$FIX360" -o "$OUT/base_$label.ivf" --video --frames "$frames" \
      $extra >/dev/null 2>&1
    # shellcheck disable=SC2086
    "$BIN" "$FIX360" -o "$OUT/new_$label.ivf" --video --frames "$frames" \
      $extra >/dev/null 2>&1
    if ! cmp -s "$OUT/base_$label.ivf" "$OUT/new_$label.ivf"; then
      echo "FAIL(identity/$label): default stream changed with the flag off"
      FAIL=1
    else
      echo "  PASS(identity/$label): $(md5 "$OUT/new_$label.ivf")"
    fi
  done
fi

if [ "$FAIL" -ne 0 ]; then
  echo "video_short_refresh_gate: FAIL"
  exit 1
fi
echo "video_short_refresh_gate: PASS ($CASES stream(s))"
