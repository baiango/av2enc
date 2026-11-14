#!/usr/bin/env bash
# CPU gate for --video-same-ref-compound (§5.4.6 num_same_ref_compound = 1,
# §5.20.7.11 read_compound_ref, §5.20.7.6 compound_mode_same_refs).
#
# Encodes a flat key+P chain with two references and the AVERAGE compound
# carrier.  Every case must report at least one same-reference compound leaf
# (otherwise the sequence field and comp_ref symbol would be inert), decode
# with pinned dav2d --strict 1, carry the expected picture count, and match the
# encoder reconstruction sample-for-sample.  The default (flag off) streams
# must stay byte-identical to --base-bin.
#
# Usage:
#   src/pipeline/video/video_same_ref_compound_gate.sh --bin build/av2enc \
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
    -h|--help) sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "video_same_ref_compound_gate: unknown argument $1" >&2; exit 2 ;;
  esac
done

[ -n "$BIN" ] || { echo "video_same_ref_compound_gate: --bin is required" >&2; exit 2; }
[ -x "$BIN" ] || { echo "video_same_ref_compound_gate: no such binary $BIN" >&2; exit 2; }
if [ -z "$DAV2D" ]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
  done
fi
[ -n "$DAV2D" ] && [ -x "$DAV2D" ] || {
  echo "video_same_ref_compound_gate: dav2d not found (set DAV2D)" >&2
  exit 77; }

OUT="${OUT:-${TMPDIR:-/tmp}/video_same_ref_compound_gate}"
mkdir -p "$OUT"
FAIL=0
CASES=0

md5() { md5sum "$1" 2>/dev/null | cut -d' ' -f1 || md5 -q "$1"; }

FIX360="$TESTDATA/shine_640x360_4f.y4m"
[ -f "$FIX360" ] || {
  echo "video_same_ref_compound_gate: missing fixture $FIX360" >&2; exit 77; }
FIX64="$OUT/clip64.y4m"
python3 "$ROOT/src/pipeline/video/crop_y4m_fixture.py" \
  --input "$TESTDATA/shine_1280x720_60f.y4m" --width 64 --height 64 \
  --frames 8 --out "$FIX64"

run_case() {
  local label=$1 input=$2 frames=$3
  shift 3
  local ivf="$OUT/$label.ivf" recon="$OUT/$label.recon.y4m"
  local dec="$OUT/$label.dec.y4m" log="$OUT/$label.log"
  echo "== $label: $*"
  if ! "$BIN" "$input" -o "$ivf" --video --frames "$frames" \
      --video-refs 2 --video-compound --video-same-ref-compound \
      --video-recon "$recon" "$@" >"$log" 2>&1; then
    echo "FAIL($label): encode failed"; tail -5 "$log"; FAIL=1; return
  fi
  local got
  got=$(sed -n 's/.*same-ref compound blocks \([0-9]*\).*/\1/p' "$log" | tail -1)
  got=${got:-0}
  if [ "$got" -le 0 ]; then
    echo "FAIL($label): no same-reference compound leaf (inert carrier)"
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
  echo "  PASS: $frames picture(s), same-ref leaves $got, md5=$recon_md5"
  CASES=$((CASES + 1))
}

run_case same_ref_64 "$FIX64" 8
run_case same_ref_360 "$FIX360" 4
run_case same_ref_360_short_refresh "$FIX360" 4 --video-short-refresh

# The flag must change the bitstream compared with the plain compound carrier.
"$BIN" "$FIX360" -o "$OUT/flag_on.ivf" --video --frames 4 --video-refs 2 \
  --video-compound --video-same-ref-compound >/dev/null 2>&1
"$BIN" "$FIX360" -o "$OUT/flag_off.ivf" --video --frames 4 --video-refs 2 \
  --video-compound >/dev/null 2>&1
if cmp -s "$OUT/flag_on.ivf" "$OUT/flag_off.ivf"; then
  echo "FAIL: --video-same-ref-compound left the stream byte-identical"; FAIL=1
else
  echo "  PASS(marker): --video-same-ref-compound changes the stream"
fi

if [ -n "$BASE_BIN" ] && [ -x "$BASE_BIN" ]; then
  for spec in "default:4:" "compound:4:--video-refs 2 --video-compound"; do
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
  echo "video_same_ref_compound_gate: FAIL"
  exit 1
fi
echo "video_same_ref_compound_gate: PASS ($CASES stream(s))"
