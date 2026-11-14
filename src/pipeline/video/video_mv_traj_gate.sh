#!/usr/bin/env bash
# CPU gate for --video-mv-traj (§5.4.6 enable_mv_traj, §7.9.2 trajectory fill,
# §7.12.2.14/.16 TrajMv derived candidates).
#
# The gated surface is the flat key+P chain with two references and
# --video-ref-mvs (the trajectory is only consumed by the derived-candidate
# projection).  Every case must report at least one trajectory-derived
# candidate, decode with pinned dav2d --strict 1, carry the expected picture
# count, and match the encoder reconstruction sample-for-sample.  The flag has
# no effect at NumTotalRefs == 1, so the off-identity check uses the same
# --video-ref-mvs command.
#
# Usage:
#   src/pipeline/video/video_mv_traj_gate.sh --bin build/av2enc \
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
    *) echo "video_mv_traj_gate: unknown argument $1" >&2; exit 2 ;;
  esac
done

[ -n "$BIN" ] || { echo "video_mv_traj_gate: --bin is required" >&2; exit 2; }
[ -x "$BIN" ] || { echo "video_mv_traj_gate: no such binary $BIN" >&2; exit 2; }
if [ -z "$DAV2D" ]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
  done
fi
[ -n "$DAV2D" ] && [ -x "$DAV2D" ] || {
  echo "video_mv_traj_gate: dav2d not found (set DAV2D)" >&2; exit 77; }

OUT="${OUT:-${TMPDIR:-/tmp}/video_mv_traj_gate}"
mkdir -p "$OUT"
FAIL=0
CASES=0

md5() { md5sum "$1" 2>/dev/null | cut -d' ' -f1 || md5 -q "$1"; }

FIX360="$TESTDATA/shine_640x360_4f.y4m"
[ -f "$FIX360" ] || {
  echo "video_mv_traj_gate: missing fixture $FIX360" >&2; exit 77; }
FIX64="$OUT/clip64.y4m"
python3 "$ROOT/src/pipeline/video/crop_y4m_fixture.py" \
  --width 64 --height 64 --frames 8 --synthetic-moving --out "$FIX64"

run_case() {
  local label=$1 input=$2 frames=$3
  shift 3
  local ivf="$OUT/$label.ivf" recon="$OUT/$label.recon.y4m"
  local dec="$OUT/$label.dec.y4m" log="$OUT/$label.log"
  echo "== $label: $*"
  if ! "$BIN" "$input" -o "$ivf" --video --frames "$frames" \
      --video-refs 2 --video-ref-mvs --video-mv-traj \
      --video-recon "$recon" "$@" >"$log" 2>&1; then
    echo "FAIL($label): encode failed"; tail -5 "$log"; FAIL=1; return
  fi
  local got
  got=$(sed -n 's/.*mv-traj derived candidates \([0-9]*\).*/\1/p' "$log" | tail -1)
  got=${got:-0}
  if [ "$got" -le 0 ]; then
    echo "FAIL($label): no trajectory-derived candidate (inert carrier)"
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
  echo "  PASS: $frames picture(s), traj-derived $got, md5=$recon_md5"
  CASES=$((CASES + 1))
}

run_case traj_64 "$FIX64" 8
run_case traj_360 "$FIX360" 4

# The flag must change the bitstream compared with the same ref-mvs stream.
"$BIN" "$FIX360" -o "$OUT/flag_on.ivf" --video --frames 4 --video-refs 2 \
  --video-ref-mvs --video-mv-traj >/dev/null 2>&1
"$BIN" "$FIX360" -o "$OUT/flag_off.ivf" --video --frames 4 --video-refs 2 \
  --video-ref-mvs >/dev/null 2>&1
if cmp -s "$OUT/flag_on.ivf" "$OUT/flag_off.ivf"; then
  echo "FAIL: --video-mv-traj left the stream byte-identical"; FAIL=1
else
  echo "  PASS(marker): --video-mv-traj changes the stream"
fi

# Named blockers: the 3-reference trajectory composition and the ref-MV bank
# are not decoder-exact; both must be rejected by name.
if "$BIN" "$FIX360" -o "$OUT/reject_r3.ivf" --video --frames 4 --video-refs 3 \
    --video-ref-mvs --video-mv-traj >"$OUT/reject_r3.log" 2>&1; then
  echo "FAIL: --video-mv-traj accepted NumTotalRefs == 3 (known residual)"
  FAIL=1
elif ! grep -q "wired for NumTotalRefs <= 2" "$OUT/reject_r3.log"; then
  echo "FAIL: --video-mv-traj refs=3 rejection lacks the named reason"
  tail -3 "$OUT/reject_r3.log"; FAIL=1
else
  echo "  PASS(reject): refs=3 trajectory is rejected by name"
fi

if "$BIN" "$FIX360" -o "$OUT/reject_bank.ivf" --video --frames 4 \
    --video-refmvbank >"$OUT/reject_bank.log" 2>&1; then
  echo "FAIL: --video-refmvbank accepted despite the known parity residual"
  FAIL=1
elif ! grep -q "not decoder-exact" "$OUT/reject_bank.log"; then
  echo "FAIL: --video-refmvbank rejection lacks the named reason"
  tail -3 "$OUT/reject_bank.log"; FAIL=1
else
  echo "  PASS(reject): --video-refmvbank is rejected by name"
fi

if [ -n "$BASE_BIN" ] && [ -x "$BASE_BIN" ]; then
  for spec in "default:4:" "refmvs:4:--video-refs 2 --video-ref-mvs"; do
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
  echo "video_mv_traj_gate: FAIL"
  exit 1
fi
echo "video_mv_traj_gate: PASS ($CASES stream(s))"
