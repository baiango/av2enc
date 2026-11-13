#!/usr/bin/env bash
# CPU gate for --video-ref-mvs (§5.4.6 enable_ref_frame_mvs, §5.18.2
# use_ref_frame_mvs, §7.9 motion-field estimation, §7.22 storage and the
# §7.12.2.7/.8 temporal MV-stack candidates).
#
# Every temporal case is encoded through the product CLI, decoded with the
# pinned dav2d at --strict 1 --threads 1, and the decoder output is compared
# sample-for-sample (Y4M MD5) against the encoder's --video-recon dump.  The
# stream must also report at least one temporal candidate, otherwise the gate
# would pass on an inert feature.
#
# When --base-bin points at a pre-change binary (the commit before the feature),
# the default invocations must be byte-identical; that is the "flag off" check.
#
# Usage:
#   src/pipeline/video/video_ref_mvs_gate.sh --bin build/av2enc \
#       [--input clip.y4m] [--dav2d /path/to/dav2d] [--base-bin old/av2enc]
#
# Exit codes: 0 pass, 1 failure, 2 usage, 77 missing fixture/decoder.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/../../.." && pwd)
BIN=""
INPUT="${AV2ENC_TESTDATA:-$ROOT/testdata}/shine_1280x720_60f.y4m"
DAV2D="${DAV2D:-}"
BASE_BIN=""

while [ $# -gt 0 ]; do
  case "$1" in
    --bin) BIN=$2; shift 2 ;;
    --input) INPUT=$2; shift 2 ;;
    --dav2d) DAV2D=$2; shift 2 ;;
    --base-bin) BASE_BIN=$2; shift 2 ;;
    -h|--help) sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "video_ref_mvs_gate: unknown argument $1" >&2; exit 2 ;;
  esac
done

[ -n "$BIN" ] || { echo "video_ref_mvs_gate: --bin is required" >&2; exit 2; }
[ -x "$BIN" ] || { echo "video_ref_mvs_gate: no such binary $BIN" >&2; exit 2; }
[ -f "$INPUT" ] || { echo "video_ref_mvs_gate: missing fixture $INPUT" >&2; exit 77; }
if [ -z "$DAV2D" ]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
  done
fi
[ -n "$DAV2D" ] && [ -x "$DAV2D" ] || {
  echo "video_ref_mvs_gate: dav2d not found (set DAV2D)" >&2; exit 77; }

OUT="${OUT:-${TMPDIR:-/tmp}/video_ref_mvs_gate}"
mkdir -p "$OUT"
FAIL=0
STREAMS=()

md5() { md5sum "$1" 2>/dev/null | cut -d' ' -f1 || md5 -q "$1"; }

# Encodes one case, requires the expected temporal-candidate count, decodes and
# compares the reconstruction.  $1 = label, $2 = expected frame count,
# $3 = want temporal candidates, $4 = require a selected temporal candidate,
# remaining args = encoder options.
run_case() {
  local label=$1 frames=$2 want_temporal=$3 want_selected=$4
  shift 4
  local ivf="$OUT/$label.ivf" recon="$OUT/$label.recon.y4m"
  local dec="$OUT/$label.dec.y4m" log="$OUT/$label.log"
  echo "== $label: $*"
  if ! "$BIN" "$INPUT" -o "$ivf" --video --frames "$frames" \
      --video-recon "$recon" "$@" >"$log" 2>&1; then
    echo "FAIL($label): encode failed"; tail -5 "$log"; FAIL=1; return
  fi
  if ! grep -q "output frame(s)" "$log"; then
    echo "FAIL($label): no encoded summary"; FAIL=1; return
  fi
  local got_temporal
  got_temporal=$(sed -n 's/.*temporal MV candidates \([0-9]*\).*/\1/p' "$log" | tail -1)
  got_temporal=${got_temporal:-0}
  if [ "$want_temporal" = "yes" ] && [ "$got_temporal" -le 0 ]; then
    echo "FAIL($label): feature inert (0 temporal candidates)"; FAIL=1; return
  fi
  if [ "$want_temporal" = "no" ] && [ "$got_temporal" -ne 0 ]; then
    echo "FAIL($label): unexpected temporal candidates"; FAIL=1; return
  fi
  local got_selected
  got_selected=$(sed -n 's/.*temporal-selected leaves \([0-9]*\).*/\1/p' "$log" | tail -1)
  got_selected=${got_selected:-0}
  if [ "$want_selected" = "yes" ] && [ "$got_selected" -le 0 ]; then
    echo "FAIL($label): no coded leaf selected a temporal candidate"
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
  # The decoded Y4M frame count is a second, independent check.
  local dec_frames
  # LC_ALL=C: BSD grep otherwise stops at the first non-UTF-8 sample byte.
  dec_frames=$(LC_ALL=C grep -a -o 'FRAME' "$dec" | wc -l | tr -d ' ')
  if [ "$dec_frames" -ne "$frames" ]; then
    echo "FAIL($label): decoded $dec_frames of $frames picture(s)"; FAIL=1; return
  fi
  echo "  PASS: $frames picture(s), temporal=$got_temporal, selected=$got_selected, md5=$recon_md5"
  STREAMS+=("$label")
}

# --- temporal cases --------------------------------------------------------
run_case p8_refs1 8 no no --video-ref-mvs
run_case p8_refs2 8 yes no --video-refs 2 --video-ref-mvs
run_case p8_refs3 8 yes no --video-refs 3 --video-ref-mvs
run_case p16_refs4 16 yes no --video-refs 4 --video-ref-mvs
run_case p8_refs2_depth4 8 yes no --video-refs 2 --video-ref-mvs --video-min-block 8
run_case p8_refs2_deblock 8 yes no --video-refs 2 --video-ref-mvs --video-deblock
run_case p8_refs2_residual 8 yes no --video-refs 2 --video-ref-mvs --video-inter-residual
run_case p8_refs2_stress 8 yes yes --video-refs 2 --video-ref-mvs-stress
run_case p8_refs3_stress 8 yes yes --video-refs 3 --video-ref-mvs-stress
run_case b4_stress 4 yes yes --b-frames 2 --video-ref-mvs-stress
run_case b4 4 yes no --b-frames 2 --video-ref-mvs
run_case b6 6 yes no --b-frames 4 --video-ref-mvs
# Temporal layers are single-reference: §7.9 returns an empty field, so the
# stream must decode but report no candidates.
run_case tl4 4 no no --video-tlayers 4 --video-ref-mvs

# --- "flag off" byte identity ---------------------------------------------
if [ -n "$BASE_BIN" ] && [ -x "$BASE_BIN" ]; then
  for spec in "default:8:" "refs2:8:--video-refs 2" "b4:4:--b-frames 2" \
              "tl4:4:--video-tlayers 4"; do
    label=${spec%%:*}; rest=${spec#*:}; frames=${rest%%:*}; extra=${rest#*:}
    # shellcheck disable=SC2086
    "$BASE_BIN" "$INPUT" -o "$OUT/base_$label.ivf" --video --frames "$frames" \
      $extra >/dev/null 2>&1
    # shellcheck disable=SC2086
    "$BIN" "$INPUT" -o "$OUT/new_$label.ivf" --video --frames "$frames" \
      $extra >/dev/null 2>&1
    if ! cmp -s "$OUT/base_$label.ivf" "$OUT/new_$label.ivf"; then
      echo "FAIL(identity/$label): default stream changed with the flag off"
      FAIL=1
    else
      echo "  PASS(identity/$label): $(md5 "$OUT/new_$label.ivf")"
    fi
  done
  # The flag must actually change the bitstream.
  "$BIN" "$INPUT" -o "$OUT/new_refmvs.ivf" --video --frames 8 \
    --video-refs 2 --video-ref-mvs >/dev/null 2>&1
  if cmp -s "$OUT/base_refs2.ivf" "$OUT/new_refmvs.ivf"; then
    echo "FAIL: --video-ref-mvs left the stream byte-identical (inert header)"
    FAIL=1
  else
    echo "  PASS(marker): --video-ref-mvs changes the stream"
  fi
fi

if [ "$FAIL" -ne 0 ]; then
  echo "video_ref_mvs_gate: FAIL"
  exit 1
fi
echo "video_ref_mvs_gate: PASS (${#STREAMS[@]} temporal stream(s))"
