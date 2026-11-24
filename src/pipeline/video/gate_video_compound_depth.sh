#!/usr/bin/env bash
# CPU gate for the compound-depth arm on the chained B (docs/compound-depth.md).
#
# The arm is env-gated and research-only:
#   AV2ENC_TIP_LEAF=1                 TIP output on the chained-B dead ends
#   AV2ENC_TIP_LEAF_FIELD_GUARD=3     fall back to the coded B when a reference
#                                     §7.22 field carries a CWP compound block
#                                     or a compound cell with a non-zero MV
#   AV2ENC_COMPOUND_RATE=1            charge the compound-mode syntax
#   AV2ENC_COMPOUND_PAIR_LIMIT=1      one NEAR_NEARMV stack pair per list
#   AV2ENC_COMPOUND_EXACT_RATE=0      the heuristic price (this gate's arm;
#                                     the promoted preset sets 1)
# plus --video-compound-near-near --video-cwp-weighted.
#
# Cases:
#   depth16     16 pictures, 17,318 B, avmdec 16/16 recon, pinned dav2d exact
#   depth60     60 pictures QP123, avmdec 60/60 recon
#   tip_base    TIP control (--video-compound-depth-off), QP121, 23,183 B
#   compound    --video-compound-depth-off + AV2ENC_TIP_LEAF=0, QP125, 23,179 B
#   flat106/110 --fast-decode pins
#
# The three --nightly-b sizes were re-recorded on 2026-11-25 (BUG-GATES-002):
# 6ec51ab4 moved the chained-B key offset -50 -> -37 on 2026-11-22, coarsening
# the key and shrinking every --nightly-b stream at a fixed QP.  All three
# still decode recon-exact (avmdec), so this is a re-record rather than a
# regression; the flat fast-decode pins are unaffected and depth60 carries no
# size pin.
#
# Usage:
#   src/pipeline/video/gate_video_compound_depth.sh --bin build-dev/av2enc \
#       [--dav2d tools/bin/dav2d-macos] [--input testdata/...y4m]
#
# Exit codes: 0 pass, 1 failure, 2 usage, 77 missing fixture/decoder.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/../../.." && pwd)
BIN=""
DAV2D="${DAV2D:-}"
INPUT="${AV2ENC_TESTDATA:-$ROOT/testdata}/nasa_480x240_30fps_60f.y4m"

while [ $# -gt 0 ]; do
  case "$1" in
    --bin) BIN=$2; shift 2 ;;
    --dav2d) DAV2D=$2; shift 2 ;;
    --input) INPUT=$2; shift 2 ;;
    -h|--help) sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "gate_video_compound_depth: unknown argument $1" >&2; exit 2 ;;
  esac
done

[ -n "$BIN" ] && [ -x "$BIN" ] || {
  echo "gate_video_compound_depth: --bin is required" >&2; exit 2; }
if [ -z "$DAV2D" ]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
  done
fi
[ -n "$DAV2D" ] && [ -x "$DAV2D" ] || {
  echo "gate_video_compound_depth: dav2d not found (set DAV2D)" >&2
  exit 77; }
[ -x "$ROOT/tools/validate_avmdec.py" ] || {
  echo "gate_video_compound_depth: tools/validate_avmdec.py missing" >&2
  exit 77; }
[ -s "$INPUT" ] || {
  echo "gate_video_compound_depth: missing fixture $INPUT" >&2; exit 77; }

OUT="${OUT:-${TMPDIR:-/tmp}/video_compound_depth_gate}"
# This gate is the joint-off depth arm; the promoted preset carries the joint
# solve (gate_video_joint_plan.sh).
export AV2ENC_RD_JOINT_P=0
mkdir -p "$OUT"
FAIL=0
PASS=0
ok() { PASS=$((PASS + 1)); echo "  PASS: $*"; }
bad() { FAIL=$((FAIL + 1)); echo "  FAIL: $*"; }

md5_of() {
  if command -v md5 >/dev/null 2>&1; then
    md5 -q "$1"
  else
    md5sum "$1" | cut -d' ' -f1
  fi
}

check_pin() {
  local name=$1 ivf=$2 bytes=$3 md5=$4
  local got_bytes got_md5
  got_bytes=$(wc -c < "$ivf" | tr -d ' ')
  got_md5=$(md5_of "$ivf")
  if [ "$got_bytes" = "$bytes" ] && [ "$got_md5" = "$md5" ]; then
    ok "$name: $got_bytes B $got_md5"
  else
    bad "$name: $got_bytes B $got_md5 (want $bytes B $md5)"
  fi
}

arm_env() {
  # This gate is the heuristic-priced depth arm: the promoted preset carries
  # AV2ENC_COMPOUND_EXACT_RATE=1, so pin the control explicitly.
  env AV2ENC_TIP_LEAF=1 AV2ENC_TIP_LEAF_FIELD_GUARD=3 \
      AV2ENC_COMPOUND_RATE=1 AV2ENC_COMPOUND_PAIR_LIMIT=1 \
      AV2ENC_COMPOUND_EXACT_RATE=0 "$@"
}

# 16-picture proof: avmdec recon equality and pinned dav2d sample equality.
arm_env "$BIN" "$INPUT" -o "$OUT/depth16.ivf" --video --frames 16 --qp 116 \
  --cpu-threads 8 --video-gops 2 --nightly-b \
  --video-compound-near-near --video-cwp-weighted \
  --video-recon "$OUT/depth16.y4m" > "$OUT/depth16.enc.log" 2>&1 || \
  bad "depth16: encode failed"
check_pin depth16 "$OUT/depth16.ivf" 17318 \
  1a5b8a5a90be82208f6211348f9397f8
python3 "$ROOT/tools/validate_avmdec.py" "$OUT/depth16.ivf" \
  --ref "$INPUT" --recon "$OUT/depth16.y4m" > "$OUT/depth16.val.log" 2>&1 && \
  ok "depth16: avmdec recon equal" || bad "depth16: avmdec recon mismatch"
"$DAV2D" --strict 1 --threads 1 -i "$OUT/depth16.ivf" \
  -o "$OUT/depth16.dec.y4m" > "$OUT/depth16.dav.log" 2>&1 || \
  bad "depth16: pinned dav2d decode failed"
if cmp -s "$OUT/depth16.y4m" "$OUT/depth16.dec.y4m"; then
  ok "depth16: pinned dav2d 16/16 sample-exact"
else
  bad "depth16: pinned dav2d recon differs"
fi

# 60-picture rate point: avmdec is the authority (dav2d has the drain defect).
arm_env "$BIN" "$INPUT" -o "$OUT/depth60.ivf" --video --frames 60 --qp 123 \
  --cpu-threads 8 --video-gops 7 --nightly-b \
  --video-compound-near-near --video-cwp-weighted \
  --video-recon "$OUT/depth60.y4m" > "$OUT/depth60.enc.log" 2>&1 || \
  bad "depth60: encode failed"
python3 "$ROOT/tools/validate_avmdec.py" "$OUT/depth60.ivf" \
  --ref "$INPUT" --recon "$OUT/depth60.y4m" > "$OUT/depth60.val.log" 2>&1 && \
  ok "depth60: avmdec 60/60 recon equal" || bad "depth60: avmdec recon mismatch"

# Controls: the promoted preset's TIP-only and compound-only arms, and the
# flat pins.  --video-compound-depth-off drops the depth arm; an explicit
# AV2ENC_TIP_LEAF=0 with it drops TIP as well.
env AV2ENC_TIP_LEAF=1 "$BIN" "$INPUT" -o "$OUT/tip_base.ivf" --video \
  --frames 60 --qp 121 --cpu-threads 8 --video-gops 7 --nightly-b \
  --video-compound-depth-off \
  > "$OUT/tip_base.log" 2>&1 || bad "tip_base: encode failed"
check_pin tip_base "$OUT/tip_base.ivf" 23183 \
  fd6a7538dd3cf3c47af37f6d1555b585

env AV2ENC_TIP_LEAF=0 "$BIN" "$INPUT" -o "$OUT/compound.ivf" --video \
  --frames 60 --qp 125 --cpu-threads 8 --video-gops 7 --nightly-b \
  --video-compound-depth-off \
  > "$OUT/compound.log" 2>&1 || bad "compound: encode failed"
check_pin compound "$OUT/compound.ivf" 23179 \
  b5330f2d60f7293ae202da319209d8d5

"$BIN" "$INPUT" -o "$OUT/flat106.ivf" --video --frames 60 --qp 106 \
  --cpu-threads 8 --fast-decode > "$OUT/flat106.log" 2>&1 || \
  bad "flat106: encode failed"
check_pin flat106 "$OUT/flat106.ivf" 25008 \
  0639b43630abb454f7be5ac494d2efb4
"$BIN" "$INPUT" -o "$OUT/flat110.ivf" --video --frames 60 --qp 110 \
  --cpu-threads 8 --fast-decode > "$OUT/flat110.log" 2>&1 || \
  bad "flat110: encode failed"
check_pin flat110 "$OUT/flat110.ivf" 23690 \
  1b99127a4cd1d386fa5600fb63c57a83

echo "compound-depth gate: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
