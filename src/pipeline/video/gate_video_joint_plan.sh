#!/usr/bin/env bash
# CPU gate for the joint plan solve (AV2ENC_RD_JOINT_P, docs/rd-joint-plan.md).
#
# The joint solve replaces the sequential partition -> leaf -> TX/levels stages
# with one objective: every candidate coding block is encoded with the coder's
# own leaf path and scored with the P leaf D (eff-perc inter CSF + chroma), the
# exact interval rate and the live inter-syntax price.  The reference tree is
# always a candidate, so the gate cannot make a stream worse under J.
#
# This is a RESEARCH gate, not the fast path: the serial raster walk costs
# ~12 min per 60-frame 240p encode.  --skip-60 keeps the fast 16-picture and
# control checks for routine runs.
#
# Pins (all measured on the release build, --cpu-threads 8):
#   flat_joint112   --video-rd-preset best --qp 112, 25,007 B, strict dav2d
#   flat_off116     AV2ENC_RD_JOINT_P=0 --video-rd-preset best --qp 116,
#                   26,334 B / fba63446670b43888c3ac7b280ad44dc (control)
#   b_joint16       --nightly-b 16 pictures, 17,222 B, avmdec + pinned dav2d
#   b_joint118/119  --nightly-b 60 pictures QP118/119, avmdec 60/60
#   b_off121        AV2ENC_RD_JOINT_P=0 --nightly-b QP121, 23,066 B /
#                   72e04413362910d48d69c46b504fb9ed (control)
#
# The four --nightly-b sizes were re-recorded on 2026-11-25 (BUG-GATES-002):
# 6ec51ab4 moved the chained-B key offset -50 -> -37 on 2026-11-22, which
# coarsens the key and shrinks every --nightly-b stream at a fixed QP.  The
# flat pins below were unaffected (they set their own -39) and still pass.
# Every re-recorded stream still decodes recon-exact (avmdec 60/60, encoder
# recon == decoded), so this is a re-record, not a regression.  The values
# agree with docs/negative-results.md and 80b1e267, which both already carry
# the post-change sizes.
#
# Usage:
#   src/pipeline/video/gate_video_joint_plan.sh --bin build-release/av2enc \
#       [--dav2d tools/bin/dav2d-macos] [--input testdata/...y4m] [--skip-60]
#
# Exit codes: 0 pass, 1 failure, 2 usage, 77 missing fixture/decoder.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/../../.." && pwd)
BIN=""
DAV2D="${DAV2D:-}"
INPUT="${AV2ENC_TESTDATA:-$ROOT/testdata}/nasa_480x240_30fps_60f.y4m"
SKIP_60=0

while [ $# -gt 0 ]; do
  case "$1" in
    --bin) BIN=$2; shift 2 ;;
    --dav2d) DAV2D=$2; shift 2 ;;
    --input) INPUT=$2; shift 2 ;;
    --skip-60) SKIP_60=1; shift ;;
    -h|--help) sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "gate_video_joint_plan: unknown argument $1" >&2; exit 2 ;;
  esac
done

[ -n "$BIN" ] && [ -x "$BIN" ] || {
  echo "gate_video_joint_plan: --bin is required" >&2; exit 2; }
if [ -z "$DAV2D" ]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
  done
fi
[ -n "$DAV2D" ] && [ -x "$DAV2D" ] || {
  echo "gate_video_joint_plan: dav2d not found (set DAV2D)" >&2; exit 77; }
[ -x "$ROOT/tools/validate_avmdec.py" ] || {
  echo "gate_video_joint_plan: tools/validate_avmdec.py missing" >&2; exit 77; }
[ -s "$INPUT" ] || {
  echo "gate_video_joint_plan: missing fixture $INPUT" >&2; exit 77; }

OUT="${OUT:-${TMPDIR:-/tmp}/video_joint_plan_gate}"
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

recon_equal() {
  local name=$1 recon=$2 decoded=$3
  if cmp -s "$recon" "$decoded"; then
    ok "$name: encoder recon == decoded payload"
  else
    bad "$name: encoder recon differs from the decoded payload"
  fi
}

# 16-picture chained-B joint proof: avmdec recon equality and pinned dav2d.
"$BIN" "$INPUT" -o "$OUT/b_joint16.ivf" --video --frames 16 --qp 116 \
  --cpu-threads 8 --video-gops 2 --nightly-b \
  --video-recon "$OUT/b_joint16.y4m" > "$OUT/b_joint16.enc.log" 2>&1 || \
  bad "b_joint16: encode failed"
check_pin b_joint16 "$OUT/b_joint16.ivf" 17222 \
  2a7fbe710542a89c3a9d2c93fab94101
python3 "$ROOT/tools/validate_avmdec.py" "$OUT/b_joint16.ivf" \
  --ref "$INPUT" --recon "$OUT/b_joint16.y4m" > "$OUT/b_joint16.val.log" 2>&1 && \
  ok "b_joint16: avmdec recon equal" || bad "b_joint16: avmdec recon mismatch"
"$DAV2D" --strict 1 --threads 1 -i "$OUT/b_joint16.ivf" \
  -o "$OUT/b_joint16.dec.y4m" > "$OUT/b_joint16.dav.log" 2>&1 || \
  bad "b_joint16: pinned dav2d decode failed"
recon_equal b_joint16 "$OUT/b_joint16.y4m" "$OUT/b_joint16.dec.y4m"

# B joint-off control: the exact-rate preset without the joint solve.
env AV2ENC_RD_JOINT_P=0 "$BIN" "$INPUT" -o "$OUT/b_off121.ivf" --video \
  --frames 60 --qp 121 --cpu-threads 8 --video-gops 7 --nightly-b \
  > "$OUT/b_off121.log" 2>&1 || bad "b_off121: encode failed"
check_pin b_off121 "$OUT/b_off121.ivf" 23066 \
  72e04413362910d48d69c46b504fb9ed

# Flat joint-off control: the 86.33 lead stack without the joint solve.
env AV2ENC_RD_JOINT_P=0 "$BIN" "$INPUT" -o "$OUT/flat_off116.ivf" \
  --frames 60 --qp 116 --cpu-threads 8 --video-rd-preset best \
  > "$OUT/flat_off116.log" 2>&1 || bad "flat_off116: encode failed"
check_pin flat_off116 "$OUT/flat_off116.ivf" 26334 \
  fba63446670b43888c3ac7b280ad44dc

if [ "$SKIP_60" -eq 0 ]; then
  # Flat joint pin: the promoted `best` preset at its matched point.  The flat
  # chain is a flat key+P stream, so the pinned dav2d is the authority.
  "$BIN" "$INPUT" -o "$OUT/flat_joint112.ivf" --frames 60 --qp 112 \
    --cpu-threads 8 --video-rd-preset best \
    --video-recon "$OUT/flat_joint112.y4m" > "$OUT/flat_joint112.enc.log" 2>&1 || \
    bad "flat_joint112: encode failed"
  check_pin flat_joint112 "$OUT/flat_joint112.ivf" 25007 \
    9a36aeb7c02733342e4ec5db300e4d6d
  "$DAV2D" --strict 1 --threads 1 -i "$OUT/flat_joint112.ivf" \
    -o "$OUT/flat_joint112.dec.y4m" > "$OUT/flat_joint112.dav.log" 2>&1 || \
    bad "flat_joint112: dav2d --strict 1 failed"
  recon_equal flat_joint112 "$OUT/flat_joint112.y4m" \
    "$OUT/flat_joint112.dec.y4m"

  # Chained-B joint bracket (avmdec is the authority on the 60-picture chain;
  # the pinned dav2d has the known §7.21.6 drain defect).
  for q in 118 119; do
    bytes=22840; md5=9c7d37ecbc358684e09d464cfd2e661b
    if [ "$q" = "119" ]; then
      bytes=22576; md5=4b622b15a0715820114765e17259868f
    fi
    "$BIN" "$INPUT" -o "$OUT/b_joint$q.ivf" --video --frames 60 \
      --qp "$q" --cpu-threads 8 --video-gops 7 --nightly-b \
      --video-recon "$OUT/b_joint$q.y4m" > "$OUT/b_joint$q.enc.log" 2>&1 || \
      bad "b_joint$q: encode failed"
    check_pin "b_joint$q" "$OUT/b_joint$q.ivf" "$bytes" "$md5"
    python3 "$ROOT/tools/validate_avmdec.py" "$OUT/b_joint$q.ivf" \
      --ref "$INPUT" --recon "$OUT/b_joint$q.y4m" \
      > "$OUT/b_joint$q.val.log" 2>&1 && \
      ok "b_joint$q: avmdec 60/60 recon equal" || \
      bad "b_joint$q: avmdec recon mismatch"
  done
fi

echo "joint plan gate: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
