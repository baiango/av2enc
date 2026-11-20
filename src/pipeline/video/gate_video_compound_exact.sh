#!/usr/bin/env bash
# CPU gate for the exact compound rate model and the compound NEWMV carrier
# (docs/compound-depth.md §7).
#
# The depth arm prices a compound block with a fixed syntax constant.  The
# exact-rate arm makes the eff-perc SyntaxProbe walk the real §5.20.7.6/.8/.13/
# .16 symbol set instead: comp_mode, is_joint, compound_mode_non_joint,
# jmvd_scale_mode, both DRL indices, the shell-coded MVD(s), comp_group_idx,
# cwp_idx and the compound interp_filter symbol.  Measured +0.261 K at
# 100 kbps on the same candidate set (the win is the price, not more pairs).
#
# The compound NEWMV carrier adds NEAR_NEWMV / NEW_NEARMV / JOINT_NEWMV so the
# plan ME MV and a ±1/8..1-pel refinement are signallable.  It is measured
# below the gate and stays off; this gate keeps it decoder-exact.
#
# Env:
#   AV2ENC_COMPOUND_EXACT_RATE=1   exact compound pricing (no new modes)
#   AV2ENC_COMPOUND_NEWMV=1        the three compound NEWMV modes (implies the
#                                  exact pricing)
#
# Cases:
#   exact16   16 pictures, 19,774 B, avmdec 16/16, pinned dav2d 16/16 exact,
#             probe fallbacks = 0
#   modes16   the NEWMV carrier 16 pictures, 19,800 B, avmdec + pinned dav2d
#             exact, and the trace proves all three modes selected
#   exact60   60 pictures QP122, 24,859 B, avmdec 60/60
#   modes60   60 pictures QP125, 24,628 B, avmdec 60/60
#   controls  TIP 86.805 pin, compound 86.607 pin, flat fast-decode pins
#
# Usage:
#   src/pipeline/video/gate_video_compound_exact.sh --bin build-release/av2enc \
#       [--dav2d tools/bin/dav2d-macos] [--input testdata/...y4m] \
#       [--skip-controls]
#
# Exit codes: 0 pass, 1 failure, 2 usage, 77 missing fixture/decoder.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/../../.." && pwd)
BIN=""
DAV2D="${DAV2D:-}"
INPUT="${AV2ENC_TESTDATA:-$ROOT/testdata}/nasa_480x240_30fps_60f.y4m"
SKIP_CONTROLS=0

while [ $# -gt 0 ]; do
  case "$1" in
    --bin) BIN=$2; shift 2 ;;
    --dav2d) DAV2D=$2; shift 2 ;;
    --input) INPUT=$2; shift 2 ;;
    --skip-controls) SKIP_CONTROLS=1; shift ;;
    -h|--help) sed -n '2,42p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "gate_video_compound_exact: unknown argument $1" >&2; exit 2 ;;
  esac
done

[ -n "$BIN" ] && [ -x "$BIN" ] || {
  echo "gate_video_compound_exact: --bin is required" >&2; exit 2; }
if [ -z "$DAV2D" ]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
  done
fi
[ -n "$DAV2D" ] && [ -x "$DAV2D" ] || {
  echo "gate_video_compound_exact: dav2d not found (set DAV2D)" >&2
  exit 77; }
[ -x "$ROOT/tools/validate_avmdec.py" ] || {
  echo "gate_video_compound_exact: tools/validate_avmdec.py missing" >&2
  exit 77; }
[ -s "$INPUT" ] || {
  echo "gate_video_compound_exact: missing fixture $INPUT" >&2; exit 77; }

OUT="${OUT:-${TMPDIR:-/tmp}/video_compound_exact_gate}"
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

base_env() {
  env AV2ENC_TIP_LEAF=1 AV2ENC_TIP_LEAF_FIELD_GUARD=3 \
      AV2ENC_COMPOUND_RATE=1 AV2ENC_COMPOUND_PAIR_LIMIT=1 "$@"
}

check_probe_exact() {
  local name=$1 log=$2
  if grep -q 'fallbacks=0' "$log"; then
    ok "$name: exact compound probe covered every leaf (fallbacks=0)"
  else
    bad "$name: probe fell back (see $log)"
  fi
}

run_case_16() {
  local name=$1 envflag=$2 bytes=$3 md5=$4
  # shellcheck disable=SC2086
  base_env $envflag "$BIN" "$INPUT" -o "$OUT/$name.ivf" --video --frames 16 \
    --qp 116 --cpu-threads 8 --video-gops 2 --nightly-b \
    --video-compound-near-near --video-cwp-weighted \
    --video-recon "$OUT/$name.y4m" --video-trace > "$OUT/$name.log" 2>&1 || \
    bad "$name: encode failed"
  check_pin "$name" "$OUT/$name.ivf" "$bytes" "$md5"
  python3 "$ROOT/tools/validate_avmdec.py" "$OUT/$name.ivf" \
    --ref "$INPUT" --recon "$OUT/$name.y4m" > "$OUT/$name.val.log" 2>&1 && \
    ok "$name: avmdec 16/16 recon equal" || \
    bad "$name: avmdec recon mismatch"
  "$DAV2D" --strict 1 --threads 1 -i "$OUT/$name.ivf" \
    -o "$OUT/$name.dec.y4m" > "$OUT/$name.dav.log" 2>&1 || \
    bad "$name: pinned dav2d decode failed"
  if cmp -s "$OUT/$name.y4m" "$OUT/$name.dec.y4m"; then
    ok "$name: pinned dav2d 16/16 sample-exact"
  else
    bad "$name: pinned dav2d recon differs"
  fi
  check_probe_exact "$name" "$OUT/$name.log"
}

# 16-picture proofs.
run_case_16 exact16 AV2ENC_COMPOUND_EXACT_RATE=1 19774 \
  51b664515a897a3b19be4131790867fa
run_case_16 modes16 AV2ENC_COMPOUND_NEWMV=1 19800 \
  424abbe40d8040fcac2eedf78bf8046a

# The modes16 trace proves each new compound mode is actually selected, not
# merely explored.
selected() {
  grep 'm2 inter' "$OUT/modes16.log" | grep 'comp=1' | grep -q -- "$1"
}
selected ' cm=1 ' && ok "modes16: NEAR_NEWMV selected" || \
  bad "modes16: NEAR_NEWMV not selected"
selected ' cm=2 ' && ok "modes16: NEW_NEARMV selected" || \
  bad "modes16: NEW_NEARMV not selected"
selected ' cj=1' && ok "modes16: JOINT_NEWMV selected" || \
  bad "modes16: JOINT_NEWMV not selected"

# 60-picture rate points (avmdec is the authority; the pinned dav2d has the
# known §7.21.6 drain defect on the chain: 17/60).
run_case_60() {
  local name=$1 envflag=$2 qp=$3 bytes=$4
  # shellcheck disable=SC2086
  base_env $envflag "$BIN" "$INPUT" -o "$OUT/$name.ivf" --video --frames 60 \
    --qp "$qp" --cpu-threads 8 --video-gops 7 --nightly-b \
    --video-compound-near-near --video-cwp-weighted \
    --video-recon "$OUT/$name.y4m" > "$OUT/$name.log" 2>&1 || \
    bad "$name: encode failed"
  local got_bytes
  got_bytes=$(wc -c < "$OUT/$name.ivf" | tr -d ' ')
  if [ "$got_bytes" = "$bytes" ]; then
    ok "$name: $got_bytes B"
  else
    bad "$name: $got_bytes B (want $bytes B)"
  fi
  python3 "$ROOT/tools/validate_avmdec.py" "$OUT/$name.ivf" \
    --ref "$INPUT" --recon "$OUT/$name.y4m" > "$OUT/$name.val.log" 2>&1 && \
    ok "$name: avmdec 60/60 recon equal" || \
    bad "$name: avmdec recon mismatch"
  check_probe_exact "$name" "$OUT/$name.log"
}

run_case_60 exact60 AV2ENC_COMPOUND_EXACT_RATE=1 122 24859
run_case_60 modes60 AV2ENC_COMPOUND_NEWMV=1 125 24628

if [ "$SKIP_CONTROLS" -eq 0 ]; then
  # Controls: the TIP-only pin, the compound-only pin and the flat pins.  The
  # promoted --nightly-b carries the depth + exact-rate arms, so the controls
  # walk back one promotion at a time.
  env AV2ENC_TIP_LEAF=1 "$BIN" "$INPUT" -o "$OUT/tip_base.ivf" --video \
    --frames 60 --qp 121 --cpu-threads 8 --video-gops 7 --nightly-b \
    --video-compound-depth-off \
    > "$OUT/tip_base.log" 2>&1 || bad "tip_base: encode failed"
  check_pin tip_base "$OUT/tip_base.ivf" 25299 \
    ca314db24a71fc1c28d7a1ba8eb1515f

  env AV2ENC_TIP_LEAF=0 "$BIN" "$INPUT" -o "$OUT/compound.ivf" --video \
    --frames 60 --qp 125 --cpu-threads 8 --video-gops 7 --nightly-b \
    --video-compound-depth-off \
    > "$OUT/compound.log" 2>&1 || bad "compound: encode failed"
  check_pin compound "$OUT/compound.ivf" 24988 \
    3dbbad9847ab661c6c154903990ade75

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
fi

echo "compound exact/NEWMV gate: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
