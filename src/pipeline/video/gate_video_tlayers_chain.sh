#!/usr/bin/env bash
# CPU gate for the chained single-key hierarchical GOPs (--video-tlayers
# --video-gops, env-gated by AV2ENC_TLAYERS_CHAIN=1; AV2ENC_TLAYERS_B=1
# selects the dyadic B block instead of the P block).
#
# --video-tlayers N codes one closed GOP.  The chained form codes
# --video-gops M blocks of N pictures under the first key: every later block
# starts with a regular anchor that references the previous block, never a
# CLK.  In the P schedule the anchor is a P picture; in the B schedule the
# block's past anchor (local 0) references the previous block's future anchor
# and the future anchor (local N-1) references the past anchor, while the
# interior B pictures reference the two anchors.  --nightly-b enables AVERAGE
# compound on the chained B carrier (the plan's past/future (0,1) pair on the
# interior pictures) and the TIP-output composition on the dead-end leaves
# (AV2ENC_TIP_LEAF=1, 0.30 dB coded-leaf floor).  --video-compound-off restores
# the pre-compound arm; AV2ENC_TIP_LEAF=0 restores the compound-only arm.
#
# Validation: chained streams are decoded with avmdec
# (tools/validate_avmdec.py, frame count + recon equality).  The pinned dav2d
# f9dafbc has a §7.21.6 drain defect for a non-key refresh that overwrites an
# implicit picture: it decodes the 16-picture proofs (the store holds every
# picture) but emits 17/60 for the 60-picture chains.  Flat key+P streams and
# the nightly pins still go through tools/validate_dav2d.sh.
#
# Cases:
#   off_reject  gate off: --video-gops is rejected for --video-tlayers
#   chain16_p   one key + two chained 8-frame P blocks, 16/16 + recon
#   chain16_b   one key + two chained 8-frame B blocks, 16/16 + recon
#   chain60_p   one key + 7x8 + 4 tail (P), 60/60 + recon
#   chain60_b   one key + 7x8 + 4 tail (B), 60/60 + recon
#   nightly_b_*  the TIP-output compound preset, 16 and 60 pictures (dav2d
#                exact at 16)
#   nightly_b_*_compound_off  the pre-compound 85.472 arm (A/B control)
#   nightly_b_*_tip_off  the compound 86.607 arm without TIP (A/B control)
#   pin106/110  the nightly gate pins are byte-identical with the gate off
#
# Usage:
#   src/pipeline/video/gate_video_tlayers_chain.sh --bin build-int/av2enc \
#       [--dav2d /path/to/dav2d] [--input clip.y4m]
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
    -h|--help) sed -n '2,37p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "gate_video_tlayers_chain: unknown argument $1" >&2; exit 2 ;;
  esac
done

[ -n "$BIN" ] && [ -x "$BIN" ] || {
  echo "gate_video_tlayers_chain: --bin is required" >&2; exit 2; }
if [ -z "$DAV2D" ]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
  done
fi
[ -n "$DAV2D" ] && [ -x "$DAV2D" ] || {
  echo "gate_video_tlayers_chain: dav2d not found (set DAV2D)" >&2; exit 77; }
[ -x "$ROOT/tools/validate_avmdec.py" ] || {
  echo "gate_video_tlayers_chain: tools/validate_avmdec.py is missing" >&2
  exit 77; }
[ -s "$INPUT" ] || {
  echo "gate_video_tlayers_chain: missing fixture $INPUT" >&2; exit 77; }
command -v python3 >/dev/null 2>&1 || {
  echo "gate_video_tlayers_chain: python3 is required" >&2; exit 77; }

OUT="${OUT:-${TMPDIR:-/tmp}/video_tlayers_chain_gate}"
mkdir -p "$OUT"
FAIL=0
PASS=0
ok() { PASS=$((PASS + 1)); echo "  PASS: $*"; }
bad() { FAIL=$((FAIL + 1)); echo "  FAIL: $*"; }

# B_LEGAL: the measured tool set that composes with the reordering schedules
# (docs/bframes.md).  The nightly-only flags that every reordering schedule
# rejects by name are listed in docs/tlayers-chain.md.
B_LEGAL=(
  --video-rd-preset quality
  --video-ccso-rd --video-ccso-bands 6 --video-ccso-lambda 22
  --video-ccso-refit 2 --video-ccso-unit-bits 2
  --video-rdoq --video-cdef --video-cdef-rd
  --video-leaf-rd --video-subpel-eighth --video-key-intra-txpart
  --video-qp-lambda
  --avg-cdf --adaptive-cdf --cdf-continuity
  --video-tile-flush-min --video-ccso-header-rd
)

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

# run_chain <label> <P|B> <frames> <gops> <pictures> <bytes|-> <md5|-> [extra flags...]
run_chain() {
  local label=$1 sched=$2 frames=$3 gops=$4 pictures=$5 want_bytes=$6 want_md5=$7
  shift 7
  local extra=("$@")
  local ivf="$OUT/$label.ivf" recon="$OUT/$label.recon.y4m"
  local log="$OUT/$label.log" dec="$OUT/$label.dec.y4m"
  local envs=(AV2ENC_TLAYERS_CHAIN=1)
  if [ "$sched" = "B" ]; then envs+=(AV2ENC_TLAYERS_B=1); fi
  echo "== $label: --video-tlayers 8 --video-gops $gops ($sched), $frames frames"
  if ! env "${envs[@]}" "$BIN" "$INPUT" -o "$ivf" \
      --video --frames "$frames" --qp 116 --cpu-threads 8 \
      --video-tlayers 8 --video-gops "$gops" \
      --video-recon "$recon" "${B_LEGAL[@]}" \
      ${extra[@]+"${extra[@]}"} >"$log" 2>&1; then
    bad "$label: encode failed"; tail -8 "$log"; return
  fi
  if [ "$want_bytes" != "-" ]; then
    local got
    got=$(wc -c < "$ivf" | tr -d ' ')
    if [ "$got" != "$want_bytes" ]; then
      bad "$label: $got bytes, expected $want_bytes"; return
    fi
  fi
  if [ "$want_md5" != "-" ]; then
    local got
    got=$(md5 -q "$ivf" 2>/dev/null || md5sum "$ivf" | cut -d' ' -f1)
    if [ "$got" != "$want_md5" ]; then
      bad "$label: md5 $got, expected $want_md5"; return
    fi
  fi
  # Chained streams: avmdec frame count + byte-exact recon (owner rule).
  if ! python3 "$ROOT/tools/validate_avmdec.py" "$ivf" --ref "$INPUT" \
      --recon "$recon" >"$OUT/$label.avmdec.log" 2>&1; then
    bad "$label: tools/validate_avmdec.py failed"
    tail -4 "$OUT/$label.avmdec.log"; return
  fi
  if [ "$pictures" = "16" ]; then
    # The 16-picture store holds every picture, so the pinned dav2d also
    # decodes it; keep that as a cross-check.
    if ! "$DAV2D" -q -i "$ivf" -o "$dec" --strict 1 --threads 1 \
        >"$OUT/$label.dav2d.log" 2>&1; then
      bad "$label: pinned dav2d rejected the 16-picture stream"
      tail -4 "$OUT/$label.dav2d.log"; return
    fi
    if ! compare_recon "$recon" "$dec" 16; then
      bad "$label: pinned dav2d recon differs on the 16-picture stream"
      return
    fi
    ok "$label: avmdec 16/16 recon ok; pinned dav2d 16/16 recon exact"
  else
    # Record what the pinned decoder does; the §7.21.6 drain defect is a
    # known decoder issue, not a gate failure (docs/validation.md).
    local frames_pin
    "$DAV2D" -q -i "$ivf" -o "$dec" --strict 1 --threads 1 \
      >"$OUT/$label.dav2d.log" 2>&1 || true
    frames_pin=$(python3 -c 'import sys; print(open(sys.argv[1], "rb").read().count(b"FRAME"))' "$dec")
    ok "$label: avmdec $pictures/$pictures recon ok (pinned dav2d: ${frames_pin:-unknown}/$pictures)"
  fi
}

# 1. Gate off: the combination must be rejected by name (no silent path).
echo "== off_reject: gate off rejects --video-gops with --video-tlayers"
if "$BIN" "$INPUT" -o "$OUT/off.ivf" --video --frames 16 --qp 116 \
    --video-tlayers 8 --video-gops 2 >"$OUT/off.log" 2>&1; then
  bad "off_reject: accepted without AV2ENC_TLAYERS_CHAIN"
else
  if grep -q "env-gated chained --video-tlayers plan" "$OUT/off.log"; then
    ok "off_reject: named rejection"
  else
    bad "off_reject: wrong rejection"; tail -4 "$OUT/off.log"
  fi
fi

# 1b. --nightly-b is the product entry point and needs a chained gop count.
echo "== nightly_b_needs_gops: --nightly-b requires --video-gops M > 1"
if "$BIN" "$INPUT" -o "$OUT/nb_bad.ivf" --video --frames 16 --qp 116 \
    --nightly-b >"$OUT/nb_bad.log" 2>&1; then
  bad "nightly_b_needs_gops: accepted without --video-gops"
else
  if grep -q "pass --video-gops M" "$OUT/nb_bad.log"; then
    ok "nightly_b_needs_gops: named rejection"
  else
    bad "nightly_b_needs_gops: wrong rejection"; tail -4 "$OUT/nb_bad.log"
  fi
fi

# run_nightly_b <label> <frames> <gops> <pictures> <qp> <bytes|-> <md5|->
run_nightly_b() {
  local label=$1 frames=$2 gops=$3 pictures=$4 qp=$5 want_bytes=$6 want_md5=$7
  shift 7
  local ivf="$OUT/$label.ivf" recon="$OUT/$label.recon.y4m"
  local log="$OUT/$label.log" dec="$OUT/$label.dec.y4m"
  echo "== $label: --nightly-b --video-gops $gops, $frames frames"
  if ! "$BIN" "$INPUT" -o "$ivf" --video --frames "$frames" \
      --qp "$qp" --cpu-threads 8 --video-gops "$gops" --nightly-b \
      --video-recon "$recon" "$@" >"$log" 2>&1; then
    bad "$label: encode failed"; tail -8 "$log"; return
  fi
  local got
  got=$(wc -c < "$ivf" | tr -d ' ')
  if [ "$want_bytes" != "-" ] && [ "$got" != "$want_bytes" ]; then
    bad "$label: $got bytes, expected $want_bytes"; return
  fi
  if [ "$want_md5" != "-" ]; then
    got=$(md5 -q "$ivf" 2>/dev/null || md5sum "$ivf" | cut -d' ' -f1)
    if [ "$got" != "$want_md5" ]; then
      bad "$label: md5 $got, expected $want_md5"; return
    fi
  fi
  if ! python3 "$ROOT/tools/validate_avmdec.py" "$ivf" --ref "$INPUT" \
      --recon "$recon" >"$OUT/$label.avmdec.log" 2>&1; then
    bad "$label: tools/validate_avmdec.py failed"
    tail -4 "$OUT/$label.avmdec.log"; return
  fi
  if [ "$pictures" = "16" ]; then
    if ! "$DAV2D" -q -i "$ivf" -o "$dec" --strict 1 --threads 1 \
        >"$OUT/$label.dav2d.log" 2>&1 ||
        ! compare_recon "$recon" "$dec" 16; then
      bad "$label: pinned dav2d recon differs on the 16-picture stream"; return
    fi
    ok "$label: avmdec 16/16 recon ok; pinned dav2d 16/16 recon exact"
  else
    local frames_pin
    "$DAV2D" -q -i "$ivf" -o "$dec" --strict 1 --threads 1 \
      >"$OUT/$label.dav2d.log" 2>&1 || true
    frames_pin=$(python3 -c 'import sys; print(open(sys.argv[1], "rb").read().count(b"FRAME"))' "$dec")
    ok "$label: avmdec $pictures/$pictures recon ok (pinned dav2d: ${frames_pin:-unknown}/$pictures)"
  fi
}

# 2. 16-picture proofs: one key + two chained 8-frame blocks.
run_chain chain16_p P 16 2 16 25827 f5485b57748a0a1afc2057f223269fb1
run_chain chain16_b B 16 2 16 25662 d5af8679202a00fafa3fd135569a28c1

# 2b. Edge path: the padded reference bank must carry the hidden bottom rows.
run_chain chain16_p_edge P 16 2 16 13165 982712c2611c5bfc709e9a34ad5c3450 \
  --video-edge-inter --video-edge-mask
run_chain chain16_b_edge B 16 2 16 13040 b269ceff4b2c4d3d94c35cba42d92455 \
  --video-edge-inter --video-edge-mask
run_chain chain16_b_edge_kq B 16 2 16 27491 4b72029de6ea916eb288101aeb959a49 \
  --video-edge-inter --video-edge-mask --video-key-qp-offset -57
run_chain chain16_b_edge_bawp B 16 2 16 13184 f8c6e722de0158e5e9e3499e78886126 \
  --video-edge-inter --video-edge-mask --video-bawp --video-bawp-carrier

# 3. 60-picture clips: one key + 7 full blocks + a 4-frame tail block.
run_chain chain60_p P 60 7 60 - -
run_chain chain60_b B 60 7 60 - -
run_chain chain60_b_edge B 60 7 60 - - \
  --video-edge-inter --video-edge-mask

# 3b. The product aggregate: --nightly-b (edge + key-QP + BAWP + AVERAGE
# compound + TIP output on the dead-end leaves + partition-RD + partition fit
# + layer-0 anchor residual).  The 16-picture proof must also be pinned-dav2d
# exact; the 60-picture streams are avmdec-only (dav2d §7.21.6 drain).  QP121
# (25,299 B) and QP122 (24,918 B) bracket the measured K 86.805 at 100 kbps.
run_nightly_b nightly_b_16 16 2 16 116 19727 \
  f6c7b79b6ddeab45b5447a7c50522685
run_nightly_b nightly_b_60_121 60 7 60 121 25299 \
  ca314db24a71fc1c28d7a1ba8eb1515f
run_nightly_b nightly_b_60_122 60 7 60 122 24918 \
  537115cf0ab942a38202f64049d24991

# 3c. Controls.  --video-compound-off drops compound and the TIP composition
# with it: the pre-compound 85.472 arm byte for byte.  An explicit
# AV2ENC_TIP_LEAF=0 keeps compound and drops only TIP: the 86.607 compound arm
# (QP125 24,988 B) byte for byte.
run_nightly_b nightly_b_16_compound_off 16 2 16 116 19759 \
  8720425796fe80f3ddce067643adfec2 --video-compound-off
run_nightly_b nightly_b_60_compound_off 60 7 60 131 24911 \
  a4c8853e3abd752503db17be769b80a4 --video-compound-off
run_nightly_b_tip_off() {
  export AV2ENC_TIP_LEAF=0
  run_nightly_b "$@"
  unset AV2ENC_TIP_LEAF
}
run_nightly_b_tip_off nightly_b_16_tip_off 16 2 16 116 19865 \
  325ce3ddad26a81efbf5ae7cc9b90f0b
run_nightly_b_tip_off nightly_b_60_tip_off 60 7 60 125 24988 \
  3dbbad9847ab661c6c154903990ade75

# 4. Gate off must keep the nightly pins byte-identical (dav2d territory).
pin_case() {
  local label=$1 qp=$2 want_bytes=$3 want_md5=$4
  local ivf="$OUT/$label.ivf" log="$OUT/$label.log"
  echo "== $label: --nightly --qp $qp"
  if ! "$BIN" "$INPUT" -o "$ivf" --video --frames 60 --qp "$qp" \
      --cpu-threads 8 --nightly >"$log" 2>&1; then
    bad "$label: encode failed"; tail -8 "$log"; return
  fi
  local got_bytes got_md5
  got_bytes=$(wc -c < "$ivf" | tr -d ' ')
  got_md5=$(md5 -q "$ivf" 2>/dev/null || md5sum "$ivf" | cut -d' ' -f1)
  if [ "$got_bytes" != "$want_bytes" ] || [ "$got_md5" != "$want_md5" ]; then
    bad "$label: $got_bytes / $got_md5, expected $want_bytes / $want_md5"
    return
  fi
  ok "$label: pin intact ($got_bytes, $got_md5)"
}
pin_case pin106 106 25008 0639b43630abb454f7be5ac494d2efb4
pin_case pin110 110 23690 1b99127a4cd1d386fa5600fb63c57a83

echo
if [ "$FAIL" -ne 0 ]; then
  echo "gate_video_tlayers_chain: $FAIL failure(s), $PASS passed" >&2
  exit 1
fi
echo "gate_video_tlayers_chain: $PASS/$PASS cases passed"
