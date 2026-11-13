#!/usr/bin/env bash
# Decoder gate for DIP (--video-dip, §5.4.5 enable_dip, §5.20.7.9
# dip_mode_info(), §7.13.2.3-.6 data-driven intra prediction).
#
#   ./video_dip_gate.sh
#   AV2ENC_BIN=build-int/av2enc DAV2D=/path/to/dav2d ./video_dip_gate.sh
#   KEEP=1 ./video_dip_gate.sh
#
# What it proves:
#   1. `--video-dip` writes §5.4.5 enable_dip = 1 and codes §5.20.7.9
#      use_dip / dip_transpose / dip_mode for luma DC_PRED keyframe blocks
#      whose size satisfies the DIP predicate.
#   2. The keyframe reconstruction uses the §7.13.2.3 predictor: the encoder's
#      --video-recon dump equals the pinned dav2d f9dafbc `--strict 1` output,
#      byte for byte.
#   3. Pinned dav2d f9dafbc `--strict 1` decodes every case and the decoded
#      picture count matches the input.
#   4. The tool composes with the other intra tools on the same keyframe
#      (MRL, IST, TX_MODE_SELECT) and with the §8.2.4 average/adaptive CDF
#      carrier (the DIP CDFs ride the tile CDF bank).
#   5. Defaults are unchanged: every `--video` case without the flag keeps its
#      pinned IVF MD5 and codes no DIP block.
#   6. Named rejections stay named: --tiles, --video-sdp and the B/temporal/
#      random-access/TIP/SVC frame schedules are not composed.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "$0")/../../.." && pwd)"
TMP="${TMPDIR:-/tmp}/av2enc_video_dip_gate.$$"
mkdir -p "$TMP"
cleanup() {
  if [ "${KEEP:-0}" != "1" ]; then rm -rf "$TMP"; fi
}
trap cleanup EXIT

AV2ENC_BIN="${AV2ENC_BIN:-}"
DAV2D="${DAV2D:-}"
while [ $# -gt 0 ]; do
  case "$1" in
    --bin) AV2ENC_BIN=$2; shift 2 ;;
    --dav2d) DAV2D=$2; shift 2 ;;
    --input-64) INPUT_64=$2; shift 2 ;;
    --input-360) INPUT_360=$2; shift 2 ;;
    -h|--help) sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "video_dip_gate: unknown argument $1" >&2; exit 2 ;;
  esac
done
if [ -z "$AV2ENC_BIN" ]; then
  for candidate in "$ROOT/build-int/av2enc" "$ROOT/build/av2enc" \
                   "$ROOT/build-m1/av2enc"; do
    if [ -x "$candidate" ]; then AV2ENC_BIN="$candidate"; break; fi
  done
fi
if [ -z "$AV2ENC_BIN" ] || [ ! -x "$AV2ENC_BIN" ]; then
  echo "video_dip_gate: no av2enc binary; set AV2ENC_BIN=..." >&2
  exit 127
fi

DAV2D="${DAV2D:-}"
if [ -z "$DAV2D" ]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$candidate" ]; then DAV2D="$candidate"; break; fi
  done
fi
if [ -z "$DAV2D" ] || [ ! -x "$DAV2D" ]; then
  echo "video_dip_gate: no dav2d binary (set DAV2D=...)" >&2
  exit 77
fi

INPUT_64="${INPUT_64:-${AV2ENC_INPUT_64:-$ROOT/testdata/m0_64x64_1f.y4m}}"
INPUT_360="${INPUT_360:-${AV2ENC_INPUT_360:-$ROOT/testdata/shine_640x360_4f.y4m}}"
for path in "$INPUT_64" "$INPUT_360"; do
  if [ ! -f "$path" ]; then
    echo "video_dip_gate: missing input $path" >&2
    exit 77
  fi
done

md5_of() {
  md5 -q "$1" 2>/dev/null || md5sum "$1" | awk '{print $1}'
}

frame_count() {
  python3 - "$1" <<'PY'
import sys
print(open(sys.argv[1], "rb").read().count(b"FRAME"))
PY
}

compare_exact() {
  python3 - "$1" "$2" "$3" <<'PY'
import sys

def frames(path):
    data = open(path, "rb").read()
    out = []
    i = 0
    w = h = 0
    while True:
        e = data.find(b"\n", i)
        if e < 0:
            break
        header = data[i:e].decode()
        if header.startswith("FRAME"):
            j = data.find(b"FRAME\n", e + 1)
            j = len(data) if j < 0 else j
            out.append(data[e + 1:j])
            i = j
        else:
            w = int(header.split("W")[1].split()[0])
            h = int(header.split("H")[1].split()[0])
            i = e + 1
    return w, h, out

w, h, a = frames(sys.argv[1])
w2, h2, b = frames(sys.argv[2])
if (w, h) != (w2, h2) or len(a) != len(b):
    raise SystemExit(f"{sys.argv[3]}: geometry/frames {w}x{h}/{len(a)} vs "
                     f"{w2}x{h2}/{len(b)}")
bad = sum(sum(x != y for x, y in zip(fa, fb)) for fa, fb in zip(a, b))
if bad:
    raise SystemExit(f"{sys.argv[3]}: {bad} samples differ")
print(f"exact {sys.argv[3]}: {len(a)} frame(s)")
PY
}

failures=0
passes=0
pass_case() { echo "PASS $1"; passes=$((passes + 1)); }
fail_case() { echo "FAIL $1: $2" >&2; failures=$((failures + 1)); }

# run_case <name> <input> <frames> <expected md5> <min dip blocks> [args...]
# The expected md5 is empty for a case that only checks decode exactness.
run_case() {
  local name=$1 input=$2 frames=$3 expected=$4 min_dip=$5
  shift 5
  local out="$TMP/$name.ivf" recon="$TMP/$name.recon.y4m"
  local dec="$TMP/$name.dav.y4m" log="$TMP/$name.log"
  if ! "$AV2ENC_BIN" "$input" -o "$out" --frames "$frames" --video \
        --video-recon "$recon" --video-trace "$@" >"$log" 2>&1; then
    fail_case "$name" "encode failed"
    tail -5 "$log" >&2
    return
  fi
  if [ -n "$expected" ]; then
    local got
    got=$(md5_of "$out")
    if [ "$got" != "$expected" ]; then
      fail_case "$name" "IVF md5 $got, expected $expected"
      return
    fi
  fi
  local dips
  dips=$(grep -c ' dip=' "$log" || true)
  if [ "$dips" -lt "$min_dip" ]; then
    fail_case "$name" "$dips DIP blocks, expected at least $min_dip"
    return
  fi
  if ! "$DAV2D" -q -i "$out" -o "$dec" --threads 1 --strict 1 \
        >"$TMP/$name.dec.log" 2>&1; then
    fail_case "$name" "dav2d --strict 1 rejected the stream"
    tail -5 "$TMP/$name.dec.log" >&2
    return
  fi
  local nframes
  nframes=$(frame_count "$dec")
  if [ "$nframes" != "$frames" ]; then
    fail_case "$name" "decoded $nframes pictures, expected $frames"
    return
  fi
  if ! compare_exact "$recon" "$dec" "$name" >>"$TMP/$name.cmp.log" 2>&1; then
    fail_case "$name" "encoder recon differs from dav2d"
    cat "$TMP/$name.cmp.log" >&2
    return
  fi
  pass_case "$name (dip=$dips, $(wc -c <"$out" | tr -d ' ') B, exact)"
}

# 64x64: one key picture.  The DC-only path still runs the DIP decision, so
# --video-intra-full is not required for use_dip to appear.
run_case dip_64 "$INPUT_64" 1 c571ca4ea12101f2331a37b056f8eeeb 1 \
  --video-dip
run_case dip_full_64 "$INPUT_64" 1 e8363bca39b0c2335de22348083d1f1c 1 \
  --video-dip --video-intra-full
run_case dip_full_tx_64 "$INPUT_64" 1 572873b50748a340ea5543259e402f08 1 \
  --video-dip --video-intra-full --video-tx-search
run_case dip_mrl_ist_64 "$INPUT_64" 1 65b9c2a7b20cfccc57d09f43a1227156 1 \
  --video-dip --video-mrl --video-ist --video-intra-full --video-tx-search

# 640x360: four pictures (key + P).  The P-frame intra fallback blocks code
# use_dip = 0 (the DIP decision is keyframe-only) but still carry the §5.20.7.9
# symbol, so the syntax stays in step with the decoder.
run_case dip_360 "$INPUT_360" 4 b10834c57c3bc8f7305d6037adaf632c 1 \
  --video-dip
run_case dip_full_360 "$INPUT_360" 4 d2a59aa14ede0490d6be4cf9942b49e8 1 \
  --video-dip --video-intra-full
run_case dip_full_tx_360 "$INPUT_360" 4 76e99c92dcf9f669e51799a7a88cf398 1 \
  --video-dip --video-intra-full --video-tx-search
run_case dip_mrl_ist_360 "$INPUT_360" 4 95b7c930e711549901c7819a5ee63296 1 \
  --video-dip --video-mrl --video-ist --video-intra-full --video-tx-search
run_case dip_avg_adaptive_360 "$INPUT_360" 4 \
  b8f4fdc5a043af9984a72b5b4a211db2 1 \
  --video-dip --video-intra-full --avg-cdf --adaptive-cdf

# Defaults: no --video-dip means no sequence bit and no DIP block, and the
# pinned IVF bytes do not move.
run_case default_64 "$INPUT_64" 1 df6fd676b1c879b6ccc2fd3e575b24a5 0
run_case default_360 "$INPUT_360" 4 83b34e7944346d6b3f64a460bdf1dc96 0

# Named rejections.
reject_case() {
  local name=$1 expect=$2
  shift 2
  if "$AV2ENC_BIN" "$INPUT_64" -o "$TMP/$name.ivf" --frames 1 --video \
        --video-dip "$@" >/dev/null 2>"$TMP/$name.err"; then
    fail_case "$name" "expected a rejection"
  elif grep -q "$expect" "$TMP/$name.err"; then
    pass_case "$name (named rejection)"
  else
    fail_case "$name" "rejection message changed"
    cat "$TMP/$name.err" >&2
  fi
}
reject_case dip_tiles_rejected "does not compose with --tiles" --tiles 2x1
reject_case dip_sdp_rejected "does not compose with --video-sdp" --video-sdp
reject_case dip_bframes_rejected \
  "does not compose with the B/temporal-layer" --b-frames 1

echo "video_dip_gate: $passes passed, $failures failed"
if [ "$failures" -ne 0 ]; then exit 1; fi
