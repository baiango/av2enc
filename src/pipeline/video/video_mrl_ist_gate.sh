#!/usr/bin/env bash
# Decoder gate for MRL (--video-mrl) and intra IST (--video-ist).
#
#   ./video_mrl_ist_gate.sh
#   AV2ENC_BIN=build-int/av2enc DAV2D=/path/to/dav2d ./video_mrl_ist_gate.sh
#   KEEP=1 ./video_mrl_ist_gate.sh
#
# What it proves:
#   1. `--video-mrl` writes §5.4.5 enable_mrls = 1 and codes §5.20.5.5
#      mrl_index / mrl_sec_index for directional luma intra blocks.  With
#      --video-intra-full the still encoder's MRL candidate loop runs; without
#      it every block stays at mrl_index 0 (the bitstream is still legal).
#   2. `--video-ist` writes §5.4.8 enable_intra_ist = 1 and codes §5.20.8.2
#      sec_tx_type / most_probable_stx_set for luma intra blocks, and the
#      encoder reconstruction uses the §7.15.3 inverse.
#   3. Pinned dav2d f9dafbc `--strict 1` decodes every case, the output picture
#      count matches, and the decoded samples equal `--video-recon` exactly.
#   4. The two tools compose on the same stream (MRL kernel selection follows
#      the block's mrl_index, §7.15.3).
#   5. Named rejections stay named: --tiles and --video-qm are not composed.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "$0")/../../.." && pwd)"
TMP="${TMPDIR:-/tmp}/av2enc_video_mrl_ist_gate.$$"
mkdir -p "$TMP"
cleanup() {
  if [ "${KEEP:-0}" != "1" ]; then rm -rf "$TMP"; fi
}
trap cleanup EXIT

AV2ENC_BIN="${AV2ENC_BIN:-}"
if [ -z "$AV2ENC_BIN" ]; then
  for candidate in "$ROOT/build-int/av2enc" "$ROOT/build/av2enc" \
                   "$ROOT/build-m1/av2enc"; do
    if [ -x "$candidate" ]; then AV2ENC_BIN="$candidate"; break; fi
  done
fi
if [ -z "$AV2ENC_BIN" ] || [ ! -x "$AV2ENC_BIN" ]; then
  echo "video_mrl_ist_gate: no av2enc binary; set AV2ENC_BIN=..." >&2
  exit 127
fi

DAV2D="${DAV2D:-}"
if [ -z "$DAV2D" ]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$candidate" ]; then DAV2D="$candidate"; break; fi
  done
fi
if [ -z "$DAV2D" ] || [ ! -x "$DAV2D" ]; then
  echo "video_mrl_ist_gate: no dav2d binary (set DAV2D=...)" >&2
  exit 77
fi

INPUT_64="${AV2ENC_INPUT_64:-$ROOT/testdata/m0_64x64_1f.y4m}"
INPUT_360="${AV2ENC_INPUT_360:-$ROOT/testdata/shine_640x360_4f.y4m}"
for path in "$INPUT_64" "$INPUT_360"; do
  if [ ! -f "$path" ]; then
    echo "video_mrl_ist_gate: missing input $path" >&2
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

# run_case <name> <input> <frames> <expected md5> <min mrl> <min ist> [args...]
run_case() {
  local name=$1 input=$2 frames=$3 expected=$4 min_mrl=$5 min_ist=$6
  shift 6
  local out="$TMP/$name.ivf" recon="$TMP/$name.recon.y4m"
  local dec="$TMP/$name.dav.y4m" log="$TMP/$name.log"
  if ! "$AV2ENC_BIN" "$input" -o "$out" --frames "$frames" --video \
        --video-recon "$recon" --video-trace "$@" >"$log" 2>&1; then
    fail_case "$name" "encode failed"
    tail -5 "$log" >&2
    return
  fi
  local got
  got=$(md5_of "$out")
  if [ "$got" != "$expected" ]; then
    fail_case "$name" "IVF md5 $got, expected $expected"
    return
  fi
  local mrl ist
  mrl=$(grep -c 'mrl=[123]' "$log" || true)
  ist=$(grep -c 'ist=[1-9]' "$log" || true)
  if [ "$mrl" -lt "$min_mrl" ]; then
    fail_case "$name" "$mrl MRL blocks, expected at least $min_mrl"
    return
  fi
  if [ "$ist" -lt "$min_ist" ]; then
    fail_case "$name" "$ist IST blocks, expected at least $min_ist"
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
  pass_case "$name (mrl=$mrl ist=$ist, $(wc -c <"$out" | tr -d ' ') B, exact)"
}

# 64x64: one key picture.
run_case mrl_64 "$INPUT_64" 1 1b8dec13ff2d5c343618b4b02b8f2194 0 0 \
  --video-mrl
run_case mrl_full_64 "$INPUT_64" 1 698ae4ee08f45a3fcea9222fe2db5f53 1 0 \
  --video-mrl --video-intra-full
run_case mrl_full_tx_64 "$INPUT_64" 1 9f1398fd1b295d23b23414e13384f152 1 0 \
  --video-mrl --video-intra-full --video-tx-search
run_case ist_64 "$INPUT_64" 1 e6c32f0f77c70c7689235600fdcbc2fc 0 0 \
  --video-ist
run_case ist_full_64 "$INPUT_64" 1 5aa715b6ea06ff14210b640f0f29dfe8 0 0 \
  --video-ist --video-intra-full
run_case ist_full_tx_64 "$INPUT_64" 1 70ff80b11b36799c31822b3b5dfa1fed 0 0 \
  --video-ist --video-intra-full --video-tx-search
run_case both_64 "$INPUT_64" 1 e9d00d0fe8573efe497f8f83cd98dc46 1 0 \
  --video-mrl --video-ist --video-intra-full --video-tx-search

# 640x360: four pictures (key + P), so the intra fallback blocks inside the P
# frames exercise the same syntax on the inter path.
run_case mrl_360 "$INPUT_360" 4 d863e1d9e5b2aaeb309d84c70e1228f7 0 0 \
  --video-mrl
run_case mrl_full_360 "$INPUT_360" 4 bd9ab61cb3550b9ea36ed50a7d367689 1 0 \
  --video-mrl --video-intra-full
run_case mrl_full_tx_360 "$INPUT_360" 4 860339f969e3d1c924f1a93596ec961f 1 0 \
  --video-mrl --video-intra-full --video-tx-search
run_case ist_360 "$INPUT_360" 4 a0bcf3e78f68c53ebacf381502ecbdbb 0 1 \
  --video-ist
run_case ist_full_360 "$INPUT_360" 4 856c18f24d47cfb4557ac39783753754 0 0 \
  --video-ist --video-intra-full
run_case ist_full_tx_360 "$INPUT_360" 4 864868d274b1fbc802d91eecc14f4f3e 0 1 \
  --video-ist --video-intra-full --video-tx-search
run_case both_360 "$INPUT_360" 4 edfc99328975fec8b37f8a2e32875dd1 1 1 \
  --video-mrl --video-ist --video-intra-full --video-tx-search

# Named rejections.
if "$AV2ENC_BIN" "$INPUT_64" -o "$TMP/reject.ivf" --frames 1 --video \
      --video-mrl --tiles 2x1 >/dev/null 2>"$TMP/reject1.err"; then
  fail_case mrl_tiles_rejected "expected a rejection"
elif grep -q "do not compose with --tiles" "$TMP/reject1.err"; then
  pass_case "mrl_tiles_rejected (named rejection)"
else
  fail_case mrl_tiles_rejected "rejection message changed"
  cat "$TMP/reject1.err" >&2
fi

if "$AV2ENC_BIN" "$INPUT_64" -o "$TMP/reject.ivf" --frames 1 --video \
      --video-ist --video-qm >/dev/null 2>"$TMP/reject2.err"; then
  fail_case ist_qm_rejected "expected a rejection"
elif grep -q "does not compose with --video-qm" "$TMP/reject2.err"; then
  pass_case "ist_qm_rejected (named rejection)"
else
  fail_case ist_qm_rejected "rejection message changed"
  cat "$TMP/reject2.err" >&2
fi

echo "video_mrl_ist_gate: $passes passed, $failures failed"
if [ "$failures" -ne 0 ]; then exit 1; fi
