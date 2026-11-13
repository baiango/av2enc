#!/usr/bin/env bash
# Decoder gate for the opt-in §5.18.7.1 segmentation / §5.18.7.8 delta-Q
# features on the CPU `--video` writer:
#
#   --video-segmentation     per-leaf segment id from a per-superblock luma
#                            variance split (SEG_LVL_ALT_Q = 0)
#   --video-delta-q          per-superblock CurrentQIndex updates
#   --video-per-segment-q    real SEG_LVL_ALT_Q driving the block quantizer
#
# Every case is encoded through the product CLI, decoded with the pinned
# dav2d --strict 1, and compared sample-for-sample against --video-recon.
# The picture count is checked (dav2d exits 0 on a truncated stream), and the
# IVF MD5s are pinned so an unintentional bitstream change fails the gate.
#
# It also pins the default-off path: `--video` without any of the new flags
# must keep the pre-change bytes.  Pass --baseline-bin with a binary built at
# the parent commit to compare directly.
#
# Usage:
#   src/pipeline/video/gate_video_deltaq.sh --bin build/av2enc \
#       [--dav2d PATH] [--baseline-bin PATH] [--input 720p.y4m]
#
# Defaults: BIN=build/av2enc, DAV2D=tools/bin/dav2d,
# INPUT=testdata/shine_1280x720_60f.y4m.  Set OUT=/tmp/prefix to keep streams.
#
# Exit codes: 0 pass, 1 failure, 2 usage, 77 missing fixture/decoder.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/../../.." && pwd)
BIN=""
DAV2D="${DAV2D:-}"
BASELINE=""
INPUT=""

while [ $# -gt 0 ]; do
  case "$1" in
    --bin) BIN=$2; shift 2 ;;
    --dav2d) DAV2D=$2; shift 2 ;;
    --baseline-bin) BASELINE=$2; shift 2 ;;
    --input) INPUT=$2; shift 2 ;;
    -h|--help)
      sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "gate_video_deltaq: unknown argument $1" >&2; exit 2 ;;
  esac
done

BIN=${BIN:-"$ROOT/build/av2enc"}
INPUT=${INPUT:-"$ROOT/testdata/shine_1280x720_60f.y4m"}
if [ -z "$DAV2D" ]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
  done
fi

[ -x "$BIN" ] || { echo "gate_video_deltaq: --bin must be an executable" >&2; exit 2; }
[ -x "$DAV2D" ] || { echo "gate_video_deltaq: no dav2d (set --dav2d)" >&2; exit 77; }
[ -f "$INPUT" ] || { echo "SKIP: missing $INPUT" >&2; exit 77; }
command -v python3 >/dev/null 2>&1 || {
  echo "gate_video_deltaq: python3 is required" >&2; exit 77; }

if [ -n "${OUT:-}" ]; then
  OUT="${OUT%*/}/video_deltaq"
else
  OUT="$(mktemp -d "${TMPDIR:-/tmp}/av2enc-video-deltaq.XXXXXX")/video_deltaq"
fi
mkdir -p "$(dirname -- "$OUT")"

PASS=0
FAIL=0
fail() {
  echo "FAIL: $*" >&2
  FAIL=$((FAIL + 1))
}

# A deterministic mixed-variance 640x360 clip: a smooth left half and a noisy
# right half, identical frames.  The P frames therefore skip (the skipped-leaf
# segment-id inference) while the per-superblock variance split uses both
# segment ids.
MIXED="${OUT}.mixed_640x360_4f.y4m"
python3 - "$MIXED" <<'PY'
import random
import sys

w, h, n = 640, 360, 4
cw, ch = w // 2, h // 2
random.seed(7)
base = bytearray(w * h)
for y in range(h):
    for x in range(w):
        if x < w // 2:
            v = 90 + (x * 40) // (w // 2) + (y * 10) // h
        else:
            v = 110 + random.randint(0, 70)
        base[y * w + x] = max(0, min(255, v))
with open(sys.argv[1], "wb") as f:
    f.write(f"YUV4MPEG2 W{w} H{h} F30:1 Ip A1:1 C420jpeg\n".encode())
    for _ in range(n):
        f.write(b"FRAME\n")
        f.write(bytes(base))
        f.write(bytes([128]) * (cw * ch))
        f.write(bytes([128]) * (cw * ch))
PY

# Case table: name | input | expected pictures | expected IVF MD5 | args...
#
# MD5s are the gated bytes of this tree.  The default (no new flag) row must
# stay at the pre-change value.
CASES=(
  "a_base|$INPUT|4|e4cf72dd13ddc2c32c2f8683153b3c81|--video --frames 4"
  "a_seg|$INPUT|4|a9b8be959de14d4a0c65f0b07acfd8a4|--video --frames 4 --video-segmentation"
  "a_dq|$INPUT|4|f37b82772a194055525d7aa3eeba4e66|--video --frames 4 --video-delta-q"
  "a_dq_r2|$INPUT|4|a8dea947e6bac38fec4fc49d3b09005c|--video --frames 4 --video-delta-q --video-delta-q-res 2 --video-inter-residual"
  "a_dq_force|$INPUT|4|661bdb721476ba2d554b61b818c77cf0|--video --frames 4 --video-delta-q --video-force-single"
  "a_dq_seg|$INPUT|4|c940328f8deec657294f0796c8ef67bd|--video --frames 4 --video-delta-q --video-segmentation --video-inter-residual"
  "a_psq|$INPUT|4|9b072e470f23256bc34e7daa3ac9eb38|--video --frames 4 --video-per-segment-q"
  "a_psq_deb|$INPUT|4|5abab13bc70deefa03a6515e66c125ae|--video --frames 4 --video-per-segment-q --video-inter-residual --video-deblock"
  "b_seg|$MIXED|4|ee7965ac6c950aa4bdd580a6faa3f495|--video --frames 4 --video-segmentation"
  "b_psq_deb|$MIXED|4|63ab06a9300fc8d840edf0ee5006d3a0|--video --frames 4 --video-per-segment-q --video-deblock"
  "b_psq_ir|$MIXED|4|399e3b33432cb81a319a43cc2834e140|--video --frames 4 --video-per-segment-q --video-inter-residual --video-tx-search --video-deblock"
  "b_dq_ir|$MIXED|4|7147a17f0095cbd1874cb41ed3a7ba14|--video --frames 4 --video-delta-q --video-inter-residual"
)

echo "== video delta-Q/segmentation gate =="
for row in "${CASES[@]}"; do
  IFS="|" read -r name input pics md5 args <<< "$row"
  # shellcheck disable=SC2206
  argv=($args)
  name_out="${OUT}.${name}"
  if ! "$BIN" "$input" "${argv[@]}" -o "${name_out}.ivf" \
      --video-recon "${name_out}.recon.y4m" >"${name_out}.log" 2>&1; then
    fail "$name: encode failed: $(tail -1 "${name_out}.log")"
    continue
  fi
  if ! "$DAV2D" -q -i "${name_out}.ivf" -o "${name_out}.dec.y4m" \
      --strict 1 --threads 1 >"${name_out}.dec.log" 2>&1; then
    fail "$name: dav2d --strict 1 rejected the stream"
    continue
  fi
  got_pics=$(python3 - "${name_out}.dec.y4m" <<'PY'
import sys

with open(sys.argv[1], "rb") as f:
    header = f.readline()
    if not header.startswith(b"YUV4MPEG2 "):
        print(0)
        raise SystemExit
    width = height = None
    for field in header.split():
        if field.startswith(b"W"):
            width = int(field[1:])
        elif field.startswith(b"H"):
            height = int(field[1:])
    cw, ch = (width + 1) // 2, (height + 1) // 2
    size = width * height + 2 * cw * ch
    count = 0
    while True:
        marker = f.readline()
        if not marker:
            break
        if marker != b"FRAME\n":
            print(-1)
            raise SystemExit
        data = f.read(size)
        if len(data) != size:
            print(-1)
            raise SystemExit
        count += 1
print(count)
PY
)
  if [ "$got_pics" -ne "$pics" ]; then
    fail "$name: decoded $got_pics picture(s), expected $pics"
    continue
  fi
  if ! cmp -s "${name_out}.recon.y4m" "${name_out}.dec.y4m"; then
    fail "$name: decoder output differs from --video-recon"
    continue
  fi
  got_md5=$(md5sum "${name_out}.ivf" | cut -c1-32)
  if [ "$got_md5" != "$md5" ]; then
    fail "$name: IVF MD5 $got_md5, expected $md5"
    continue
  fi
  echo "PASS $name: $got_pics picture(s), recon exact, md5 $got_md5"
  PASS=$((PASS + 1))
done

# Default-off byte identity against a pre-change binary (optional, strong).
if [ -n "$BASELINE" ]; then
  if [ ! -x "$BASELINE" ]; then
    fail "--baseline-bin is not executable"
  else
    "$BIN" "$INPUT" --video --frames 4 -o "${OUT}.new_base.ivf" >/dev/null 2>&1
    "$BASELINE" "$INPUT" --video --frames 4 -o "${OUT}.old_base.ivf" >/dev/null 2>&1
    if cmp -s "${OUT}.new_base.ivf" "${OUT}.old_base.ivf"; then
      echo "PASS default-off stream is byte-identical to --baseline-bin"
      PASS=$((PASS + 1))
    else
      fail "default-off stream differs from --baseline-bin"
    fi
  fi
else
  echo "SKIP --baseline-bin comparison (default byte identity is still pinned by a_base)"
fi

# The delta-Q plan really moves CurrentQIndex (a header-only delta_q_present
# with an all-zero ramp would pass the decode check above).
"$BIN" "$INPUT" --video --frames 2 --video-delta-q --video-trace \
  -o "${OUT}.trace_dq.ivf" 2>"${OUT}.trace_dq.log" >/dev/null
MOVED=$(grep -cE 'deltaq sb=[0-9]+ .*reduced=-?[1-9]' "${OUT}.trace_dq.log" || true)
if [ "${MOVED:-0}" -gt 0 ]; then
  echo "PASS delta-Q coded $MOVED non-zero per-superblock step(s)"
  PASS=$((PASS + 1))
else
  fail "delta-Q trace shows no non-zero reduced delta"
fi

# Skip-first-leaf hazard: with --video-force-single every superblock is one
# 64x64 leaf, and the P frames skip.  §5.20.5.11 codes no delta for a
# MiSize == SbSize skip, so the plan must drop those steps (the writer verifies
# the achieved qindex).  The decode above already proves the stream stays exact;
# this asserts the hazard was actually exercised.
"$BIN" "$INPUT" --video --frames 2 --video-delta-q --video-force-single \
  --video-trace -o "${OUT}.trace_skip.ivf" 2>"${OUT}.trace_skip.log" >/dev/null
DROPPED=$(grep -c 'dropped(full-SB skip)' "${OUT}.trace_skip.log" || true)
if [ "${DROPPED:-0}" -gt 0 ]; then
  echo "PASS skip-first-leaf hazard: $DROPPED full-SB-skip step(s) dropped"
  PASS=$((PASS + 1))
else
  fail "skip-first-leaf hazard was not exercised by --video-force-single"
fi

# Per-segment-Q uses both segment ids on the mixed clip and replays the
# §5.20.7.4 neighbour prediction for skipped leaves.
"$BIN" "$MIXED" --video --frames 2 --video-per-segment-q --video-trace \
  -o "${OUT}.trace_seg.ivf" 2>"${OUT}.trace_seg.log" >/dev/null
SEG0=$(grep -c 'segment=0' "${OUT}.trace_seg.log" || true)
SEG1=$(grep -c 'segment=1' "${OUT}.trace_seg.log" || true)
INFERRED=$(grep -c 'segq inferred' "${OUT}.trace_seg.log" || true)
if [ "${SEG0:-0}" -gt 0 ] && [ "${SEG1:-0}" -gt 0 ]; then
  echo "PASS per-segment-Q used segment 0 and 1 ($SEG0/$SEG1 lines)"
  PASS=$((PASS + 1))
else
  fail "per-segment-Q did not use both segment ids ($SEG0/$SEG1)"
fi
if [ "${INFERRED:-0}" -gt 0 ]; then
  echo "PASS skipped leaves replay the inferred segment id ($INFERRED)"
  PASS=$((PASS + 1))
else
  fail "no skipped leaf exercised the inferred segment id"
fi

# --- usage rejections (exit 2 before the output file is opened) ------------
reject() {
  local label=$1 needle=$2
  shift 2
  local sentinel="${OUT}.sentinel.ivf"
  printf 'SENTINEL\n' > "$sentinel"
  set +e
  "$BIN" "$@" >"${OUT}.reject.log" 2>&1
  local rc=$?
  set -e
  if [ "$rc" -ne 2 ]; then
    fail "$label: expected exit 2, got $rc"
    return
  fi
  if ! grep -qF "$needle" "${OUT}.reject.log"; then
    fail "$label: error does not mention '$needle'"
    return
  fi
  if [ "$(cat "$sentinel")" != "SENTINEL" ]; then
    fail "$label: exit 2 modified the -o file"
    return
  fi
  echo "PASS usage: $label"
  PASS=$((PASS + 1))
}

reject "per-segment-q + delta-q" "does not compose" \
  "$INPUT" -o "${OUT}.sentinel.ivf" --video --frames 4 \
  --video-per-segment-q --video-delta-q
reject "delta-q-res without delta-q" "requires --video-delta-q" \
  "$INPUT" -o "${OUT}.sentinel.ivf" --video --frames 4 \
  --video-delta-q-res 2
reject "seg-delta without a consumer" "requires --video-per-segment-q" \
  "$INPUT" -o "${OUT}.sentinel.ivf" --video --frames 4 --video-seg-delta 4
reject "delta-q + qm" "does not compose with --video-qm" \
  "$INPUT" -o "${OUT}.sentinel.ivf" --video --frames 4 \
  --video-delta-q --video-qm
reject "delta-q + cdf chain" "CDF chain" \
  "$INPUT" -o "${OUT}.sentinel.ivf" --video --frames 4 \
  --video-delta-q --avg-cdf
reject "delta-q + tiles" "does not support --tiles > 1x1" \
  "$INPUT" -o "${OUT}.sentinel.ivf" --video --frames 4 \
  --video-delta-q --tiles 2x1
reject "delta-q + rc crf" "fixed base_q_idx" \
  "$INPUT" -o "${OUT}.sentinel.ivf" --video --frames 4 \
  --video-delta-q --rc crf --crf 32
reject "delta-q + b frames" "B/temporal-layer" \
  "$INPUT" -o "${OUT}.sentinel.ivf" --video --frames 6 \
  --video-delta-q --b-frames 2
reject "delta-q 10-bit" "only wired on the 8-bit 4:2:0" \
  "$INPUT" -o "${OUT}.sentinel.ivf" --video --frames 4 \
  --bit-depth 10 --video-delta-q
reject "per-segment-q lossless segment" "lossless" \
  "$INPUT" -o "${OUT}.sentinel.ivf" --video --frames 1 --qp 8 \
  --video-per-segment-q --video-seg-delta -8

echo
if [ "$FAIL" -ne 0 ]; then
  echo "FAIL: $FAIL failure(s), $PASS pass(es)" >&2
  exit 1
fi
echo "PASS: $PASS checks (decode + recon + MD5 + usage)"
