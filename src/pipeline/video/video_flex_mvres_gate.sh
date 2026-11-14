#!/usr/bin/env bash
# CPU decoder gate for --video-flex-mvres (§5.4.6 enable_flex_mvres,
# §5.18.2 UsePerBlockMvPrecision, §5.20.7.6 use_most_probable_precision /
# pb_mv_precision).
#
# Two deterministic fixtures (64x64 and 640x360, four frames each, a translating
# texture that produces exact per-block MVDs) are encoded through the product
# CLI with --video-refs 2 --video-inter-residual --video-tx-search, decoded with
# the pinned dav2d at --strict 1 --threads 1, and compared sample-for-sample
# against the encoder's --video-recon dump.  The stream must report at least
# one flex-precision leaf, otherwise the flag would be inert.
#
# The default (flag off) streams are pinned by MD5, which is the
# "default streams byte-identical" check.  Passing --base-bin additionally
# compares a pre-change binary byte-for-byte.
#
# Usage:
#   src/pipeline/video/video_flex_mvres_gate.sh --bin build-int/av2enc \
#       [--dav2d /path/to/dav2d] [--base-bin old/av2enc]
#
# Exit codes: 0 pass, 1 failure, 2 usage, 77 missing decoder.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/../../.." && pwd)
BIN=""
BASE_BIN=""
DAV2D="${DAV2D:-}"
OUT="${OUT:-${TMPDIR:-/tmp}/video_flex_mvres_gate}"

DEFAULT64_PIN=b04bcb9b452e17841b077e96491c18a1
DEFAULT360_PIN=aa81dd3a69098c8f6f18eb8d24c119fc
REFS2_64_PIN=320a799618fa46e8bc5bb8d611f38c93
REFS2_360_PIN=bbd36a7fff8e07c592b28f5dda1a8c87

while [ $# -gt 0 ]; do
  case "$1" in
    --bin) BIN=$2; shift 2 ;;
    --dav2d) DAV2D=$2; shift 2 ;;
    --base-bin) BASE_BIN=$2; shift 2 ;;
    -h|--help) sed -n '2,23p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "video_flex_mvres_gate: unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -n "$BIN" ] && [ -x "$BIN" ] || {
  echo "video_flex_mvres_gate: --bin is required" >&2; exit 2; }
if [ -z "$DAV2D" ]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
  done
fi
[ -n "$DAV2D" ] && [ -x "$DAV2D" ] || {
  echo "video_flex_mvres_gate: dav2d not found (set DAV2D)" >&2; exit 77; }

mkdir -p "$OUT"
FAIL=0
md5() { md5sum "$1" 2>/dev/null | cut -d' ' -f1 || md5 -q "$1"; }
pass() { echo "PASS: $*"; }
fail() { echo "FAIL: $*" >&2; FAIL=1; }

# ---------------------------------------------------------------------------
# Deterministic fixtures: a translating texture so the ME MVDs are exact
# multiples of the coarser precision radix (2 luma samples per frame).
# ---------------------------------------------------------------------------
python3 - "$OUT" <<'PY'
import sys
out = sys.argv[1]

def write(name, w, h, frames):
    hdr = ("YUV4MPEG2 W%d H%d F30:1 Ip A1:1 C420jpeg\n" % (w, h)).encode()
    blob = bytearray(hdr)
    for y, u, v in frames:
        blob += b"FRAME\n" + bytes(y) + bytes(u) + bytes(v)
    open(out + "/" + name, "wb").write(blob)

def frame(w, h, shift):
    y = bytearray(w * h)
    for j in range(h):
        for i in range(w):
            x = (i + shift) % w
            y[j * w + i] = (x * 3 + j * 2 + (x // 8) * 17 + (j // 8) * 11) % 256
    u = bytearray((w // 2) * (h // 2))
    v = bytearray((w // 2) * (h // 2))
    for j in range(h // 2):
        for i in range(w // 2):
            u[j * (w // 2) + i] = (64 + i * 2 + shift) % 256
            v[j * (w // 2) + i] = (160 + j * 2 + shift) % 256
    return y, u, v

write("mvres64_4f.y4m", 64, 64, [frame(64, 64, t * 2) for t in range(4)])
write("mvres360_4f.y4m", 640, 360,
      [frame(640, 360, t * 2) for t in range(4)])
PY
FIX64="$OUT/mvres64_4f.y4m"
FIX360="$OUT/mvres360_4f.y4m"

# Default-stream pins: the flags are off, so these must not move.
check_pin() {
  local label=$1 input=$2 pin=$3
  shift 3
  local ivf="$OUT/pin_$label.ivf"
  "$BIN" "$input" -o "$ivf" --video --frames 4 "$@" >/dev/null 2>&1 || {
    fail "pin $label: encode failed"; return; }
  local got
  got=$(md5 "$ivf")
  if [ "$got" != "$pin" ]; then
    fail "pin $label: default stream changed ($got != $pin)"
  else
    pass "pin $label: $got"
  fi
}
check_pin 64 "$FIX64" "$DEFAULT64_PIN"
check_pin 360 "$FIX360" "$DEFAULT360_PIN"
check_pin 64_refs2 "$FIX64" "$REFS2_64_PIN" \
  --video-refs 2 --video-inter-residual --video-tx-search
check_pin 360_refs2 "$FIX360" "$REFS2_360_PIN" \
  --video-refs 2 --video-inter-residual --video-tx-search

# One exact-decoder case.  $1 label, $2 input, $3 minimum flex leaves,
# remaining args = encoder options.
run_case() {
  local label=$1 input=$2 want=$3
  shift 3
  local ivf="$OUT/$label.ivf" recon="$OUT/$label.recon.y4m"
  local dec="$OUT/$label.dec.y4m" log="$OUT/$label.log"
  echo "== $label: $*"
  if ! "$BIN" "$input" -o "$ivf" --video --frames 4 \
      --video-recon "$recon" "$@" >"$log" 2>&1; then
    fail "$label: encode failed"; tail -5 "$log"; return
  fi
  local got
  got=$(sed -n 's/.*flex-precision leaves \([0-9]*\).*/\1/p' "$log" | tail -1)
  got=${got:-0}
  if [ "$got" -lt "$want" ]; then
    fail "$label: feature inert ($got < $want flex-precision leaves)"; return
  fi
  if ! "$DAV2D" -q -i "$ivf" -o "$dec" --strict 1 --threads 1; then
    fail "$label: dav2d --strict 1 rejected the stream"; return
  fi
  local recon_md5 dec_md5
  recon_md5=$(md5 "$recon"); dec_md5=$(md5 "$dec")
  if [ "$recon_md5" != "$dec_md5" ]; then
    fail "$label: recon mismatch (encoder $recon_md5 decoder $dec_md5)"
    return
  fi
  local frames
  frames=$(LC_ALL=C grep -a -o 'FRAME' "$dec" | wc -l | tr -d ' ')
  if [ "$frames" -ne 4 ]; then
    fail "$label: decoded $frames of 4 picture(s)"; return
  fi
  pass "$label: 4 pictures, flex=$got, md5=$recon_md5"
}

run_case flex64_eighth "$FIX64" 1 \
  --video-refs 2 --video-inter-residual --video-tx-search --video-flex-mvres
run_case flex360_eighth "$FIX360" 1 \
  --video-refs 2 --video-inter-residual --video-tx-search --video-flex-mvres
run_case flex64_quarter "$FIX64" 1 \
  --video-refs 2 --video-inter-residual --video-tx-search \
  --video-mv-precision quarter --video-flex-mvres
run_case flex360_quarter "$FIX360" 1 \
  --video-refs 2 --video-inter-residual --video-tx-search \
  --video-mv-precision quarter --video-flex-mvres

# Named rejections: compositions the block writer cannot code exactly.
reject() {
  local label=$1 want=$2
  shift 2
  local log="$OUT/reject_$label.log"
  if "$BIN" "$FIX64" -o "$OUT/reject_$label.ivf" --video --frames 4 "$@" \
      >"$log" 2>&1; then
    fail "reject/$label: command unexpectedly succeeded"
  elif ! grep -q "$want" "$log"; then
    fail "reject/$label: missing rejection '$want'"; tail -3 "$log"
  else
    pass "reject/$label: $want"
  fi
}
reject bframes "flat key+P chain" --video-flex-mvres --b-frames 2
reject compound "does not compose" --video-flex-mvres --video-compound \
  --video-refs 2
reject bawp "does not compose" --video-flex-mvres --video-bawp
reject tiles "single tile" --video-flex-mvres --tiles 2x1

# Optional pre-change binary byte identity with every flag off.
if [ -n "$BASE_BIN" ] && [ -x "$BASE_BIN" ]; then
  for spec in "64:$FIX64:" "360:$FIX360:" \
              "64r2:$FIX64:--video-refs 2 --video-inter-residual --video-tx-search"; do
    label=${spec%%:*}; rest=${spec#*:}; input=${rest%%:*}; extra=${rest#*:}
    # shellcheck disable=SC2086
    "$BASE_BIN" "$input" -o "$OUT/base_$label.ivf" --video --frames 4 \
      $extra >/dev/null 2>&1
    # shellcheck disable=SC2086
    "$BIN" "$input" -o "$OUT/new_$label.ivf" --video --frames 4 \
      $extra >/dev/null 2>&1
    if cmp -s "$OUT/base_$label.ivf" "$OUT/new_$label.ivf"; then
      pass "identity/$label: flag off byte-identical"
    else
      fail "identity/$label: default stream changed with the flag off"
    fi
  done
fi

if [ "$FAIL" -ne 0 ]; then
  echo "video_flex_mvres_gate: FAILED" >&2
  exit 1
fi
echo "video_flex_mvres_gate: OK"
