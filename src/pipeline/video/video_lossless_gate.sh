#!/usr/bin/env bash
# Decoder gate for the lossless video path (--video-lossless).
#
#   ./video_lossless_gate.sh
#   AV2ENC_BIN=build/av2enc DAV2D=/path/to/dav2d ./video_lossless_gate.sh
#   KEEP=1 ./video_lossless_gate.sh
#
# What it proves:
#   1. `--video-lossless` routes the plain multi-frame sequence to the §5.18.2
#      lossless inter writer: its own sequence/frame headers (base_q_idx = 0,
#      CodedLossless = 1, TxMode = ONLY_4X4), DPCM, reversible WHT/IDTX and
#      FSC coefficient syntax.
#   2. Pinned dav2d --strict 1 --threads 1 decodes every picture and the
#      decoded samples equal `--video-recon` exactly, for the all-inter
#      stream, the mixed intra/inter stream (--video-lossless-intra-period 4)
#      and the forced-FSC path (--lossless-fsc).
#   3. Four and eight frame streams are IVF-pinned.
#   4. A `--video` stream is unchanged (this is an independent writer) and the
#      video tools the lossless writer does not implement are rejected by name.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "$0")/../../.." && pwd)"
TMP="${TMPDIR:-/tmp}/av2enc_video_lossless_gate.$$"
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
  echo "video_lossless_gate: no av2enc binary; set AV2ENC_BIN=..." >&2
  exit 127
fi

DAV2D="${DAV2D:-}"
if [ -z "$DAV2D" ]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$candidate" ]; then DAV2D="$candidate"; break; fi
  done
fi
if [ -z "$DAV2D" ] || [ ! -x "$DAV2D" ]; then
  echo "video_lossless_gate: no dav2d binary (set DAV2D=...)" >&2
  exit 77
fi

INPUT="${AV2ENC_INPUT:-$ROOT/testdata/shine_1280x720_60f.y4m}"
if [ ! -f "$INPUT" ]; then
  echo "video_lossless_gate: missing input $INPUT" >&2
  exit 77
fi

md5_of() {
  md5 -q "$1" 2>/dev/null || md5sum "$1" | awk '{print $1}'
}

y4m_payload() {
  python3 - "$1" <<'PY'
import sys
data = open(sys.argv[1], "rb").read()
out = bytearray()
i = data.find(b"FRAME\n")
while i >= 0:
    j = data.find(b"FRAME\n", i + 6)
    out += data[i + 6:j if j >= 0 else len(data)]
    i = j
sys.stdout.buffer.write(out)
PY
}

compare_exact() {
  python3 - "$1" "$2" "$3" <<'PY'
import sys

def payload(path):
    data = open(path, "rb").read()
    out = bytearray()
    i = data.find(b"FRAME\n")
    while i >= 0:
        j = data.find(b"FRAME\n", i + 6)
        out += data[i + 6:j if j >= 0 else len(data)]
        i = j
    return bytes(out)

a = payload(sys.argv[1])
b = payload(sys.argv[2])
if len(a) != len(b):
    raise SystemExit(f"{sys.argv[3]}: {len(a)} != {len(b)} bytes")
bad = sum(1 for x, y in zip(a, b) if x != y)
if bad:
    raise SystemExit(f"{sys.argv[3]}: {bad}/{len(a)} samples differ")
print(f"exact {sys.argv[3]}: {len(a)} samples")
PY
}

failures=0
passes=0
pass_case() { echo "PASS $1"; passes=$((passes + 1)); }
fail_case() { echo "FAIL $1: $2" >&2; failures=$((failures + 1)); }

# run_case <name> <frames> <expected md5> [encoder args...]
run_case() {
  local name=$1 frames=$2 expected=$3; shift 3
  local out="$TMP/$name.ivf" recon="$TMP/$name.recon.y4m" dec="$TMP/$name.dav.y4m"
  if ! "$AV2ENC_BIN" "$INPUT" -o "$out" --frames "$frames" \
        --video-lossless --video-recon "$recon" "$@" >"$TMP/$name.log" 2>&1; then
    fail_case "$name" "encode failed"; cat "$TMP/$name.log" >&2; return
  fi
  local got; got=$(md5_of "$out")
  if [ "$got" != "$expected" ]; then
    fail_case "$name" "IVF md5 $got, expected $expected"; return
  fi
  if ! "$DAV2D" -q -i "$out" -o "$dec" --threads 1 --strict 1 \
        >"$TMP/$name.dec.log" 2>&1; then
    fail_case "$name" "dav2d --strict 1 rejected the stream"
    cat "$TMP/$name.dec.log" >&2; return
  fi
  if ! compare_exact "$recon" "$dec" "$name" >>"$TMP/$name.cmp.log" 2>&1; then
    fail_case "$name" "encoder recon differs from dav2d"
    cat "$TMP/$name.cmp.log" >&2; return
  fi
  pass_case "$name ($(wc -c <"$out" | tr -d ' ') B, md5 $expected)"
}

# All-inter (IDTX/FSC on every P block), mixed DPCM/WHT, forced FSC, 8 frames.
run_case all_inter_4 4 3569437af68c2ff640e6deda5be2b5a5
run_case mixed_4 4 59cf90c17e97639bfd9ed8d06670a964 \
  --video-lossless-intra-period 4
run_case forced_fsc_4 4 31e50ec0d113cc02d1383664ff371124 \
  --video-lossless-intra-period 4 --lossless-fsc
run_case all_inter_8 8 19c4b81811692127c108fad1c6ec7f47

# Independent writer: a normal --video stream must not move.
BASE="$TMP/base.ivf"
if ! "$AV2ENC_BIN" "$INPUT" -o "$BASE" --frames 4 --video \
      >"$TMP/base.log" 2>&1; then
  fail_case base "default --video encode failed"
else
  BASE_MD5=$(md5_of "$BASE")
  if [ "$BASE_MD5" != "e4cf72dd13ddc2c32c2f8683153b3c81" ]; then
    fail_case base "IVF md5 $BASE_MD5, expected e4cf72dd13ddc2c32c2f8683153b3c81"
  else
    pass_case "base (--video unchanged)"
  fi
fi

# Named rejections for tools the lossless writer does not apply.
reject_flag() {
  local label=$1 needle=$2; shift 2
  if "$AV2ENC_BIN" "$INPUT" -o "$TMP/reject.ivf" --frames 4 \
        --video-lossless "$@" >/dev/null 2>"$TMP/reject.err"; then
    fail_case "$label" "expected a rejection"
  elif grep -q "$needle" "$TMP/reject.err"; then
    pass_case "$label (named rejection)"
  else
    fail_case "$label" "rejection message changed"
    cat "$TMP/reject.err" >&2
  fi
}
reject_flag reject_qm "not applied by the --video-lossless writer" --video-qm
reject_flag reject_inter_residual \
  "not applied by the --video-lossless writer" --video-inter-residual
reject_flag reject_tiles "not applied by the --video-lossless writer" --tiles 2x1
reject_flag reject_qp "is Q=0" --qp 90

if [ "$failures" -ne 0 ]; then
  echo "video_lossless_gate: $failures failed, $passes passed" >&2
  exit 1
fi
echo "video_lossless_gate: $passes passed, 0 failed"
if [ "${KEEP:-0}" = "1" ]; then echo "streams: $TMP"; fi
