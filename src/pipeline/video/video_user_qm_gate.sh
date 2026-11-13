#!/usr/bin/env bash
# Decoder gate for the video user-defined QM carrier (--video-user-qm).
#
#   ./video_user_qm_gate.sh
#   AV2ENC_BIN=build/av2enc AVM_DEC=/path/to/avmdec ./video_user_qm_gate.sh
#   KEEP=1 ./video_user_qm_gate.sh
#
# What it proves:
#   1. `--video-user-qm` emits a §5.13 OBU_QUANTIZATION_MATRIX (type 22) after
#      the sequence header and before the key frame, with an identity (all-32)
#      user matrix for level 0.
#   2. Pinned dav2d f9dafbc rejects every QM OBU ("QM OBUs with user-defined
#      QMs are currently unsupported"), so this carrier is gated with avmdec.
#   3. avmdec decodes all four pictures and the decoded samples equal the
#      encoder's --video-recon dump exactly (the user QM OBU and the §7.14.4
#      dequant state agree).
#   4. `--video` and `--video-qm` stay byte-identical (the new flag is off).
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "$0")/../../.." && pwd)"
TMP="${TMPDIR:-/tmp}/av2enc_video_user_qm_gate.$$"
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
  echo "video_user_qm_gate: no av2enc binary; set AV2ENC_BIN=..." >&2
  exit 127
fi

DAV2D="${DAV2D:-}"
if [ -z "$DAV2D" ]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$candidate" ]; then DAV2D="$candidate"; break; fi
  done
fi
if [ -z "$DAV2D" ] || [ ! -x "$DAV2D" ]; then
  echo "video_user_qm_gate: no dav2d binary (set DAV2D=...)" >&2
  exit 77
fi

AVM_DEC="${AVM_DEC:-${AV2ENC_AVMDEC:-}}"
if [ -z "$AVM_DEC" ]; then
  for candidate in "$ROOT/tools/bin/avm-macos/avmdec" \
                   "$ROOT/tools/bin/avmdec" \
                   /Users/arin/work/avm_src/build/avmdec; do
    if [ -x "$candidate" ]; then AVM_DEC="$candidate"; break; fi
  done
fi
if [ -z "$AVM_DEC" ] || [ ! -x "$AVM_DEC" ]; then
  echo "video_user_qm_gate: avmdec is required for the user QM carrier" >&2
  exit 77
fi

INPUT="${AV2ENC_INPUT:-$ROOT/testdata/shine_1280x720_60f.y4m}"
FRAMES=4
if [ ! -f "$INPUT" ]; then
  echo "video_user_qm_gate: missing input $INPUT" >&2
  exit 77
fi

md5_of() {
  md5 -q "$1" 2>/dev/null || md5sum "$1" | awk '{print $1}'
}

# y4m_payload <file>: concatenated frame payloads, ignoring every header line.
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

# list_obus <ivf> -> "type:length" per line.
list_obus() {
  python3 - "$1" <<'PY'
import struct, sys
data = open(sys.argv[1], "rb").read()
if data[:4] != b"DKIF":
    raise SystemExit("not an IVF file")
size = struct.unpack_from("<I", data, 32)[0]
payload = data[44:44 + size]
i = 0
while i < len(payload):
    value = 0
    shift = 0
    while True:
        byte = payload[i]
        i += 1
        value |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            break
        shift += 7
    obu = payload[i:i + value]
    i += value
    print(f"{(obu[0] >> 2) & 0x1F}:{len(obu)}")
PY
}

compare_exact() {
  python3 - "$1" "$2" "$3" <<'PY'
import sys

def y4m_payload(path):
    data = open(path, "rb").read()
    out = bytearray()
    i = data.find(b"FRAME\n")
    while i >= 0:
        j = data.find(b"FRAME\n", i + 6)
        out += data[i + 6:j if j >= 0 else len(data)]
        i = j
    return bytes(out)

a = y4m_payload(sys.argv[1])
b = open(sys.argv[2], "rb").read()
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

# --- 1. Base and default-matrix streams: dav2d, recon exact, pinned MD5s -----
run_dav2d_case_base() {
  local name=$1 expected=$2; shift 2
  local out="$TMP/$name.ivf" dec="$TMP/$name.dav.y4m" recon="$TMP/$name.recon.y4m"
  if ! "$AV2ENC_BIN" "$INPUT" -o "$out" --frames "$FRAMES" --video \
        --video-recon "$recon" "$@" >"$TMP/$name.log" 2>&1; then
    fail_case "$name" "encode failed"; cat "$TMP/$name.log" >&2; return
  fi
  local got; got=$(md5_of "$out")
  if [ "$got" != "$expected" ]; then
    fail_case "$name" "IVF md5 $got, expected $expected"; return
  fi
  if ! "$DAV2D" -q -i "$out" -o "$dec" --threads 1 --strict 1 \
        >"$TMP/$name.dec.log" 2>&1; then
    fail_case "$name" "dav2d --strict 1 rejected the stream"; return
  fi
  y4m_payload "$dec" >"$TMP/$name.dec.raw"
  if ! compare_exact "$recon" "$TMP/$name.dec.raw" "$name" \
        >>"$TMP/$name.cmp.log" 2>&1; then
    fail_case "$name" "encoder recon differs from dav2d"
    cat "$TMP/$name.cmp.log" >&2; return
  fi
  pass_case "$name ($(wc -c <"$out" | tr -d ' ') B, md5 $expected)"
}

run_dav2d_case_base base e4cf72dd13ddc2c32c2f8683153b3c81
run_dav2d_case_base qm 25b029a72820aa7f44eaa348468054d5 --video-qm

# --- 2. User QM carrier ------------------------------------------------------
USER="$TMP/user.ivf"
URECON="$TMP/user.recon.y4m"
if ! "$AV2ENC_BIN" "$INPUT" -o "$USER" --frames "$FRAMES" --video \
      --video-user-qm --video-recon "$URECON" >"$TMP/user.log" 2>&1; then
  fail_case user_qm_encode "encode failed"; cat "$TMP/user.log" >&2
else
  UMD5=$(md5_of "$USER")
  if [ "$UMD5" != "2bba524225209c18b7af16ddf298f493" ]; then
    fail_case user_qm_pin "IVF md5 $UMD5, expected 2bba524225209c18b7af16ddf298f493"
  else
    pass_case "user_qm_pin (md5 $UMD5)"
  fi
  if ! list_obus "$USER" | head -4 | grep -q '^22:'; then
    fail_case user_qm_obu "no OBU_QUANTIZATION_MATRIX (type 22) after the sequence header"
  else
    pass_case "user_qm_obu (TD, SEQ, 22, CLK)"
  fi
  if list_obus "$USER" | head -4 | paste -sd, - | grep -q '^2:.*,1:.*,22:.*,4:'; then
    pass_case "user_qm_order (type 22 precedes the key frame)"
  else
    fail_case user_qm_order "unexpected key unit OBU order"
  fi
  # Pinned dav2d must reject the OBU with the documented message.
  if "$DAV2D" -q -i "$USER" -o "$TMP/user.dav.y4m" --threads 1 --strict 1 \
        >"$TMP/user.dav.log" 2>&1; then
    fail_case user_qm_dav2d "dav2d unexpectedly accepted a user-QM OBU"
  elif grep -q "QM OBUs with user-defined QMs are currently unsupported" \
        "$TMP/user.dav.log"; then
    pass_case "user_qm_dav2d (rejected by pinned dav2d)"
  else
    fail_case user_qm_dav2d "rejection message changed"
    cat "$TMP/user.dav.log" >&2
  fi
  # avmdec is the decoder for this carrier: 4 pictures, sample-exact recon.
  if ! "$AVM_DEC" --codec=av2 --i420 -o "$TMP/user.avm.yuv" "$USER" \
        >"$TMP/user.avm.log" 2>&1; then
    fail_case user_qm_avmdec "decode failed"; cat "$TMP/user.avm.log" >&2
  else
    HAVE=$(wc -c <"$TMP/user.avm.yuv" | tr -d ' ')
    TOTAL=$(python3 - "$URECON" "$FRAMES" <<'PY'
import re, sys
header = open(sys.argv[1], "rb").readline().decode("ascii")
w = int(re.search(r" W(\d+)", header).group(1))
h = int(re.search(r" H(\d+)", header).group(1))
print(w * h * 3 // 2 * int(sys.argv[2]))
PY
)
    if [ "$HAVE" -ne "$TOTAL" ]; then
      fail_case user_qm_pictures "avmdec produced $HAVE bytes, expected $TOTAL ($FRAMES pictures)"
    else
      pass_case "user_qm_pictures (avmdec, $FRAMES pictures)"
    fi
    if ! compare_exact "$URECON" "$TMP/user.avm.yuv" user_recon \
          >>"$TMP/user.cmp.log" 2>&1; then
      fail_case user_qm_recon "encoder recon differs from avmdec"
      cat "$TMP/user.cmp.log" >&2
    else
      pass_case "user_qm_recon (avmdec == --video-recon)"
    fi
  fi
fi

if [ "$failures" -ne 0 ]; then
  echo "video_user_qm_gate: $failures failed, $passes passed" >&2
  exit 1
fi
echo "video_user_qm_gate: $passes passed, 0 failed"
if [ "${KEEP:-0}" = "1" ]; then echo "streams: $TMP"; fi
