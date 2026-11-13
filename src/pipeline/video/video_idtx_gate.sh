#!/usr/bin/env bash
# Decoder gate for the 2-D identity intra transform (--video-idtx) and the
# named rejection of the one-axis 1-D intra types (--video-idtx-1d).
#
#   ./video_idtx_gate.sh
#   AV2ENC_BIN=build/av2enc DAV2D=/path/to/dav2d ./video_idtx_gate.sh
#   KEEP=1 ./video_idtx_gate.sh
#
# What it proves:
#   1. `--video-idtx` writes §5.4.8 enable_idtx_intra = 1, codes the
#      §5.20.5.5 fsc_mode = 0 symbol for every luma coding block <= 32x32, and
#      lets the TX search select the 2-D identity type (9) on long-side
#      transforms.  `--video-trace` reports `idtx leaf` for those blocks.
#   2. Pinned dav2d --strict 1 --threads 1 decodes the stream and the decoded
#      samples equal `--video-recon` exactly.
#   3. `--video-tx-search` without `--video-idtx` never selects identity types
#      (the §5.4.8 sequence bit is 0), so its stream stays conformant.
#   4. `--video-idtx-1d` is rejected by name with the pinned-decoder repro:
#      dav2d rejects negative H_DCT/V_DCT coefficients under --strict 1 (the
#      `hv_neg.ivf` case in src/codec/intra/supplemental/).
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "$0")/../../.." && pwd)"
TMP="${TMPDIR:-/tmp}/av2enc_video_idtx_gate.$$"
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
  echo "video_idtx_gate: no av2enc binary; set AV2ENC_BIN=..." >&2
  exit 127
fi

DAV2D="${DAV2D:-}"
if [ -z "$DAV2D" ]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$candidate" ]; then DAV2D="$candidate"; break; fi
  done
fi
if [ -z "$DAV2D" ] || [ ! -x "$DAV2D" ]; then
  echo "video_idtx_gate: no dav2d binary (set DAV2D=...)" >&2
  exit 77
fi

INPUT="${AV2ENC_INPUT:-$ROOT/testdata/shine_1280x720_60f.y4m}"
if [ ! -f "$INPUT" ]; then
  echo "video_idtx_gate: missing input $INPUT" >&2
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

# run_case <name> <frames> <expect idtx leaves|-> <expected md5> [args...]
run_case() {
  local name=$1 frames=$2 idtx=$3 expected=$4; shift 4
  local out="$TMP/$name.ivf" recon="$TMP/$name.recon.y4m"
  local dec="$TMP/$name.dav.y4m" log="$TMP/$name.log"
  if ! "$AV2ENC_BIN" "$INPUT" -o "$out" --frames "$frames" --video \
        --video-recon "$recon" --video-trace "$@" >"$log" 2>&1; then
    fail_case "$name" "encode failed"; cat "$log" >&2; return
  fi
  local got; got=$(md5_of "$out")
  if [ "$got" != "$expected" ]; then
    fail_case "$name" "IVF md5 $got, expected $expected"; return
  fi
  local leaves; leaves=$(grep -c "idtx leaf" "$log" || true)
  if [ "$idtx" = "-" ]; then
    if [ "$leaves" -ne 0 ]; then
      fail_case "$name" "selected $leaves identity leaf(s) with enable_idtx_intra = 0"
      return
    fi
    pass_case "$name identity filter (0 IDTX leaves)"
  else
    if [ "$leaves" -ne "$idtx" ]; then
      fail_case "$name" "selected $leaves IDTX leaves, expected $idtx"
      return
    fi
    pass_case "$name IDTX leaves ($leaves)"
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
  pass_case "$name ($(wc -c <"$out" | tr -d ' ') B, md5 $expected, exact)"
}

# §5.4.8 fix: the TX search must not select IDTX when the sequence bit is 0.
run_case tx_search_4 4 - be155c435a6c4d591bb5b7ab7d084c7e --video-tx-search
run_case tx_search_idtx_4 4 4 13c96481b49d6150d7cb9fa4240f5c69 \
  --video-tx-search --video-idtx
run_case tx_search_idtx_8 8 8 c4e7ceab9bf98e672c992b30b93aa871 \
  --video-tx-search --video-idtx

BASE="$TMP/base.ivf"
if ! "$AV2ENC_BIN" "$INPUT" -o "$BASE" --frames 4 --video \
      >"$TMP/base.log" 2>&1; then
  fail_case base "default encode failed"
else
  BASE_MD5=$(md5_of "$BASE")
  if [ "$BASE_MD5" != "e4cf72dd13ddc2c32c2f8683153b3c81" ]; then
    fail_case base "IVF md5 $BASE_MD5, expected e4cf72dd13ddc2c32c2f8683153b3c81"
  else
    pass_case "base (--video unchanged)"
  fi
fi

# Named rejections.
if "$AV2ENC_BIN" "$INPUT" -o "$TMP/reject.ivf" --frames 4 --video \
      --video-idtx >/dev/null 2>"$TMP/reject1.err"; then
  fail_case idtx_requires_tx_search "expected a rejection"
elif grep -q "requires --video-tx-search" "$TMP/reject1.err"; then
  pass_case "idtx_requires_tx_search (named rejection)"
else
  fail_case idtx_requires_tx_search "rejection message changed"
  cat "$TMP/reject1.err" >&2
fi

if "$AV2ENC_BIN" "$INPUT" -o "$TMP/reject.ivf" --frames 4 --video \
      --video-idtx-1d >/dev/null 2>"$TMP/reject2.err"; then
  fail_case idtx_1d_rejected "expected a rejection"
elif grep -q "rejects every negative H_DCT/V_DCT coefficient under --strict 1" \
      "$TMP/reject2.err"; then
  pass_case "idtx_1d_rejected (pinned-decoder repro named)"
else
  fail_case idtx_1d_rejected "rejection message changed"
  cat "$TMP/reject2.err" >&2
fi

if [ "$failures" -ne 0 ]; then
  echo "video_idtx_gate: $failures failed, $passes passed" >&2
  exit 1
fi
echo "video_idtx_gate: $passes passed, 0 failed"
if [ "${KEEP:-0}" = "1" ]; then echo "streams: $TMP"; fi
