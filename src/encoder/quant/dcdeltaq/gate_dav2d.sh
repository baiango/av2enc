#!/usr/bin/env bash
# Decoder gate for the DC/AC delta-Q header splice.
#
#   ./gate_dav2d.sh
#   DAV2D=/path/to/dav2d AV2ENC_BIN=/path/to/av2enc ./gate_dav2d.sh
#   KEEP=1 ./gate_dav2d.sh
#
# What it proves:
#   1. A byte-identical header rebuild (--identity) leaves the M1 stream
#      unchanged.
#   2. Non-zero Y-DC, UV-DC and UV-AC deltas in both sequence and frame headers
#      decode with `dav2d --strict 1 --threads 1`, and the reconstruction is
#      sample-identical to the reference stream.  The flat DC-only source makes
#      base_q_idx 1 +7 deltas the same quantizer as the reference QP 8 stream
#      (Y AC stays at qlookup(1), but has no coefficients).
#   3. `separate_uv_delta_q = 1` with diff_uv_delta = 1 (different U/V
#      deltas) also decodes and matches.
#   4. A variant whose effective Y-DC quantizer changes still decodes (and
#      changes pixels), proving the deltas are not silently ignored.
set -euo pipefail
cd "$(dirname "$0")"

ROOT="$(cd ../../../.. && pwd)"
TMP="${TMPDIR:-/tmp}/av2enc_dcdeltaq_gate.$$"
mkdir -p "$TMP"
cleanup() {
  if [ "${KEEP:-0}" != "1" ]; then rm -rf "$TMP"; fi
}
trap cleanup EXIT

CXX="${CXX:-c++}"
AV2ENC_BIN="${AV2ENC_BIN:-}"
if [ -z "$AV2ENC_BIN" ]; then
  for candidate in "$ROOT/build-m1/av2enc" "$ROOT/build/av2enc" "$ROOT/build-m0/av2enc"; do
    if [ -x "$candidate" ]; then AV2ENC_BIN="$candidate"; break; fi
  done
fi
if [ -z "$AV2ENC_BIN" ] || [ ! -x "$AV2ENC_BIN" ]; then
  echo "dcdeltaq gate: no av2enc binary; set AV2ENC_BIN=..." >&2
  exit 127
fi

DAV2D="${DAV2D:-}"
if [ -z "$DAV2D" ]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$candidate" ]; then DAV2D="$candidate"; break; fi
  done
fi
if [ -z "$DAV2D" ] || [ ! -x "$DAV2D" ]; then
  echo "dcdeltaq gate: no dav2d binary; set DAV2D=..." >&2
  exit 127
fi

GATE="${GATE_BIN:-$TMP/dcdeltaq_gate}"
"$CXX" -std=c++20 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow \
  -I"$ROOT/src" \
  dcdeltaq_gate.cpp syntax.cpp quant.cpp \
  "$ROOT/src/bitstream/bitwriter.cpp" \
  "$ROOT/src/bitstream/m0_headers.cpp" \
  "$ROOT/src/encoder/qm/qm.cpp" \
  "$ROOT/src/encoder/m1/m1_headers.cpp" \
  -o "$GATE"

# Deterministic flat 64x64 4:2:0 source: every residual is a constant (DC-only).
SRC="$TMP/flat64.y4m"
python3 - "$SRC" <<'PY'
import sys
w = h = 64
y = bytes([200]) * (w * h)
u = bytes([150]) * ((w // 2) * (h // 2))
# V is a constant 128, so its residual is zero for every quantizer; this lets
# the separate-UV variant use different U/V deltas (diff_uv_delta = 1) while
# keeping V's reconstruction independent of q.
v = bytes([128]) * ((w // 2) * (h // 2))
with open(sys.argv[1], "wb") as f:
    f.write(b"YUV4MPEG2 W64 H64 F1:1 Ip A1:1 C420\nFRAME\n" + y + u + v)
PY

BASE="$TMP/base_qp8.ivf"
"$AV2ENC_BIN" "$SRC" -o "$BASE" --m1 --qp 8 --m1-single --m1-dc --no-deblock >/dev/null

decode() {
  "$DAV2D" -q -i "$1" --muxer yuv4mpeg2 -o "$2" --strict 1 --threads 1
}

decode "$BASE" "$TMP/ref.y4m"

# 1. Identity rebuild.
"$GATE" "$BASE" "$TMP/identity.ivf" --identity --ref-qp 8 >/dev/null
if ! cmp -s "$BASE" "$TMP/identity.ivf"; then
  echo "FAIL: identity header rebuild changed the stream" >&2
  exit 1
fi
echo "identity rebuild: byte-identical"

# 2. equal_ac_dc_q = 0, UV DC/AC shared.
"$GATE" "$BASE" "$TMP/variant_shared.ivf" \
  --ref-qp 8 --base-q 1 --ydc 7 --udc 7 --uac 7 >/dev/null
decode "$TMP/variant_shared.ivf" "$TMP/variant_shared.y4m"
if ! cmp -s "$TMP/ref.y4m" "$TMP/variant_shared.y4m"; then
  echo "FAIL: shared-UV delta stream differs from the reference recon" >&2
  exit 1
fi
echo "dav2d strict: Y-DC/UV-DC/UV-AC deltas, recon 0 diffs"

# 3. separate_uv_delta_q = 1 with diff_uv_delta = 1.
"$GATE" "$BASE" "$TMP/variant_separate.ivf" \
  --ref-qp 8 --base-q 1 --ydc 7 --udc 7 --vdc -3 --uac 7 --vac -3 \
  --separate-uv >/dev/null
decode "$TMP/variant_separate.ivf" "$TMP/variant_separate.y4m"
if ! cmp -s "$TMP/ref.y4m" "$TMP/variant_separate.y4m"; then
  echo "FAIL: separate_uv_delta_q stream differs from the reference recon" >&2
  exit 1
fi
echo "dav2d strict: separate_uv_delta_q = 1, diff_uv_delta = 1, recon 0 diffs"

# 3b. Non-zero sequence *base* deltas: Y -7 + frame 14, UV -5 + frame 12.
"$GATE" "$BASE" "$TMP/variant_base.ivf" \
  --ref-qp 8 --base-q 1 --base-ydc -7 --base-uvdc -5 --base-uvac -5 \
  --ydc 14 --udc 12 --uac 12 >/dev/null
decode "$TMP/variant_base.ivf" "$TMP/variant_base.y4m"
if ! cmp -s "$TMP/ref.y4m" "$TMP/variant_base.y4m"; then
  echo "FAIL: sequence base-delta stream differs from the reference recon" >&2
  exit 1
fi
echo "dav2d strict: non-zero sequence base deltas, recon 0 diffs"

# 4. A Y-DC delta that changes the effective quantizer must decode and change.
"$GATE" "$BASE" "$TMP/variant_changed.ivf" \
  --ref-qp 8 --base-q 1 --ydc 8 --udc 7 --uac 7 >/dev/null
decode "$TMP/variant_changed.ivf" "$TMP/variant_changed.y4m"
if cmp -s "$TMP/ref.y4m" "$TMP/variant_changed.y4m"; then
  echo "FAIL: a changed Y-DC quantizer did not change the reconstruction" >&2
  exit 1
fi
echo "dav2d strict: changed Y-DC delta is applied"

echo "dcdeltaq gate: OK"
if [ "${KEEP:-0}" = "1" ]; then
  echo "streams: $TMP"
fi
