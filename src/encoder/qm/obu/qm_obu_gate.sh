#!/usr/bin/env bash
# Decoder gate for the AV2 quantizer-matrix OBU state machine.
#
#   ./qm_obu_gate.sh
#   AV2ENC_BIN=/path/to/av2enc AVM_DEC=/path/to/avmdec DAV2D=/path/to/dav2d \
#     ./qm_obu_gate.sh
#   KEEP=1 ./qm_obu_gate.sh        # keep the generated streams
#   SKIP_TESTS=1 ./qm_obu_gate.sh  # skip build_test.sh
#
# What it proves:
#   1. Unit tests: payload/frame-header round-trips, pinned goldens and the
#      conformance rejection cases (duplicate level, second qm_bit_map == 0,
#      zero coefficient, bad trailing bits, ...).
#   2. dav2d accepts using_qmatrix = 1 with the normative default matrices when
#      no QM OBU is present.  The pinned dav2d rejects OBU type 22, so the
#      default-matrix check must use the no-OBU path.
#   3. avmdec (which does parse QM OBUs) accepts:
#        identity        -> pixels identical to the base stream
#        reset-identity  -> identical (qm_bit_map = 0 + one user level)
#        multi           -> identical (reset + default level + user level)
#        default         -> differs from base, and equals the no-OBU default
#                           matrix decode (qm_is_default_flag agrees with the
#                           implicit default)
#      An all-32 user matrix is the identity (Round2(q*32,5) == q) and an 8x8
#      4:2:0 picture only has 8x8/4x4 transforms, so identity must be exact.
set -euo pipefail
cd "$(dirname "$0")"

ROOT="$(cd ../../../.. && pwd)"
TMP="${TMPDIR:-/tmp}/av2enc_qm_obu_gate.$$"
mkdir -p "$TMP"
cleanup() {
  if [ "${KEEP:-0}" != "1" ]; then rm -rf "$TMP"; fi
}
trap cleanup EXIT

if [ "${SKIP_TESTS:-0}" != "1" ]; then
  ./build_test.sh
fi

CXX="${CXX:-c++}"
AV2ENC_BIN="${AV2ENC_BIN:-}"
if [ -z "$AV2ENC_BIN" ]; then
  for candidate in "$ROOT/build-m1/av2enc" "$ROOT/build/av2enc" "$ROOT/build-m0/av2enc"; do
    if [ -x "$candidate" ]; then AV2ENC_BIN="$candidate"; break; fi
  done
fi
if [ -z "$AV2ENC_BIN" ] || [ ! -x "$AV2ENC_BIN" ]; then
  echo "qm_obu_gate: no av2enc binary; set AV2ENC_BIN=..." >&2
  exit 127
fi

DAV2D="${DAV2D:-}"
if [ -z "$DAV2D" ]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$candidate" ]; then DAV2D="$candidate"; break; fi
  done
fi
AVM_DEC="${AVM_DEC:-}"
if [ -z "$AVM_DEC" ]; then
  for candidate in "$ROOT/tools/bin/avm-macos/avmdec" "$ROOT/tools/bin/avmdec"; do
    if [ -x "$candidate" ]; then AVM_DEC="$candidate"; break; fi
  done
fi

GATE="${QM_OBU_GATE_BIN:-$TMP/qm_obu_gate}"
if [ ! -x "$GATE" ]; then
  "$CXX" -std=c++20 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow \
    -I"$ROOT/src" \
    qm_obu_gate.cpp qm_obu.cpp \
    "$ROOT/src/bitstream/bitwriter.cpp" \
    "$ROOT/src/bitstream/m0_headers.cpp" \
    "$ROOT/src/encoder/m1/m1_headers.cpp" \
    -o "$GATE"
fi

"$GATE" --selftest

# Deterministic 8x8 4:2:0 input.
SRC="$TMP/qm_obu_8x8.y4m"
python3 - "$SRC" <<'PY'
import sys
w = h = 8
y = bytes(((x * 17 + yy * 29) & 255) for yy in range(h) for x in range(w))
u = bytes(((x * 13 + yy * 41 + 80) & 255)
          for yy in range(h // 2) for x in range(w // 2))
v = bytes(((x * 47 + yy * 7 + 40) & 255)
          for yy in range(h // 2) for x in range(w // 2))
with open(sys.argv[1], "wb") as f:
    f.write(b"YUV4MPEG2 W8 H8 F1:1 Ip A1:1 C420\nFRAME\n" + y + u + v)
PY

BASE="$TMP/base.ivf"
"$AV2ENC_BIN" "$SRC" -o "$BASE" --m1 --qp 90 >/dev/null

dav2d_md5() {
  "$DAV2D" -q -i "$1" --muxer md5 -o - 2>/dev/null | tail -1
}
avm_md5() {
  "$AVM_DEC" --codec=av2 --md5 --limit=1 "$1" 2>/dev/null |
    tail -1 | awk '{print $1}'
}

for level in 0 5 14; do
  for mode in identity default no-obu; do
    "$GATE" "$BASE" "$TMP/${mode}_$level.ivf" \
      --width 8 --height 8 --qp 90 --level "$level" --mode "$mode" >/dev/null
  done
done
for mode in reset-identity multi; do
  "$GATE" "$BASE" "$TMP/${mode}_5.ivf" \
    --width 8 --height 8 --qp 90 --level 5 --mode "$mode" >/dev/null
done

if [ -n "$DAV2D" ]; then
  BASE_MD5=$(dav2d_md5 "$BASE")
  NO_OBU_MD5=$(dav2d_md5 "$TMP/no-obu_0.ivf")
  if [ -z "$NO_OBU_MD5" ]; then
    echo "FAIL: dav2d did not decode using_qmatrix = 1 (no QM OBU)" >&2
    exit 1
  fi
  if [ "$NO_OBU_MD5" = "$BASE_MD5" ]; then
    echo "FAIL: default QM did not change pixels" >&2
    exit 1
  fi
  # Pinned dav2d rejects OBU type 22; make that explicit so a future decoder
  # change is visible rather than silently skipping the check.
  if [ -n "$(dav2d_md5 "$TMP/default_0.ivf")" ]; then
    echo "note: dav2d now accepts QM OBUs; the no-OBU check remains valid" >&2
  fi
  echo "dav2d: using_qmatrix default matrices OK ($NO_OBU_MD5 != $BASE_MD5)"
else
  echo "dav2d: skipped (no decoder)"
fi

if [ -n "$AVM_DEC" ]; then
  BASE_AVM=$(avm_md5 "$BASE")
  for level in 0 5 14; do
    MD5=$(avm_md5 "$TMP/identity_$level.ivf")
    if [ -z "$MD5" ] || [ "$MD5" != "$BASE_AVM" ]; then
      echo "FAIL: level $level identity decoded '$MD5'," \
           "expected '$BASE_AVM'" >&2
      exit 1
    fi
    echo "avmdec: level $level identity == base"
  done
  for mode in reset-identity multi; do
    MD5=$(avm_md5 "$TMP/${mode}_5.ivf")
    if [ -z "$MD5" ] || [ "$MD5" != "$BASE_AVM" ]; then
      echo "FAIL: $mode decoded '$MD5', expected identity '$BASE_AVM'" >&2
      exit 1
    fi
    echo "avmdec: $mode == base (identity matrix)"
  done

  for level in 0 5 14; do
    DEFAULT_AVM=$(avm_md5 "$TMP/default_$level.ivf")
    NO_OBU_AVM=$(avm_md5 "$TMP/no-obu_$level.ivf")
    if [ -z "$DEFAULT_AVM" ] || [ "$DEFAULT_AVM" = "$BASE_AVM" ]; then
      echo "FAIL: level $level default QM OBU did not change pixels" >&2
      exit 1
    fi
    if [ "$DEFAULT_AVM" != "$NO_OBU_AVM" ]; then
      echo "FAIL: level $level qm_is_default_flag=1 ($DEFAULT_AVM)" \
           "disagrees with the implicit default ($NO_OBU_AVM)" >&2
      exit 1
    fi
    echo "avmdec: level $level default OBU == no-OBU default, differs from base"
  done
else
  echo "avmdec: skipped (QM OBU gate needs it; dav2d rejects OBU type 22)"
fi

echo "qm obu gate: OK"
if [ "${KEEP:-0}" = "1" ]; then
  echo "streams: $TMP"
fi
