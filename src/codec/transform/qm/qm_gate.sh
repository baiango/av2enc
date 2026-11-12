#!/usr/bin/env bash
# Decoder gate for AV2 quantizer matrices.
#
#   ./qm_gate.sh
#   AV2ENC_BIN=/path/to/av2enc AVM_DEC=/path/to/avmdec ./qm_gate.sh
#   KEEP=1 ./qm_gate.sh          # keep the generated streams
#
# What it proves:
#   1. dav2d accepts using_qmatrix = 1 with the normative default matrices
#      (no QM OBU; the pinned dav2d rejects all QM OBUs).
#   2. avmdec accepts a §5.13 QM OBU with user-defined matrices for levels
#      0/5/14.  An all-32 user matrix is the identity (q2 = Round2(q*32,5) = q),
#      so decoded pixels must match the base stream exactly.
#   3. The same levels signalled as qm_is_default_flag = 1 decode successfully
#      and change pixels, proving the QM scaling is actually applied.
#
# An 8x8 picture keeps every transform at 8x8 or 4x4, which is the only range
# where user QM data is used (§7.14.4 useUserQm).  For larger transforms the
# default Quantizer_Matrix tables always apply.
set -euo pipefail
cd "$(dirname "$0")"

ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
TMP="${TMPDIR:-/tmp}/av2enc_qm_gate.$$"
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
  echo "qm_gate: no av2enc binary; set AV2ENC_BIN=..." >&2
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

QM_GATE="${QM_GATE_BIN:-$TMP/qm_gate}"
if [ ! -x "$QM_GATE" ]; then
  "$CXX" -std=c++20 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow \
    -I"$ROOT/src" \
    qm_gate.cpp qm.cpp \
    "$ROOT/src/core/bitstream/bitwriter.cpp" \
    "$ROOT/src/core/bitstream/obu.cpp" \
    "$ROOT/src/core/bitstream/basic_headers.cpp" \
    "$ROOT/src/pipeline/still/still_headers.cpp" \
    -o "$QM_GATE"
fi

# Deterministic 8x8 4:2:0 input.
SRC="$TMP/qm_8x8.y4m"
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
"$AV2ENC_BIN" "$SRC" -o "$BASE" --still --qp 90 >/dev/null

dav2d_md5() {
  "$DAV2D" -q -i "$1" --muxer md5 -o - 2>/dev/null | tail -1
}
avm_md5() {
  "$AVM_DEC" --codec=av2 --md5 --limit=1 "$1" 2>/dev/null |
    tail -1 | awk '{print $1}'
}

if [ -n "$DAV2D" ]; then
  BASE_MD5=$(dav2d_md5 "$BASE")
  "$QM_GATE" "$BASE" "$TMP/default_no_obu.ivf" \
    --width 8 --height 8 --qp 90 --level 0 --default --no-qm-obu >/dev/null
  DEFAULT_NO_OBU_MD5=$(dav2d_md5 "$TMP/default_no_obu.ivf")
  if [ -z "$DEFAULT_NO_OBU_MD5" ]; then
    echo "FAIL: dav2d did not decode using_qmatrix = 1" >&2
    exit 1
  fi
  if [ "$DEFAULT_NO_OBU_MD5" = "$BASE_MD5" ]; then
    echo "FAIL: default QM did not change pixels" >&2
    exit 1
  fi
  echo "dav2d: using_qmatrix default matrices OK ($DEFAULT_NO_OBU_MD5 != $BASE_MD5)"

  # Larger blocks: 64x64 exercises the default-table (not user-QM) branch.
  SRC64="$ROOT/testdata/basic_64x64_1f.y4m"
  if [ -f "$SRC64" ]; then
    BASE64="$TMP/base64.ivf"
    "$AV2ENC_BIN" "$SRC64" -o "$BASE64" --still --qp 90 >/dev/null
    "$QM_GATE" "$BASE64" "$TMP/default64.ivf" \
      --width 64 --height 64 --qp 90 --level 0 --default --no-qm-obu >/dev/null
    BASE64_MD5=$(dav2d_md5 "$BASE64")
    DEF64_MD5=$(dav2d_md5 "$TMP/default64.ivf")
    if [ -z "$DEF64_MD5" ] || [ "$DEF64_MD5" = "$BASE64_MD5" ]; then
      echo "FAIL: 64x64 default QM did not decode/change pixels" >&2
      exit 1
    fi
    echo "dav2d: 64x64 default matrices OK ($DEF64_MD5 != $BASE64_MD5)"
  fi
else
  echo "dav2d: skipped (no decoder)"
fi

if [ -n "$AVM_DEC" ]; then
  BASE_MD5_AVM=$(avm_md5 "$BASE")
  for level in 0 5 14; do
    "$QM_GATE" "$BASE" "$TMP/id_$level.ivf" \
      --width 8 --height 8 --qp 90 --level "$level" >/dev/null
    ID_MD5=$(avm_md5 "$TMP/id_$level.ivf")
    if [ -z "$ID_MD5" ] || [ "$ID_MD5" != "$BASE_MD5_AVM" ]; then
      echo "FAIL: level $level user identity QM: got '$ID_MD5'," \
           "expected '$BASE_MD5_AVM'" >&2
      exit 1
    fi
    "$QM_GATE" "$BASE" "$TMP/def_$level.ivf" \
      --width 8 --height 8 --qp 90 --level "$level" --default >/dev/null
    DEF_MD5=$(avm_md5 "$TMP/def_$level.ivf")
    if [ -z "$DEF_MD5" ] || [ "$DEF_MD5" = "$BASE_MD5_AVM" ]; then
      echo "FAIL: level $level default QM did not change pixels" >&2
      exit 1
    fi
    echo "avmdec: level $level user identity == base; default QM differs"
  done
else
  echo "avmdec: skipped (user-defined QM OBU gate needs it; dav2d rejects QM OBUs)"
fi

echo "qm gate: OK"
if [ "${KEEP:-0}" = "1" ]; then
  echo "streams: $TMP"
fi
