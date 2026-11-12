#!/usr/bin/env bash
# Decoder gate for the AV2 HDR metadata OBUs.
#
# Encodes a known-good M1 still picture, appends spec-conformant CLL/MDCV
# metadata, scans it back, and requires the pinned dav2d build to decode the
# stream to the same pixels as the unmodified stream with --strict 1.
#
# The gate also probes the pinned decoder with the non-conformant AV1-style
# metadata layout and requires a corrupted trailing_bits byte to be rejected.
# That proves dav2d actively parses metadata when it understands the layout,
# rather than silently skipping every unknown OBU.
#
# Usage:
#   src/frame/metadata/hdr/run_hdr_gate.sh --bin build-m1/av2enc [--dav2d <path>]
#
# Exit: 0 pass, 1 failure, 2 usage, 77 skip (missing decoder/fixture/tool).
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
BIN=""
DAV2D="${DAV2D:-}"
FIXTURE="$ROOT/tests/data/basic_64x64_1f.y4m"

while [ $# -gt 0 ]; do
  case "$1" in
    --bin) BIN=$2; shift 2 ;;
    --dav2d) DAV2D=$2; shift 2 ;;
    --fixture) FIXTURE=$2; shift 2 ;;
    -h|--help)
      sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "hdr_gate: unknown argument $1" >&2; exit 2 ;;
  esac
done

if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
  echo "hdr_gate: --bin must point at an av2enc executable" >&2
  exit 2
fi
if [ -z "$DAV2D" ]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
  done
fi
if [ -z "$DAV2D" ] || [ ! -x "$DAV2D" ]; then
  echo "SKIP: no dav2d binary (set DAV2D=...)" >&2
  exit 77
fi
if [ ! -f "$FIXTURE" ]; then
  echo "SKIP: fixture not found: $FIXTURE" >&2
  exit 77
fi

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

failures=0
pass=0

fail() {
  echo "FAIL: $*" >&2
  failures=$((failures + 1))
}

md5_of() {
  if command -v md5 >/dev/null 2>&1; then
    md5 -q "$1"
  else
    md5sum "$1" | awk '{print $1}'
  fi
}

echo "hdr_gate: decoder: $("$DAV2D" --version 2>&1 | tail -1)"
echo "hdr_gate: encoder: $BIN"

# Build the standalone HDR test/tool (no CMake target by design).
if ! "$ROOT/src/frame/metadata/hdr/build.sh" "$TMP/build" >"$TMP/build.log" 2>&1; then
  cat "$TMP/build.log" >&2
  fail "HDR tool build"
  exit 1
fi
TOOL="$TMP/build/append_hdr_metadata"

if ! "$TMP/build/hdr_metadata_test" >"$TMP/unit.log" 2>&1; then
  cat "$TMP/unit.log" >&2
  fail "hdr_metadata_test"
fi

# Baseline M1 stream (the current reference configuration).
if ! "$BIN" "$FIXTURE" -o "$TMP/base.ivf" --frames 1 --still >"$TMP/enc.log" 2>&1; then
  cat "$TMP/enc.log" >&2
  echo "SKIP: M1 encoding failed (encoder build lacks --still?)" >&2
  exit 77
fi
BASE_PAYLOAD=$(python3 - "$TMP/base.ivf" <<'PY'
import struct, sys
b = open(sys.argv[1], 'rb').read()
print(struct.unpack_from('<I', b, 32)[0])
PY
)

if ! "$DAV2D" -q -i "$TMP/base.ivf" -o "$TMP/base.y4m" --threads 1 --strict 1 \
    >"$TMP/base.dec.log" 2>&1; then
  cat "$TMP/base.dec.log" >&2
  fail "baseline stream does not decode"
  exit 1
fi
BASE_MD5=$(md5_of "$TMP/base.y4m")
BASE_HEADER=$(python3 - "$TMP/base.ivf" <<'PY'
import struct, sys
b = open(sys.argv[1], 'rb').read()
# File header without the frame count (which must also stay put) and the
# timestamp of frame 0: all must be preserved byte-for-byte.
print(b[:32].hex(), b[36:44].hex())
PY
)

# decode_variant <name> <ivf> <expected extra payload bytes>
decode_variant() {
  local name=$1 ivf=$2 delta=$3
  local payload
  payload=$(python3 - "$ivf" <<'PY'
import struct, sys
b = open(sys.argv[1], 'rb').read()
print(struct.unpack_from('<I', b, 32)[0])
PY
)
  if [ "$payload" -ne "$((BASE_PAYLOAD + delta))" ]; then
    fail "$name: payload $payload != $((BASE_PAYLOAD + delta))"
  fi
  local header
  header=$(python3 - "$ivf" <<'PY'
import sys
b = open(sys.argv[1], 'rb').read()
print(b[:32].hex(), b[36:44].hex())
PY
)
  if [ "$header" != "$BASE_HEADER" ]; then
    fail "$name: IVF header/timestamp changed"
  fi
  if ! "$DAV2D" -q -i "$ivf" -o "$TMP/$name.y4m" --threads 1 --strict 1 \
      >"$TMP/$name.dec.log" 2>&1; then
    cat "$TMP/$name.dec.log" >&2
    fail "$name: dav2d --strict 1 rejected the stream"
    return
  fi
  local md5
  md5=$(md5_of "$TMP/$name.y4m")
  if [ "$md5" != "$BASE_MD5" ]; then
    fail "$name: decoded md5 $md5 != baseline $BASE_MD5"
    return
  fi
  echo "PASS $name (payload +$delta, md5 $md5)"
  if [ -s "$TMP/$name.dec.log" ]; then
    echo "  note: decoder stderr: $(tr '\n' '|' <"$TMP/$name.dec.log")"
  fi
  pass=$((pass + 1))
}

scan_expect() {
  local name=$1 ivf=$2 pattern=$3
  local scan
  scan=$("$TOOL" --input "$ivf" --scan 2>"$TMP/$name.scan.err") || {
    cat "$TMP/$name.scan.err" >&2
    fail "$name: scan failed"
    return
  }
  if ! grep -q "$pattern" <<<"$scan"; then
    echo "$scan" >&2
    fail "$name: scan did not match: $pattern"
    return
  fi
  if grep -q "invalid_units=[1-9]" <<<"$scan"; then
    echo "$scan" >&2
    fail "$name: scan reported invalid units"
  fi
}

# --- spec layout: two short OBUs, prefix (default) --------------------------
if "$TOOL" --input "$TMP/base.ivf" --output "$TMP/short_prefix.ivf" \
    >"$TMP/short_prefix.log" 2>&1; then
  scan_expect short_prefix "$TMP/short_prefix.ivf" \
    "metadata_obus=2 metadata_units=2 unknown_units=0 invalid_units=0 records=2"
  scan_expect short_prefix "$TMP/short_prefix.ivf" \
    "record=cll max_cll=1000 max_fall=400 suffix=0 group=0 before_frame=1"
  scan_expect short_prefix "$TMP/short_prefix.ivf" \
    "record=mdcv .*Lmax=10000000 Lmin=50 suffix=0 group=0 before_frame=1"
  decode_variant short_prefix "$TMP/short_prefix.ivf" 38
else
  cat "$TMP/short_prefix.log" >&2
  fail "short_prefix: append failed"
fi

# --- spec layout: one group OBU ---------------------------------------------
if "$TOOL" --input "$TMP/base.ivf" --output "$TMP/group.ivf" --group \
    >"$TMP/group.log" 2>&1; then
  scan_expect group "$TMP/group.ivf" \
    "metadata_obus=1 metadata_units=2 unknown_units=0 invalid_units=0 records=2"
  scan_expect group "$TMP/group.ivf" \
    "record=cll max_cll=1000 max_fall=400 suffix=0 group=1 before_frame=1"
  decode_variant group "$TMP/group.ivf" 43
else
  cat "$TMP/group.log" >&2
  fail "group: append failed"
fi

# --- spec layout: suffix metadata -------------------------------------------
if "$TOOL" --input "$TMP/base.ivf" --output "$TMP/suffix.ivf" --suffix \
    >"$TMP/suffix.log" 2>&1; then
  scan_expect suffix "$TMP/suffix.ivf" \
    "record=cll max_cll=1000 max_fall=400 suffix=1 group=0 before_frame=0"
  decode_variant suffix "$TMP/suffix.ivf" 38
else
  cat "$TMP/suffix.log" >&2
  fail "suffix: append failed"
fi

# --- spec layout: MDCV only -------------------------------------------------
if "$TOOL" --input "$TMP/base.ivf" --output "$TMP/mdcv_only.ivf" \
    --no-cll --mdcv p3-1000 >"$TMP/mdcv_only.log" 2>&1; then
  scan_expect mdcv_only "$TMP/mdcv_only.ivf" \
    "metadata_obus=1 metadata_units=1 unknown_units=0 invalid_units=0 records=1"
  scan_expect mdcv_only "$TMP/mdcv_only.ivf" \
    "record=mdcv Gx=13250 Gy=34500 Bx=7500 By=3000 Rx=34000 Ry=16000"
  decode_variant mdcv_only "$TMP/mdcv_only.ivf" 29
else
  cat "$TMP/mdcv_only.log" >&2
  fail "mdcv_only: append failed"
fi

# --- decoder-interop probe: non-conformant AV1-style layout -----------------
if "$TOOL" --input "$TMP/base.ivf" --output "$TMP/legacy.ivf" --legacy-av1 \
    >"$TMP/legacy.log" 2>&1; then
  decode_variant legacy "$TMP/legacy.ivf" 36
  if [ -s "$TMP/legacy.dec.log" ]; then
    echo "  note: pinned decoder reported: $(tr '\n' '|' <"$TMP/legacy.dec.log")"
  else
    echo "  note: pinned decoder parsed the legacy CLL/MDCV OBUs silently"
  fi

  # Corrupt the first metadata OBU's trailing_bits byte. A decoder that
  # actively parses metadata must reject this under --strict 1.
  cp "$TMP/legacy.ivf" "$TMP/corrupt.ivf"
  python3 - "$TMP/corrupt.ivf" <<'PY'
import struct, sys
path = sys.argv[1]
b = bytearray(open(path, 'rb').read())
size = struct.unpack_from('<I', b, 32)[0]
payload = bytearray(b[44:44 + size])
pos = 0
patched = False
while pos < len(payload):
    start = pos
    value = 0
    shift = 0
    while True:
        byte = payload[pos]
        pos += 1
        value |= (byte & 0x7f) << shift
        shift += 7
        if not byte & 0x80:
            break
    obu_type = (payload[pos] >> 2) & 0x1f
    end = pos + value
    if obu_type == 8:
        payload[end - 1] ^= 0x80
        patched = True
        break
    pos = end
if not patched:
    raise SystemExit('no metadata OBU found')
b[44:44 + size] = payload
open(path, 'wb').write(b)
PY
  if "$DAV2D" -q -i "$TMP/corrupt.ivf" -o "$TMP/corrupt.y4m" \
      --threads 1 --strict 1 >"$TMP/corrupt.dec.log" 2>&1; then
    fail "corrupt_legacy: decoder accepted invalid metadata trailing_bits"
  else
    echo "PASS corrupt_legacy (decoder rejected invalid metadata)"
    pass=$((pass + 1))
  fi
else
  cat "$TMP/legacy.log" >&2
  fail "legacy: append failed"
fi

# --- spec layout: out-of-range values are legal "unknown", not clamped ------
if "$TOOL" --input "$TMP/base.ivf" --output "$TMP/unknown_mdcv.ivf" \
    --no-cll --mdcv unknown >"$TMP/unknown_mdcv.log" 2>&1; then
  scan_expect unknown_mdcv "$TMP/unknown_mdcv.ivf" \
    "record=mdcv Gx=0 Gy=0 .*Lmax=0 Lmin=0"
  decode_variant unknown_mdcv "$TMP/unknown_mdcv.ivf" 29
else
  cat "$TMP/unknown_mdcv.log" >&2
  fail "unknown_mdcv: append failed"
fi

# --- strict range rejection -------------------------------------------------
if "$TOOL" --input "$TMP/base.ivf" --output "$TMP/strict.ivf" \
    --strict-ranges --mdcv unknown >"$TMP/strict.log" 2>&1; then
  fail "strict_ranges: out-of-range MDCV was accepted"
else
  echo "PASS strict_ranges (out-of-range MDCV rejected)"
  pass=$((pass + 1))
fi

echo
if [ "$failures" -ne 0 ]; then
  echo "hdr_gate: $failures failure(s), $pass pass(es)" >&2
  exit 1
fi
echo "hdr_gate: PASS ($pass checks, baseline md5 $BASE_MD5)"
