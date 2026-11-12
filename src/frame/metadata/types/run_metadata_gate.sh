#!/usr/bin/env bash
# Decoder gate for the AV2 v1.0.0 non-HDR metadata unit types.
#
# Builds the module and its unit test, obtains a valid M2 stream (the M2 gate
# synthesises a 64x64 two-frame P-frame clip), then for every implemented type:
#
#   1. appends it as prefix metadata (frame 0) and suffix metadata (frame 1);
#   2. rescans the stream with the module's self-parser and checks the type,
#      placement, flags and decoded values;
#   3. decodes with the pinned dav2d (--strict 1 --threads 1) and requires the
#      decoded Y4M to be byte-identical to the unmodified stream.
#
# It also covers one group OBU, all types at once, cancel units, a mixed
# group/prefix/suffix combination, and a corrupted-trailing_bits rejection.
#
# The pinned dav2d snapshot (f9dafbc, 2026-11-10) does not implement AV2
# §5.17.2/§5.17.3. It reads OBU type 8 as the pre-1.0 AV1-style metadata OBU
# and therefore logs "Unknown Metadata OBU type 32" (the AV2 options byte is
# interpreted as metadata_type) and ignores the OBU. For OBU type 9 it logs
# "Unknown OBU type 9 of size ...". The gate accepts those logs: the metadata
# is gated on decode-invariance plus self-parse, never on decoder support.
#
# Usage:
#   src/frame/metadata/types/run_metadata_gate.sh
#   src/frame/metadata/types/run_metadata_gate.sh --ivf /tmp/m2.ivf
#   src/frame/metadata/types/run_metadata_gate.sh --dav2d tools/bin/dav2d-macos
#
# Exit: 0 pass, 1 failure, 2 usage, 77 skip (no decoder / no M2 stream).
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
DAV2D="${DAV2D:-}"
IVF=""
FRAMES=2

while [ $# -gt 0 ]; do
  case "$1" in
    --ivf) IVF=$2; shift 2 ;;
    --dav2d) DAV2D=$2; shift 2 ;;
    --frames) FRAMES=$2; shift 2 ;;
    -h|--help)
      sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "metadata_gate: unknown argument $1" >&2; exit 2 ;;
  esac
done

if [ -z "$DAV2D" ]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
  done
fi
if [ -z "$DAV2D" ] || [ ! -x "$DAV2D" ]; then
  echo "SKIP: no dav2d binary (set DAV2D=...)" >&2
  exit 77
fi

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

failures=0
pass=0
DECODER_NOTE=""

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

echo "metadata_gate: decoder: $("$DAV2D" --version 2>&1 | tail -1)"

# --- build the module, unit test and tool (no CMake target by design) -------
if ! "$ROOT/src/frame/metadata/types/build.sh" "$TMP/build" \
    >"$TMP/build.log" 2>&1; then
  cat "$TMP/build.log" >&2
  echo "metadata_gate: build failed" >&2
  exit 1
fi
TOOL="$TMP/build/append_metadata_types"

if ! "$TMP/build/metadata_types_test" >"$TMP/unit.log" 2>&1; then
  cat "$TMP/unit.log" >&2
  fail "metadata_types_test"
else
  echo "PASS metadata_types_test ($(grep -c '^PASS ' "$TMP/unit.log") cases)"
  pass=$((pass + 1))
fi

# --- obtain a valid M2 stream ----------------------------------------------
BASE="$TMP/video_base.ivf"
if [ -n "$IVF" ]; then
  if [ ! -f "$IVF" ]; then
    echo "metadata_gate: --ivf not found: $IVF" >&2
    exit 2
  fi
  cp "$IVF" "$BASE"
else
  if ! OUT="$TMP/video_gate_bin" "$ROOT/src/pipeline/video/build_and_test.sh" \
      --frames "$FRAMES" --out "$BASE" >"$TMP/video_build.log" 2>&1; then
    cat "$TMP/video_build.log" >&2
    echo "SKIP: could not build/encode a valid M2 stream" >&2
    exit 77
  fi
fi
if [ ! -s "$BASE" ]; then
  echo "SKIP: M2 baseline stream is empty" >&2
  exit 77
fi

if ! "$DAV2D" -q -i "$BASE" -o "$TMP/base.y4m" --threads 1 --strict 1 \
    >"$TMP/base.dec.log" 2>&1; then
  cat "$TMP/base.dec.log" >&2
  fail "baseline M2 stream does not decode"
  echo "metadata_gate: cannot gate without a decodable baseline" >&2
  exit 1
fi
BASE_MD5=$(md5_of "$TMP/base.y4m")

# --- helpers ---------------------------------------------------------------

# ivf_check <base> <variant> <modified-frame> <name>
# Verifies the 32-byte file header and every frame timestamp are unchanged,
# and that only the selected frame payload grew.
ivf_check() {
  local base=$1 variant=$2 frame=$3 name=$4
  local result
  result=$(python3 - "$base" "$variant" "$frame" <<'PY'
import struct
import sys

def parse(path):
    data = open(path, 'rb').read()
    if data[:4] != b'DKIF':
        raise SystemExit('not an IVF file')
    header = data[:32]
    pos = 32
    sizes = []
    stamps = []
    while pos < len(data):
        size, stamp = struct.unpack_from('<IQ', data, pos)
        pos += 12
        if pos + size > len(data):
            raise SystemExit('truncated frame payload')
        sizes.append(size)
        stamps.append(stamp)
        pos += size
    return header, sizes, stamps

h1, s1, t1 = parse(sys.argv[1])
h2, s2, t2 = parse(sys.argv[2])
frame = int(sys.argv[3])
if h1 != h2:
    raise SystemExit('IVF file header changed')
if len(s1) != len(s2):
    raise SystemExit('frame count changed')
if t1 != t2:
    raise SystemExit('frame timestamps changed')
for index, (a, b) in enumerate(zip(s1, s2)):
    if index == frame:
        if b <= a:
            raise SystemExit('modified frame did not grow')
    elif a != b:
        raise SystemExit(f'frame {index} payload changed')
print(f'ok frame={frame} payload {s1[frame]}->{s2[frame]}')
PY
) || {
    fail "$name: $result"
    return
  }
  echo "  $result"
}

# decode_check <name> <ivf>
decode_check() {
  local name=$1 ivf=$2
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
  if [ -s "$TMP/$name.dec.log" ]; then
    local note
    note=$(sort -u "$TMP/$name.dec.log" | tr '\n' '|')
    if [ "$note" != "$DECODER_NOTE" ]; then
      echo "  decoder note: $note"
      DECODER_NOTE=$note
    fi
  fi
  echo "PASS $name (pixels unchanged, md5 $md5)"
  pass=$((pass + 1))
}

# scan_check <name> <ivf> <record-pattern> [<value-pattern>]
scan_check() {
  local name=$1 ivf=$2 record_pattern=$3 value_pattern=${4:-}
  local scan
  if ! scan=$("$TOOL" --input "$ivf" --scan --values \
      2>"$TMP/$name.scan.err"); then
    cat "$TMP/$name.scan.err" >&2
    fail "$name: self-scan failed"
    return
  fi
  if ! grep -q "$record_pattern" <<<"$scan"; then
    printf '%s\n' "$scan" >&2
    fail "$name: scan did not match: $record_pattern"
    return
  fi
  if [ -n "$value_pattern" ] && ! grep -q "$value_pattern" <<<"$scan"; then
    printf '%s\n' "$scan" >&2
    fail "$name: decoded value did not match: $value_pattern"
    return
  fi
  if grep -q "invalid_units=[1-9]" <<<"$scan"; then
    printf '%s\n' "$scan" >&2
    fail "$name: scan reported invalid units"
  fi
}

# add_metadata <name> [tool options...]
add_metadata() {
  local name=$1
  shift
  if ! "$TOOL" --input "$BASE" --output "$TMP/$name.ivf" "$@" \
      >"$TMP/$name.append.log" 2>&1; then
    cat "$TMP/$name.append.log" >&2
    fail "$name: append failed"
    return 1
  fi
  return 0
}

record_name() {
  case "$1" in
    frame_hash) echo "decoded_frame_hash" ;;
    banding) echo "banding_hints" ;;
    icc) echo "icc_profile" ;;
    temporal_point) echo "temporal_point_info" ;;
    user_data) echo "user_data_unregistered" ;;
    *) echo "$1" ;;
  esac
}

expected_value() {
  case "$1" in
    itut_t35)
      echo "country_code=181 extension_byte=0 payload_size=5 payload=003c000104" ;;
    timecode)
      echo "counting_type=0 full=1 discontinuity=0 cnt_dropped=0 n_frames=42 seconds_flag=1 seconds=7 minutes_flag=1 minutes=3 hours_flag=1 hours=1 offset_length=0 offset=0" ;;
    frame_hash)
      echo "hash_type=0 per_plane=1 has_grain=0 is_monochrome=0 hash=000102030405060708090a0b0c0d0e0f" ;;
    banding)
      echo "coding=1 source=0 hints=1 three_color=0 band_units=1 rows=2 cols=3 varying=1 band_block=1 units=6" ;;
    icc)
      echo "size=16 icc_major=4 icc_minor=3" ;;
    scan_type)
      echo "pic_struct_type=1 source_scan_type_idc=2 duplicate=0" ;;
    temporal_point)
      echo "frame_presentation_time=1000" ;;
    user_data)
      echo "uuid=000102030405060708090a0b0c0d0e0f payload_size=4 payload=deadbeef" ;;
    *) echo "" ;;
  esac
}

# --- per-type prefix and suffix --------------------------------------------
TYPES=(itut_t35 timecode frame_hash banding icc scan_type temporal_point user_data)

for type in "${TYPES[@]}"; do
  value=$(expected_value "$type")
  record=$(record_name "$type")

  name="type_${type}_prefix"
  if add_metadata "$name" --type "$type"; then
    ivf_check "$BASE" "$TMP/$name.ivf" 0 "$name"
    scan_check "$name" "$TMP/$name.ivf" \
      "record=$record suffix=0 group=0 before_frame=1 cancel=0 .*layer=current persistence=global priority=0" \
      "$value"
    decode_check "$name" "$TMP/$name.ivf"
  fi

  name="type_${type}_suffix"
  if add_metadata "$name" --type "$type" --suffix --frame 1; then
    ivf_check "$BASE" "$TMP/$name.ivf" 1 "$name"
    scan_check "$name" "$TMP/$name.ivf" \
      "record=$record suffix=1 group=0 before_frame=0 cancel=0 .*layer=current persistence=global priority=0" \
      "$value"
    decode_check "$name" "$TMP/$name.ivf"
  fi
done

# --- persistence / layer variants (one representative type each) -----------
name="persistence_layers"
if add_metadata "$name" --type timecode --layer global --persistence basic; then
  ivf_check "$BASE" "$TMP/$name.ivf" 0 "$name"
  scan_check "$name" "$TMP/$name.ivf" \
    "record=timecode suffix=0 group=0 before_frame=1 cancel=0 .*layer=global persistence=basic"
  decode_check "$name" "$TMP/$name.ivf"
fi

# --- group OBU prefix and suffix -------------------------------------------
GROUP_TYPES="itut_t35,timecode,frame_hash,banding,icc,scan_type,user_data"
name="group_prefix"
if add_metadata "$name" --group --type "$GROUP_TYPES"; then
  ivf_check "$BASE" "$TMP/$name.ivf" 0 "$name"
  scan_check "$name" "$TMP/$name.ivf" \
    "metadata_obus=1 metadata_units=7 supported_units=7 unsupported_units=0 invalid_units=0"
  scan_check "$name" "$TMP/$name.ivf" \
    "record=decoded_frame_hash suffix=0 group=1 before_frame=1 cancel=0 .*priority=0" \
    "$(expected_value frame_hash)"
  decode_check "$name" "$TMP/$name.ivf"
fi

name="group_suffix"
if add_metadata "$name" --group --suffix --frame 1 --type "$GROUP_TYPES"; then
  ivf_check "$BASE" "$TMP/$name.ivf" 1 "$name"
  scan_check "$name" "$TMP/$name.ivf" \
    "metadata_obus=1 metadata_units=7 supported_units=7 unsupported_units=0 invalid_units=0"
  scan_check "$name" "$TMP/$name.ivf" \
    "record=itut_t35 suffix=1 group=1 before_frame=0 cancel=0" \
    "$(expected_value itut_t35)"
  decode_check "$name" "$TMP/$name.ivf"
fi

# --- all eight types in one stream, prefix and suffix ----------------------
name="all_prefix"
if add_metadata "$name"; then
  ivf_check "$BASE" "$TMP/$name.ivf" 0 "$name"
  scan_check "$name" "$TMP/$name.ivf" \
    "metadata_obus=8 metadata_units=8 supported_units=8 unsupported_units=0 invalid_units=0"
  decode_check "$name" "$TMP/$name.ivf"
fi

name="all_suffix"
if add_metadata "$name" --suffix --frame 1; then
  ivf_check "$BASE" "$TMP/$name.ivf" 1 "$name"
  scan_check "$name" "$TMP/$name.ivf" \
    "metadata_obus=8 metadata_units=8 supported_units=8 unsupported_units=0 invalid_units=0"
  decode_check "$name" "$TMP/$name.ivf"
fi

# --- group with a temporal-point-info entry must be refused ----------------
if "$TOOL" --input "$BASE" --output "$TMP/group_tpi.ivf" --group \
    >"$TMP/group_tpi.log" 2>&1; then
  fail "group_tpi: temporal point info was accepted in a group OBU"
else
  if ! grep -q "TEMPORAL_POINT_INFO" "$TMP/group_tpi.log"; then
    cat "$TMP/group_tpi.log" >&2
    fail "group_tpi: unexpected rejection reason"
  else
    echo "PASS group_tpi (group-carried temporal point info rejected)"
    pass=$((pass + 1))
  fi
fi

# --- cancel units ----------------------------------------------------------
name="cancel_prefix"
if add_metadata "$name" --cancel; then
  ivf_check "$BASE" "$TMP/$name.ivf" 0 "$name"
  scan_check "$name" "$TMP/$name.ivf" \
    "metadata_obus=8 metadata_units=8 supported_units=0 unsupported_units=0 invalid_units=0 cancelled_units=8"
  decode_check "$name" "$TMP/$name.ivf"
fi

name="group_cancel_prefix"
if add_metadata "$name" --group --cancel --type "$GROUP_TYPES"; then
  ivf_check "$BASE" "$TMP/$name.ivf" 0 "$name"
  scan_check "$name" "$TMP/$name.ivf" \
    "metadata_obus=1 metadata_units=7 supported_units=0 unsupported_units=0 invalid_units=0 cancelled_units=7"
  decode_check "$name" "$TMP/$name.ivf"
fi

# --- malformed self-parse rejection ---------------------------------------
# Flip the trailing_bits byte of the first OBU_METADATA_SHORT in frame 0.
cp "$TMP/type_itut_t35_prefix.ivf" "$TMP/corrupt.ivf" 2>/dev/null || true
if [ -f "$TMP/corrupt.ivf" ]; then
  if python3 - "$TMP/corrupt.ivf" <<'PY'
import struct
import sys

path = sys.argv[1]
data = bytearray(open(path, 'rb').read())
size = struct.unpack_from('<I', data, 32)[0]
start = 44
payload = data[start:start + size]
pos = 0
while pos < len(payload):
    obu_start = pos
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
        payload[end - 1] ^= 0x80  # trailing_bits must be 0x80
        data[start:start + size] = payload
        open(path, 'wb').write(data)
        raise SystemExit(0)
    pos = end
    if pos <= obu_start:
        break
raise SystemExit('no metadata OBU found')
PY
  then
    if "$TOOL" --input "$TMP/corrupt.ivf" --scan >"$TMP/corrupt.scan" 2>&1; then
      fail "corrupt_trailing_bits: scanner accepted invalid metadata"
    else
      echo "PASS corrupt_trailing_bits (self-parser rejected invalid trailing_bits)"
      pass=$((pass + 1))
    fi
  else
    fail "corrupt_trailing_bits: could not patch the stream"
  fi
else
  fail "corrupt_trailing_bits: prefix variant missing"
fi

echo
if [ "$failures" -ne 0 ]; then
  echo "metadata_gate: $failures failure(s), $pass pass(es)" >&2
  exit 1
fi
echo "metadata_gate: PASS ($pass checks, baseline md5 $BASE_MD5)"
echo "metadata_gate: dav2d ignores AV2 metadata OBUs; gated on decode-invariance + self-parse"
