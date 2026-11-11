#!/usr/bin/env bash
# Decoder gate for the AV2 v1.0.0 metadata gaps in this directory:
#   * metadata cancel units (metadata_type signalled, no body);
#   * muh_layer_idc == LAYER_VALUES with muh_xlayer_map f(32) +
#     muh_mlayer_map f(8), in both the local-xlayer and GLOBAL_XLAYER_ID forms;
#   * muh_header_extension_byte f(8).
#
# The gate builds the module and its unit tests, obtains a valid M2 stream
# (the M2 gate synthesises a 64x64 two-frame P-frame clip), inserts the gap
# metadata into frame 0 (prefix) and frame 1 (suffix), self-parses every
# variant, and requires the pinned dav2d decode to be byte-identical to the
# unmodified stream.
#
# The pinned dav2d snapshot (f9dafbc, 2026-11-10) does not implement AV2
# §5.17.2/§5.17.3. It logs "Unknown Metadata OBU type ..."/"Unknown OBU type 9"
# and skips the OBU by its length, which is exactly why decode-invariance plus
# self-parse is the gate, not decoder-side metadata validation.
#
# Usage:
#   src/encoder/metadata/gaps/run_gap_gate.sh
#   src/encoder/metadata/gaps/run_gap_gate.sh --ivf /tmp/m2.ivf
#   src/encoder/metadata/gaps/run_gap_gate.sh --dav2d tools/bin/dav2d-macos
#
# Exit: 0 pass, 1 failure, 2 usage, 77 skip (no decoder / no M2 stream).
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
DAV2D="${DAV2D:-}"
IVF=""
FRAMES=2

while [ $# -gt 0 ]; do
  case "$1" in
    --ivf) IVF=$2; shift 2 ;;
    --dav2d) DAV2D=$2; shift 2 ;;
    --frames) FRAMES=$2; shift 2 ;;
    -h|--help)
      sed -n '2,27p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "gap_gate: unknown argument $1" >&2; exit 2 ;;
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

echo "gap_gate: decoder: $("$DAV2D" --version 2>&1 | tail -1)"

# --- build the module, unit tests and gate tool ----------------------------
if ! "$ROOT/src/encoder/metadata/gaps/build.sh" "$TMP/build" \
    >"$TMP/build.log" 2>&1; then
  cat "$TMP/build.log" >&2
  echo "gap_gate: build failed" >&2
  exit 1
fi

if ! "$TMP/build/metadata_gaps_test" >"$TMP/md.log" 2>&1; then
  cat "$TMP/md.log" >&2
  fail "metadata_gaps_test"
else
  echo "PASS metadata_gaps_test ($(grep -c '^PASS ' "$TMP/md.log") cases)"
  pass=$((pass + 1))
fi

if ! "$TMP/build/reference_state_test" >"$TMP/rs.log" 2>&1; then
  cat "$TMP/rs.log" >&2
  fail "reference_state_test"
else
  echo "PASS reference_state_test ($(grep -c '^PASS ' "$TMP/rs.log") cases)"
  pass=$((pass + 1))
fi

# --- obtain a valid M2 stream ----------------------------------------------
BASE="$TMP/m2_base.ivf"
if [ -n "$IVF" ]; then
  if [ ! -f "$IVF" ]; then
    echo "gap_gate: --ivf not found: $IVF" >&2
    exit 2
  fi
  cp "$IVF" "$BASE"
else
  if ! OUT="$TMP/m2_gate_bin" "$ROOT/src/encoder/m2/build_and_test.sh" \
      --frames "$FRAMES" --out "$BASE" >"$TMP/m2_build.log" 2>&1; then
    cat "$TMP/m2_build.log" >&2
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
  echo "gap_gate: cannot gate without a decodable baseline" >&2
  exit 1
fi
BASE_MD5=$(md5_of "$TMP/base.y4m")
echo "gap_gate: baseline md5 $BASE_MD5"

# --- variants --------------------------------------------------------------
if ! "$TMP/build/gap_stream_gate" --base "$BASE" --out-dir "$TMP/variants" \
    >"$TMP/variants.log" 2>&1; then
  cat "$TMP/variants.log" >&2
  fail "gap_stream_gate self-parse checks"
  echo "gap_gate: FAIL" >&2
  exit 1
fi
cat "$TMP/variants.log"
pass=$((pass + 1))

# ivf_check <base> <variant> <modified-frame> <name>
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

# variant_check <name> <modified-frame>
variant_check() {
  local name=$1 frame=$2
  local ivf="$TMP/variants/$name.ivf"
  if [ ! -s "$ivf" ]; then
    fail "$name: variant was not produced"
    return
  fi
  ivf_check "$BASE" "$ivf" "$frame" "$name"
  decode_check "$name" "$ivf"
}

variant_check cancel_prefix 0
variant_check layer_local_prefix 0
variant_check layer_global_prefix 0
variant_check layer_values_suffix 1

# The tool already proves the corrupt variant is rejected by the self-parser;
# assert that the file exists so the check cannot silently disappear.
if [ ! -s "$TMP/variants/corrupt_trailing.ivf" ]; then
  fail "corrupt_trailing: rejection variant missing"
else
  echo "PASS corrupt_trailing (self-parser rejection exercised)"
  pass=$((pass + 1))
fi

echo
if [ "$failures" -ne 0 ]; then
  echo "gap_gate: $failures failure(s), $pass pass(es)" >&2
  exit 1
fi
echo "gap_gate: PASS ($pass checks)"
