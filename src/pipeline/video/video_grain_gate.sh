#!/usr/bin/env bash
# Decoder gate for video film-grain signalling (--video-grain).
#
#   ./video_grain_gate.sh
#   AV2ENC_BIN=build/av2enc DAV2D=/path/to/dav2d ./video_grain_gate.sh
#   KEEP=1 ./video_grain_gate.sh
#
# What it proves:
#   1. `--video-grain` writes sequence-header film_grain_params_present = 1,
#      one §5.14 film grain OBU (type 23) before the coded-frame OBU of every
#      temporal unit, and the §5.18.10.1 per-frame apply_grain/fgm_id/seed.
#   2. Pinned dav2d --strict 1 decodes the stream, and `--filmgrain 0` output
#      equals the encoder's --video-recon dump exactly (grain is an output-side
#      synthesis, never part of the reference reconstruction).
#   3. `--filmgrain 1` decodes the same picture count and differs from the
#      pre-grain reconstruction, so the model is actually applied.
#   4. `--video` streams stay byte-identical when the flag is off.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "$0")/../../.." && pwd)"
TMP="${TMPDIR:-/tmp}/av2enc_video_grain_gate.$$"
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
  echo "video_grain_gate: no av2enc binary; set AV2ENC_BIN=..." >&2
  exit 127
fi

DAV2D="${DAV2D:-}"
if [ -z "$DAV2D" ]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$candidate" ]; then DAV2D="$candidate"; break; fi
  done
fi
if [ -z "$DAV2D" ] || [ ! -x "$DAV2D" ]; then
  echo "video_grain_gate: no dav2d binary (set DAV2D=...)" >&2
  exit 77
fi

INPUT="${AV2ENC_INPUT:-$ROOT/testdata/shine_1280x720_60f.y4m}"
if [ ! -f "$INPUT" ]; then
  echo "video_grain_gate: missing input $INPUT" >&2
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

# List every OBU per IVF frame as "frame:type:length".
list_obus() {
  python3 - "$1" <<'PY'
import struct, sys
data = open(sys.argv[1], "rb").read()
off = 32
frame = 0
while off + 12 <= len(data):
    size = struct.unpack_from("<I", data, off)[0]
    off += 12
    payload = data[off:off + size]
    off += size
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
        print(f"{frame}:{(obu[0] >> 2) & 0x1F}:{value}")
    frame += 1
PY
}

compare_raw() {
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

run_case() {
  local frames=$1 expected=$2
  local out="$TMP/grain${frames}.ivf"
  local recon="$TMP/grain${frames}.recon.y4m"
  local pre="$TMP/grain${frames}.pre.y4m"
  local post="$TMP/grain${frames}.post.y4m"
  if ! "$AV2ENC_BIN" "$INPUT" -o "$out" --frames "$frames" --video \
        --video-grain --video-recon "$recon" >"$TMP/grain${frames}.log" 2>&1; then
    fail_case "grain${frames}_encode" "encode failed"
    cat "$TMP/grain${frames}.log" >&2
    return
  fi
  local got; got=$(md5_of "$out")
  if [ "$got" != "$expected" ]; then
    fail_case "grain${frames}_pin" "IVF md5 $got, expected $expected"
  else
    pass_case "grain${frames}_pin (md5 $got)"
  fi
  # One film grain OBU per temporal unit, before the coded-frame OBU.
  local grain_obus frames_seen
  grain_obus=$(list_obus "$out" | grep -c ':23:' || true)
  frames_seen=$(list_obus "$out" | cut -d: -f1 | sort -u | wc -l | tr -d ' ')
  if [ "$grain_obus" -ne "$frames_seen" ] || [ "$frames_seen" -ne "$frames" ]; then
    fail_case "grain${frames}_obus" \
      "$grain_obus grain OBU(s) over $frames_seen frame unit(s), expected $frames"
  else
    pass_case "grain${frames}_obus ($frames per-unit type-23 OBUs)"
  fi
  if ! list_obus "$out" | python3 -c '
import collections, sys
pos = collections.defaultdict(list)
for line in sys.stdin:
    frame, kind, _ = line.strip().split(":")
    pos[int(frame)].append(int(kind))
bad = []
for frame, kinds in pos.items():
    grain = [i for i, k in enumerate(kinds) if k == 23]
    frames = [i for i, k in enumerate(kinds) if k in (4, 6, 7, 8, 10, 11, 12, 13, 14, 21)]
    if len(grain) != 1 or len(frames) != 1 or grain[0] != frames[0] - 1:
        bad.append((frame, kinds))
if bad:
    print(bad, file=sys.stderr)
    sys.exit(1)
'; then
    fail_case "grain${frames}_order" "film grain OBU is not immediately before the frame OBU"
  else
    pass_case "grain${frames}_order (film grain OBU precedes the frame OBU)"
  fi
  # Pre-grain decode must equal the encoder reconstruction.
  if ! "$DAV2D" -q -i "$out" -o "$pre" --threads 1 --strict 1 --filmgrain 0 \
        >"$TMP/grain${frames}.pre.log" 2>&1; then
    fail_case "grain${frames}_dav2d" "dav2d --filmgrain 0 rejected the stream"
    cat "$TMP/grain${frames}.pre.log" >&2
    return
  fi
  y4m_payload "$pre" >"$TMP/grain${frames}.pre.raw"
  if ! compare_raw "$recon" "$TMP/grain${frames}.pre.raw" recon \
        >>"$TMP/grain${frames}.cmp.log" 2>&1; then
    fail_case "grain${frames}_recon" "encoder recon differs from dav2d --filmgrain 0"
    cat "$TMP/grain${frames}.cmp.log" >&2
  else
    pass_case "grain${frames}_recon (dav2d --filmgrain 0 == --video-recon)"
  fi
  # Grain application: same picture count, different samples.
  if ! "$DAV2D" -q -i "$out" -o "$post" --threads 1 --strict 1 --filmgrain 1 \
        >"$TMP/grain${frames}.post.log" 2>&1; then
    fail_case "grain${frames}_apply" "dav2d --filmgrain 1 rejected the stream"
    cat "$TMP/grain${frames}.post.log" >&2
    return
  fi
  if ! python3 - "$pre" "$post" "$frames" <<'PY'
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
want = int(sys.argv[3])
frame = 1280 * 720 * 3 // 2
if len(a) != len(b):
    raise SystemExit("picture count or geometry changed with film grain")
if len(a) != frame * want:
    raise SystemExit(f"{len(a)} decoded bytes, expected {frame * want}")
same = all(x == y for x, y in zip(a, b))
if same:
    raise SystemExit("film grain application changed no samples")
print("grain applied")
PY
  then
    fail_case "grain${frames}_apply" "film grain synthesis was not observable"
  else
    pass_case "grain${frames}_apply (--filmgrain 1 differs, same picture count)"
  fi
}

# ------------------------------------------------------------------ cases ---
BASE="$TMP/base.ivf"
if ! "$AV2ENC_BIN" "$INPUT" -o "$BASE" --frames 4 --video \
      >"$TMP/base.log" 2>&1; then
  fail_case base "default encode failed"
else
  BASE_MD5=$(md5_of "$BASE")
  if [ "$BASE_MD5" != "e4cf72dd13ddc2c32c2f8683153b3c81" ]; then
    fail_case base "IVF md5 $BASE_MD5, expected e4cf72dd13ddc2c32c2f8683153b3c81"
  else
    pass_case "base (flag off, md5 unchanged)"
  fi
fi

run_case 4 e60ded0a500a856984cd1b575c89103b
run_case 8 e4c9dbd7e105160aea90c66947b69bbb

# Usage rejection: film grain is not applied by the tiled or GPU writers.
if "$AV2ENC_BIN" "$INPUT" -o "$TMP/reject.ivf" --frames 4 --video \
      --video-grain --tiles 2x1 >/dev/null 2>"$TMP/reject.err"; then
  fail_case tiles_reject "expected --video-grain --tiles to be rejected"
elif grep -q "does not compose with --tiles" "$TMP/reject.err"; then
  pass_case "tiles_reject (named rejection)"
else
  fail_case tiles_reject "rejection message changed"
  cat "$TMP/reject.err" >&2
fi

if [ "$failures" -ne 0 ]; then
  echo "video_grain_gate: $failures failed, $passes passed" >&2
  exit 1
fi
echo "video_grain_gate: $passes passed, 0 failed"
if [ "${KEEP:-0}" = "1" ]; then echo "streams: $TMP"; fi
