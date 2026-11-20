#!/usr/bin/env bash
# Gate for the CCSO parameter-traffic model (--video-ccso-header-rd).
#
# Checks, on the 60-frame 240p30 nightly point (QP110):
#   * the flag-off pin is byte-identical (24,885 B / 1b99127a...);
#   * the flag-on stream shrinks at the same QP and still decodes under
#     dav2d --strict 1 with 60/60 pictures, and the visible encoder recon equals
#     the decoded payload;
#   * AV2ENC_TRACE_CCSO shows the §5.18.7.12 row itself shrinking: the traced
#     ccso_params() bits per frame drop by at least 25% (measured 4,798 vs
#     8,441 bits over 60 frames), and the band count is no longer pinned at 64;
#   * the named rejections: --video-gpu, --still and a non-4:2:0/8-bit core.
#
# Usage: tools/gate_video_ccso_overhead.sh [--bin build/av2enc] [--dav2d PATH]
# Exit: 0 pass, 1 failure, 2 usage, 77 missing fixture/decoder.
set -uo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/.." && pwd)
BIN=""
DAV2D="${DAV2D:-}"
TESTDATA="${AV2ENC_TESTDATA:-$ROOT/testdata}"

while [ $# -gt 0 ]; do
    case "$1" in
        --bin) BIN=${2:-}; shift 2 ;;
        --dav2d) DAV2D=${2:-}; shift 2 ;;
        -h|--help) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "gate_video_ccso_overhead: unknown argument $1" >&2; exit 2 ;;
    esac
done

BIN=${BIN:-"$ROOT/build/av2enc"}
[ -x "$BIN" ] || { echo "gate_video_ccso_overhead: --bin must point at av2enc" >&2; exit 2; }
if [ -z "$DAV2D" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
    done
fi
[ -n "$DAV2D" ] && [ -x "$DAV2D" ] || {
    echo "gate_video_ccso_overhead: SKIP (exit 77): no dav2d binary" >&2; exit 77; }

NASA="$TESTDATA/nasa_480x240_30fps_60f.y4m"
PUB_CLIP="$TESTDATA/pub_640x360_4f.y4m"
[ -f "$PUB_CLIP" ] || PUB_CLIP="$TESTDATA/pub_640x360_1f.y4m"
[ -f "$NASA" ] || {
    echo "gate_video_ccso_overhead: SKIP (exit 77): no 240p60 fixture" >&2
    exit 77; }

TMP=$(mktemp -d "${TMPDIR:-/tmp}/av2enc-ccso.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
FAILED=0
fail() { echo "FAIL: $*" >&2; FAILED=1; }

md5_of() { python3 - "$1" <<'PY'
import hashlib, sys
print(hashlib.md5(open(sys.argv[1], 'rb').read()).hexdigest())
PY
}

echo "== CCSO parameter overhead gate =="

# --- named rejections -----------------------------------------------------
if "$BIN" "$PUB_CLIP" -o "$TMP/gpu.ivf" --video --video-gpu \
        --video-ccso-header-rd >/dev/null 2>"$TMP/gpu.err"; then
    fail "--video-ccso-header-rd accepted --video-gpu"
fi
grep -q "CPU-measured CCSO decision model" "$TMP/gpu.err" ||
    fail "the --video-gpu rejection did not name the CPU-only contract"

if "$BIN" "$PUB_CLIP" -o "$TMP/still.ivf" --still \
        --video-ccso-header-rd >/dev/null 2>"$TMP/still.err"; then
    fail "--video-ccso-header-rd accepted --still"
fi
grep -q "video options require --video" "$TMP/still.err" ||
    fail "the --still rejection did not name the --video requirement"

if "$BIN" "$PUB_CLIP" -o "$TMP/ten.ivf" --video --bit-depth 10 \
        --video-ccso-header-rd >/dev/null 2>"$TMP/ten.err"; then
    fail "--video-ccso-header-rd accepted a 10-bit core"
fi
grep -q "8-bit 4:2:0" "$TMP/ten.err" ||
    fail "the 10-bit rejection did not name the 8-bit 4:2:0 contract"

# --- flag-off pin, flag-on stream -----------------------------------------
AV2ENC_TRACE_CCSO=1 "$BIN" "$NASA" -o "$TMP/pin.ivf" --video --frames 60 \
    --qp 110 --cpu-threads 8 --fast-decode --video-ccso-header-off \
    >/dev/null 2>"$TMP/pin.trace" ||
    fail "the flag-off nightly encode failed"

AV2ENC_TRACE_CCSO=1 "$BIN" "$NASA" -o "$TMP/hdr.ivf" --video --frames 60 \
    --qp 110 --cpu-threads 8 --fast-decode --video-ccso-header-rd \
    --video-recon "$TMP/hdr.y4m" >/dev/null 2>"$TMP/hdr.trace" ||
    fail "the --video-ccso-header-rd encode failed"

PIN_BYTES=$(wc -c <"$TMP/pin.ivf" | tr -d ' ')
PIN_MD5=$(md5_of "$TMP/pin.ivf")
HDR_BYTES=$(wc -c <"$TMP/hdr.ivf" | tr -d ' ')
[ "$PIN_BYTES" = "24058" ] || fail "flag-off stream is $PIN_BYTES B, expected 24058"
[ "$PIN_MD5" = "d389d3def444dfd01a68a483b07d447f" ] ||
    fail "flag-off stream md5 is $PIN_MD5, expected d389d3de..."
[ "$HDR_BYTES" -lt "$PIN_BYTES" ] ||
    fail "flag-on stream ($HDR_BYTES B) is not smaller than the pin ($PIN_BYTES B)"

# --- strict decode + visible recon ---------------------------------------
"$DAV2D" -q --strict 1 -i "$TMP/hdr.ivf" -o "$TMP/decoded.y4m" --threads 1 \
    >"$TMP/strict.log" 2>&1 || {
    fail "--video-ccso-header-rd stream failed dav2d --strict 1"
    cat "$TMP/strict.log" >&2
}
PICTURES=$(python3 -c 'import sys; print(open(sys.argv[1], "rb").read().count(b"FRAME"))' \
    "$TMP/decoded.y4m" 2>/dev/null || echo 0)
[ "$PICTURES" = "60" ] || fail "decoded $PICTURES/60 pictures"
python3 - "$TMP/hdr.y4m" "$TMP/decoded.y4m" <<'PY' || fail "encoder recon != decoded output"
import sys

def payload(path):
    data = open(path, 'rb').read()
    nl = data.index(b'\n') + 1
    return data[nl:].replace(b'FRAME\n', b'')

a, b = payload(sys.argv[1]), payload(sys.argv[2])
if a != b:
    print(f"recon {len(a)} B != decoded {len(b)} B", file=sys.stderr)
    sys.exit(1)
print(f"visible recon matches decoded output ({len(a)} B)")
PY

# --- the row itself: bits before/after -----------------------------------
python3 - "$TMP/pin.trace" "$TMP/hdr.trace" "$HDR_BYTES" <<'PY' || FAILED=1
import re, sys

def rows(path):
    bits = 0
    frames = 0
    bands = set()
    enabled = 0
    for line in open(path):
        m = re.match(r'ccso f=(\d+) key=\d flag=(\d) bits=(\d+)', line)
        if m:
            frames += 1
            bits += int(m.group(3))
            enabled += m.group(2) == '1'
            continue
        m = re.match(r'  ccso p\d bands=(\d+)', line)
        if m:
            bands.add(int(m.group(1)))
    return frames, bits, enabled, bands

f0, b0, e0, bands0 = rows(sys.argv[1])
f1, b1, e1, bands1 = rows(sys.argv[2])
print(f"ccso_params() row: {b0} bits/{f0} frames -> {b1} bits/{f1} frames "
      f"({100.0 * (b0 - b1) / max(1, b0):.1f}% fewer), "
      f"enabled planes/frames {e0}->{e1}")
print(f"band counts before {sorted(bands0)} after {sorted(bands1)}")
ok = True
if f0 != 60 or f1 != 60:
    print(f"trace saw {f0}/{f1} frames, expected 60", file=sys.stderr)
    ok = False
if not b1 < b0 * 0.75:
    print(f"row bits {b1} are not >=25% below {b0}", file=sys.stderr)
    ok = False
if bands0 != {64}:
    print(f"flag-off band counts are {sorted(bands0)}, expected {{64}}", file=sys.stderr)
    ok = False
if bands1 == {64}:
    print("flag-on still pins every plane at 64 bands", file=sys.stderr)
    ok = False
sys.exit(0 if ok else 1)
PY

if [ "$FAILED" -ne 0 ]; then
    echo "gate_video_ccso_overhead: FAILED" >&2
    exit 1
fi
echo "gate_video_ccso_overhead: OK (pin, row bits, band choice, decode, recon)"
