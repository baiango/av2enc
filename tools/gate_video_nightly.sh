#!/usr/bin/env bash
# Gate for --nightly, the CPU-only algorithmic RD aggregate.
#
# Checks:
#   * --nightly expands to the documented flags (same IVF bytes as spelling
#     every flag out);
#   * the flag-less default and the stable quality preset keep their pins;
#   * --nightly decodes under dav2d --strict 1 and the visible encoder recon
#     equals the decoded payload;
#   * --nightly --video-gpu and --nightly --still are rejected by name.
#
# Usage: tools/gate_video_nightly.sh [--bin build/av2enc] [--dav2d PATH]
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
        -h|--help) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "gate_video_nightly: unknown argument $1" >&2; exit 2 ;;
    esac
done

BIN=${BIN:-"$ROOT/build/av2enc"}
[ -x "$BIN" ] || { echo "gate_video_nightly: --bin must point at av2enc" >&2; exit 2; }
if [ -z "$DAV2D" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
    done
fi
[ -n "$DAV2D" ] && [ -x "$DAV2D" ] || {
    echo "gate_video_nightly: SKIP (exit 77): no dav2d binary" >&2; exit 77; }
export DAV2D

NASA="$TESTDATA/nasa_480x240_30fps_60f.y4m"
SHINE="$TESTDATA/shine_640x360_4f.y4m"
[ -f "$SHINE" ] || SHINE="$TESTDATA/shine_640x360_1f.y4m"
if [ ! -f "$SHINE" ]; then
    echo "gate_video_nightly: SKIP (exit 77): no short clip in $TESTDATA" >&2
    exit 77
fi

TMP=$(mktemp -d "${TMPDIR:-/tmp}/av2enc-nightly.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
FAILED=0
fail() { echo "FAIL: $*" >&2; FAILED=1; }

NIGHTLY_FLAGS=(--video-rd-preset quality --video-ccso-rd --video-edge-inter)

echo "== nightly RD gate =="

# --- usage rejections -----------------------------------------------------
if "$BIN" "$SHINE" -o "$TMP/gpu.ivf" --nightly --video-gpu >/dev/null 2>"$TMP/gpu.err"; then
    fail "--nightly accepted --video-gpu"
fi
grep -q "nightly is CPU-only" "$TMP/gpu.err" ||
    fail "--nightly --video-gpu did not report the CPU-only rejection"

if "$BIN" "$SHINE" -o "$TMP/still.ivf" --nightly --still >/dev/null 2>"$TMP/still.err"; then
    fail "--nightly accepted --still"
fi
grep -q "nightly is a CPU --video mode" "$TMP/still.err" ||
    fail "--nightly --still did not report the video-only rejection"

# --- expansion identity ---------------------------------------------------
FRAMES=4
if ! "$BIN" "$SHINE" -o "$TMP/nightly.ivf" --nightly --frames "$FRAMES" \
        --qp 190 >"$TMP/nightly.log" 2>&1; then
    fail "--nightly encode failed"
    cat "$TMP/nightly.log" >&2
fi
if ! "$BIN" "$SHINE" -o "$TMP/explicit.ivf" --video --frames "$FRAMES" \
        --qp 190 "${NIGHTLY_FLAGS[@]}" >"$TMP/explicit.log" 2>&1; then
    fail "the explicit nightly flags failed"
fi
if [ -f "$TMP/nightly.ivf" ] && [ -f "$TMP/explicit.ivf" ]; then
    cmp -s "$TMP/nightly.ivf" "$TMP/explicit.ivf" ||
        fail "--nightly differs from its documented flag expansion"
fi

# --- default and quality pins --------------------------------------------
if [ -f "$NASA" ]; then
    if ! "$BIN" "$NASA" -o "$TMP/default.ivf" --video --frames 60 \
            --qp 189 --cpu-threads 8 >"$TMP/default.log" 2>&1; then
        fail "the flag-less baseline encode failed"
    fi
    if [ -f "$TMP/default.ivf" ]; then
        BYTES=$(wc -c <"$TMP/default.ivf" | tr -d ' ')
        MD5=$(python3 - "$TMP/default.ivf" <<'PY'
import hashlib, sys
print(hashlib.md5(open(sys.argv[1], 'rb').read()).hexdigest())
PY
)
        [ "$BYTES" = "25009" ] || fail "default stream is $BYTES B, expected 25009"
        [ "$MD5" = "a1d0dba1259b20a4b628d4cd1e14c33c" ] ||
            fail "default stream md5 is $MD5, expected a1d0dba1..."
    fi

    if ! "$BIN" "$NASA" -o "$TMP/quality.ivf" --video --frames 60 \
            --qp 190 --cpu-threads 8 --video-rd-preset quality \
            >"$TMP/quality.log" 2>&1; then
        fail "the quality preset encode failed"
    fi
    if [ -f "$TMP/quality.ivf" ]; then
        BYTES=$(wc -c <"$TMP/quality.ivf" | tr -d ' ')
        MD5=$(python3 - "$TMP/quality.ivf" <<'PY'
import hashlib, sys
print(hashlib.md5(open(sys.argv[1], 'rb').read()).hexdigest())
PY
)
        [ "$BYTES" = "25141" ] || fail "quality stream is $BYTES B, expected 25141"
        [ "$MD5" = "3984677b9b665952984b07b7cf75c6ea" ] ||
            fail "quality stream md5 is $MD5, expected 3984677b..."
    fi
fi

# --- strict decode + visible recon ---------------------------------------
if ! "$DAV2D" -q --strict 1 -i "$TMP/nightly.ivf" -o "$TMP/decoded.y4m" \
        --threads 1 >"$TMP/strict.log" 2>&1; then
    fail "--nightly stream failed dav2d --strict 1"
    cat "$TMP/strict.log" >&2
fi
if ! "$BIN" "$SHINE" -o "$TMP/recon.ivf" --nightly --frames "$FRAMES" \
        --qp 190 --video-recon "$TMP/recon.y4m" >"$TMP/recon.log" 2>&1; then
    fail "the --nightly --video-recon encode failed"
elif ! python3 - "$TMP/recon.y4m" "$TMP/decoded.y4m" <<'PY'
import sys

def payload(path):
    data = open(path, 'rb').read()
    nl = data.index(b'\n') + 1
    return data[nl:].replace(b'FRAME\n', b'')

a, b = payload(sys.argv[1]), payload(sys.argv[2])
if len(a) != len(b):
    print(f"recon payload {len(a)} B != decoded {len(b)} B", file=sys.stderr)
    sys.exit(1)
if a != b:
    diff = sum(1 for x, y in zip(a, b) if x != y)
    print(f"recon differs from decoded output in {diff} byte(s)", file=sys.stderr)
    sys.exit(1)
print(f"visible recon matches decoded output ({len(a)} B)")
PY
then
    fail "decoder output differs from the encoder reconstruction"
fi

if [ "$FAILED" -ne 0 ]; then
    echo "gate_video_nightly: FAILED" >&2
    exit 1
fi
echo "gate_video_nightly: OK (expansion, pins, strict decode, visible recon)"
exit 0
