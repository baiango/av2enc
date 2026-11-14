#!/usr/bin/env bash
# Gate for the opt-in video coefficient RDOQ (--video-rdoq).
#
# What it checks:
#   * the flag is rejected without --video-inter-residual and with --video-tcq;
#   * the flag-less 480x240 60-frame default and preset streams keep their
#     recorded sizes (25,009 B / a1d0dba1... and 25,141 B);
#   * --video-rdoq changes the stream (same QP, smaller) and decodes with the
#     pinned dav2d under --strict 1;
#   * the whole-frame decoder output equals the encoder reconstruction.
#
# Usage:
#   tools/gate_video_rdoq.sh [--bin build-c/av2enc] [--dav2d PATH]
#
# Exit codes: 0 pass, 1 failure, 2 usage, 77 missing fixture/decoder.
set -uo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/.." && pwd)
BIN=""
DAV2D="${DAV2D:-}"
TESTDATA="${AV2ENC_TESTDATA:-$ROOT/testdata}"

while [ $# -gt 0 ]; do
    case "$1" in
        --bin) BIN=${2:-}; shift 2 ;;
        --dav2d) DAV2D=${2:-}; shift 2 ;;
        -h|--help) sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "gate_video_rdoq: unknown argument $1" >&2; exit 2 ;;
    esac
done

BIN=${BIN:-"$ROOT/build-c/av2enc"}
[ -x "$BIN" ] || { echo "gate_video_rdoq: --bin must point at av2enc" >&2; exit 2; }
if [ -z "$DAV2D" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
    done
fi
[ -n "$DAV2D" ] && [ -x "$DAV2D" ] || {
    echo "gate_video_rdoq: SKIP (exit 77): no dav2d binary" >&2; exit 77; }
export DAV2D

NASA="$TESTDATA/nasa_480x240_30fps_60f.y4m"
SHINE="$TESTDATA/shine_640x360_4f.y4m"
[ -f "$SHINE" ] || SHINE="$TESTDATA/shine_640x360_1f.y4m"
if [ ! -f "$NASA" ] && [ ! -f "$SHINE" ]; then
    echo "gate_video_rdoq: SKIP (exit 77): no fixture in $TESTDATA" >&2
    exit 77
fi
CLIP=${NASA:-$SHINE}
FRAMES=60
[ -f "$NASA" ] || FRAMES=4

TMP=$(mktemp -d "${TMPDIR:-/tmp}/av2enc-rdoq.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
FAILED=0
fail() { echo "FAIL: $*" >&2; FAILED=1; }

echo "== video RDOQ gate =="

# --- usage errors ---------------------------------------------------------
if "$BIN" "$CLIP" -o "$TMP/reject.ivf" --video --frames 1 --qp 189 \
        --video-rdoq >/dev/null 2>"$TMP/reject1.err"; then
    fail "--video-rdoq was accepted without --video-inter-residual"
fi
grep -q "needs --video-inter-residual" "$TMP/reject1.err" ||
    fail "the missing --video-inter-residual rejection was not reported"

if "$BIN" "$CLIP" -o "$TMP/reject.ivf" --video --frames 1 --qp 189 \
        --video-inter-residual --video-tcq --video-rdoq \
        >/dev/null 2>"$TMP/reject2.err"; then
    fail "--video-rdoq was accepted with --video-tcq"
fi
grep -q "does not compose with --video-tcq" "$TMP/reject2.err" ||
    fail "the --video-tcq rejection was not reported"
[ -e "$TMP/reject.ivf" ] && fail "a rejected run created its output file"

# --- pinned flag-less streams are untouched ------------------------------
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
        [ "$BYTES" = "25009" ] ||
            fail "default 480x240 stream is $BYTES B, expected 25009 B"
        [ "$MD5" = "a1d0dba1259b20a4b628d4cd1e14c33c" ] ||
            fail "default 480x240 stream md5 is $MD5, expected a1d0dba1..."
    fi
fi

# --- RDOQ stream changes, shrinks, decodes and matches the recon ----------
if ! "$BIN" "$CLIP" -o "$TMP/preset.ivf" --video --frames "$FRAMES" \
        --qp 190 --cpu-threads 8 --video-rd-preset quality \
        >"$TMP/preset.log" 2>&1; then
    fail "the preset encode failed"
fi
if ! "$BIN" "$CLIP" -o "$TMP/rdoq.ivf" --video --frames "$FRAMES" \
        --qp 190 --cpu-threads 8 --video-rd-preset quality --video-rdoq \
        --video-recon "$TMP/enc.y4m" >"$TMP/rdoq.log" 2>&1; then
    fail "the RDOQ encode failed"
    cat "$TMP/rdoq.log" >&2
fi
if [ -f "$TMP/preset.ivf" ] && [ -f "$TMP/rdoq.ivf" ]; then
    cmp -s "$TMP/preset.ivf" "$TMP/rdoq.ivf" &&
        fail "--video-rdoq did not change the stream"
    PB=$(wc -c <"$TMP/preset.ivf" | tr -d ' ')
    RB=$(wc -c <"$TMP/rdoq.ivf" | tr -d ' ')
    [ "$RB" -lt "$PB" ] ||
        fail "--video-rdoq stream is $RB B, not smaller than preset $PB B"
    if [ -f "$NASA" ] && [ "$FRAMES" = "60" ]; then
        [ "$PB" = "25141" ] ||
            fail "preset 480x240 stream is $PB B, expected 25141 B"
    fi
fi

if ! "$DAV2D" -q --strict 1 -i "$TMP/rdoq.ivf" -o "$TMP/dec.y4m" \
        --threads 1 >"$TMP/strict.log" 2>&1; then
    fail "the RDOQ stream failed dav2d --strict 1"
    cat "$TMP/strict.log" >&2
fi
if ! python3 - "$TMP/enc.y4m" "$TMP/dec.y4m" <<'PY'
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
print(f"recon matches decoded output ({len(a)} B)")
PY
then
    fail "decoder output differs from the encoder reconstruction"
fi

if [ "$FAILED" -ne 0 ]; then
    echo "gate_video_rdoq: FAILED" >&2
    exit 1
fi
echo "gate_video_rdoq: OK (usage, pinned identity, stream change, decode, recon)"
exit 0
