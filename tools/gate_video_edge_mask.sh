#!/usr/bin/env bash
# Gate for --video-edge-mask (bottom-edge residual mask).
#
# The mask treats the coded rows of a bottom-straddling leaf below the visible
# picture edge as decoder-invisible: their residual is zero before the
# transform and the leaf's mode/MV/skip metrics score only the visible rows.
# No syntax changes, so the decoder path is untouched.
#
# Checks:
#   * the named rejections (needs --video-edge-inter; CPU only; single tile;
#     --video mode only);
#   * on an SB-aligned height the mask is inert (byte-identical stream);
#   * on 480x240 the masked stream is smaller at the same QP than the
#     edge-inter stream;
#   * dav2d --strict 1 decodes the masked stream and the visible encoder
#     reconstruction equals the decoded payload byte for byte (deblock/CCSO
#     included: the preset enables both).
#
# Usage: tools/gate_video_edge_mask.sh [--bin build/av2enc] [--dav2d PATH]
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
        -h|--help) sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "gate_video_edge_mask: unknown argument $1" >&2; exit 2 ;;
    esac
done

BIN=${BIN:-"$ROOT/build/av2enc"}
[ -x "$BIN" ] || { echo "gate_video_edge_mask: --bin must point at av2enc" >&2; exit 2; }
if [ -z "$DAV2D" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
    done
fi
[ -n "$DAV2D" ] && [ -x "$DAV2D" ] || {
    echo "gate_video_edge_mask: SKIP (exit 77): no dav2d binary" >&2; exit 77; }

NASA="$TESTDATA/nasa_480x240_30fps_60f.y4m"
ALIGNED="$TESTDATA/shine_256x256_1f.y4m"
[ -f "$NASA" ] || {
    echo "gate_video_edge_mask: SKIP (exit 77): no $NASA" >&2; exit 77; }
[ -f "$ALIGNED" ] || {
    echo "gate_video_edge_mask: SKIP (exit 77): no $ALIGNED" >&2; exit 77; }

TMP=$(mktemp -d "${TMPDIR:-/tmp}/av2enc-edge-mask.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
FAILED=0
fail() { echo "FAIL: $*" >&2; FAILED=1; }

echo "== video edge mask gate =="

# --- named rejections ----------------------------------------------------
if "$BIN" "$NASA" -o "$TMP/reject.ivf" --video --frames 1 --qp 190 \
        --video-edge-mask >/dev/null 2>"$TMP/need.err"; then
    fail "--video-edge-mask was accepted without --video-edge-inter"
fi
grep -q "needs --video-edge-inter" "$TMP/need.err" ||
    fail "the --video-edge-inter requirement was not named"
if "$BIN" "$NASA" -o "$TMP/reject.ivf" --video --frames 1 --qp 190 \
        --video-edge-inter --video-edge-mask --video-gpu \
        >/dev/null 2>"$TMP/gpu.err"; then
    fail "--video-edge-mask was accepted with --video-gpu"
fi
grep -q "CPU-only" "$TMP/gpu.err" ||
    fail "the --video-gpu rejection did not name the CPU-only path"
if "$BIN" "$NASA" -o "$TMP/reject.ivf" --video --frames 1 --qp 190 \
        --video-edge-inter --video-edge-mask --tiles 2x1 \
        >/dev/null 2>"$TMP/tiles.err"; then
    fail "--video-edge-mask was accepted with --tiles"
fi
grep -q "single-tile" "$TMP/tiles.err" ||
    fail "the --tiles rejection did not name the single-tile requirement"
if "$BIN" "$NASA" -o "$TMP/reject.ivf" --still --video-edge-inter \
        --video-edge-mask >/dev/null 2>"$TMP/still.err"; then
    fail "--video-edge-mask was accepted on the --still writer"
fi
grep -q "CPU --video encoder" "$TMP/still.err" ||
    fail "the --still rejection did not point at --video"
[ -e "$TMP/reject.ivf" ] && fail "a rejected run created its output file"

# --- inert when the coded extent never crosses the visible bottom --------
# 256x256 is four whole 64x64 superblock rows, so no leaf straddles the
# bottom edge and the mask must not change a single byte.
if ! "$BIN" "$ALIGNED" -o "$TMP/a_plain.ivf" --video --frames 1 --qp 190 \
        --cpu-threads 8 --video-edge-inter >"$TMP/a_plain.log" 2>&1; then
    fail "the SB-aligned control encode failed"
fi
if ! "$BIN" "$ALIGNED" -o "$TMP/a_mask.ivf" --video --frames 1 --qp 190 \
        --cpu-threads 8 --video-edge-inter --video-edge-mask \
        >"$TMP/a_mask.log" 2>&1; then
    fail "the SB-aligned masked encode failed"
fi
if [ -f "$TMP/a_plain.ivf" ] && [ -f "$TMP/a_mask.ivf" ]; then
    cmp -s "$TMP/a_plain.ivf" "$TMP/a_mask.ivf" ||
        fail "the mask changed an SB-aligned (no straddle) stream"
fi

# --- size, decode, visible recon on a straddling frame -------------------
FRAMES=8
COMMON=(--video --frames "$FRAMES" --qp 190 --cpu-threads 8
        --video-rd-preset quality --video-edge-inter)
if ! "$BIN" "$NASA" -o "$TMP/plain.ivf" "${COMMON[@]}" >"$TMP/plain.log" 2>&1; then
    fail "the plain edge-inter encode failed"
fi
if ! "$BIN" "$NASA" -o "$TMP/mask.ivf" "${COMMON[@]}" --video-edge-mask \
        --video-recon "$TMP/recon.y4m" --video-trace \
        >"$TMP/mask.log" 2>"$TMP/mask.trace"; then
    fail "the masked encode failed"
    cat "$TMP/mask.log" >&2
fi

if [ ! -s "$TMP/mask.ivf" ]; then
    fail "the masked encode produced no stream"
else
    PLAIN_BYTES=$(wc -c <"$TMP/plain.ivf" | tr -d ' ')
    MASK_BYTES=$(wc -c <"$TMP/mask.ivf" | tr -d ' ')
    echo "same-QP bytes: plain=$PLAIN_BYTES mask=$MASK_BYTES"
    [ "$MASK_BYTES" -lt "$PLAIN_BYTES" ] ||
        fail "the masked stream is not smaller than the plain edge stream"
    INTER=$(grep -c "m2 inter r=48" "$TMP/mask.trace" || true)
    echo "bottom-row inter leaves: inter=$INTER (frames=$FRAMES)"
    [ "$INTER" -ge $(( (FRAMES - 1) * 8 )) ] ||
        fail "only $INTER bottom-row inter leaves in the masked stream"
fi

if ! "$DAV2D" -q --strict 1 -i "$TMP/mask.ivf" -o "$TMP/decoded.y4m" \
        --threads 1 >"$TMP/strict.log" 2>&1; then
    fail "the masked stream failed dav2d --strict 1"
    cat "$TMP/strict.log" >&2
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
    echo "gate_video_edge_mask: FAILED" >&2
    exit 1
fi
echo "gate_video_edge_mask: OK (rejections, inert-aligned, size, strict decode, visible recon)"
