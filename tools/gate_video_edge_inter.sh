#!/usr/bin/env bash
# Gate for --video-edge-inter (opt-in bottom-edge inter).
#
# What it checks:
#   * the bottom 64x64 band is inter-coded on P frames and intra only on the
#     keyframe (trace counts on the 480x240 fixture);
#   * the stream decodes with pinned dav2d under --strict 1;
#   * the encoder reconstruction equals the decoder output for every visible
#     sample (the SB-aligned hidden rows are outside the Y4M payload);
#   * at the same QP the edge path is far smaller than the preset alone (the
#     whole point of the change);
#   * the flag is rejected by name for --video-gpu, --tiles and --still.
#
# Usage:
#   tools/gate_video_edge_inter.sh [--bin build-g/av2enc] [--dav2d PATH]
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
        -h|--help) sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "gate_video_edge_inter: unknown argument $1" >&2; exit 2 ;;
    esac
done

BIN=${BIN:-"$ROOT/build-g/av2enc"}
[ -x "$BIN" ] || { echo "gate_video_edge_inter: --bin must point at av2enc" >&2; exit 2; }
if [ -z "$DAV2D" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
    done
fi
[ -n "$DAV2D" ] && [ -x "$DAV2D" ] || {
    echo "gate_video_edge_inter: SKIP (exit 77): no dav2d binary" >&2; exit 77; }

NASA="$TESTDATA/nasa_480x240_30fps_60f.y4m"
if [ ! -f "$NASA" ]; then
    echo "gate_video_edge_inter: SKIP (exit 77): no $NASA" >&2
    exit 77
fi

TMP=$(mktemp -d "${TMPDIR:-/tmp}/av2enc-edge-inter.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
FAILED=0
fail() { echo "FAIL: $*" >&2; FAILED=1; }

echo "== video edge inter gate =="

# --- named rejections ----------------------------------------------------
if "$BIN" "$NASA" -o "$TMP/reject.ivf" --video --frames 1 --qp 190 \
        --video-edge-inter --video-gpu >/dev/null 2>"$TMP/gpu.err"; then
    fail "--video-edge-inter was accepted with --video-gpu"
fi
grep -q "CPU-only" "$TMP/gpu.err" ||
    fail "the --video-gpu rejection did not name the CPU-only path"
if "$BIN" "$NASA" -o "$TMP/reject.ivf" --video --frames 1 --qp 190 \
        --video-edge-inter --tiles 2x1 >/dev/null 2>"$TMP/tiles.err"; then
    fail "--video-edge-inter was accepted with --tiles"
fi
grep -q "single-tile" "$TMP/tiles.err" ||
    fail "the --tiles rejection did not name the single-tile requirement"
if "$BIN" "$NASA" -o "$TMP/reject.ivf" --still --video-edge-inter \
        >/dev/null 2>"$TMP/still.err"; then
    fail "--video-edge-inter was accepted on the --still writer"
fi
grep -q "CPU --video encoder" "$TMP/still.err" ||
    fail "the --still rejection did not point at --video"
[ -e "$TMP/reject.ivf" ] && fail "a rejected run created its output file"

# --- encode/decode/recon -------------------------------------------------
FRAMES=8
COMMON=(--video --frames "$FRAMES" --qp 190 --cpu-threads 8
        --video-rd-preset quality)
if ! "$BIN" "$NASA" -o "$TMP/edge.ivf" "${COMMON[@]}" --video-edge-inter \
        --video-recon "$TMP/recon.y4m" --video-trace \
        >"$TMP/edge.log" 2>"$TMP/edge.trace"; then
    fail "the --video-edge-inter encode failed"
    cat "$TMP/edge.log" >&2
fi
if ! "$BIN" "$NASA" -o "$TMP/preset.ivf" "${COMMON[@]}" \
        >"$TMP/preset.log" 2>&1; then
    fail "the preset-only encode failed"
fi

if [ ! -s "$TMP/edge.ivf" ]; then
    fail "the edge encode produced no stream"
else
    # Every P-frame bottom leaf must try inter.  The keyframe is intra by
    # definition, so at most its leaves stay intra.
    INTER=$(grep -c "m2 inter r=48" "$TMP/edge.trace" || true)
    INTRA=$(grep -c "m2 intra r=48" "$TMP/edge.trace" || true)
    echo "bottom-row leaves: inter=$INTER intra=$INTRA (frames=$FRAMES)"
    [ "$INTER" -ge $(( (FRAMES - 1) * 8 )) ] ||
        fail "only $INTER bottom-row inter leaves (expected >= $(( (FRAMES - 1) * 8 )))"
    [ "$INTRA" -le 8 ] ||
        fail "$INTRA bottom-row intra leaves (expected <= 8 keyframe leaves)"
    EDGE_BYTES=$(wc -c <"$TMP/edge.ivf" | tr -d ' ')
    PRESET_BYTES=$(wc -c <"$TMP/preset.ivf" | tr -d ' ')
    echo "same-QP bytes: preset=$PRESET_BYTES edge=$EDGE_BYTES"
    [ "$EDGE_BYTES" -lt $(( PRESET_BYTES * 3 / 4 )) ] ||
        fail "the edge stream is not below 75% of the preset stream"
fi

if ! "$DAV2D" -q --strict 1 -i "$TMP/edge.ivf" -o "$TMP/decoded.y4m" \
        --threads 1 >"$TMP/strict.log" 2>&1; then
    fail "the edge stream failed dav2d --strict 1"
    cat "$TMP/strict.log" >&2
elif [ ! -s "$TMP/decoded.y4m" ]; then
    fail "dav2d --strict 1 produced no output"
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
    print(f"visible recon differs from decoded output in {diff} byte(s)",
          file=sys.stderr)
    sys.exit(1)
print(f"visible recon matches decoded output ({len(a)} B)")
PY
then
    fail "decoder output differs from the encoder reconstruction"
fi

if [ "$FAILED" -ne 0 ]; then
    echo "gate_video_edge_inter: FAILED" >&2
    exit 1
fi
echo "gate_video_edge_inter: OK (trace, bytes, decode, visible recon, rejections)"
exit 0
