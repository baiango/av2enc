#!/usr/bin/env bash
# Generated-content smoke encode.
#
# CI must never depend on copyrighted media, and the pinned testdata/ clips are
# gitignored anyway, so this generates its own input:
#   - ffmpeg testsrc2 when ffmpeg is available, else
#   - a tiny Python Y4M writer (GitHub's macOS runners ship no ffmpeg).
#
# Usage: tools/ci_encode_smoke.sh <av2enc> [frames] [threads] [--trace]
# Exit: 0 ok, 1 encode failed, 2 setup problem, 77 no generator available.
set -uo pipefail

BIN=${1:-build-release/av2enc}
FRAMES=${2:-8}
THREADS=${3:-8}
TRACE=${4:-}

[ -x "$BIN" ] || { echo "ci_encode_smoke: no binary: $BIN" >&2; exit 2; }

TMP=$(mktemp -d "${TMPDIR:-/tmp}/av2enc-smoke.XXXXXX")
trap 'rm -rf "$TMP"' EXIT

# Synthetic content, generated rather than sampled from any video.
if command -v ffmpeg >/dev/null 2>&1; then
    ffmpeg -hide_banner -loglevel error \
        -f lavfi -i "testsrc2=size=320x240:rate=30:duration=1" \
        -pix_fmt yuv420p -f yuv4mpegpipe "$TMP/in.y4m" || {
        echo "ci_encode_smoke: could not generate the input" >&2; exit 2; }
elif command -v python3 >/dev/null 2>&1; then
    python3 - "$TMP/in.y4m" "$FRAMES" <<'PY'
import sys

path, frames = sys.argv[1], int(sys.argv[2])
w, h = 320, 240
with open(path, "wb") as f:
    f.write(f"YUV4MPEG2 W{w} H{h} F30:1 Ip A1:1 C420jpeg\n".encode())
    for n in range(frames):
        y = bytearray(w * h)
        for r in range(h):
            base = (r * 255) // max(1, h - 1)
            row = r * w
            for c in range(w):
                y[row + c] = (base + c // 4) & 0xFF
        # Moving box: enough inter motion for the tool paths to fire.
        x0 = (n * 7) % max(1, w - 32)
        y0 = (n * 5) % max(1, h - 32)
        for r in range(y0, min(h, y0 + 32)):
            row = r * w
            for c in range(x0, min(w, x0 + 32)):
                y[row + c] = 235
        chroma = bytes([128]) * (w // 2 * h // 2)
        f.write(b"FRAME\n")
        f.write(bytes(y))
        f.write(chroma)
        f.write(chroma)
PY
else
    echo "ci_encode_smoke: neither ffmpeg nor python3 available; skipping" >&2
    exit 77
fi

ARGS=(--video --frames "$FRAMES" --cpu-threads "$THREADS" --qp 106 --fast-decode)
[ "$TRACE" = "--trace" ] && ARGS+=(--video-trace)

if ! "$BIN" "$TMP/in.y4m" -o "$TMP/out.ivf" "${ARGS[@]}" \
    >"$TMP/out.log" 2>"$TMP/err.log"; then
    echo "ci_encode_smoke: encode failed" >&2
    tail -n 10 "$TMP/out.log" "$TMP/err.log" >&2
    exit 1
fi
if [ ! -s "$TMP/out.ivf" ]; then
    echo "ci_encode_smoke: empty output" >&2; exit 1
fi

# TSan builds report races on stderr; surface them here rather than only in the
# workflow's grep.
if grep -q 'WARNING: ThreadSanitizer' "$TMP/err.log"; then
    echo "ci_encode_smoke: ThreadSanitizer reported a race" >&2
    grep -A8 'WARNING: ThreadSanitizer' "$TMP/err.log" | head -30 >&2
    exit 1
fi

echo "ci_encode_smoke: ok ($(wc -c <"$TMP/out.ivf" | tr -d ' ') bytes, ${FRAMES} frames, ${THREADS} threads)"
