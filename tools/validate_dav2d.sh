#!/usr/bin/env bash
# Validate that an AV2 IVF stream decodes with dav2d.
#
# Usage:
#   tools/validate_dav2d.sh out.ivf [source.y4m]
#
# Env:
#   DAV2D=/path/to/dav2d     override decoder
#   THREADS=8                decoder threads (default 0 = auto)
#   KEEP=1                   keep decoded Y4M (prints path)
#   EXPECTED_MD5=...         fail if decoded MD5 differs
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 out.ivf [source.y4m]" >&2
    exit 2
fi

IVF=$1
SRC=${2:-}
ROOT=$(cd "$(dirname "$0")/.." && pwd)

if [ -z "${DAV2D:-}" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then
            DAV2D=$candidate
            break
        fi
    done
fi

if [ -z "${DAV2D:-}" ] || [ ! -x "$DAV2D" ]; then
    echo "error: dav2d not found; set DAV2D=/path/to/dav2d" >&2
    exit 127
fi

TMPDIR=$(mktemp -d)
trap 'if [ "${KEEP:-0}" != "1" ]; then rm -rf "$TMPDIR"; fi' EXIT
OUT="$TMPDIR/decoded.y4m"

echo "decoder: $($DAV2D --version 2>&1 | tail -1)"
echo "input:   $IVF"

"$DAV2D" -q -i "$IVF" -o "$OUT" --threads "${THREADS:-0}"

if [ ! -s "$OUT" ]; then
    echo "error: dav2d produced no output" >&2
    exit 1
fi

ffprobe -v error -f yuv4mpegpipe -i "$OUT" \
    -count_frames \
    -show_entries stream=width,height,pix_fmt,nb_read_frames \
    -of default=nw=1

if [ -n "${EXPECTED_MD5:-}" ]; then
    GOT=$("$DAV2D" -q -i "$IVF" --muxer md5 -o - 2>/dev/null | tail -1)
    if [ "$GOT" != "$EXPECTED_MD5" ]; then
        echo "error: decoded MD5 mismatch: got $GOT, expected $EXPECTED_MD5" >&2
        exit 1
    fi
    echo "md5: $GOT (ok)"
fi

if [ -n "$SRC" ]; then
    echo "quality:"
    ffmpeg -hide_banner -v info -i "$OUT" -i "$SRC" \
        -lavfi "[0:v][1:v]psnr" -f null - 2>&1 \
        | grep -E "PSNR|average" | tail -3
fi

if [ "${KEEP:-0}" = "1" ]; then
    KEEP_PATH="$ROOT/decoded.y4m"
    cp "$OUT" "$KEEP_PATH"
    echo "decoded: $KEEP_PATH"
fi

echo "OK: dav2d decoded $IVF"
