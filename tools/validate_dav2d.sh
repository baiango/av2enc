#!/usr/bin/env bash
# Validate that an AV2 IVF stream decodes with the pinned dav2d build.
#
# Usage:
#   tools/validate_dav2d.sh out.ivf [source.y4m]
#
# Env:
#   DAV2D=/path/to/dav2d     override decoder (else tools/bin/dav2d[-macos])
#   THREADS=8                decoder threads (default 1 = deterministic)
#   KEEP=1                   keep decoded Y4M (prints path)
#   EXPECTED_MD5=...         fail if decoded MD5 differs
#   DAV2D_ARGS="..."         replace the default decode arguments entirely
#   ALLOW_NONOUTPUT_FRAMES=1 decode fewer Y4M frames than the IVF header
#                            declares (streams with hidden/leading pictures)
#
# Exit codes: 0 ok, 1 decode/quality failure, 2 usage, 127 no decoder.
# See docs/validation.md for the decoder pin and the gate policy.
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 out.ivf [source.y4m]" >&2
    exit 2
fi

IVF=$1
SRC=${2:-}
ROOT=$(cd "$(dirname "$0")/.." && pwd)

if [ ! -f "$IVF" ]; then
    echo "error: no such file: $IVF" >&2
    exit 2
fi

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

# --- container sanity (does not decode) -------------------------------------
IVF_FRAMES=""
if command -v python3 >/dev/null 2>&1; then
    python3 - "$IVF" <<'PY'
import struct, sys
p = sys.argv[1]
with open(p, "rb") as f:
    h = f.read(32)
if len(h) >= 32 and h[:4] == b"DKIF":
    w, hgt = struct.unpack_from("<HH", h, 12)
    count = struct.unpack_from("<I", h, 24)[0]
    print(f"ivf:     {'%dx%d' % (w, hgt)}, {count} frame(s), fourcc {h[8:12].decode('latin1')}")
else:
    print("ivf:     warning: not a DKIF header (raw OBU stream?)")
PY
    IVF_FRAMES=$(python3 - "$IVF" <<'PY'
import struct, sys
with open(sys.argv[1], "rb") as f:
    h = f.read(32)
print(struct.unpack_from("<I", h, 24)[0]
      if len(h) >= 32 and h[:4] == b"DKIF" else "")
PY
)
fi

TMPDIR=$(mktemp -d)
trap 'if [ "${KEEP:-0}" != "1" ]; then rm -rf "$TMPDIR"; fi' EXIT
OUT="$TMPDIR/decoded.y4m"

echo "decoder: $($DAV2D --version 2>&1 | tail -1)"
echo "input:   $IVF"

if [ -n "${DAV2D_ARGS:-}" ]; then
    # shellcheck disable=SC2086
    "$DAV2D" $DAV2D_ARGS
else
    if ! "$DAV2D" -q -i "$IVF" -o "$OUT" --threads "${THREADS:-1}"; then
        echo "error: dav2d failed to decode $IVF; decoder help follows" >&2
        "$DAV2D" --help 2>&1 | head -40 >&2 || true
        exit 1
    fi
fi

if [ ! -s "$OUT" ]; then
    echo "error: dav2d produced no output" >&2
    exit 1
fi

FRAMES=""
if command -v ffprobe >/dev/null 2>&1; then
    ffprobe -v error -f yuv4mpegpipe -i "$OUT" \
        -count_frames \
        -show_entries stream=width,height,pix_fmt,nb_read_frames \
        -of default=nw=1
    FRAMES=$(ffprobe -v error -f yuv4mpegpipe -i "$OUT" \
        -count_frames -show_entries stream=nb_read_frames \
        -of default=nw=1:nk=1)
else
    echo "warning: ffprobe not found; counting Y4M FRAME markers" >&2
    FRAMES=$(grep -a -c '^FRAME' "$OUT" 2>/dev/null || echo 0)
fi

# The decoder exits 0 after dropping frames it cannot parse, so a truncated or
# desynced multi-frame stream used to pass this validator with frame 0 only.
# The IVF header is authoritative for the number of temporal units; every one
# must be accounted for in the decoded output unless the stream intentionally
# codes non-output (e.g. leading/OLK) pictures, which the caller can declare.
if [ -n "$IVF_FRAMES" ] && [ -n "$FRAMES" ]; then
    if [ "$FRAMES" -lt "$IVF_FRAMES" ] &&
       [ "${ALLOW_NONOUTPUT_FRAMES:-0}" != "1" ]; then
        echo "error: dav2d decoded $FRAMES frame(s) but the IVF header" \
             "declares $IVF_FRAMES; later temporal units were dropped" \
             "(set ALLOW_NONOUTPUT_FRAMES=1 for streams with hidden pictures)" >&2
        exit 1
    fi
    if [ "$FRAMES" -gt "$IVF_FRAMES" ]; then
        echo "error: dav2d decoded $FRAMES frame(s), more than the IVF header" \
             "declares ($IVF_FRAMES)" >&2
        exit 1
    fi
fi

if [ -n "${EXPECTED_MD5:-}" ]; then
    GOT=$("$DAV2D" -q -i "$IVF" --muxer md5 -o - 2>/dev/null | tail -1)
    if [ "$GOT" != "$EXPECTED_MD5" ]; then
        echo "error: decoded MD5 mismatch: got $GOT, expected $EXPECTED_MD5" >&2
        exit 1
    fi
    echo "md5: $GOT (ok)"
fi

if [ -n "$SRC" ]; then
    if ! command -v ffmpeg >/dev/null 2>&1; then
        echo "warning: ffmpeg not found; skipping PSNR" >&2
    else
        SRC_FRAMES=""
        if command -v ffprobe >/dev/null 2>&1; then
            SRC_FRAMES=$(ffprobe -v error -f yuv4mpegpipe -i "$SRC" \
                -count_frames -show_entries stream=nb_read_frames \
                -of default=nw=1:nk=1)
        fi
        if [ -n "$FRAMES" ] && [ -n "$SRC_FRAMES" ] && [ "$FRAMES" != "$SRC_FRAMES" ]; then
            echo "warning: decoded $FRAMES frame(s) vs reference $SRC_FRAMES;" \
                 "skipping PSNR (use tools/quality.py for normalized metrics)" >&2
        else
            echo "quality:"
            FRAME_ARGS=()
            if [ -n "$FRAMES" ]; then
                FRAME_ARGS=(-frames:v "$FRAMES")
            fi
            ffmpeg -hide_banner -v info -i "$OUT" -i "$SRC" \
                -lavfi "[0:v][1:v]psnr" "${FRAME_ARGS[@]}" -f null - 2>&1 \
                | grep -E "PSNR|average" | tail -3
        fi
    fi
fi

if [ "${KEEP:-0}" = "1" ]; then
    KEEP_PATH="$ROOT/decoded.y4m"
    cp "$OUT" "$KEEP_PATH"
    echo "decoded: $KEEP_PATH"
fi

echo "OK: dav2d decoded $IVF"
