#!/usr/bin/env bash
# Screen-content gate for `--still-gpu` palette (§5.20.8) and IntraBC
# (§5.20.5.3/.4).
#
# Natural fixtures almost never select palette or IBC, so their byte parity
# only proves the flag plumbing.  This wrapper generates flat-fill/stripe and
# repeated-tile Y4M inputs, then runs `verify_gpu.sh` (CPU vs GPU frame bytes,
# `dav2d --strict 1`, decoded-plane identity) for every tool:
#
#   palette       64x64, 640x360
#   ibc           384x128, 640x360
#   palette+ibc   384x128
#
# Usage:
#   tools/still_gpu/verify_screen_content.sh [--encoder PATH] [--decoder PATH]
#
# Environment: same as verify_gpu.sh (AV2ENC, DAV2D, NSYS, ...).
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ENCODER=${AV2ENC:-$ROOT/build-cuda/av2enc}
DECODER=${DAV2D:-}
PY=${PYTHON:-python3}

while [ $# -gt 0 ]; do
    case "$1" in
        --encoder) ENCODER=$2; shift 2 ;;
        --decoder) DECODER=$2; shift 2 ;;
        -h|--help) sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "verify_screen_content: unknown argument $1" >&2; exit 2 ;;
    esac
done

if [ ! -x "$ENCODER" ]; then
    echo "verify_screen_content: encoder not executable: $ENCODER" >&2
    exit 2
fi
if [ -z "$DECODER" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DECODER=$candidate; break; fi
    done
fi
if [ -z "$DECODER" ] || [ ! -x "$DECODER" ]; then
    echo "verify_screen_content: no dav2d binary (set DAV2D=...)" >&2
    exit 77
fi

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

"$PY" - "$TMP" <<'PY'
import sys, pathlib
out = pathlib.Path(sys.argv[1])

def write_y4m(path, w, h):
    """Flat fills, 8-pixel stripes and icon squares (palette-friendly)."""
    y = bytearray(w * h)
    for yy in range(h):
        for xx in range(w):
            if yy < h // 2:
                v = 32 if xx < w // 2 else 200
                if (xx // 8) % 2 == 0 and yy >= h // 8:
                    v = 96
            else:
                v = 64 if (xx // 16) % 2 == 0 else 160
                if (yy // 8) % 4 == 0:
                    v = 240
            y[yy * w + xx] = v
    cw, ch = (w + 1) // 2, (h + 1) // 2
    with open(path, "wb") as f:
        f.write(b"YUV4MPEG2 W%d H%d F25:1 Ip A0:0 C420\n" % (w, h))
        f.write(b"FRAME\n")
        f.write(bytes(y)); f.write(bytes([96]) * (cw * ch))
        f.write(bytes([160]) * (cw * ch))

def write_ibc_y4m(path, w, h):
    """A 32x32 tile repeated horizontally/vertically: exact-copy friendly."""
    y = bytearray(w * h)
    palette = [64, 96, 160, 200, 32, 240]
    for yy in range(h):
        for xx in range(w):
            tx, ty = (xx // 32) % 3, (yy // 32) % 2
            v = palette[(tx + ty * 3) % 6]
            if (xx % 32) < 8 and (yy % 32) < 8:
                v = 16
            if (xx % 32) >= 24 and (yy % 32) >= 24:
                v = 224
            y[yy * w + xx] = v
    cw, ch = (w + 1) // 2, (h + 1) // 2
    with open(path, "wb") as f:
        f.write(b"YUV4MPEG2 W%d H%d F25:1 Ip A0:0 C420\n" % (w, h))
        f.write(b"FRAME\n")
        f.write(bytes(y)); f.write(bytes([100]) * (cw * ch))
        f.write(bytes([150]) * (cw * ch))

write_y4m(out / "scc_64x64.y4m", 64, 64)
write_y4m(out / "scc_640x360.y4m", 640, 360)
write_ibc_y4m(out / "ibc_384x128.y4m", 384, 128)
write_ibc_y4m(out / "ibc_640x360.y4m", 640, 360)
PY

failures=0
run() {
    local input=$1 args=$2 label=$3
    echo "== $label ($args)"
    if ! "$ROOT/tools/still_gpu/verify_gpu.sh" "$input" \
            --encoder "$ENCODER" --decoder "$DECODER" \
            --still-args "$args"; then
        echo "FAIL: $label" >&2
        failures=$((failures + 1))
    fi
}

run "$TMP/scc_64x64.y4m"      "--still-palette"            "palette 64x64"
run "$TMP/scc_640x360.y4m"    "--still-palette"            "palette 640x360"
run "$TMP/ibc_384x128.y4m"    "--still-ibc"                "ibc 384x128"
run "$TMP/ibc_640x360.y4m"    "--still-ibc"                "ibc 640x360"
run "$TMP/ibc_384x128.y4m"    "--still-ibc --still-palette" "palette+ibc 384x128"

if [ "$failures" -ne 0 ]; then
    echo "verify_screen_content: $failures case(s) failed" >&2
    exit 1
fi
echo "verify_screen_content: PASS (5 cases)"
