#!/usr/bin/env bash
# CLI gate for multi-tile video (--video --tiles).
#
# Cases:
#   * default --video 720p 6-frame stream stays on the wave-2 pin
#     (c6d588647aacb2275dfaea41a00fb501);
#   * 720p 6 frames at 2x1/2x2/4x2/8x2 decodes with `dav2d --strict 1` and its
#     decoded planes equal the encoder's --video-recon output sample for
#     sample (the per-tile splitter resets entropy/context state per tile);
#   * every stream passes tools/validate_dav2d.sh;
#   * unsupported combinations are rejected by name (16x2, multi-ref,
#     non-eighth-pel precision, frame-type modes).
#
# Usage:
#   src/pipeline/video/gate_cli_video_tiles.sh --bin build-int/av2enc
#   DAV2D=/path/to/dav2d src/pipeline/video/gate_cli_video_tiles.sh --bin ...
#
# Exit codes: 0 pass, 1 failure, 2 usage, 77 missing fixture/decoder.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${HERE}/../../.." && pwd)"
BIN=""
DAV2D="${DAV2D:-}"

while [ $# -gt 0 ]; do
    case "$1" in
        --bin) BIN=$2; shift 2 ;;
        --dav2d) DAV2D=$2; shift 2 ;;
        -h|--help)
            sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "gate_cli_video_tiles: unknown argument $1" >&2; exit 2 ;;
    esac
done

if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
    echo "gate_cli_video_tiles: --bin must point at an av2enc executable" >&2
    exit 2
fi
if [ -z "$DAV2D" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
    done
fi
if [ -z "$DAV2D" ] || [ ! -x "$DAV2D" ]; then
    echo "gate_cli_video_tiles: no dav2d binary (set DAV2D=...)" >&2
    exit 77
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "gate_cli_video_tiles: python3 is required for the recon comparison" >&2
    exit 77
fi

INPUT="$ROOT/testdata/shine_1280x720_60f.y4m"
if [ ! -f "$INPUT" ]; then
    echo "SKIP: missing $INPUT" >&2
    exit 77
fi

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
export DAV2D

md5_of() {
    if command -v md5 >/dev/null 2>&1; then
        md5 -q "$1"
    else
        md5sum "$1" | awk '{print $1}'
    fi
}

failures=0
pass=0
fail() {
    echo "FAIL: $*" >&2
    failures=$((failures + 1))
}
ok() {
    echo "PASS $*"
    pass=$((pass + 1))
}

compare_recon() {
    python3 - "$1" "$2" "$3" <<'PY'
import sys


def read_y4m(path):
    frames = []
    with open(path, "rb") as f:
        header = f.readline()
        if not header.startswith(b"YUV4MPEG2 "):
            raise SystemExit(f"{path}: not a Y4M file")
        width = height = None
        for field in header.split()[1:]:
            if field.startswith(b"W"):
                width = int(field[1:])
            elif field.startswith(b"H"):
                height = int(field[1:])
        if width is None or height is None:
            raise SystemExit(f"{path}: missing dimensions")
        cw = (width + 1) // 2
        ch = (height + 1) // 2
        size = width * height + 2 * cw * ch
        while True:
            line = f.readline()
            if not line:
                break
            if not line.startswith(b"FRAME"):
                raise SystemExit(f"{path}: bad frame marker")
            data = f.read(size)
            if len(data) != size:
                raise SystemExit(f"{path}: truncated frame")
            frames.append(data)
    return frames


expected = read_y4m(sys.argv[1])
decoded = read_y4m(sys.argv[2])
if len(expected) != len(decoded):
    raise SystemExit(
        f"frame count mismatch: encoder recon {len(expected)} vs decoder {len(decoded)}")
diffs = 0
first = None
for index, (a, b) in enumerate(zip(expected, decoded)):
    if a == b:
        continue
    for pos, (x, y) in enumerate(zip(a, b)):
        if x != y:
            diffs += 1
            if first is None:
                first = (index, pos, x, y)
if diffs:
    print(f"  first diff: frame={first[0]} offset={first[1]} "
          f"dec={first[2]} enc={first[3]}")
    raise SystemExit(1)
print(f"  recon compare {sys.argv[3]}: {len(expected)} frame(s), 0 sample diff(s)")
PY
}

COMMON=(--video --frames 6 --video-max-depth 3)

# --- default stream is still the wave-2 pin ------------------------------
"$BIN" "$INPUT" -o "$TMP/base.ivf" "${COMMON[@]}" >/dev/null
base_md5=$(md5_of "$TMP/base.ivf")
if [ "$base_md5" != "c6d588647aacb2275dfaea41a00fb501" ]; then
    fail "default --video MD5 moved: $base_md5"
else
    ok "default --video 720p pin $base_md5"
fi

# --- tiled layouts: strict decode + sample-exact recon -------------------
for LAYOUT in 2x1 2x2 4x2 8x2; do
    IVF="$TMP/tiles_${LAYOUT}.ivf"
    RECON="$TMP/tiles_${LAYOUT}.recon.y4m"
    DEC="$TMP/tiles_${LAYOUT}.decoded.y4m"
    LOG="$TMP/tiles_${LAYOUT}.log"
    if ! "$BIN" "$INPUT" -o "$IVF" "${COMMON[@]}" --tiles "$LAYOUT" \
            --video-recon "$RECON" >"$LOG" 2>&1; then
        cat "$LOG" >&2
        fail "--tiles $LAYOUT: encode failed"
        continue
    fi
    if ! "$ROOT/tools/validate_dav2d.sh" "$IVF" >"$TMP/validate_${LAYOUT}.log" 2>&1; then
        cat "$TMP/validate_${LAYOUT}.log" >&2
        fail "--tiles $LAYOUT: validate_dav2d failed"
        continue
    fi
    if ! "$DAV2D" -q -i "$IVF" -o "$DEC" --strict 1 --threads 1 2>"$LOG"; then
        cat "$LOG" >&2
        fail "--tiles $LAYOUT: dav2d --strict 1 failed"
        continue
    fi
    if ! compare_recon "$RECON" "$DEC" "$LAYOUT"; then
        fail "--tiles $LAYOUT: decoded output differs from the encoder recon"
        continue
    fi
    ok "--tiles $LAYOUT: 6 frames strict-decoded, recon exact"
done

# --- rejections ----------------------------------------------------------
expect_reject() {
    local label=$1
    shift
    if "$BIN" "$INPUT" -o "$TMP/reject.ivf" "$@" >"$TMP/reject.log" 2>&1; then
        fail "$label: expected a rejection"
        return
    fi
    if ! grep -qi "error" "$TMP/reject.log"; then
        fail "$label: rejection did not print an error"
        return
    fi
    ok "$label: rejected"
}

expect_reject "--tiles 16x2" "${COMMON[@]}" --tiles 16x2
expect_reject "--tiles + --video-refs 2" "${COMMON[@]}" --tiles 2x2 \
    --video-refs 2
expect_reject "--tiles + quarter-pel" "${COMMON[@]}" --tiles 2x2 \
    --video-mv-precision quarter
expect_reject "--tiles + --video-sef" "${COMMON[@]}" --tiles 2x2 --video-sef

echo
if [ "$failures" -ne 0 ]; then
    echo "FAIL: $failures failure(s), $pass passed" >&2
    exit 1
fi
echo "PASS: video tiles ($pass cases)"
