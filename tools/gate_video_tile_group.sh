#!/usr/bin/env bash
# Gate for the per-frame §5.19 tile_group_obu() header surface:
#
#   * the flag-off pins stay byte-identical (25,009 / 25,141 / 24,885);
#   * `--video-tile-group-compact` (§5.4.2 seq_tile_info_present_flag = 1 with
#     allow_tile_info_change = 0) codes zero per-frame tile bits: every frame's
#     header body is exactly 3 bits shorter than the flag-off stream, and the
#     measured 220 bits of §5.19 byte_alignment() padding are unchanged;
#   * the compact stream is 40 bytes smaller than the nightly pin at 60 frames
#     (23 bytes of frame-header payload out, 1 byte in for the sequence header);
#   * `dav2d --strict 1` decodes both streams, the picture count is 60/60 and the
#     decoded payload is byte-identical (a header-only change: the tile payloads
#     are untouched), and the encoder reconstruction equals the decode;
#   * the flag is rejected with --tiles > 1x1 and with the B/TIP/frame-type
#     wrappers (they build their own sequence header).
#
# The bit-level check reads the AV2ENC_TRACE_FRAME_OBU trace
# (`tile_group_header body_bits=<n> pad_bits=<p>`), so a change that spends the
# alignment padding instead of removing bits fails here instead of passing on a
# smaller file.
#
# Usage: tools/gate_video_tile_group.sh [--bin build/av2enc] [--dav2d PATH]
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
        -h|--help) sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "gate_video_tile_group: unknown argument $1" >&2; exit 2 ;;
    esac
done

BIN=${BIN:-"$ROOT/build/av2enc"}
[ -x "$BIN" ] || { echo "gate_video_tile_group: --bin must point at av2enc" >&2; exit 2; }
if [ -z "$DAV2D" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
    done
fi
[ -n "$DAV2D" ] && [ -x "$DAV2D" ] || {
    echo "gate_video_tile_group: SKIP (exit 77): no dav2d binary" >&2; exit 77; }
command -v python3 >/dev/null 2>&1 || {
    echo "gate_video_tile_group: SKIP (exit 77): no python3" >&2; exit 77; }

NASA="$TESTDATA/nasa_480x240_30fps_60f.y4m"
PUB_CLIP="$TESTDATA/pub_640x360_1f.y4m"
if [ ! -f "$NASA" ] || [ ! -f "$PUB_CLIP" ]; then
    echo "gate_video_tile_group: SKIP (exit 77): missing testdata clips" >&2
    exit 77
fi

TMP=$(mktemp -d "${TMPDIR:-/tmp}/av2enc-tile-group.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
FAILED=0
fail() { echo "FAIL: $*" >&2; FAILED=1; }

md5_of() { md5 -q "$1" 2>/dev/null || md5sum "$1" | cut -d' ' -f1; }
size_of() { wc -c <"$1" | tr -d ' '; }

echo "== tile_group_obu header gate =="

# --- flag-off pins --------------------------------------------------------
"$BIN" "$NASA" -o "$TMP/default.ivf" --video --frames 60 --qp 189 \
    --cpu-threads 8 >"$TMP/default.log" 2>&1 ||
    fail "the flag-less baseline encode failed"
[ "$(size_of "$TMP/default.ivf")" = "25009" ] ||
    fail "default stream is $(size_of "$TMP/default.ivf") B, expected 25009"
[ "$(md5_of "$TMP/default.ivf")" = "a1d0dba1259b20a4b628d4cd1e14c33c" ] ||
    fail "default stream md5 drifted"

"$BIN" "$NASA" -o "$TMP/quality.ivf" --video --frames 60 --qp 190 \
    --cpu-threads 8 --video-rd-preset quality >"$TMP/quality.log" 2>&1 ||
    fail "the quality preset encode failed"
[ "$(size_of "$TMP/quality.ivf")" = "25141" ] ||
    fail "quality stream is $(size_of "$TMP/quality.ivf") B, expected 25141"
[ "$(md5_of "$TMP/quality.ivf")" = "3984677b9b665952984b07b7cf75c6ea" ] ||
    fail "quality stream md5 drifted"

AV2ENC_TRACE_FRAME_OBU=1 "$BIN" "$NASA" -o "$TMP/base.ivf" --video \
    --frames 60 --qp 110 --cpu-threads 8 --nightly --video-tile-group-off \
    >"$TMP/base.log" 2>"$TMP/base.trace" ||
    fail "the nightly pin encode failed"
[ "$(size_of "$TMP/base.ivf")" = "23735" ] ||
    fail "nightly stream is $(size_of "$TMP/base.ivf") B, expected 23735"
[ "$(md5_of "$TMP/base.ivf")" = "6741f4589cf5674047b9b9bb8badb513" ] ||
    fail "nightly stream md5 drifted"

# --- compact stream -------------------------------------------------------
AV2ENC_TRACE_FRAME_OBU=1 "$BIN" "$NASA" -o "$TMP/compact.ivf" --video \
    --frames 60 --qp 110 --cpu-threads 8 --nightly --video-tile-group-compact \
    --video-recon "$TMP/compact_recon.y4m" \
    >"$TMP/compact.log" 2>"$TMP/compact.trace" ||
    fail "the --video-tile-group-compact encode failed"
[ "$(size_of "$TMP/compact.ivf")" = "23690" ] ||
    fail "compact stream is $(size_of "$TMP/compact.ivf") B, expected 23690"
[ "$(md5_of "$TMP/compact.ivf")" = "1b99127a4cd1d386fa5600fb63c57a83" ] ||
    fail "compact stream md5 drifted"

python3 - "$TMP/base.trace" "$TMP/compact.trace" <<'PY'
import sys

def headers(path):
    out = []
    for line in open(path):
        parts = line.split()
        if parts and parts[0] == 'tile_group_header':
            fields = dict(p.split('=') for p in parts[1:])
            out.append((int(fields['body_bits']), int(fields['pad_bits'])))
    return out

base = headers(sys.argv[1])
compact = headers(sys.argv[2])
errors = []
if len(base) != 60 or len(compact) != 60:
    errors.append(f"expected 60 frames in both traces, got {len(base)}/{len(compact)}")
for index, (before, after) in enumerate(zip(base, compact)):
    if after[0] != before[0] - 3:
        errors.append(f"frame {index}: body {after[0]} bits, expected {before[0] - 3}")
base_pad = sum(p for _, p in base)
compact_pad = sum(p for _, p in compact)
# The sequence header eats the 4 new bits and the frames lose 3 bits each, so
# the padding shrinks by at most 3*60 bits plus one byte of sequence-header
# alignment (measured 188 bits on the current pin).
if not 0 <= base_pad - compact_pad <= 3 * 60 + 8:
    errors.append(f"alignment padding {compact_pad} outside the legal range")
print(f"header body bits: {sum(b for b, _ in base)} -> "
      f"{sum(b for b, _ in compact)} ({len(base)} frames, -3 bits each)")
print(f"alignment padding: {base_pad} -> {compact_pad} bits")
if errors:
    for error in errors:
        print("FAIL:", error, file=sys.stderr)
    sys.exit(1)
PY
[ $? -eq 0 ] || fail "the per-frame tile bits are not the expected 3-bit removal"

# --- decode: strict, picture count, decoded payload ----------------------
"$DAV2D" -q --strict 1 -i "$TMP/base.ivf" -o "$TMP/base.y4m" --threads 1 \
    >"$TMP/base.dec" 2>&1 || fail "the nightly pin failed dav2d --strict 1"
"$DAV2D" -q --strict 1 -i "$TMP/compact.ivf" -o "$TMP/compact.y4m" --threads 1 \
    >"$TMP/compact.dec" 2>&1 || fail "the compact stream failed dav2d --strict 1"

python3 - "$TMP/base.y4m" "$TMP/compact.y4m" "$TMP/compact_recon.y4m" <<'PY'
import sys

def payload(path):
    data = open(path, 'rb').read()
    nl = data.index(b'\n') + 1
    return data[nl:].replace(b'FRAME\n', b''), data.count(b'FRAME')

base, base_frames = payload(sys.argv[1])
compact, compact_frames = payload(sys.argv[2])
recon, _ = payload(sys.argv[3])
errors = []
if base_frames != 60 or compact_frames != 60:
    errors.append(f"decoded pictures {base_frames}/{compact_frames}, expected 60/60")
if len(compact) != len(base):
    errors.append(f"decoded payload {len(compact)} B != {len(base)} B")
elif compact != base:
    errors.append("decoded payload differs from the flag-off pin")
if recon != compact:
    errors.append("encoder reconstruction differs from the decoded output")
print(f"decoded pictures: {compact_frames}/60, payload {len(compact)} B, "
      f"identical to the flag-off pin, recon matches")
for error in errors:
    print("FAIL:", error, file=sys.stderr)
sys.exit(1 if errors else 0)
PY
[ $? -eq 0 ] || fail "the compact stream lost decoder/recon evidence"

# --- rejections -----------------------------------------------------------
if "$BIN" "$PUB_CLIP" -o "$TMP/tiles.ivf" --video --frames 1 --qp 190 \
        --tiles 2x1 --video-tile-group-compact >/dev/null 2>"$TMP/tiles.err"; then
    fail "--video-tile-group-compact was accepted with --tiles 2x1"
fi
grep -q "gated on a single tile" "$TMP/tiles.err" ||
    fail "the multi-tile rejection did not name the single-tile contract"
if "$BIN" "$PUB_CLIP" -o "$TMP/bframes.ivf" --video --frames 2 --qp 190 \
        --b-frames 1 --video-tile-group-compact >/dev/null 2>"$TMP/bframes.err"; then
    fail "--video-tile-group-compact was accepted with --b-frames"
fi
grep -q "plain key+P chain" "$TMP/bframes.err" ||
    fail "the B-frame rejection did not name the plain key+P chain"

# --- geometry sweep: the reuse condition must hold at every SbSize --------
GEN=""
for candidate in "$ROOT/build-t/tests/gen_y4m" "$ROOT/build/tests/gen_y4m" \
                 "$ROOT/build-m1/tests/gen_y4m"; do
    if [ -x "$candidate" ]; then GEN=$candidate; break; fi
done
if [ -n "$GEN" ]; then
    # 100x70 is the MI-edge geometry, 102x72 exercises the §5.18.1 intra SB
    # clamp against a 256 sequence SB, 640x362 a 128 sequence SB with a
    # partial bottom superblock row.
    for case in "100x70:" "102x72:--video-sb256" "640x362:--video-sb128"; do
        clip=${case%%:*}; extra=${case#*:}
        w=${clip%x*}; h=${clip#*x}
        "$GEN" "$w" "$h" "$TMP/one.y4m" >/dev/null 2>&1 || continue
        python3 - "$TMP/one.y4m" <<'PY'
import sys
path = sys.argv[1]
data = open(path, 'rb').read()
nl = data.index(b'\n') + 1
open(path, 'wb').write(data[:nl] + data[nl:] * 4)
PY
        tag=$(printf '%s%s' "$clip" "$extra" | tr -cd 'a-z0-9')
        "$BIN" "$TMP/one.y4m" -o "$TMP/gb-$tag.ivf" --video --frames 4 \
            --qp 180 --cpu-threads 8 $extra >/dev/null 2>&1 ||
            { fail "geometry $clip $extra: flag-off encode failed"; continue; }
        "$BIN" "$TMP/one.y4m" -o "$TMP/gc-$tag.ivf" --video --frames 4 \
            --qp 180 --cpu-threads 8 $extra --video-tile-group-compact \
            >/dev/null 2>&1 ||
            { fail "geometry $clip $extra: compact encode failed"; continue; }
        "$DAV2D" -q --strict 1 -i "$TMP/gb-$tag.ivf" -o "$TMP/gb.y4m" \
            --threads 1 >/dev/null 2>&1 ||
            { fail "geometry $clip $extra: baseline decode failed"; continue; }
        "$DAV2D" -q --strict 1 -i "$TMP/gc-$tag.ivf" -o "$TMP/gc.y4m" \
            --threads 1 >/dev/null 2>&1 ||
            { fail "geometry $clip $extra: compact decode failed"; continue; }
        cmp -s "$TMP/gb.y4m" "$TMP/gc.y4m" ||
            fail "geometry $clip $extra: decoded output differs"
    done
else
    echo "geometry sweep skipped (no gen_y4m binary)"
fi

if [ "$FAILED" -ne 0 ]; then
    echo "gate_video_tile_group: FAILED" >&2
    exit 1
fi
echo "gate_video_tile_group: OK (pins, -3 bits/frame, -22 B, strict decode, recon, rejections)"
exit 0
