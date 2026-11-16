#!/usr/bin/env bash
# Gate for --video-skip-collapse (opt-in static-superblock partition collapse).
#
# A P-frame superblock whose coded leaves are all plain zero-motion inter skips
# predicts every 4x4 cell by copying the reference at the same position, so one
# superblock-sized skip leaf reproduces exactly the same pixels while coding one
# partition symbol per superblock instead of a tree.  `--video-skip-collapse`
# rewrites those superblocks after the partition solve and re-collects the
# frame; a rewritten leaf that does not come back as a plain zero-motion skip is
# restored.
#
# What a fully-skipped P frame still pays after the rewrite (60 identical
# frames, --nightly): partition 35 symbols / 7.2 bits, is_inter 33 / 0.05 bits,
# skip 33 / 0.04 bits, mode 33 / 0.12 bits -> 7.4 information bits, a 3-byte
# tile.  Before the rewrite: 78 partition symbols / 32.1 bits and a 7-byte
# tile.  The 2-byte SB-level floor (--video-force-single) is the §8.2.4 marker
# plus padding.
#
# Checks:
#   * the named rejections (--video-gpu, --tiles > 1x1, --video-delta-q);
#   * flag off keeps the three pinned streams (default QP189, quality QP190,
#     nightly QP110) byte-identical;
#   * on a 60-frame all-identical clip the flag cuts the steady-state P frame
#     from 21 B to 17 B of IVF framing and shrinks the whole stream;
#   * the flag-on static stream decodes under dav2d --strict 1 and its visible
#     reconstruction equals the flag-off reconstruction byte for byte (the
#     rewrite is reconstruction-preserving on static content);
#   * the flag-on real clip decodes --strict 1, the visible encoder
#     reconstruction equals the decoded payload, and the stream is smaller.
#
# Usage: src/pipeline/video/video_skip_collapse_gate.sh [--bin build/av2enc]
#                                                      [--dav2d PATH]
# Exit: 0 pass, 1 failure, 2 usage, 77 missing fixture/decoder.
set -uo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/../../.." && pwd)
BIN=""
DAV2D="${DAV2D:-}"
TESTDATA="${AV2ENC_TESTDATA:-$ROOT/testdata}"

while [ $# -gt 0 ]; do
    case "$1" in
        --bin) BIN=${2:-}; shift 2 ;;
        --dav2d) DAV2D=${2:-}; shift 2 ;;
        -h|--help) sed -n '2,36p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "video_skip_collapse_gate: unknown argument $1" >&2; exit 2 ;;
    esac
done

BIN=${BIN:-"$ROOT/build/av2enc"}
[ -x "$BIN" ] || {
    echo "video_skip_collapse_gate: --bin must point at av2enc" >&2; exit 2; }
if [ -z "$DAV2D" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
    done
fi
[ -n "$DAV2D" ] && [ -x "$DAV2D" ] || {
    echo "video_skip_collapse_gate: SKIP (exit 77): no dav2d binary" >&2
    exit 77; }

NASA="$TESTDATA/nasa_480x240_30fps_60f.y4m"
[ -f "$NASA" ] || {
    echo "video_skip_collapse_gate: SKIP (exit 77): no $NASA" >&2; exit 77; }

TMP=$(mktemp -d "${TMPDIR:-/tmp}/av2enc-skip-collapse.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
FAILED=0
fail() { echo "FAIL: $*" >&2; FAILED=1; }

QP=110
THREADS=8

echo "== video skip-collapse gate =="

# --- named rejections -----------------------------------------------------
reject() {  # reject <tag> <needle> <args...>
    local tag=$1 needle=$2; shift 2
    if "$BIN" "$NASA" -o "$TMP/$tag.ivf" "$@" >/dev/null 2>"$TMP/$tag.err"; then
        fail "--video-skip-collapse was accepted with $tag"
    fi
    grep -q "$needle" "$TMP/$tag.err" ||
        fail "the $tag rejection did not name the requirement"
}
reject gpu "CPU partition decision" --video --frames 4 --qp "$QP" \
    --video-skip-collapse --video-gpu
reject tiles "single-tile" --video --frames 4 --qp "$QP" \
    --video-skip-collapse --tiles 2x1
reject deltaq "full-SB skip leaf cannot code a delta" --video --frames 4 \
    --qp "$QP" --video-skip-collapse --video-delta-q
[ -e "$TMP/gpu.ivf" ] && fail "a rejected run created its output file"

# --- flag off keeps the three pins ---------------------------------------
md5() { python3 - "$1" <<'PY'
import hashlib, sys
print(hashlib.md5(open(sys.argv[1], 'rb').read()).hexdigest())
PY
}
if ! "$BIN" "$NASA" -o "$TMP/default.ivf" --video --frames 60 --qp 189 \
        --cpu-threads "$THREADS" >"$TMP/default.log" 2>&1; then
    fail "the default QP189 encode failed"
fi
[ "$(wc -c <"$TMP/default.ivf" | tr -d ' ')" = "25009" ] ||
    fail "default QP189 is not 25009 B"
[ "$(md5 "$TMP/default.ivf")" = "a1d0dba1259b20a4b628d4cd1e14c33c" ] ||
    fail "default QP189 pin moved"

if ! "$BIN" "$NASA" -o "$TMP/quality.ivf" --video --frames 60 --qp 190 \
        --cpu-threads "$THREADS" --video-rd-preset quality \
        >"$TMP/quality.log" 2>&1; then
    fail "the quality preset encode failed"
fi
[ "$(wc -c <"$TMP/quality.ivf" | tr -d ' ')" = "25141" ] ||
    fail "quality QP190 is not 25141 B"
[ "$(md5 "$TMP/quality.ivf")" = "3984677b9b665952984b07b7cf75c6ea" ] ||
    fail "quality QP190 pin moved"

if ! "$BIN" "$NASA" -o "$TMP/nightly.ivf" --video --frames 60 --qp "$QP" \
        --cpu-threads "$THREADS" --nightly >"$TMP/nightly.log" 2>&1; then
    fail "the nightly encode failed"
fi
[ "$(wc -c <"$TMP/nightly.ivf" | tr -d ' ')" = "23690" ] ||
    fail "nightly is not 23690 B"
[ "$(md5 "$TMP/nightly.ivf")" = "1b99127a4cd1d386fa5600fb63c57a83" ] ||
    fail "nightly pin moved"

# --- the static floor -----------------------------------------------------
# 60 copies of frame 0: nothing changes, so every P frame is a full skip.
python3 - "$NASA" "$TMP/static.y4m" 60 <<'PY' || fail "static clip build failed"
import sys
src, dst, copies = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(src, 'rb') as f:
    header = f.readline()
    assert f.readline() == b'FRAME\n'
    frame = f.read(480 * 240 * 3 // 2)
with open(dst, 'wb') as f:
    f.write(header)
    for _ in range(copies):
        f.write(b'FRAME\n')
        f.write(frame)
PY

if ! "$BIN" "$TMP/static.y4m" -o "$TMP/static_off.ivf" --video --frames 60 \
        --qp "$QP" --cpu-threads "$THREADS" --nightly \
        --video-recon "$TMP/static_off_recon.y4m" >"$TMP/static_off.log" 2>&1; then
    fail "the static flag-off encode failed"
fi
if ! "$BIN" "$TMP/static.y4m" -o "$TMP/static_on.ivf" --video --frames 60 \
        --qp "$QP" --cpu-threads "$THREADS" --nightly \
        --video-skip-collapse --video-recon "$TMP/static_on_recon.y4m" \
        >"$TMP/static_on.log" 2>&1; then
    fail "the static flag-on encode failed"
fi

per_frame() {  # per_frame <ivf>: prints "<last> <meanP>"
    python3 - "$1" <<'PY'
import struct, sys
data = open(sys.argv[1], 'rb').read()
pos = 32
sizes = []
while pos + 12 <= len(data):
    sizes.append(struct.unpack('<I', data[pos:pos + 4])[0])
    pos += 12 + sizes[-1]
if len(sizes) < 2:
    print("0 0")
    sys.exit(0)
print(f"{sizes[-1]} {sum(sizes[1:]) / (len(sizes) - 1):.2f}")
PY
}

read -r OFF_LAST OFF_MEAN <<<"$(per_frame "$TMP/static_off.ivf")"
read -r ON_LAST ON_MEAN <<<"$(per_frame "$TMP/static_on.ivf")"
echo "static P frame: off=${OFF_MEAN} B on=${ON_MEAN} B (steady ${OFF_LAST} -> ${ON_LAST} B, incl. 12 B IVF framing)"
[ "$OFF_LAST" -ge 13 ] || fail "the flag-off static floor is below 13 B/frame"
[ "$ON_LAST" -le 18 ] ||
    fail "the flag-on static floor is $ON_LAST B/frame, expected <= 12"
[ "$ON_MEAN" \< "$OFF_MEAN" ] || fail "the flag did not shrink the static clip"
[ "$(wc -c <"$TMP/static_on.ivf" | tr -d ' ')" -lt \
  "$(wc -c <"$TMP/static_off.ivf" | tr -d ' ')" ] ||
    fail "the flag-on static stream is not smaller"

# --- decode-exactness -----------------------------------------------------
check_recon() {  # check_recon <tag> <ivf> <recon>
    if ! "$DAV2D" -q --strict 1 -i "$2" -o "$TMP/$1_dec.y4m" --threads 1 \
            >"$TMP/$1_dec.log" 2>&1; then
        fail "$1 stream failed dav2d --strict 1"
        cat "$TMP/$1_dec.log" >&2
        return 1
    fi
    python3 - "$3" "$TMP/$1_dec.y4m" "$1" <<'PY'
import sys

def payload(path):
    data = open(path, 'rb').read()
    return data[data.index(b'\n') + 1:].replace(b'FRAME\n', b'')

a, b = payload(sys.argv[1]), payload(sys.argv[2])
if len(a) != len(b) or a != b:
    print(f"{sys.argv[3]}: recon {len(a)} B != decoded {len(b)} B", file=sys.stderr)
    sys.exit(1)
print(f"{sys.argv[3]}: visible recon matches decoded output ({len(a)} B)")
PY
}

check_recon static_off "$TMP/static_off.ivf" "$TMP/static_off_recon.y4m" ||
    fail "the static flag-off stream is not recon-exact"
check_recon static_on "$TMP/static_on.ivf" "$TMP/static_on_recon.y4m" ||
    fail "the static flag-on stream is not recon-exact"

# On fully static content the rewrite must not touch one pixel.
python3 - "$TMP/static_off_recon.y4m" "$TMP/static_on_recon.y4m" <<'PY'
import sys

def payload(path):
    data = open(path, 'rb').read()
    return data[data.index(b'\n') + 1:].replace(b'FRAME\n', b'')

off, on = payload(sys.argv[1]), payload(sys.argv[2])
if off != on:
    diff = sum(1 for x, y in zip(off, on) if x != y)
    print(f"static recon differs in {diff} byte(s)", file=sys.stderr)
    sys.exit(1)
print(f"static recon is unchanged by the collapse ({len(on)} B)")
PY
[ $? -eq 0 ] || fail "the collapse changed the static reconstruction"

# --- the real clip --------------------------------------------------------
if ! "$BIN" "$NASA" -o "$TMP/real_on.ivf" --video --frames 60 --qp "$QP" \
        --cpu-threads "$THREADS" --nightly --video-skip-collapse \
        --video-recon "$TMP/real_on_recon.y4m" >"$TMP/real_on.log" 2>&1; then
    fail "the real-clip flag-on encode failed"
fi
check_recon real_on "$TMP/real_on.ivf" "$TMP/real_on_recon.y4m" ||
    fail "the real-clip flag-on stream is not recon-exact"
REAL_ON=$(wc -c <"$TMP/real_on.ivf" | tr -d ' ')
echo "real clip: pin=23690 B collapse=$REAL_ON B"
[ "$REAL_ON" -lt 23690 ] ||
    fail "the real-clip flag-on stream is not smaller than the pin"

# --- symbol cost meter ----------------------------------------------------
# AV2ENC_SYMBOL_COST=1 turns on the diagnostic table.  It must not change a
# single byte of the stream.
if ! AV2ENC_SYMBOL_COST=1 "$BIN" "$TMP/static.y4m" -o "$TMP/meter.ivf" \
        --video --frames 60 --qp "$QP" --cpu-threads "$THREADS" --nightly \
        --video-skip-collapse >"$TMP/meter.log" 2>&1; then
    fail "the metered encode failed"
fi
grep -q "symbol_cost tile_bytes=" "$TMP/meter.log" ||
    fail "AV2ENC_SYMBOL_COST produced no table"
[ "$(md5 "$TMP/meter.ivf")" = "$(md5 "$TMP/static_on.ivf")" ] ||
    fail "the symbol-cost meter changed the stream"
LAST_COST=$(grep "symbol_cost tile_bytes=" "$TMP/meter.log" | tail -1)
echo "steady-state symbols: $LAST_COST"
case "$LAST_COST" in
    *"partition=32/"*) ;;
    *) fail "the steady-state partition symbol count is not 32" ;;
esac

if [ "$FAILED" -ne 0 ]; then
    echo "video_skip_collapse_gate: FAILED" >&2
    exit 1
fi
echo "video_skip_collapse_gate: OK (rejections, pins, static floor, recon, real clip, meter)"
