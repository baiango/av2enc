#!/usr/bin/env bash
# Extended partitions on the nightly chain (--nightly --video-ext-partitions).
#
# 1. Flag-off pins are byte-identical: default QP189 25,009 B, quality QP190
#    25,141 B, nightly QP110 24,113 B.  The composition must not move them.
# 2. The nightly + extended stream decodes all 60 pictures under
#    `dav2d --strict 1` and the decoded payload equals --video-recon byte for
#    byte.  This is the whole chain: carried CDFs (the extended
#    do_ext_partition / do_uneven_4way_partition rows ride the §8.2 image),
#    edge mask, BAWP carrier, drift refresh, deblock + CDEF + CCSO.
# 3. The extended search is actually used and stays inside the safe geometry:
#    the QP120 keyframe tree has HORZ_3 and VERT_3 nodes, no 4-wide child and
#    no vertical uneven node below a 64-wide parent (the documented dav2d
#    bottom-edge desync).  The keyframe is smaller than the rect-only pin.
# 4. The bottom-edge keyframe at --video-min-block 8 (where the 3-way family
#    first becomes reachable at 32x32) decodes one picture with an exact
#    reconstruction.
# 5. Rejections that are still true: --video-sb256, --tiles, B/temporal frame
#    schedules, segmentation/delta-Q and --video-ref-mvs stay named errors.
#
# Usage: gate_video_ext_nightly.sh --bin build/av2enc [--dav2d PATH]
# Exit: 0 pass, 1 fail, 2 usage, 77 skip.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
BIN=""
DAV2D="${DAV2D:-}"
TESTDATA="${AV2ENC_TESTDATA:-$ROOT/testdata}"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

while [ $# -gt 0 ]; do
    case "$1" in
        --bin) BIN=$2; shift 2 ;;
        --dav2d) DAV2D=$2; shift 2 ;;
        -h|--help) sed -n '2,27p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "gate_video_ext_nightly: unknown argument $1" >&2; exit 2 ;;
    esac
done
[ -n "$BIN" ] && [ -x "$BIN" ] || {
    echo "gate_video_ext_nightly: --bin required" >&2; exit 2; }
if [ -z "$DAV2D" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
    done
fi
[ -n "$DAV2D" ] && [ -x "$DAV2D" ] || {
    echo "gate_video_ext_nightly: no dav2d" >&2; exit 77; }

NASA="$TESTDATA/nasa_480x240_30fps_60f.y4m"
[ -f "$NASA" ] || { echo "gate_video_ext_nightly: missing $NASA" >&2; exit 77; }

PIN_DEFAULT_BYTES=25009
PIN_DEFAULT_MD5=a1d0dba1259b20a4b628d4cd1e14c33c
PIN_QUALITY_BYTES=25141
PIN_QUALITY_MD5=3984677b9b665952984b07b7cf75c6ea
PIN_NIGHTLY_BYTES=23690
PIN_NIGHTLY_MD5=1b99127a4cd1d386fa5600fb63c57a83
EXT_Q110_BYTES=23475
EXT_Q110_MD5=001f410d1a97c3a7af686214769f9800
PIN_KEY_BYTES=15928
PIN_KEY_MD5=7394fd90c4c0a008ca8d602301371238
EXT_KEY_BYTES=15632
EXT_KEY_MD5=0a4810f5f452a63bf88784f6d8975282
EXT_KEY8_BYTES=14463
EXT_KEY8_MD5=04e6ef58493a69ea36b3415e4b9001cd

md5_of() { md5 -q "$1" 2>/dev/null || md5sum "$1" | awk '{print $1}'; }

# dav2d exits 0 even after failing pictures, so count the decoded frames and
# compare the payload.
decode_exact() {
    local ivf=$1 recon=$2 expect=$3 tag=$4
    "$DAV2D" --strict 1 -i "$ivf" -o "$TMP/$tag.dec.y4m" \
        >"$TMP/$tag.dec.log" 2>&1 || true
    local got
    got=$(sed -n 's/^Decoded \([0-9]*\)\/\([0-9]*\) frames.*/\1 \2/p' \
              "$TMP/$tag.dec.log" | tail -1 | awk '{print $1}')
    if [ "${got:-0}" != "$expect" ]; then
        echo "FAIL $tag: dav2d decoded ${got:-0}/$expect pictures" >&2
        tail -5 "$TMP/$tag.dec.log" >&2
        exit 1
    fi
    if grep -q "Invalid argument" "$TMP/$tag.dec.log"; then
        echo "FAIL $tag: dav2d reported failed pictures" >&2
        exit 1
    fi
    if ! cmp -s "$recon" "$TMP/$tag.dec.y4m"; then
        echo "FAIL $tag: decoded payload differs from --video-recon" >&2
        exit 1
    fi
}

expect_bytes() {
    local file=$1 bytes=$2 tag=$3
    local got
    got=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file")
    if [ "$got" != "$bytes" ]; then
        echo "FAIL $tag: $got bytes, expected $bytes" >&2
        exit 1
    fi
}

expect_rejected() {
    local tag=$1; shift
    if "$BIN" "$NASA" -o "$TMP/reject.ivf" --video --frames 2 --qp 110 \
            --video-ext-partitions "$@" >"$TMP/reject.log" 2>&1; then
        echo "FAIL $tag: combination was accepted" >&2
        exit 1
    fi
}

# --- 1. flag-off pins ------------------------------------------------------
"$BIN" "$NASA" -o "$TMP/default.ivf" --video --frames 60 --qp 189 \
    >"$TMP/default.log" 2>&1
expect_bytes "$TMP/default.ivf" "$PIN_DEFAULT_BYTES" "default pin"
[ "$(md5_of "$TMP/default.ivf")" = "$PIN_DEFAULT_MD5" ] || {
    echo "FAIL default pin md5" >&2; exit 1; }

"$BIN" "$NASA" -o "$TMP/quality.ivf" --video --frames 60 --qp 190 \
    --video-rd-preset quality >"$TMP/quality.log" 2>&1
expect_bytes "$TMP/quality.ivf" "$PIN_QUALITY_BYTES" "quality pin"
[ "$(md5_of "$TMP/quality.ivf")" = "$PIN_QUALITY_MD5" ] || {
    echo "FAIL quality pin md5" >&2; exit 1; }

"$BIN" "$NASA" -o "$TMP/nightly.ivf" --video --frames 60 --qp 110 \
    --cpu-threads 8 --nightly >"$TMP/nightly.log" 2>&1
expect_bytes "$TMP/nightly.ivf" "$PIN_NIGHTLY_BYTES" "nightly pin"
[ "$(md5_of "$TMP/nightly.ivf")" = "$PIN_NIGHTLY_MD5" ] || {
    echo "FAIL nightly pin md5" >&2; exit 1; }

# --- 2. composed stream ----------------------------------------------------
"$BIN" "$NASA" -o "$TMP/ext.ivf" --video --frames 60 --qp 110 \
    --cpu-threads 8 --nightly --video-ext-partitions \
    --video-recon "$TMP/ext.recon.y4m" >"$TMP/ext.log" 2>&1 || {
    echo "FAIL: --nightly --video-ext-partitions encode" >&2
    cat "$TMP/ext.log" >&2; exit 1; }
expect_bytes "$TMP/ext.ivf" "$EXT_Q110_BYTES" "ext nightly"
[ "$(md5_of "$TMP/ext.ivf")" = "$EXT_Q110_MD5" ] || {
    echo "FAIL ext nightly md5" >&2; exit 1; }
if [ "$EXT_Q110_BYTES" -ge "$PIN_NIGHTLY_BYTES" ]; then
    echo "FAIL: extended stream is not smaller than the pin" >&2
    exit 1
fi
decode_exact "$TMP/ext.ivf" "$TMP/ext.recon.y4m" 60 ext

# --- 3. keyframe tree: composed, extended and inside the safe geometry -----
AV2ENC_PART_TREE="$TMP/ext-key.tree" \
"$BIN" "$NASA" -o "$TMP/ext-key.ivf" --video --frames 1 --qp 120 \
    --cpu-threads 8 --nightly --video-ext-partitions \
    --video-recon "$TMP/ext-key.recon.y4m" >"$TMP/ext-key.log" 2>&1 || {
    echo "FAIL: extended QP120 keyframe encode" >&2
    cat "$TMP/ext-key.log" >&2; exit 1; }
expect_bytes "$TMP/ext-key.ivf" "$EXT_KEY_BYTES" "ext keyframe"
[ "$(md5_of "$TMP/ext-key.ivf")" = "$EXT_KEY_MD5" ] || {
    echo "FAIL ext keyframe md5" >&2; exit 1; }
decode_exact "$TMP/ext-key.ivf" "$TMP/ext-key.recon.y4m" 1 ext-key

"$BIN" "$NASA" -o "$TMP/base-key.ivf" --video --frames 1 --qp 120 \
    --cpu-threads 8 --nightly >"$TMP/base-key.log" 2>&1
expect_bytes "$TMP/base-key.ivf" "$PIN_KEY_BYTES" "rect QP120 keyframe"
[ "$(md5_of "$TMP/base-key.ivf")" = "$PIN_KEY_MD5" ] || {
    echo "FAIL rect QP120 keyframe md5" >&2; exit 1; }

horz3=$(grep -c "chosen=HORZ_3" "$TMP/ext-key.tree" || true)
vert3=$(grep -c "chosen=VERT_3" "$TMP/ext-key.tree" || true)
if [ "$horz3" -lt 1 ] || [ "$vert3" -lt 1 ]; then
    echo "FAIL: extended keyframe tree has no HORZ_3/VERT_3 node" >&2
    grep "chosen=" "$TMP/ext-key.tree" | awk '{print $9}' | sort | uniq -c >&2
    exit 1
fi
# The 4-wide 1/8 vertical child is a documented dav2d bottom-edge desync.  The
# encoder keeps the vertical uneven family to w >= 64; a 4-wide leaf must never
# appear in the emitted tree.
if grep -Eq " (w|h)=4 " "$TMP/ext-key.tree"; then
    echo "FAIL: extended tree coded a 4-wide/4-high child" >&2
    exit 1
fi

# --- 4. bottom-edge min-block 8 keyframe -----------------------------------
AV2ENC_PART_TREE="$TMP/ext-key8.tree" \
"$BIN" "$NASA" -o "$TMP/ext-key8.ivf" --video --frames 1 --qp 120 \
    --cpu-threads 8 --nightly --video-ext-partitions --video-min-block 8 \
    --video-recon "$TMP/ext-key8.recon.y4m" >"$TMP/ext-key8.log" 2>&1 || {
    echo "FAIL: extended min-block 8 keyframe encode" >&2
    cat "$TMP/ext-key8.log" >&2; exit 1; }
decode_exact "$TMP/ext-key8.ivf" "$TMP/ext-key8.recon.y4m" 1 ext-key8
expect_bytes "$TMP/ext-key8.ivf" "$EXT_KEY8_BYTES" "ext keyframe min-block 8"
[ "$(md5_of "$TMP/ext-key8.ivf")" = "$EXT_KEY8_MD5" ] || {
    echo "FAIL ext keyframe min-block 8 md5" >&2; exit 1; }
if grep -q "chosen=VERT_4A\|chosen=VERT_4B" "$TMP/ext-key8.tree"; then
    # Only legal when the 64-wide parent keeps the 1/8 subsize at 8 pixels.
    awk '$9 ~ /^chosen=VERT_4/ && $4 != "w=64" { bad=1 } END { exit bad }' \
        "$TMP/ext-key8.tree" || {
        echo "FAIL: vertical uneven node below a 64-wide parent" >&2
        exit 1; }
fi

# --- 4b. 16-row bottom superblock (the 720p bottom-edge case) --------------
# 80 rows = one 64-row superblock row plus a 16-row partial row.  This is the
# geometry that has no legal PARTITION_NONE and made the extended solver fail
# before it mirrored the rect solver's boundary relax.
python3 - "$TMP/bottom80.y4m" <<'PY'
import sys
w, h = 128, 80
hdr = b"YUV4MPEG2 W%d H%d F30:1 Ip A1:1 C420jpeg\n" % (w, h)
size = (w * h * 3) // 2
with open(sys.argv[1], "wb") as f:
    f.write(hdr)
    for frame in range(2):
        plane = bytearray(size)
        for y in range(h):
            for x in range(w):
                plane[y * w + x] = (x * 3 + y * 2 + frame * 7) & 0xFF
        for y in range(h // 2):
            for x in range(w // 2):
                plane[w * h + y * (w // 2) + x] = 96 + (x + y) % 32
                plane[w * h + (w * h) // 4 + y * (w // 2) + x] = 96
        f.write(b"FRAME\n")
        f.write(bytes(plane))
PY
"$BIN" "$TMP/bottom80.y4m" -o "$TMP/bottom80.ivf" --video --frames 2 \
    --qp 110 --cpu-threads 8 --nightly --video-ext-partitions \
    --video-recon "$TMP/bottom80.recon.y4m" >"$TMP/bottom80.log" 2>&1 || {
    echo "FAIL: 16-row bottom superblock encode" >&2
    cat "$TMP/bottom80.log" >&2; exit 1; }
decode_exact "$TMP/bottom80.ivf" "$TMP/bottom80.recon.y4m" 2 bottom80

# --- 5. rejections that remain true ---------------------------------------
expect_rejected "ext+sb256" --video-sb256
expect_rejected "ext+tiles" --tiles 2x1
expect_rejected "ext+b-frames" --b-frames 2
expect_rejected "ext+ref-mvs" --video-ref-mvs
expect_rejected "ext+segmentation" --video-segmentation

echo "PASS ext nightly: $EXT_Q110_BYTES B, 60/60 pictures, recon exact"
echo "PASS ext keyframe: $EXT_KEY_BYTES B (rect $PIN_KEY_BYTES B), 1 picture"
echo "PASS keyframe tree: HORZ_3=$horz3 VERT_3=$vert3, no 4-wide child"
echo "PASS min-block 8 bottom edge, 16-row bottom SB and rejections"
