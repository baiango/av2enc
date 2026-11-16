#!/usr/bin/env bash
# Regenerate the public-domain (NASA) replacements for the retired
# promo-derived fixtures under testdata/.
#
# Sources and licence: docs/testdata-provenance.md ("Approved sources").
# Every recipe below records the exact seek offset, filter chain and frame
# count; the output bytes are part of the fixture identity, so do not "clean
# up" a filter or bump ffmpeg without re-recording the byte/md5 pins in the
# gates that use the fixture.
#
# Usage:
#   tools/make_testdata_pub.sh                # write into $AV2ENC_TESTDATA (testdata/)
#   tools/make_testdata_pub.sh --check        # regenerate to a temp dir and compare
#
# Env: SPACE_SRC, TRAIN_SRC (source clips), AV2ENC_TESTDATA, FFMPEG.
#
# The sources are not redistributed. The training clip supplies the
# small/portrait-ish stills and the 4-frame clip (real 30 fps detail and
# motion); the 8K clip supplies 720p/1080p.
set -euo pipefail

ROOT=$(cd "$(dirname "$(dirname "${BASH_SOURCE[0]}")")" && pwd)
TD=${AV2ENC_TESTDATA:-"$ROOT/testdata"}
FFMPEG=${FFMPEG:-ffmpeg}
SPACE_SRC=${SPACE_SRC:-/Users/arin/pctransfer/First-8K-Video-from-Space_orig.mp4}
TRAIN_SRC=${TRAIN_SRC:-/Users/arin/pctransfer/loral_ohara_training_orig.mp4}

CHECK=0
case "${1:-}" in
    --check) CHECK=1 ;;
    "") ;;
    *) echo "usage: $0 [--check]" >&2; exit 2 ;;
esac

for src in "$SPACE_SRC" "$TRAIN_SRC"; do
    [ -f "$src" ] || { echo "make_testdata_pub: missing source $src" >&2; exit 2; }
done

OUT=$TD
CLEANUP=""
if [ "$CHECK" -eq 1 ]; then
    OUT=$(mktemp -d "${TMPDIR:-/tmp}/pub-fixtures.XXXXXX")
    CLEANUP=$OUT
fi
trap '[ -n "$CLEANUP" ] && rm -rf "$CLEANUP"' EXIT
mkdir -p "$OUT"

# clip <src> <ss> <W> <H> <fps> <frames> <out>
clip() {
    local src=$1 ss=$2 w=$3 h=$4 fps=$5 frames=$6 out=$7
    "$FFMPEG" -hide_banner -loglevel error -nostdin \
        -ss "$ss" -i "$src" -frames:v "$frames" \
        -vf "fps=$fps,scale=$w:$h:in_range=full:out_range=limited,setsar=1,format=yuv420p" \
        -pix_fmt yuv420p -an -f yuv4mpegpipe "$OUT/$out" -y
}

# --- stills: training clip, t=305 s ----------------------------------------
# Native-resolution crops, not whole-frame downscales: the still gates check
# that CDEF / loop restoration / GDF / RDOQ change the stream, and a 64x64
# downscale of a 4K frame is so smooth that every one of them becomes a no-op.
# The three windows nest: 64x64 (2048,576) sits inside 256x256 (1984,416),
# which sits inside 640x360 (1984,384).  The source is limited-range 4:2:0, so
# in_range=limited:out_range=limited is an identity range map (unlike the 8K
# source, which is yuvj420p and needs full->limited).
crop_still() {
    local w=$1 h=$2 x=$3 y=$4 fps=$5 out=$6
    "$FFMPEG" -hide_banner -loglevel error -nostdin \
        -ss 305 -i "$TRAIN_SRC" -frames:v 1 \
        -vf "fps=$fps,crop=$w:$h:$x:$y,scale=$w:$h:in_range=limited:out_range=limited,setsar=1,format=yuv420p" \
        -pix_fmt yuv420p -an -f yuv4mpegpipe "$OUT/$out" -y
}
# 64x64 keeps the retired fixture's F30/1 A1:1 header convention.
crop_still 64 64 2048 576 30 pub_64x64_1f.y4m
crop_still 256 256 1984 416 60 pub_256x256_1f.y4m
crop_still 640 360 1984 384 60 pub_640x360_1f.y4m

# --- 4-frame 360p clip: training clip, t=545 s, native 30 fps ---------------
# Whole-frame downscale so the four pictures cover the scene's motion (a static
# crop window would be nearly inert).  30 fps (not 60) so the four pictures are
# four real source frames; the motion gates require non-zero trajectory/MV
# activity and duplicated frames would make them inert.  t=545 (not the stills'
# 305) because this window is one of the few where BOTH the same-reference
# compound and the IST tool are actually selected: 13 same-ref leaves and 2 IST
# blocks, versus 0 / 0 at 305.  The training source is limited-range, so the
# range map is the identity.
"$FFMPEG" -hide_banner -loglevel error -nostdin \
    -ss 545 -i "$TRAIN_SRC" -frames:v 4 \
    -vf "fps=30,scale=640:360:in_range=limited:out_range=limited,setsar=1,format=yuv420p" \
    -pix_fmt yuv420p -an -f yuv4mpegpipe "$OUT/pub_640x360_4f.y4m" -y

# --- 720p clip: 8K clip, t=55 s (luma entropy 0.93), 60 pictures -----------
clip "$SPACE_SRC" 55 1280 720 60 60 pub_1280x720_60f.y4m

# --- 1080p clip: 8K clip, t=135 s (luma entropy 0.93), 10 pictures ---------
clip "$SPACE_SRC" 135 1920 1080 60 10 pub_1920x1080_10f.y4m

if [ "$CHECK" -eq 1 ]; then
    fail=0
    for f in pub_64x64_1f pub_256x256_1f pub_640x360_1f pub_640x360_4f \
             pub_1280x720_60f pub_1920x1080_10f; do
        if ! cmp -s "$OUT/$f.y4m" "$TD/$f.y4m"; then
            echo "make_testdata_pub: $f.y4m differs from $TD" >&2
            fail=1
        fi
    done
    [ "$fail" -eq 0 ] || exit 1
    echo "make_testdata_pub: $TD is reproducible"
    exit 0
fi

echo "make_testdata_pub: wrote 6 fixtures to $OUT"
for f in pub_64x64_1f pub_256x256_1f pub_640x360_1f pub_640x360_4f \
         pub_1280x720_60f pub_1920x1080_10f; do
    printf '  %-24s %10s B  %s\n' "$f.y4m" "$(wc -c <"$OUT/$f.y4m" | tr -d ' ')" \
        "$(head -1 "$OUT/$f.y4m")"
done
