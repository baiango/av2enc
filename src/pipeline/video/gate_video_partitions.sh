#!/usr/bin/env bash
# Product gate for the video partition tools:
#   * --video-sb128  (§5.4.3 use_128x128_superblock)
#   * --video-sb256  (§5.4.3 use_256x256_superblock; inter frames only)
#   * --video-ext-partitions (enable_ext_partitions + uneven 4-way)
#
# Every case must:
#   1. encode without error (picture count printed by the encoder matches),
#   2. decode with the pinned dav2d under --strict 1,
#   3. produce one decoded frame per input frame, and
#   4. have --video-recon byte-identical to the decoder's planes.
#
# Default 64x64 streams are covered by the pinned MD5s at the top.
#
# Usage:
#   DAV2D=/path/to/dav2d src/pipeline/video/gate_video_partitions.sh [build-dir]
#   src/pipeline/video/gate_video_partitions.sh --bin build/av2enc [--decoder dav2d]
#
# Env: DAV2D, KEEP=1 to keep the artifacts.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
BUILD=$ROOT/build-int
BIN=""
DECODER=${DAV2D:-}
while [ $# -gt 0 ]; do
    case "$1" in
        --bin) BIN=${2:-}; shift 2 ;;
        --decoder) DECODER=${2:-}; shift 2 ;;
        --build) BUILD=${2:-}; shift 2 ;;
        -h|--help) sed -n '2,12p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) BUILD=$1; shift ;;
    esac
done
if [ -z "$BIN" ]; then BIN=$BUILD/av2enc; fi
if [ -z "$DECODER" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DECODER=$candidate; break; fi
    done
fi
if [ ! -x "$BIN" ]; then echo "gate: no encoder at $BIN" >&2; exit 2; fi
if [ -z "$DECODER" ] || [ ! -x "$DECODER" ]; then
    echo "gate: SKIP (exit 77): no dav2d (set DAV2D)" >&2
    exit 77
fi

TMP=$(mktemp -d)
trap 'if [ "${KEEP:-0}" != "1" ]; then rm -rf "$TMP"; fi' EXIT
TESTDATA=$ROOT/testdata
FAIL=0
CASES=0

plane_compare() {  # recon.y4m decoded.y4m
    python3 - "$1" "$2" <<'PY'
import sys
def frames(p):
    b=open(p,'rb').read()
    i=b.index(b'\n'); hdr=b[:i].split(); i+=1
    w=int([x for x in hdr if x.startswith(b'W')][0][1:])
    h=int([x for x in hdr if x.startswith(b'H')][0][1:])
    n=w*h*3//2; out=[]
    while i < len(b):
        i=b.index(b'\n',i)+1
        out.append(b[i:i+n]); i+=n
    return out
a=frames(sys.argv[1]); d=frames(sys.argv[2])
if len(a)!=len(d):
    print(f"frame count {len(a)} != {len(d)}"); sys.exit(1)
if a!=d:
    print("reconstruction differs from decoded picture"); sys.exit(1)
print(f"{len(a)} frame(s), recon==decoded")
PY
}

case_run() {  # name source expected_frames extra...
    local name=$1 source=$2 expect=$3; shift 3
    local ivf=$TMP/$name.ivf recon=$TMP/$name.recon.y4m dec=$TMP/$name.dec.y4m
    local log=$TMP/$name.log
    CASES=$((CASES+1))
    if ! "$BIN" "$source" -o "$ivf" --video-recon "$recon" "$@" >"$log" 2>&1; then
        echo "FAIL $name: encode"; tail -5 "$log"; FAIL=$((FAIL+1)); return
    fi
    local got
    got=$(sed -n 's/.*frames: *\([0-9][0-9]*\) temporal unit(s), \([0-9][0-9]*\) coded frame(s).*/\2/p' "$log")
    if [ "$got" != "$expect" ]; then
        echo "FAIL $name: coded $got frame(s), expected $expect"; FAIL=$((FAIL+1)); return
    fi
    if ! "$DECODER" -i "$ivf" -o "$dec" --threads 1 --strict 1 >"$TMP/$name.dec.log" 2>&1; then
        echo "FAIL $name: dav2d rejected the stream"; tail -3 "$TMP/$name.dec.log"
        FAIL=$((FAIL+1)); return
    fi
    local cmp
    if ! cmp=$(plane_compare "$recon" "$dec"); then
        echo "FAIL $name: $cmp"; FAIL=$((FAIL+1)); return
    fi
    echo "PASS $name: $cmp"
}

# --- 128x128 superblocks --------------------------------------------------
case_run sb128_360 "$TESTDATA/shine_640x360_1f.y4m" 1 \
    --video --frames 1 --qp 90 --video-sb128
case_run sb128_360_4f "$TESTDATA/shine_640x360_4f.y4m" 4 \
    --video --frames 4 --qp 90 --video-sb128
case_run sb128_360_single "$TESTDATA/shine_640x360_4f.y4m" 3 \
    --video --frames 3 --qp 90 --video-sb128 --video-force-single
case_run sb128_360_deblock "$TESTDATA/shine_640x360_1f.y4m" 1 \
    --video --frames 1 --qp 90 --video-sb128 --video-deblock
case_run sb128_256 "$TESTDATA/shine_256x256_1f.y4m" 1 \
    --video --frames 1 --qp 90 --video-sb128
case_run sb128_720 "$TESTDATA/shine_1280x720_60f.y4m" 2 \
    --video --frames 2 --qp 90 --video-sb128 --video-deblock
case_run sb128_1080 "$TESTDATA/shine_1920x1080_10f.y4m" 1 \
    --video --frames 1 --qp 90 --video-sb128

# --- 256x256 superblocks (inter frames; intra keys clamp to 128) ----------
case_run sb256_360 "$TESTDATA/shine_640x360_1f.y4m" 1 \
    --video --frames 1 --qp 90 --video-sb256
case_run sb256_360_3f "$TESTDATA/shine_640x360_4f.y4m" 3 \
    --video --frames 3 --qp 90 --video-sb256
case_run sb256_360_single "$TESTDATA/shine_640x360_4f.y4m" 3 \
    --video --frames 3 --qp 90 --video-sb256 --video-force-single
case_run sb256_360_deblock "$TESTDATA/shine_640x360_4f.y4m" 2 \
    --video --frames 2 --qp 90 --video-sb256 --video-deblock
case_run sb256_256 "$TESTDATA/shine_256x256_1f.y4m" 1 \
    --video --frames 1 --qp 90 --video-sb256
case_run sb256_1080 "$TESTDATA/shine_1920x1080_10f.y4m" 1 \
    --video --frames 1 --qp 90 --video-sb256

# --- extended / uneven 4-way partitions -----------------------------------
case_run ext_360 "$TESTDATA/shine_640x360_1f.y4m" 1 \
    --video --frames 1 --qp 90 --video-ext-partitions
case_run ext_360_4f "$TESTDATA/shine_640x360_4f.y4m" 4 \
    --video --frames 4 --qp 90 --video-ext-partitions
case_run ext_360_deblock "$TESTDATA/shine_640x360_1f.y4m" 1 \
    --video --frames 1 --qp 90 --video-ext-partitions --video-deblock
case_run ext_360_min8 "$TESTDATA/shine_640x360_4f.y4m" 3 \
    --video --frames 3 --qp 120 --video-ext-partitions --video-min-block 8
case_run ext_256 "$TESTDATA/shine_256x256_1f.y4m" 1 \
    --video --frames 1 --qp 90 --video-ext-partitions
case_run ext_720 "$TESTDATA/shine_1280x720_60f.y4m" 2 \
    --video --frames 2 --qp 90 --video-ext-partitions
case_run ext_1080 "$TESTDATA/shine_1920x1080_10f.y4m" 1 \
    --video --frames 1 --qp 90 --video-ext-partitions
case_run ext_sb128_360 "$TESTDATA/shine_640x360_1f.y4m" 1 \
    --video --frames 1 --qp 90 --video-ext-partitions --video-sb128
case_run ext_sb128_4f "$TESTDATA/shine_640x360_4f.y4m" 4 \
    --video --frames 4 --qp 90 --video-ext-partitions --video-sb128 \
    --video-deblock

# --- default 64x64 identity (pinned) --------------------------------------
def_md5() {  # name expected source extra...
    local name=$1 want=$2 source=$3; shift 3
    local ivf=$TMP/$name.ivf; CASES=$((CASES+1))
    if ! "$BIN" "$source" -o "$ivf" "$@" >/dev/null 2>&1; then
        echo "FAIL $name: encode"; FAIL=$((FAIL+1)); return
    fi
    local got; got=$(md5 -q "$ivf" 2>/dev/null || md5sum "$ivf" | cut -d' ' -f1)
    if [ "$got" != "$want" ]; then
        echo "FAIL $name: default stream moved ($got != $want)"; FAIL=$((FAIL+1)); return
    fi
    echo "PASS $name: default 64x64 pin $want"
}
def_md5 default_360 ef85e55bd85357f965169051dd691612 \
    "$TESTDATA/shine_640x360_1f.y4m" --video --frames 1 --qp 90
def_md5 default_360_deblock 078c29e0df452c7cad625c718028cd04 \
    "$TESTDATA/shine_640x360_1f.y4m" --video --frames 1 --qp 90 --video-deblock

echo
if [ "$FAIL" -ne 0 ]; then
    echo "video partition gate: FAIL ($FAIL/$CASES)"; exit 1
fi
echo "video partition gate: PASS ($CASES cases, $(basename "$DECODER"))"
