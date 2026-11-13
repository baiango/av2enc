#!/usr/bin/env bash
# Product gate for §5.20.3.1 sub-8 luma leaves and shared-chroma ownership on
# the CPU --video writer (--video-min-block 4).
#
# Every case must:
#   1. encode without error and report the expected coded picture count,
#   2. actually select a sub-8 leaf (4xN / Nx4) when EXPECT_SUB8 is set,
#   3. decode with the pinned dav2d under --strict 1,
#   4. produce one decoded frame per input frame, and
#   5. have --video-recon byte-identical to the decoder's planes.
#
# The gate runs the rect tree, the extended/uneven families, SDP, the full
# intra search, deblocking, 128x128 superblocks and tiling at 64x64 and 640x360,
# depths 5 and 6.  `--video-min-block 8` cases are regression controls: they
# must stay exact and must not select a sub-8 leaf.
#
# Named rejections pinned here: the CLI range, and the pre-existing
# --video-sdp + --video-intra-full combination (not decoder-exact at any
# min-block, see docs/negative-results.md).
#
# Usage:
#   DAV2D=/path/to/dav2d src/pipeline/video/gate_video_sub8.sh [build-dir]
#   src/pipeline/video/gate_video_sub8.sh --bin build/av2enc [--decoder dav2d]
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
        -h|--help) sed -n '2,26p' "${BASH_SOURCE[0]}"; exit 0 ;;
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
    local trace=()
    if [ "${EXPECT_SUB8:-0}" = 1 ] || [ "${EXPECT_NO_SUB8:-0}" = 1 ]; then
        trace=(--video-trace)
    fi
    if ! "$BIN" "$source" -o "$ivf" --video-recon "$recon" \
            "${trace[@]}" "$@" >"$log" 2>&1; then
        echo "FAIL $name: encode"; tail -5 "$log"; FAIL=$((FAIL+1)); return
    fi
    local got
    got=$(sed -n 's/.*frames: *\([0-9][0-9]*\) temporal unit(s), \([0-9][0-9]*\) coded frame(s).*/\2/p' "$log")
    if [ "$got" != "$expect" ]; then
        echo "FAIL $name: coded $got frame(s), expected $expect"; FAIL=$((FAIL+1)); return
    fi
    if [ "${EXPECT_SUB8:-0}" = 1 ]; then
        if ! grep -Eq " (4x[0-9]+|[0-9]+x4) " "$log"; then
            echo "FAIL $name: no sub-8 leaf selected"; FAIL=$((FAIL+1)); return
        fi
    elif [ "${EXPECT_NO_SUB8:-0}" = 1 ]; then
        if grep -Eq " (4x[0-9]+|[0-9]+x4) " "$log"; then
            echo "FAIL $name: min-block 8 selected a sub-8 leaf"
            FAIL=$((FAIL+1)); return
        fi
    fi
    if ! "$DECODER" -i "$ivf" -o "$dec" --threads 1 --strict 1 \
            >"$TMP/$name.dec.log" 2>&1; then
        echo "FAIL $name: dav2d rejected the stream"; tail -3 "$TMP/$name.dec.log"
        FAIL=$((FAIL+1)); return
    fi
    local cmp
    if ! cmp=$(plane_compare "$recon" "$dec"); then
        echo "FAIL $name: $cmp"; FAIL=$((FAIL+1)); return
    fi
    echo "PASS $name: $cmp"
}

run_reject() {  # name expected_stderr extra...
    local name=$1 expected=$2; shift 2
    CASES=$((CASES+1))
    local out=$TMP/$name.ivf
    if "$BIN" "$TESTDATA/shine_640x360_1f.y4m" -o "$out" --video --frames 1 \
            "$@" >"$TMP/$name.log" 2>&1; then
        echo "FAIL $name: expected a rejection, encode succeeded"
        FAIL=$((FAIL+1)); return
    fi
    if ! grep -qF -- "$expected" "$TMP/$name.log"; then
        echo "FAIL $name: rejection does not mention '$expected'"
        tail -5 "$TMP/$name.log"; FAIL=$((FAIL+1)); return
    fi
    echo "PASS $name (rejected: $expected)"
}

# --- sub-8 rect tree at the two gated depths ------------------------------
EXPECT_SUB8=1 case_run sub8_64_d5 "$TESTDATA/m0_64x64_1f.y4m" 1 \
    --video --frames 1 --qp 90 --video-min-block 4 --video-max-depth 5
EXPECT_SUB8=1 case_run sub8_64_d6 "$TESTDATA/m0_64x64_1f.y4m" 1 \
    --video --frames 1 --qp 90 --video-min-block 4 --video-max-depth 6
EXPECT_SUB8=1 case_run sub8_360_d5 "$TESTDATA/shine_640x360_1f.y4m" 1 \
    --video --frames 1 --qp 90 --video-min-block 4 --video-max-depth 5
EXPECT_SUB8=1 case_run sub8_360_d6 "$TESTDATA/shine_640x360_1f.y4m" 1 \
    --video --frames 1 --qp 90 --video-min-block 4 --video-max-depth 6
EXPECT_SUB8=1 case_run sub8_360_4f_d5 "$TESTDATA/shine_640x360_4f.y4m" 4 \
    --video --frames 4 --qp 90 --video-min-block 4 --video-max-depth 5
EXPECT_SUB8=1 case_run sub8_360_4f_d6 "$TESTDATA/shine_640x360_4f.y4m" 4 \
    --video --frames 4 --qp 90 --video-min-block 4 --video-max-depth 6

# --- shared-chroma owners through the opt-in tools ------------------------
EXPECT_SUB8=1 case_run sub8_360_ext_d6 "$TESTDATA/shine_640x360_1f.y4m" 1 \
    --video --frames 1 --qp 90 --video-min-block 4 --video-max-depth 6 \
    --video-ext-partitions
EXPECT_SUB8=1 case_run sub8_360_ext_4f "$TESTDATA/shine_640x360_4f.y4m" 4 \
    --video --frames 4 --qp 90 --video-min-block 4 --video-max-depth 5 \
    --video-ext-partitions
EXPECT_SUB8=1 case_run sub8_360_sdp_d6 "$TESTDATA/shine_640x360_1f.y4m" 1 \
    --video --frames 1 --qp 90 --video-min-block 4 --video-max-depth 6 \
    --video-sdp
EXPECT_SUB8=1 case_run sub8_360_intra_full_d5 "$TESTDATA/shine_640x360_1f.y4m" 1 \
    --video --frames 1 --qp 90 --video-min-block 4 --video-max-depth 5 \
    --video-intra-full
EXPECT_SUB8=1 case_run sub8_64_intra_full_d6 "$TESTDATA/m0_64x64_1f.y4m" 1 \
    --video --frames 1 --qp 90 --video-min-block 4 --video-max-depth 6 \
    --video-intra-full
EXPECT_SUB8=1 case_run sub8_360_deblock_d6 "$TESTDATA/shine_640x360_1f.y4m" 1 \
    --video --frames 1 --qp 90 --video-min-block 4 --video-max-depth 6 \
    --video-deblock
EXPECT_SUB8=1 case_run sub8_360_sb128_d6 "$TESTDATA/shine_640x360_1f.y4m" 1 \
    --video --frames 1 --qp 90 --video-min-block 4 --video-max-depth 6 \
    --video-sb128
EXPECT_SUB8=1 case_run sub8_360_tiles "$TESTDATA/shine_640x360_1f.y4m" 1 \
    --video --frames 1 --qp 90 --video-min-block 4 --video-max-depth 6 \
    --tiles 2x2
EXPECT_SUB8=1 case_run sub8_360_tx_search "$TESTDATA/shine_640x360_1f.y4m" 1 \
    --video --frames 1 --qp 90 --video-min-block 4 --video-max-depth 6 \
    --video-tx-search

# --- min-block 8 regression controls (no sub-8 leaf) ----------------------
EXPECT_NO_SUB8=1 case_run min8_64_d6 "$TESTDATA/m0_64x64_1f.y4m" 1 \
    --video --frames 1 --qp 90 --video-min-block 8 --video-max-depth 6
EXPECT_NO_SUB8=1 case_run min8_360_d6 "$TESTDATA/shine_640x360_1f.y4m" 1 \
    --video --frames 1 --qp 90 --video-min-block 8 --video-max-depth 6
EXPECT_NO_SUB8=1 case_run min8_360_4f "$TESTDATA/shine_640x360_4f.y4m" 4 \
    --video --frames 4 --qp 90 --video-min-block 8 --video-max-depth 6
EXPECT_NO_SUB8=1 case_run min8_360_ext "$TESTDATA/shine_640x360_4f.y4m" 4 \
    --video --frames 4 --qp 90 --video-min-block 8 --video-max-depth 6 \
    --video-ext-partitions

# --- named rejections -----------------------------------------------------
run_reject min_block_3 "--video-min-block must be an integer in [4, 64]" \
    --video-min-block 3
run_reject min_block_65 "--video-min-block must be an integer in [4, 64]" \
    --video-min-block 65
run_reject sdp_intra_full "--video-sdp does not compose with --video-intra-full" \
    --video-min-block 4 --video-max-depth 6 --video-sdp --video-intra-full
run_reject idtx "--video-min-block 4 does not compose with --video-idtx" \
    --video-min-block 4 --video-max-depth 6 --video-tx-search --video-idtx

echo
echo "gate_video_sub8: $CASES case(s), $FAIL failure(s)"
[ "$FAIL" -eq 0 ]
