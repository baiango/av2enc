#!/usr/bin/env bash
# Cross-frame CDF carry gate (--cdf-continuity).
#
# §5.18.2 / §6.17.2 (which CDF image a frame installs), §7.4.4
# (load_cdfs/save_cdfs), §7.5 (frame_end_update_cdf) and §8.2.4
# (exit_symbol copy/avg_cdf).
#
# 1. §5.18.2 forced reset.  choose_primary_secondary_ref_frame() only counts a
#    reference whose FrameType is INTER_FRAME, so the first inter frame after a
#    keyframe has PRIMARY_REF_NONE derived and the decoder forces
#    disable_cross_frame_cdf_init = 1: it installs the §9.3 image even though
#    the header signals primary_ref_frame = 0.  The encoder must reset the same
#    frame.  A key+P chain with --cdf-continuity must decode under
#    dav2d --strict 1 and reproduce the encoder reconstruction exactly (this is
#    the frame whose carried image used to desynchronise the whole chain).
# 2. §8.2 CDF image completeness.  A 60-frame --fast-decode stream with the carried
#    image decodes all 60 pictures (dav2d --strict 1, frame count not exit
#    status) and the decoded payload equals --video-recon byte for byte.  The
#    same decisions with the reset-per-frame image are 24,904 B, so the carry
#    must be smaller at identical reconstruction.  The same 60-frame stream
#    with each loop-restoration tool (--video-wiener, --video-chroma-lr,
#    --video-pc-wiener) must decode 60/60 too: the six §6.17.2 LR CDF rows are
#    part of the carried image.  The same holds for the fourteen compound rows
#    (§5.20.7.10/.6): a 60-frame compound P chain with --cdf-continuity must
#    decode 60/60, reproduce the encoder recon and stay below its reset image.
# 3. Static-clip floor.  With every picture identical the carried image needs no
#    per-frame re-seed, so the per-P-frame IVF size must be well below the
#    reset path's (measured 20.5 B vs 31.0 B).
# 4. Rejections.  --cdf-continuity without --avg-cdf/--adaptive-cdf, and with
#    the reset-per-frame opt-in, are usage errors.
#
# Usage: gate_video_cdf_carry.sh --bin build/av2enc [--dav2d PATH]
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
        -h|--help) sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "gate_video_cdf_carry: unknown argument $1" >&2; exit 2 ;;
    esac
done
[ -n "$BIN" ] && [ -x "$BIN" ] || {
    echo "gate_video_cdf_carry: --bin required" >&2; exit 2; }
if [ -z "$DAV2D" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
    done
fi
[ -n "$DAV2D" ] && [ -x "$DAV2D" ] || {
    echo "gate_video_cdf_carry: no dav2d" >&2; exit 77; }

NASA="$TESTDATA/nasa_480x240_30fps_60f.y4m"
PUB_CLIP="$TESTDATA/pub_640x360_4f.y4m"
[ -f "$PUB_CLIP" ] || PUB_CLIP="$TESTDATA/pub_640x360_1f.y4m"
for f in "$NASA" "$PUB_CLIP"; do
    [ -f "$f" ] || { echo "gate_video_cdf_carry: missing $f" >&2; exit 77; }
done
# Reference operating point with the reset-per-frame image (docs/video-nightly.md).
RESET_PIN_BYTES=23737
# The carried image at the same point: same decisions, −2.77% rate.
CARRY_PIN_BYTES=23025
CARRY_PIN_MD5=cf98ea9bad3f81bd9a20133409279318

md5_of() { md5 -q "$1" 2>/dev/null || md5sum "$1" | awk '{print $1}'; }

# dav2d exits 0 even after failing pictures, so count the decoded frames.
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

# --- 1. §5.18.2 forced reset on the first inter frame ---------------------
if ! "$BIN" "$PUB_CLIP" -o "$TMP/chain.ivf" --video --p-frames 3 --qp 150 \
        --avg-cdf --adaptive-cdf --cdf-continuity \
        --video-recon "$TMP/chain.y4m" >"$TMP/chain.log" 2>&1; then
    echo "FAIL: carry encode on a key+P chain" >&2
    cat "$TMP/chain.log" >&2; exit 1
fi
decode_exact "$TMP/chain.ivf" "$TMP/chain.y4m" 4 chain

# --- 2. full 60-frame carry stream at the nightly operating point ----------
if ! "$BIN" "$NASA" -o "$TMP/carry.ivf" --video --frames 60 --qp 110 \
        --cpu-threads 8 --fast-decode --cdf-continuity \
        --video-key-qp-offset -25 \
        --video-recon "$TMP/carry.y4m" >"$TMP/carry.log" 2>&1; then
    echo "FAIL: --fast-decode --cdf-continuity encode" >&2
    cat "$TMP/carry.log" >&2; exit 1
fi
decode_exact "$TMP/carry.ivf" "$TMP/carry.y4m" 60 carry
CARRY_BYTES=$(wc -c <"$TMP/carry.ivf" | tr -d ' ')
if [ "$CARRY_BYTES" -ge "$RESET_PIN_BYTES" ]; then
    echo "FAIL: carried image is $CARRY_BYTES B, not below the $RESET_PIN_BYTES" \
         "B reset pin" >&2
    exit 1
fi
if [ "$CARRY_BYTES" != "$CARRY_PIN_BYTES" ]; then
    echo "FAIL: carried stream is $CARRY_BYTES B, expected $CARRY_PIN_BYTES B" >&2
    exit 1
fi
CARRY_MD5=$(md5_of "$TMP/carry.ivf")
if [ "$CARRY_MD5" != "$CARRY_PIN_MD5" ]; then
    echo "FAIL: carried stream md5 is $CARRY_MD5, expected $CARRY_PIN_MD5" >&2
    exit 1
fi

# --- 2b. §6.17.2 loop-restoration CDF rows ride the carried image ----------
# init_non_coeff_cdfs() names UseWienerNsCdf, WienerNsLengthCdf,
# WienerNsUvSymCdf, WienerNsBaseCdf, UsePcWienerCdf and FlexRestorationTypeCdf,
# so §7.4.4 load_cdfs()/§7.5 save_cdfs() must copy them like every other row.
# When they were missing from the encoder's image, every frame restarted its
# tile LR CDFs at the §9.3 defaults while the decoder started from the
# reference frame's adapted rows: the first frame that coded an LR unit symbol
# after that (frame 3 here) desynchronised and dav2d reported
# "Error decoding frame: Invalid argument" for every later picture.
# Each arm must decode all 60 pictures under --strict 1.  The Wiener and
# chroma-LR reconstructions must also match --video-recon exactly.  PC-Wiener
# is decode-checked only: pinned dav2d's pixel classifier differs from the
# v1.0.0 §7.20.4 box features on a few samples where the 6x6 feature window
# crosses a 64x64 restoration-unit edge (pre-existing; the same stream decodes,
# see docs/negative-results.md).
decode_count() {
    local ivf=$1 expect=$2 tag=$3
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
}
for lr in wiener chroma-lr pc-wiener; do
    case "$lr" in
        wiener) LR_FLAG=--video-wiener ;;
        chroma-lr) LR_FLAG=--video-chroma-lr ;;
        pc-wiener) LR_FLAG=--video-pc-wiener ;;
    esac
    if ! "$BIN" "$NASA" -o "$TMP/lr-$lr.ivf" --video --frames 60 \
            --qp 110 --cpu-threads 8 --fast-decode --cdf-continuity \
            --video-key-qp-offset -25 "$LR_FLAG" \
            --video-recon "$TMP/lr-$lr.y4m" >"$TMP/lr-$lr.log" 2>&1; then
        echo "FAIL: --fast-decode --cdf-continuity $LR_FLAG encode" >&2
        cat "$TMP/lr-$lr.log" >&2
        exit 1
    fi
    if cmp -s "$TMP/lr-$lr.ivf" "$TMP/carry.ivf"; then
        echo "FAIL: $LR_FLAG did not change the carried stream" >&2
        exit 1
    fi
    if [ "$lr" = pc-wiener ]; then
        decode_count "$TMP/lr-$lr.ivf" 60 "lr-$lr"
        echo "PASS cdf-carry LR pc-wiener: 60/60 pictures (decode only)"
    else
        decode_exact "$TMP/lr-$lr.ivf" "$TMP/lr-$lr.y4m" 60 "lr-$lr"
        echo "PASS cdf-carry LR $lr: 60/60 pictures, recon exact"
    fi
done

# --- 2c. §5.20.7.10/.6 compound CDF rows ride the carried image ------------
# read_compound_mode_info()/read_inter_block_mode_info() also adapt
# CompModeCdf, CompRef0Cdf, CompRef1Cdf, IsJointCdf, CompoundModeNonJointCdf,
# CompoundModeSameRefsCdf, UseOptflowCdf, CompGroupIdxCdf, CompoundTypeCdf,
# CwpIdxCdf and the four wedge rows; §7.4.4 load_cdfs()/§7.5 save_cdfs() name
# every one of them.  When they were missing from the encoder's packed image,
# --cdf-continuity + --video-compound restarted each frame at the §9.3
# defaults while the decoder loaded the reference frame's adapted rows: only
# 3/60 pictures decoded and every later frame reported "Error decoding frame:
# Invalid argument".  The masked P-chain carrier codes comp_mode,
# comp_group_idx, compound_type and the wedge_* rows (256 wedge + 69 diffwtd
# leaves at this point), so this arm exercises the mask rows as well as the
# plain AVERAGE mode.
COMPOUND_CARRY_PIN_BYTES=15780
COMPOUND_CARRY_PIN_MD5=de29f3112fe5f7716937cd63c7b73b46
if ! "$BIN" "$NASA" -o "$TMP/compound.ivf" --video --frames 60 \
        --cpu-threads 8 --qp 197 --video-refs 2 --video-compound \
        --video-compound-wedge --video-compound-diffwtd \
        --avg-cdf --adaptive-cdf --cdf-continuity \
        --video-recon "$TMP/compound.y4m" >"$TMP/compound.log" 2>&1; then
    echo "FAIL: compound carry encode" >&2
    cat "$TMP/compound.log" >&2
    exit 1
fi
decode_exact "$TMP/compound.ivf" "$TMP/compound.y4m" 60 compound
COMPOUND_BYTES=$(wc -c <"$TMP/compound.ivf" | tr -d ' ')
if [ "$COMPOUND_BYTES" != "$COMPOUND_CARRY_PIN_BYTES" ]; then
    echo "FAIL: compound carried stream is $COMPOUND_BYTES B, expected" \
         "$COMPOUND_CARRY_PIN_BYTES B" >&2
    exit 1
fi
COMPOUND_MD5=$(md5_of "$TMP/compound.ivf")
if [ "$COMPOUND_MD5" != "$COMPOUND_CARRY_PIN_MD5" ]; then
    echo "FAIL: compound carried stream md5 is $COMPOUND_MD5, expected" \
         "$COMPOUND_CARRY_PIN_MD5" >&2
    exit 1
fi
# Same decisions with the reset-per-frame image (17,304 B measured): the
# carried image must stay below it, exactly like the §8.2 nightly arm.
if ! "$BIN" "$NASA" -o "$TMP/compound-reset.ivf" --video --frames 60 \
        --cpu-threads 8 --qp 197 --video-refs 2 --video-compound \
        --video-compound-wedge --video-compound-diffwtd \
        --avg-cdf --adaptive-cdf \
        --video-recon "$TMP/compound-reset.y4m" >"$TMP/compound-reset.log" 2>&1; then
    echo "FAIL: compound reset encode" >&2
    cat "$TMP/compound-reset.log" >&2
    exit 1
fi
COMPOUND_RESET_BYTES=$(wc -c <"$TMP/compound-reset.ivf" | tr -d ' ')
if [ "$COMPOUND_BYTES" -ge "$COMPOUND_RESET_BYTES" ]; then
    echo "FAIL: compound carried image is $COMPOUND_BYTES B, not below the" \
         "$COMPOUND_RESET_BYTES B reset stream" >&2
    exit 1
fi
echo "PASS cdf-carry compound: 60/60 pictures, recon exact," \
     "$COMPOUND_BYTES B (reset $COMPOUND_RESET_BYTES B)"

# --- 3. static-clip per-P-frame floor -------------------------------------
python3 - "$PUB_CLIP" "$TMP/static.y4m" 16 <<'PY'
import sys
data = open(sys.argv[1], 'rb').read()
end = data.index(b'\n') + 1
parts = data[:end].decode().split()
w = int(parts[1][1:]); h = int(parts[2][1:])
size = (w * h * 3) // 2
first = data[end + 5:end + 5 + size]  # after the first "FRAME\n"
with open(sys.argv[2], 'wb') as f:
    f.write(data[:end])
    for _ in range(int(sys.argv[3])):
        f.write(b'FRAME\n')
        f.write(first)
PY
mean_p_bytes() {
    python3 - "$1" <<'PY'
import struct, sys
data = open(sys.argv[1], 'rb').read()
off, sizes = 32, []
while off + 12 <= len(data):
    sz = struct.unpack('<I', data[off:off + 4])[0]
    sizes.append(sz)
    off += 12 + sz
p = sizes[1:]
if not p:
    sys.exit('no inter frames')
print(f"{sum(p) / len(p):.1f}")
PY
}
"$BIN" "$TMP/static.y4m" -o "$TMP/static-reset.ivf" --video --frames 16 \
    --qp 110 --cpu-threads 8 --fast-decode --video-cdf-per-frame-qp \
    >"$TMP/static-reset.log" 2>&1 || {
    echo "FAIL: static reset encode" >&2; cat "$TMP/static-reset.log" >&2
    exit 1; }
"$BIN" "$TMP/static.y4m" -o "$TMP/static-carry.ivf" --video --frames 16 \
    --qp 110 --cpu-threads 8 --fast-decode --cdf-continuity \
    --video-key-qp-offset -25 --video-recon "$TMP/static-carry.y4m" >"$TMP/static-carry.log" 2>&1 || {
    echo "FAIL: static carry encode" >&2; cat "$TMP/static-carry.log" >&2
    exit 1; }
decode_exact "$TMP/static-carry.ivf" "$TMP/static-carry.y4m" 16 static
RESET_FLOOR=$(mean_p_bytes "$TMP/static-reset.ivf")
CARRY_FLOOR=$(mean_p_bytes "$TMP/static-carry.ivf")
python3 - "$RESET_FLOOR" "$CARRY_FLOOR" <<'PY' || exit 1
import sys
reset, carry = float(sys.argv[1]), float(sys.argv[2])
if carry > reset * 0.8:
    sys.exit(f'FAIL: carried static P-frame floor {carry:.1f} B is not well '
             f'below the reset floor {reset:.1f} B')
PY

# --- 4. rejections --------------------------------------------------------
if "$BIN" "$PUB_CLIP" -o "$TMP/nocdf.ivf" --video --p-frames 2 \
        --cdf-continuity >/dev/null 2>"$TMP/nocdf.err"; then
    echo "FAIL: --cdf-continuity was accepted without the CDF chain" >&2
    exit 1
fi
grep -q -- "requires --avg-cdf and --adaptive-cdf" "$TMP/nocdf.err" || {
    echo "FAIL: the --cdf-continuity chain rejection message changed" >&2
    cat "$TMP/nocdf.err" >&2; exit 1; }
if "$BIN" "$PUB_CLIP" -o "$TMP/perframe.ivf" --video --p-frames 2 --qp 190 \
        --avg-cdf --adaptive-cdf --video-cdf-per-frame-qp --cdf-continuity \
        >/dev/null 2>"$TMP/perframe.err"; then
    echo "FAIL: --cdf-continuity was accepted with --video-cdf-per-frame-qp" >&2
    exit 1
fi
grep -q -- "does not compose with --cdf-continuity" "$TMP/perframe.err" || {
    echo "FAIL: the per-frame-QP carry rejection message changed" >&2
    cat "$TMP/perframe.err" >&2; exit 1; }

echo "PASS video cdf-carry: chain 4 pictures, 60-frame carry $CARRY_BYTES B" \
     "(reset pin $RESET_PIN_BYTES B, md5 $CARRY_MD5); static P-frame floor" \
     "$CARRY_FLOOR B vs $RESET_FLOOR B"
