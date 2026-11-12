#!/usr/bin/env bash
# Intra-tool gate: encode -> dav2d --strict 1 decode -> encoder-reconstruction
# exactness, for every opt-in tool integrated on the `integration/intra` branch.
#
# Usage:
#   tools/still_verify/intra_gate.sh --bin build-int/av2enc \
#       [--dump build-int/still_recon_dump] [--pin] [--only <substring>]
#
#   --pin     print the IVF MD5 of every case (for docs/m1-features.md)
#   --only S  run only cases whose name contains S
#
# Every case is default-off on the CLI, so the pinned `--still` baseline cases
# also run here; a moved baseline MD5 fails the case.
#
# Env: DAV2D overrides the decoder path.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BIN=""
DUMP=""
PIN=0
ONLY=""
DAV2D="${DAV2D:-}"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

while [ $# -gt 0 ]; do
    case "$1" in
        --bin) BIN=$2; shift 2 ;;
        --dump) DUMP=$2; shift 2 ;;
        --pin) PIN=1; shift ;;
        --only) ONLY=$2; shift 2 ;;
        --dav2d) DAV2D=$2; shift 2 ;;
        -h|--help)
            sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "intra_gate: unknown argument $1" >&2; exit 2 ;;
    esac
done

if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
    echo "intra_gate: --bin must point at an av2enc executable" >&2
    exit 2
fi
if [ -z "$DAV2D" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
    done
fi
if [ -z "$DAV2D" ] || [ ! -x "$DAV2D" ]; then
    echo "intra_gate: no dav2d binary (set DAV2D=...)" >&2
    exit 77
fi
if [ -z "$DUMP" ] || [ ! -x "$DUMP" ]; then
    echo "intra_gate: --dump must point at still_recon_dump" >&2
    exit 2
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "SKIP: python3 is required for the recon comparison" >&2
    exit 77
fi
have_fixture=0
for fixture in "$ROOT/tests/data/basic_64x64_1f.y4m" \
               "$ROOT/tests/data/basic_640x360_1f.y4m"; do
    [ -f "$fixture" ] && have_fixture=1
done
if [ "$have_fixture" -eq 0 ]; then
    echo "SKIP: tests/data fixtures are missing" >&2
    exit 77
fi

failures=0
pass=0
skipped=0

# run_case <fixture> <name> <encode flags...>
run_case() {
    local fixture=$1 name=$2
    shift 2
    if [ -n "$ONLY" ] && [[ "$name" != *"$ONLY"* ]]; then
        return 0
    fi
    local input="$ROOT/$fixture"
    if [[ "$fixture" == /* ]]; then
        input="$fixture"
    fi
    local out="$TMP/${name}.ivf"
    local dec="$TMP/${name}.dec.y4m"
    local recon="$TMP/${name}.recon.y4m"
    if [ ! -f "$input" ]; then
        echo "SKIP $name: missing $fixture" >&2
        skipped=$((skipped + 1))
        return 0
    fi
    if ! "$BIN" "$input" -o "$out" --frames 1 "$@" >"$TMP/${name}.enc.log" 2>&1; then
        echo "FAIL $name: encode failed" >&2
        cat "$TMP/${name}.enc.log" >&2
        failures=$((failures + 1))
        return 0
    fi
    if [ -n "${EXPECT_IBC:-}" ]; then
        local got_ibc
        got_ibc=$(sed -n 's/^  ibc:     \([0-9][0-9]*\) exact-copy.*/\1/p' \
            "$TMP/${name}.enc.log" | tail -1)
        if [ -z "$got_ibc" ] || [ "$got_ibc" -lt "$EXPECT_IBC" ]; then
            echo "FAIL $name: expected >= $EXPECT_IBC IBC block(s), got ${got_ibc:-0}" >&2
            failures=$((failures + 1))
            return 0
        fi
    fi
    if ! "$DAV2D" -q -i "$out" -o "$dec" --threads 1 --strict 1 \
            >"$TMP/${name}.dec.log" 2>&1; then
        echo "FAIL $name: dav2d --strict 1 rejected the stream" >&2
        cat "$TMP/${name}.dec.log" >&2
        failures=$((failures + 1))
        return 0
    fi
    if ! "$DUMP" -i "$input" -o "$recon" "$@" >"$TMP/${name}.dump.log" 2>&1; then
        echo "FAIL $name: still_recon_dump failed" >&2
        cat "$TMP/${name}.dump.log" >&2
        failures=$((failures + 1))
        return 0
    fi
    local cmp
    cmp=$(python3 - "$dec" "$recon" <<'PY'
import sys

def read_y4m(path):
    with open(path, "rb") as f:
        data = f.read()
    pos = data.index(b"\n") + 1
    frames = []
    while pos < len(data):
        if data[pos:pos + 5] != b"FRAME":
            raise SystemExit("bad frame marker in %s" % path)
        pos = data.index(b"\n", pos) + 1
        frames.append(data[pos:])
        break
    return frames[0] if frames else b""

a = read_y4m(sys.argv[1])
b = read_y4m(sys.argv[2])
if len(a) != len(b):
    print("size %d != %d" % (len(a), len(b)))
    raise SystemExit(0)
diffs = [abs(x - y) for x, y in zip(a, b) if x != y]
if not diffs:
    print("exact")
else:
    print("%d diffs, max %d" % (len(diffs), max(diffs)))
PY
)
    local md5
    md5=$(md5 -q "$out" 2>/dev/null || md5sum "$out" | awk '{print $1}')
    if [ "$cmp" != "exact" ]; then
        echo "FAIL $name: recon mismatch ($cmp)" >&2
        failures=$((failures + 1))
        return 0
    fi
    if [ -n "${EXPECT_MD5:-}" ] && [ "$md5" != "$EXPECT_MD5" ]; then
        echo "FAIL $name: IVF md5 $md5, expected $EXPECT_MD5" >&2
        failures=$((failures + 1))
        return 0
    fi
    if [ "$PIN" = "1" ]; then
        echo "PASS $name ($(wc -c <"$out" | tr -d ' ') B, md5 $md5, recon exact)"
    else
        echo "PASS $name ($(wc -c <"$out" | tr -d ' ') B, recon exact)"
    fi
    pass=$((pass + 1))
}

# run_recon_case <fixture> <name> <encode flags...>
# Uses the product CLI's own --recon dump instead of still_recon_dump, so it can
# exercise the SDP/ext/sub-8/segmentation/cdef combinations that the hand-written
# dump parser does not expose.
run_recon_case() {
    local fixture=$1 name=$2
    shift 2
    if [ -n "$ONLY" ] && [[ "$name" != *"$ONLY"* ]]; then
        return 0
    fi
    local input="$ROOT/$fixture"
    if [[ "$fixture" == /* ]]; then
        input="$fixture"
    fi
    local out="$TMP/${name}.ivf"
    local dec="$TMP/${name}.dec.y4m"
    local recon="$TMP/${name}.recon.y4m"
    if [ ! -f "$input" ]; then
        echo "SKIP $name: missing $fixture" >&2
        skipped=$((skipped + 1))
        return 0
    fi
    if ! "$BIN" "$input" -o "$out" --frames 1 "$@" --recon "$recon" \
            >"$TMP/${name}.enc.log" 2>&1; then
        echo "FAIL $name: encode failed" >&2
        cat "$TMP/${name}.enc.log" >&2
        failures=$((failures + 1))
        return 0
    fi
    if [ -n "${EXPECT_IBC:-}" ]; then
        local got_ibc
        got_ibc=$(sed -n 's/^  ibc:     \([0-9][0-9]*\) exact-copy.*/\1/p' \
            "$TMP/${name}.enc.log" | tail -1)
        if [ -z "$got_ibc" ] || [ "$got_ibc" -lt "$EXPECT_IBC" ]; then
            echo "FAIL $name: expected >= $EXPECT_IBC IBC block(s), got ${got_ibc:-0}" >&2
            failures=$((failures + 1))
            return 0
        fi
    fi
    if ! "$DAV2D" -q -i "$out" -o "$dec" --threads 1 --strict 1 \
            >"$TMP/${name}.dec.log" 2>&1; then
        echo "FAIL $name: dav2d --strict 1 rejected the stream" >&2
        cat "$TMP/${name}.dec.log" >&2
        failures=$((failures + 1))
        return 0
    fi
    local cmp
    cmp=$(python3 - "$dec" "$recon" <<'PY'
import sys

def read_y4m(path):
    with open(path, "rb") as f:
        data = f.read()
    pos = data.index(b"\n") + 1
    while pos < len(data):
        if data[pos:pos + 5] != b"FRAME":
            raise SystemExit("bad frame marker in %s" % path)
        pos = data.index(b"\n", pos) + 1
        return data[pos:]
    return b""

a = read_y4m(sys.argv[1])
b = read_y4m(sys.argv[2])
if len(a) != len(b):
    print("size %d != %d" % (len(a), len(b)))
    raise SystemExit(0)
diffs = [abs(x - y) for x, y in zip(a, b) if x != y]
print("exact" if not diffs else "%d diffs, max %d" % (len(diffs), max(diffs)))
PY
)
    if [ "$cmp" != "exact" ]; then
        echo "FAIL $name: recon mismatch ($cmp)" >&2
        failures=$((failures + 1))
        return 0
    fi
    echo "PASS $name (recon exact)"
    pass=$((pass + 1))
}

# run_reject <name> <expected stderr substring> <encode flags...>
run_reject() {
    local name=$1 expected=$2
    shift 2
    if [ -n "$ONLY" ] && [[ "$name" != *"$ONLY"* ]]; then
        return 0
    fi
    local out="$TMP/${name}.ivf"
    if "$BIN" "$ROOT/tests/data/basic_64x64_1f.y4m" -o "$out" --frames 1 \
            "$@" >"$TMP/${name}.log" 2>&1; then
        echo "FAIL $name: expected a rejection, encode succeeded" >&2
        failures=$((failures + 1))
        return 0
    fi
    if ! grep -q "$expected" "$TMP/${name}.log"; then
        echo "FAIL $name: rejection message does not mention '$expected'" >&2
        cat "$TMP/${name}.log" >&2
        failures=$((failures + 1))
        return 0
    fi
    echo "PASS $name (rejected at the CLI)"
    pass=$((pass + 1))
}

BASELINE=(--still)

# Synthetic screen-content fixtures (flat fills, stripes, icon squares).  The
# natural fixtures almost never select palette, so the tool needs its own
# content to prove the syntax path.
python3 - "$TMP" <<'PY'
import sys, pathlib
out = pathlib.Path(sys.argv[1])

def write_y4m(path, w, h):
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
    u = bytes([96]) * (cw * ch)
    v = bytes([160]) * (cw * ch)
    with open(path, "wb") as f:
        f.write(b"YUV4MPEG2 W%d H%d F25:1 Ip A0:0 C420\n" % (w, h))
        f.write(b"FRAME\n")
        f.write(bytes(y))
        f.write(u)
        f.write(v)

write_y4m(out / "scc_64x64.y4m", 64, 64)
write_y4m(out / "scc_128x96.y4m", 128, 96)

# IBC fixture: a 32x32 tile repeated horizontally/vertically with a flat 8x8
# icon and a 224 corner.  Blocks at the same modulo repeat exactly, so the
# exact-copy-only encoder selects intra block copy.
def write_ibc_y4m(path, w, h):
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
        f.write(bytes(y))
        f.write(bytes([100]) * (cw * ch))
        f.write(bytes([150]) * (cw * ch))

write_ibc_y4m(out / "ibc_384x128.y4m", 384, 128)
PY
SCC64="$TMP/scc_64x64.y4m"
SCC128="$TMP/scc_128x96.y4m"
IBC384="$TMP/ibc_384x128.y4m"

# --- default-off regressions (must stay byte-identical everywhere) ----------
run_case tests/data/basic_64x64_1f.y4m baseline_64 "${BASELINE[@]}"
run_case tests/data/basic_640x360_1f.y4m baseline_360 "${BASELINE[@]}"

# --- integrated intra tools -------------------------------------------------
run_case tests/data/basic_64x64_1f.y4m ibp_64 "${BASELINE[@]}" --still-ibp
run_case tests/data/basic_640x360_1f.y4m ibp_360 "${BASELINE[@]}" --still-ibp
run_case tests/data/basic_64x64_1f.y4m ibp_fullintra_64 "${BASELINE[@]}" \
    --still-ibp --still-full-intra
run_case tests/data/basic_640x360_1f.y4m ibp_fullintra_360 "${BASELINE[@]}" \
    --still-ibp --still-full-intra
run_case tests/data/basic_64x64_1f.y4m ibp_rdoq_64 "${BASELINE[@]}" \
    --still-ibp --still-rdoq
run_case tests/data/basic_640x360_1f.y4m ibp_cdef_360 "${BASELINE[@]}" \
    --still-ibp --still-cdef
run_case tests/data/basic_64x64_1f.y4m mrl_64 "${BASELINE[@]}" --still-mrl
run_case tests/data/basic_640x360_1f.y4m mrl_360 "${BASELINE[@]}" --still-mrl
run_case tests/data/basic_64x64_1f.y4m mrl_txtype_64 "${BASELINE[@]}" \
    --still-mrl --still-tx-type
run_case tests/data/basic_640x360_1f.y4m mrl_deblock_360 "${BASELINE[@]}" \
    --still-mrl --still-cdef
run_case tests/data/basic_64x64_1f.y4m cfl_ds1_64 "${BASELINE[@]}" \
    --still-cfl-ds 1
run_case tests/data/basic_640x360_1f.y4m cfl_ds1_360 "${BASELINE[@]}" \
    --still-cfl-ds 1
run_case tests/data/basic_64x64_1f.y4m cfl_ds2_64 "${BASELINE[@]}" \
    --still-cfl-ds 2
run_case tests/data/basic_640x360_1f.y4m cfl_ds2_360 "${BASELINE[@]}" \
    --still-cfl-ds 2
run_case tests/data/basic_64x64_1f.y4m cfl_explicit_64 "${BASELINE[@]}" \
    --still-cfl-explicit
run_case tests/data/basic_640x360_1f.y4m cfl_explicit_360 "${BASELINE[@]}" \
    --still-cfl-explicit
run_case tests/data/basic_64x64_1f.y4m cfl_explicit_ds1_64 "${BASELINE[@]}" \
    --still-cfl-explicit --still-cfl-ds 1
run_case tests/data/basic_64x64_1f.y4m dip_64 "${BASELINE[@]}" --still-dip
run_case tests/data/basic_640x360_1f.y4m dip_360 "${BASELINE[@]}" --still-dip
run_case tests/data/basic_64x64_1f.y4m dip_cdef_64 "${BASELINE[@]}" \
    --still-dip --still-cdef
run_case tests/data/basic_640x360_1f.y4m dip_ibp_360 "${BASELINE[@]}" \
    --still-dip --still-ibp
# Palette (§5.20.8): the 128x96 synthetic fixture drives 8x32 direction-0
# leaves whose identity_row_y == 1 rows exercise the §5.20.8.1 cache split and
# the §5.20.8.4 line copy (docs/negative-results.md reproducer).
run_case tests/data/basic_64x64_1f.y4m palette_64 "${BASELINE[@]}" \
    --still-palette
run_case tests/data/basic_640x360_1f.y4m palette_360 "${BASELINE[@]}" \
    --still-palette
EXPECT_MD5=47bfd8db98aaa08df240e0fe674fe0e5 \
    run_case "$SCC64" palette_scc64 "${BASELINE[@]}" --still-palette
EXPECT_MD5=daf0532f15fc4d37f5ead6938943c074 \
    run_case "$SCC128" palette_scc128 "${BASELINE[@]}" --still-palette
EXPECT_MD5=ec81d5d3b4dd69e656d9e40b3912c712 \
    run_case "$SCC128" palette_scc128_dip "${BASELINE[@]}" --still-palette \
    --still-dip
EXPECT_MD5=7db8078929a6043313700ea4ecf6261d \
    run_case "$SCC128" palette_scc128_depth5 "${BASELINE[@]}" --still-palette \
    --still-depth 5
# Intra block copy (§5.20.5.3/.4): the 384x128 fixture repeats a 32x32 tile, so
# the exact-copy-only decision selects NEARMV blocks.  The gate also checks that
# at least one IBC block was selected (EXPECT_IBC).
# The two narrow fixtures exercise the use_intrabc = 0 syntax only: below 129
# luma samples wide there is no in-frame source at the §5.20.5.1 delay distance
# (see docs/m1-features.md), so they intentionally run without EXPECT_IBC.
run_case "$SCC128" ibc_scc128 "${BASELINE[@]}" --still-ibc
run_case "$SCC64" ibc_scc64 "${BASELINE[@]}" --still-ibc
EXPECT_IBC=1 EXPECT_MD5=dc5fca8c44f78e51850afe7bf1ac8983 \
    run_case "$IBC384" ibc_384x128 "${BASELINE[@]}" --still-ibc
EXPECT_IBC=1 EXPECT_MD5=de0e3a238163e751a6ae6c6b5ef5b4f5 \
    run_case "$IBC384" ibc_384x128_palette "${BASELINE[@]}" --still-ibc \
    --still-palette
EXPECT_IBC=1 EXPECT_MD5=3ba3788644e5b6456c34a060da970752 \
    run_case "$IBC384" ibc_384x128_depth5 "${BASELINE[@]}" --still-ibc \
    --still-depth 5
run_case tests/data/basic_640x360_1f.y4m ibc_360 "${BASELINE[@]}" --still-ibc
run_reject ibc_no_still "requires --still" --still-ibc
run_reject palette_no_still "requires --still" --still-palette
run_reject ibc_sb128 "requires 64x64 superblocks" --still --still-ibc \
    --still-sb128
run_reject ibc_search_without_ibc "requires --still-ibc" --still \
    --still-ibc-search 16

# 128x128 superblock grid: the roots still split to 64x64 coding blocks, so
# this exercises the assembled 128-SB path with a real reconstruction compare.
run_case tests/data/basic_640x360_1f.y4m sb128_360 "${BASELINE[@]}" \
    --still-sb128
run_case "$SCC128" sb128_scc128 "${BASELINE[@]}" --still-sb128
run_case "$SCC128" sb128_palette "${BASELINE[@]}" --still-sb128 \
    --still-palette
run_case tests/data/basic_64x64_1f.y4m mhccp_64 "${BASELINE[@]}" --still-mhccp
run_case tests/data/basic_640x360_1f.y4m mhccp_360 "${BASELINE[@]}" --still-mhccp
run_case tests/data/basic_64x64_1f.y4m mhccp_explicit_64 "${BASELINE[@]}" \
    --still-mhccp --still-cfl-explicit
run_case "$SCC128" mhccp_scc128 "${BASELINE[@]}" --still-mhccp

# --- shared-chroma ownership / context (BUG-CHROMA-001) --------------------
# SDP CHROMA_PART reads YMode/AngleDeltaY from the co-located luma tree, so a
# directional luma mode must stay decoder-exact.  4-wide ext/sub-8 leaves use
# the dynamic y_mode list and the full UV index list; MHCCP's cfl_mh_dir CDF
# context is Size_Group[MiSize]; ext/SDP must carry the per-SB segment id.
run_recon_case tests/data/basic_64x64_1f.y4m sdp_full_intra_64 \
    "${BASELINE[@]}" --still-sdp --still-full-intra
run_recon_case tests/data/basic_640x360_1f.y4m sdp_full_intra_360 \
    "${BASELINE[@]}" --still-sdp --still-full-intra
run_recon_case tests/data/basic_64x64_1f.y4m sdp_mrl_64 \
    "${BASELINE[@]}" --still-sdp --still-mrl
run_recon_case tests/data/basic_64x64_1f.y4m sdp_mhccp_64 \
    "${BASELINE[@]}" --still-sdp --still-mhccp
run_recon_case tests/data/basic_64x64_1f.y4m ext_full_intra_64 \
    "${BASELINE[@]}" --still-ext-partitions --still-full-intra
run_recon_case tests/data/basic_640x360_1f.y4m ext_full_intra_360 \
    "${BASELINE[@]}" --still-ext-partitions --still-full-intra
run_recon_case tests/data/basic_64x64_1f.y4m ext_mhccp_64 \
    "${BASELINE[@]}" --still-ext-partitions --still-mhccp
run_recon_case tests/data/basic_640x360_1f.y4m ext_mhccp_360 \
    "${BASELINE[@]}" --still-ext-partitions --still-mhccp
run_recon_case tests/data/basic_64x64_1f.y4m sub8_full_intra_64 \
    "${BASELINE[@]}" --still-sub8 --still-full-intra
run_recon_case tests/data/basic_640x360_1f.y4m sub8_full_intra_360 \
    "${BASELINE[@]}" --still-sub8 --still-full-intra
run_recon_case tests/data/basic_64x64_1f.y4m sub8_mhccp_64 \
    "${BASELINE[@]}" --still-sub8 --still-mhccp
run_recon_case tests/data/basic_640x360_1f.y4m ext_segq_360 \
    "${BASELINE[@]}" --still-ext-partitions --still-per-segment-q \
    --still-seg-delta 40
run_recon_case tests/data/basic_640x360_1f.y4m sdp_segq_360 \
    "${BASELINE[@]}" --still-sdp --still-per-segment-q --still-seg-delta 40
EXPECT_IBC=1 run_recon_case "$IBC384" ibc_cdef_384 "${BASELINE[@]}" \
    --still-ibc --still-cdef
EXPECT_IBC=1 run_recon_case "$IBC384" ibc_cdef_one_384 "${BASELINE[@]}" \
    --still-ibc --still-cdef --still-cdef-strengths 1
EXPECT_IBC=1 run_recon_case "$IBC384" ibc_cdef_onskip_384 "${BASELINE[@]}" \
    --still-ibc --still-cdef --still-cdef-on-skip-txfm

echo
echo "intra_gate: $pass passed, $failures failed, $skipped skipped"
[ "$failures" -eq 0 ]
