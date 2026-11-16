#!/usr/bin/env bash
# Loop-restoration feature gate: pinned IVF MD5s + dav2d --strict 1 decode +
# encoder-recon exactness for the opt-in §7.20 Wiener NS paths.
#
# Kept separate from feature_gate.sh because this branch is not allowed to
# change the published baseline (refactor/baseline.json pins feature_gate.sh's
# 30-case list).  Every case is opt-in on the CLI, so the default pinned
# streams cannot move.
#
# Usage:
#   tools/still_verify/loop_restoration_gate.sh --bin build/av2enc \
#       [--dump build/still_dump] [--quick]
#
#   --dump    still_dump harness; defaults to still_dump next to --bin
#   --quick   skip the 1080p fixtures (testdata is gitignored)
#
# Build the harness with: cmake --build <build> --target still_dump
#
# Env: DAV2D overrides the decoder path; AV2ENC_TESTDATA overrides testdata/.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BIN=""
DUMP=""
DAV2D="${DAV2D:-}"
TESTDATA="${AV2ENC_TESTDATA:-$ROOT/testdata}"
QUICK=0
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

while [ $# -gt 0 ]; do
    case "$1" in
        --bin) BIN=$2; shift 2 ;;
        --dump) DUMP=$2; shift 2 ;;
        --dav2d) DAV2D=$2; shift 2 ;;
        --quick) QUICK=1; shift ;;
        -h|--help)
            sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "loop_restoration_gate: unknown argument $1" >&2; exit 2 ;;
    esac
done

if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
    echo "loop_restoration_gate: --bin must point at an av2enc executable" >&2
    exit 2
fi
# The harness must be linked against the same core library as --bin, so the
# implicit lookup only considers the build directory that holds the binary.
if [ -z "$DUMP" ]; then
    candidate="$(dirname "$BIN")/still_dump"
    if [ -x "$candidate" ]; then DUMP=$candidate; fi
fi
if [ -n "$DUMP" ] && [ ! -x "$DUMP" ]; then
    echo "loop_restoration_gate: --dump $DUMP is not executable" >&2
    exit 2
fi
if [ -z "$DUMP" ]; then
    echo "loop_restoration_gate: no still_dump harness in $(dirname "$BIN");" \
         "recon exactness will not be checked" >&2
    echo "  build one with: cmake --build <build> --target still_dump" >&2
fi
if [ -z "$DAV2D" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
    done
fi
if [ -z "$DAV2D" ] || [ ! -x "$DAV2D" ]; then
    echo "loop_restoration_gate: no dav2d binary (set DAV2D=...)" >&2
    exit 77
fi
export DAV2D
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

# Film grain is decoder-side output, so the recon comparison always decodes
# with synthesis disabled.  The wrapper is harmless for non-grain streams.
NOGRAIN="$TMP/dav2d_nograin"
printf '#!/bin/sh\nexec "%s" --filmgrain 0 "$@"\n' "$DAV2D" >"$NOGRAIN"
chmod +x "$NOGRAIN"

failures=0
pass=0

fixture_path() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s\n' "$ROOT/$1" ;;
    esac
}

# run_case <fixture> <name> <expected_md5|-> <dump args...> -- <encode args...>
# An expected MD5 of "-" skips the pin and only requires decode + recon
# exactness (used by the partial-unit geometry cases, whose streams are new).
run_case() {
    local fixture=$1 name=$2 expected=$3
    shift 3
    local dump_args=()
    while [ "$1" != "--" ]; do dump_args+=("$1"); shift; done
    shift
    local input
    input=$(fixture_path "$fixture")
    local out="$TMP/${name}.ivf"
    local dec="$TMP/${name}.y4m"
    if [ ! -f "$input" ]; then
        echo "SKIP $name: missing $fixture" >&2
        return 0
    fi
    if ! "$BIN" "$input" -o "$out" --frames 1 "$@" >"$TMP/${name}.log" 2>&1; then
        echo "FAIL $name: encode failed" >&2
        cat "$TMP/${name}.log" >&2
        failures=$((failures + 1))
        return 0
    fi
    local got=- pin=unpinned
    if [ "$expected" != "-" ]; then
        got=$(md5 -q "$out" 2>/dev/null || md5sum "$out" | awk '{print $1}')
        pin=$expected
        if [ "$got" != "$expected" ]; then
            echo "FAIL $name: IVF md5 $got, expected $expected" >&2
            failures=$((failures + 1))
            return 0
        fi
    fi
    if ! "$NOGRAIN" -q -i "$out" -o "$dec" --threads 1 --strict 1 \
            >"$TMP/${name}.dec.log" 2>&1; then
        echo "FAIL $name: dav2d --strict 1 rejected the stream" >&2
        cat "$TMP/${name}.dec.log" >&2
        failures=$((failures + 1))
        return 0
    fi
    if [ -n "$DUMP" ]; then
        if ! "$DUMP" -i "$input" -o "$TMP/${name}.dump" \
                ${dump_args[@]+"${dump_args[@]}"} \
                >"$TMP/${name}.dump.log" 2>&1; then
            echo "FAIL $name: still_dump failed" >&2
            cat "$TMP/${name}.dump.log" >&2
            failures=$((failures + 1))
            return 0
        fi
        if ! python3 "$ROOT/tools/still_verify/verify.py" \
                --ivf "$out" --dump "$TMP/${name}.dump" --decoder "$NOGRAIN" \
                --tsv >"$TMP/${name}.verify.log" 2>&1; then
            echo "FAIL $name: recon/decoder mismatch" >&2
            cat "$TMP/${name}.verify.log" >&2
            failures=$((failures + 1))
            return 0
        fi
    fi
    echo "PASS $name ($(wc -c <"$out" | tr -d ' ') B, $pin)"
    pass=$((pass + 1))
}

# --- 64x64 -----------------------------------------------------------------
run_case tests/data/basic_64x64_1f.y4m still_64_wiener \
    03d7722ec7bbf0deffc7dab25b748004 --wiener -- \
    --still --still-wiener
run_case tests/data/basic_64x64_1f.y4m still_64_chroma_lr \
    d67f32ff94430eff9819ce31c84da833 --chroma-lr -- \
    --still --still-chroma-lr
run_case tests/data/basic_64x64_1f.y4m still_64_pc_wiener \
    4ae2ccd4b28800cde774ea03d88c7bee --pc-wiener -- \
    --still --still-pc-wiener
run_case tests/data/basic_64x64_1f.y4m still_64_gdf \
    19cb8676d6a2f3367cf2766920fe6b9e --gdf -- \
    --still --still-gdf
run_case tests/data/basic_64x64_1f.y4m still_64_ccso_edge \
    b35555e6a4bb62ab7c422894879b2c34 --ccso-edge -- \
    --still --still-ccso-edge

# --- 640x360 ---------------------------------------------------------------
run_case tests/data/basic_640x360_1f.y4m still_360_wiener \
    f47e0f57b9bf7e630285fc40189b01ff --wiener -- \
    --still --still-wiener
run_case tests/data/basic_640x360_1f.y4m still_360_chroma_lr \
    476a3673836163121839b72755ec7f94 --chroma-lr -- \
    --still --still-chroma-lr
run_case tests/data/basic_640x360_1f.y4m still_360_wiener_chroma \
    3d1d1cf0c4ccf7cacfec233c97ba6dfa --wiener --chroma-lr -- \
    --still --still-wiener --still-chroma-lr
run_case tests/data/basic_640x360_1f.y4m still_360_cdef_ccso_wiener \
    0f173ffd137716f368b3944079efc2e4 --cdef --ccso --wiener --chroma-lr -- \
    --still --still-cdef --still-ccso --still-wiener --still-chroma-lr
run_case tests/data/basic_640x360_1f.y4m still_360_pc_wiener \
    08836c1181c4fe5ce219b3658dbb8a45 --pc-wiener -- \
    --still --still-pc-wiener
run_case tests/data/basic_640x360_1f.y4m still_360_cdef_ccso_pc_wiener \
    63f4592a88a82b724819984e6bf8fa50 --cdef --ccso --pc-wiener -- \
    --still --still-cdef --still-ccso --still-pc-wiener
run_case tests/data/basic_640x360_1f.y4m still_360_gdf \
    d26304dddfeaf299190150ef4c2b872f --gdf -- \
    --still --still-gdf
run_case tests/data/basic_640x360_1f.y4m still_360_cdef_ccso_gdf \
    cf0cc9155f9df319c6ce0ff778f5cc74 --cdef --ccso --gdf -- \
    --still --still-cdef --still-ccso --still-gdf

# --- partial restoration unit at the frame edge ---------------------------
# §5.20.10.4 count_units_in_frame() rounds the unit count to nearest, so the
# last frame band can fall outside every unit when its extent is below
# unit_size/2 (720 = 11*64 + 16; 70 = 1*64 + 6; 70 = 2*32 + 6 for chroma).
# read_lr() then codes nothing for a superblock at that origin and §7.20.1
# clamps the band to the last unit.  The writer used to abort instead.
run_case testdata/src_100x70.y4m still_70_wiener \
    - --wiener -- --still --still-wiener
run_case testdata/src_100x70.y4m still_70_chroma_lr \
    - --chroma-lr -- --still --still-chroma-lr
run_case testdata/src_100x70.y4m still_70_pc_wiener \
    - --pc-wiener -- --still --still-pc-wiener
run_case "$TESTDATA/pub_1280x720_60f.y4m" still_720_wiener \
    - --wiener -- --still --still-wiener
run_case "$TESTDATA/pub_1280x720_60f.y4m" still_720_chroma_lr \
    - --chroma-lr -- --still --still-chroma-lr
run_case "$TESTDATA/pub_1280x720_60f.y4m" still_720_pc_wiener \
    - --pc-wiener -- --still --still-pc-wiener

# --- per-plane chroma FrameRestorationType ---------------------------------
# §5.18.7.11 codes one chroma tool list but a separate FrameRestorationType
# per plane, so U can be RESTORE_NONE while V is RESTORE_WIENER_NONSEP.  This
# qp1 depth-4 plan picks exactly that split; the header used to derive the
# shared chroma flag from U alone and then write no V tool symbol, so the
# decoder desynchronised at the tile.
run_case tests/data/basic_640x360_1f.y4m still_360_qp1_chroma_lr_split \
    - --qp 1 --depth 4 --chroma-lr -- \
    --still --qp 1 --still-depth 4 --still-chroma-lr

# --- §7.20 pre-GDF CdefFrame / LrFrame split ------------------------------
# GDF runs after Wiener/PC-Wiener and samples CdefFrame, so the restoration
# output is a distinct buffer from the GDF source.  These combinations were
# rejected before the split; each one is recon-exact.
run_case tests/data/basic_64x64_1f.y4m still_64_gdf_wiener \
    f9b5ae5ae0dd69b297d521c8a2262a21 --gdf --wiener -- \
    --still --still-gdf --still-wiener
run_case tests/data/basic_64x64_1f.y4m still_64_gdf_chroma \
    aaaeff190f572d357c0f4509ee234294 --gdf --chroma-lr -- \
    --still --still-gdf --still-chroma-lr
run_case tests/data/basic_640x360_1f.y4m still_360_gdf_wiener \
    f1c53f17546dc7cf300372995363c049 --gdf --wiener -- \
    --still --still-gdf --still-wiener
run_case tests/data/basic_640x360_1f.y4m still_360_gdf_chroma \
    ea6e72df90453328daadb2a983b5a086 --gdf --chroma-lr -- \
    --still --still-gdf --still-chroma-lr
run_case tests/data/basic_640x360_1f.y4m still_360_gdf_wiener_chroma \
    722b602f18687db227d491c16fde232e --gdf --wiener --chroma-lr -- \
    --still --still-gdf --still-wiener --still-chroma-lr
run_case tests/data/basic_640x360_1f.y4m still_360_cdef_ccso_gdf_lr \
    0754c2b88f42dd89d4542b53da641542 --cdef --ccso --gdf --wiener --chroma-lr -- \
    --still --still-cdef --still-ccso --still-gdf --still-wiener --still-chroma-lr
run_case tests/data/basic_640x360_1f.y4m still_360_gdf_pc_wiener \
    bff046f4a1930fa1532a3285ab91ff36 --gdf --pc-wiener -- \
    --still --still-gdf --still-pc-wiener

# --- PC-Wiener luma + chroma Wiener NS ------------------------------------
# Plane 0 is RESTORE_PC_WIENER and planes 1/2 are RESTORE_WIENER_NONSEP; the
# frame header codes both rows and the tile carries use_pc_wiener for luma
# units followed by use_wiener_ns for chroma units.
run_case tests/data/basic_64x64_1f.y4m still_64_pc_wiener_chroma \
    19a0f9fe8f0d2a9f80139f1449a9ec12 --pc-wiener --chroma-lr -- \
    --still --still-pc-wiener --still-chroma-lr
run_case tests/data/basic_640x360_1f.y4m still_360_cdef_ccso_pc_wiener_chroma \
    1c59d674d282144fc178a73f6ee9d518 --cdef --ccso --pc-wiener --chroma-lr -- \
    --still --still-cdef --still-ccso --still-pc-wiener --still-chroma-lr
run_case tests/data/basic_640x360_1f.y4m still_360_gdf_pc_wiener_chroma \
    343f79ae03294a44f01780e1891539f0 --gdf --pc-wiener --chroma-lr -- \
    --still --still-gdf --still-pc-wiener --still-chroma-lr
run_case tests/data/basic_640x360_1f.y4m still_360_cdef_ccso_gdf_pc_wiener_chroma \
    3cb418d3ce4d5e236a849386f8d52f4b --cdef --ccso --gdf --pc-wiener --chroma-lr -- \
    --still --still-cdef --still-ccso --still-gdf --still-pc-wiener --still-chroma-lr
run_case tests/data/basic_640x360_1f.y4m still_360_ccso_edge \
    f79ae7d94fa2c7626bc66632577af192 --ccso-edge -- \
    --still --still-ccso-edge
run_case tests/data/basic_640x360_1f.y4m still_360_cdef_ccso_edge \
    d2745679d25558e095dbf4ee4e5b8ac5 --cdef --ccso-edge -- \
    --still --still-cdef --still-ccso-edge

# --- CDEF cdef_on_skip_txfm frame bit (CDEF_ON_SKIP_TXFM_ADAPTIVE) --------
# A single-picture header infers ADAPTIVE, so the frame bit is coded.  Intra
# frames derive skip = 0 either way, so only the header bit moves; the recon
# stays exact.  The sequence-level DISABLED/ALWAYS_ON modes are not codeable
# with single_picture_header_flag = 1; the M2 tiled gate covers those.
run_case tests/data/basic_64x64_1f.y4m still_64_cdef_skip_txfm \
    56c2ec8e60434138f6e94f5e287323fb --cdef --cdef-on-skip-txfm -- \
    --still --still-cdef --still-cdef-on-skip-txfm
run_case tests/data/basic_640x360_1f.y4m still_360_cdef_skip_txfm \
    498eb2182e39bcf6918bcde5464c9bc4 --cdef --cdef-on-skip-txfm -- \
    --still --still-cdef --still-cdef-on-skip-txfm
run_case tests/data/basic_640x360_1f.y4m still_360_cdef_ccso_lr_skip_txfm \
    e631b9aba5a1b5358cece27f6abb00f6 --cdef --cdef-on-skip-txfm --ccso --wiener --chroma-lr -- \
    --still --still-cdef --still-ccso --still-wiener --still-chroma-lr \
    --still-cdef-on-skip-txfm

# --- optional 1080p (testdata is a gitignored symlink) --------------------
if [ "$QUICK" -eq 0 ]; then
    run_case "$TESTDATA/pub_1920x1080_10f.y4m" still_hd_wiener_chroma \
        98b14b6a25db5c4c77416906b925be96 --wiener --chroma-lr -- \
        --still --still-wiener --still-chroma-lr
    run_case "$TESTDATA/pub_1920x1080_10f.y4m" still_hd_pc_wiener \
        12749d3e325192de967ab481a2e99fc6 --pc-wiener -- \
        --still --still-pc-wiener
    run_case "$TESTDATA/pub_1920x1080_10f.y4m" still_hd_gdf \
        fe970eaae16606e579688386a2892811 --gdf -- \
        --still --still-gdf
    run_case "$TESTDATA/pub_1920x1080_10f.y4m" still_hd_gdf_wiener_chroma \
        0a0c2dca4bb808b1b8c89ebb1ff03070 --gdf --wiener --chroma-lr -- \
        --still --still-gdf --still-wiener --still-chroma-lr
    run_case "$TESTDATA/pub_1920x1080_10f.y4m" still_hd_pc_wiener_chroma \
        6ee1392ce2781016e8c9b2d746793cb6 --pc-wiener --chroma-lr -- \
        --still --still-pc-wiener --still-chroma-lr
    run_case "$TESTDATA/pub_1920x1080_10f.y4m" still_hd_ccso_edge \
        812c0f027fb2442c7593836101e36c13 --ccso-edge -- \
        --still --still-ccso-edge
else
    echo "loop_restoration_gate: --quick, skipping 1080p cases"
fi

# --- film grain (decoder-side output) -------------------------------------
# run_case verifies the IVF MD5 and recon exactness with --filmgrain 0; this
# wrapper additionally proves the decoder actually synthesizes grain.
run_grain_case() {
    local fixture=$1 name=$2 expected=$3
    shift 3
    local input
    input=$(fixture_path "$fixture")
    if [ ! -f "$input" ]; then
        echo "SKIP $name: missing $fixture" >&2
        return 0
    fi
    run_case "$fixture" "$name" "$expected" --grain -- "$@"
    local out="$TMP/${name}.ivf"
    local off="$TMP/${name}.grain_off.y4m"
    local on="$TMP/${name}.grain_on.y4m"
    if ! "$NOGRAIN" -q -i "$out" -o "$off" --threads 1 --strict 1 \
            >"$TMP/${name}.off.log" 2>&1 ||
        ! "$DAV2D" -q -i "$out" -o "$on" --threads 1 --strict 1 \
            --filmgrain 1 >"$TMP/${name}.on.log" 2>&1; then
        echo "FAIL $name: grain decode failed" >&2
        failures=$((failures + 1))
        return 0
    fi
    if cmp -s "$off" "$on"; then
        echo "FAIL $name: --filmgrain 1 produced no synthesized grain" >&2
        failures=$((failures + 1))
        return 0
    fi
    echo "PASS $name: grain synthesized"
    pass=$((pass + 1))
}

run_grain_case tests/data/basic_64x64_1f.y4m still_64_grain \
    54153cc8c187b7034e67aaea72237366 --still --still-grain
run_grain_case tests/data/basic_640x360_1f.y4m still_360_grain \
    1e0f2775d39764041482ebe7e052bbf1 --still --still-grain
run_grain_case tests/data/basic_64x64_1f.y4m still_64_grain_adaptive \
    023216546d768a971d4a2982d6d3f534 --still --still-grain --still-grain-adaptive
run_grain_case tests/data/basic_640x360_1f.y4m still_360_grain_adaptive \
    05e22ce4446ac5b4bc9525c43870bf1f --still --still-grain --still-grain-adaptive
if [ "$QUICK" -eq 0 ]; then
    run_grain_case "$TESTDATA/pub_1920x1080_10f.y4m" still_hd_grain \
        052af22ce78f949602021ba69bc65852 --still --still-grain
    run_grain_case "$TESTDATA/pub_1920x1080_10f.y4m" still_hd_grain_adaptive \
        40d3199500890571624f7b9d47028c57 --still --still-grain --still-grain-adaptive
fi

echo "loop_restoration_gate: $pass passed, $failures failed"
[ "$failures" -eq 0 ]
