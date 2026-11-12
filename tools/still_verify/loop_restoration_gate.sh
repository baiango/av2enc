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

# run_case <fixture> <name> <expected_md5> <dump args...> -- <encode args...>
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
    local got
    got=$(md5 -q "$out" 2>/dev/null || md5sum "$out" | awk '{print $1}')
    if [ "$got" != "$expected" ]; then
        echo "FAIL $name: IVF md5 $got, expected $expected" >&2
        failures=$((failures + 1))
        return 0
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
    echo "PASS $name ($(wc -c <"$out" | tr -d ' ') B, md5 $expected)"
    pass=$((pass + 1))
}

# --- 64x64 -----------------------------------------------------------------
run_case tests/data/basic_64x64_1f.y4m still_64_wiener \
    f2c005d7873c9c493e1b41473c8f3a54 --wiener -- \
    --still --still-wiener
run_case tests/data/basic_64x64_1f.y4m still_64_chroma_lr \
    0e291675117f627783606d4bf0b3ca43 --chroma-lr -- \
    --still --still-chroma-lr
run_case tests/data/basic_64x64_1f.y4m still_64_pc_wiener \
    508ab8ec82f6b5fc0a1b6a1980eb0c50 --pc-wiener -- \
    --still --still-pc-wiener
run_case tests/data/basic_64x64_1f.y4m still_64_gdf \
    1714218f7dab3dbaaa334885a38c2a88 --gdf -- \
    --still --still-gdf
run_case tests/data/basic_64x64_1f.y4m still_64_ccso_edge \
    b83169960ce4e75c6ee5c1d933901923 --ccso-edge -- \
    --still --still-ccso-edge

# --- 640x360 ---------------------------------------------------------------
run_case tests/data/basic_640x360_1f.y4m still_360_wiener \
    4bbede8ad74b6552a2404e5c23431adb --wiener -- \
    --still --still-wiener
run_case tests/data/basic_640x360_1f.y4m still_360_chroma_lr \
    e6a83f2fec7bc4537b868cbbdc389504 --chroma-lr -- \
    --still --still-chroma-lr
run_case tests/data/basic_640x360_1f.y4m still_360_wiener_chroma \
    a7a4c06967bd4704dbb72f4de8e79f91 --wiener --chroma-lr -- \
    --still --still-wiener --still-chroma-lr
run_case tests/data/basic_640x360_1f.y4m still_360_cdef_ccso_wiener \
    2f4ffb0c0dc62d713f5a6a9f5e236240 --cdef --ccso --wiener --chroma-lr -- \
    --still --still-cdef --still-ccso --still-wiener --still-chroma-lr
run_case tests/data/basic_640x360_1f.y4m still_360_pc_wiener \
    5236b0dd917d7cd62832b9d79b88f007 --pc-wiener -- \
    --still --still-pc-wiener
run_case tests/data/basic_640x360_1f.y4m still_360_cdef_ccso_pc_wiener \
    cffddedb6fac60b29d38e2caf5f9b6f4 --cdef --ccso --pc-wiener -- \
    --still --still-cdef --still-ccso --still-pc-wiener
run_case tests/data/basic_640x360_1f.y4m still_360_gdf \
    b07c1327de42ec038e25dc7c74561984 --gdf -- \
    --still --still-gdf
run_case tests/data/basic_640x360_1f.y4m still_360_cdef_ccso_gdf \
    ebe9f970bd7a3a73e4b38ae8e3393354 --cdef --ccso --gdf -- \
    --still --still-cdef --still-ccso --still-gdf

# --- §7.20 pre-GDF CdefFrame / LrFrame split ------------------------------
# GDF runs after Wiener/PC-Wiener and samples CdefFrame, so the restoration
# output is a distinct buffer from the GDF source.  These combinations were
# rejected before the split; each one is recon-exact.
run_case tests/data/basic_64x64_1f.y4m still_64_gdf_wiener \
    37569cd29b22dedb926d31fad17d4aa2 --gdf --wiener -- \
    --still --still-gdf --still-wiener
run_case tests/data/basic_64x64_1f.y4m still_64_gdf_chroma \
    a153d0b08391512160ef3ab141071895 --gdf --chroma-lr -- \
    --still --still-gdf --still-chroma-lr
run_case tests/data/basic_640x360_1f.y4m still_360_gdf_wiener \
    fd67ddab058eb86485eb9cdb986ffbb2 --gdf --wiener -- \
    --still --still-gdf --still-wiener
run_case tests/data/basic_640x360_1f.y4m still_360_gdf_chroma \
    636137a428463bf0a6ecf59375c09a5a --gdf --chroma-lr -- \
    --still --still-gdf --still-chroma-lr
run_case tests/data/basic_640x360_1f.y4m still_360_gdf_wiener_chroma \
    a55b3f920a57d3872dd7a336d664a65f --gdf --wiener --chroma-lr -- \
    --still --still-gdf --still-wiener --still-chroma-lr
run_case tests/data/basic_640x360_1f.y4m still_360_cdef_ccso_gdf_lr \
    1dfb02e1ccc42d31e8176caae1ad0f9f --cdef --ccso --gdf --wiener --chroma-lr -- \
    --still --still-cdef --still-ccso --still-gdf --still-wiener --still-chroma-lr
run_case tests/data/basic_640x360_1f.y4m still_360_gdf_pc_wiener \
    6baecfd7d4ed2bd6f2328358f394a7f8 --gdf --pc-wiener -- \
    --still --still-gdf --still-pc-wiener

# --- PC-Wiener luma + chroma Wiener NS ------------------------------------
# Plane 0 is RESTORE_PC_WIENER and planes 1/2 are RESTORE_WIENER_NONSEP; the
# frame header codes both rows and the tile carries use_pc_wiener for luma
# units followed by use_wiener_ns for chroma units.
run_case tests/data/basic_64x64_1f.y4m still_64_pc_wiener_chroma \
    afb66bbe431d39f8a5f6867ef6a2b865 --pc-wiener --chroma-lr -- \
    --still --still-pc-wiener --still-chroma-lr
run_case tests/data/basic_640x360_1f.y4m still_360_cdef_ccso_pc_wiener_chroma \
    d23bc36b9d85ec34e70ddce53baccdfe --cdef --ccso --pc-wiener --chroma-lr -- \
    --still --still-cdef --still-ccso --still-pc-wiener --still-chroma-lr
run_case tests/data/basic_640x360_1f.y4m still_360_gdf_pc_wiener_chroma \
    ba08c8ef70953cf54de375ddc8b9cdd4 --gdf --pc-wiener --chroma-lr -- \
    --still --still-gdf --still-pc-wiener --still-chroma-lr
run_case tests/data/basic_640x360_1f.y4m still_360_cdef_ccso_gdf_pc_wiener_chroma \
    c706c3709a5ffc936f62f161e2986347 --cdef --ccso --gdf --pc-wiener --chroma-lr -- \
    --still --still-cdef --still-ccso --still-gdf --still-pc-wiener --still-chroma-lr
run_case tests/data/basic_640x360_1f.y4m still_360_ccso_edge \
    f0f3fb9693d9cb5bf2f86cef305004bb --ccso-edge -- \
    --still --still-ccso-edge
run_case tests/data/basic_640x360_1f.y4m still_360_cdef_ccso_edge \
    a06c1765e37827cbb840ed018b154037 --cdef --ccso-edge -- \
    --still --still-cdef --still-ccso-edge

# --- CDEF cdef_on_skip_txfm frame bit (CDEF_ON_SKIP_TXFM_ADAPTIVE) --------
# A single-picture header infers ADAPTIVE, so the frame bit is coded.  Intra
# frames derive skip = 0 either way, so only the header bit moves; the recon
# stays exact.  The sequence-level DISABLED/ALWAYS_ON modes are not codeable
# with single_picture_header_flag = 1; the M2 tiled gate covers those.
run_case tests/data/basic_64x64_1f.y4m still_64_cdef_skip_txfm \
    cf86fce480936dd5abb4ece5f4983ba4 --cdef --cdef-on-skip-txfm -- \
    --still --still-cdef --still-cdef-on-skip-txfm
run_case tests/data/basic_640x360_1f.y4m still_360_cdef_skip_txfm \
    8815c209c2b9e1838a7c56ebb4c3f8bd --cdef --cdef-on-skip-txfm -- \
    --still --still-cdef --still-cdef-on-skip-txfm
run_case tests/data/basic_640x360_1f.y4m still_360_cdef_ccso_lr_skip_txfm \
    5e823112b09a8feada7f0965b70c4980 --cdef --cdef-on-skip-txfm --ccso --wiener --chroma-lr -- \
    --still --still-cdef --still-ccso --still-wiener --still-chroma-lr \
    --still-cdef-on-skip-txfm

# --- optional 1080p (testdata is a gitignored symlink) --------------------
if [ "$QUICK" -eq 0 ]; then
    run_case "$TESTDATA/shine_1920x1080_10f.y4m" still_hd_wiener_chroma \
        1b266d52a1a77e858a9c9b1e052a1364 --wiener --chroma-lr -- \
        --still --still-wiener --still-chroma-lr
    run_case "$TESTDATA/shine_1920x1080_10f.y4m" still_hd_pc_wiener \
        448ce9a6613192fc703efba24274d19c --pc-wiener -- \
        --still --still-pc-wiener
    run_case "$TESTDATA/shine_1920x1080_10f.y4m" still_hd_gdf \
        1a0abbc45131695752100f6434b41b3a --gdf -- \
        --still --still-gdf
    run_case "$TESTDATA/shine_1920x1080_10f.y4m" still_hd_gdf_wiener_chroma \
        a4cfd4cb671dd7f38ccbcdf1f76e58c5 --gdf --wiener --chroma-lr -- \
        --still --still-gdf --still-wiener --still-chroma-lr
    run_case "$TESTDATA/shine_1920x1080_10f.y4m" still_hd_pc_wiener_chroma \
        6270eb72216b965d90584501da119680 --pc-wiener --chroma-lr -- \
        --still --still-pc-wiener --still-chroma-lr
    run_case "$TESTDATA/shine_1920x1080_10f.y4m" still_hd_ccso_edge \
        f5febd934e5e310630d9098e6794b93b --ccso-edge -- \
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
    2c487ae7f9abb1f51d33c335dc07feac --still --still-grain
run_grain_case tests/data/basic_640x360_1f.y4m still_360_grain \
    b23601e81d0600debc5b6d84f91be11a --still --still-grain
run_grain_case tests/data/basic_64x64_1f.y4m still_64_grain_adaptive \
    9db402e9d668c1143451e34d47b9b2aa --still --still-grain --still-grain-adaptive
run_grain_case tests/data/basic_640x360_1f.y4m still_360_grain_adaptive \
    78dfef3a53b7b315fa9150ad915f1ef6 --still --still-grain --still-grain-adaptive
if [ "$QUICK" -eq 0 ]; then
    run_grain_case "$TESTDATA/shine_1920x1080_10f.y4m" still_hd_grain \
        3b380c0cfed0f5538724465cf02297ed --still --still-grain
    run_grain_case "$TESTDATA/shine_1920x1080_10f.y4m" still_hd_grain_adaptive \
        80f5de0949b8800c7cff48f3447455ab --still --still-grain --still-grain-adaptive
fi

echo "loop_restoration_gate: $pass passed, $failures failed"
[ "$failures" -eq 0 ]
