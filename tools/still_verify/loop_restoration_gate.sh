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
#   tools/still_verify/loop_restoration_gate.sh --bin build-int/av2enc \
#       --dump build-int/still_dump
#
# Env: DAV2D overrides the decoder path.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BIN=""
DUMP=""
DAV2D="${DAV2D:-}"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

while [ $# -gt 0 ]; do
    case "$1" in
        --bin) BIN=$2; shift 2 ;;
        --dump) DUMP=$2; shift 2 ;;
        --dav2d) DAV2D=$2; shift 2 ;;
        -h|--help)
            sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "loop_restoration_gate: unknown argument $1" >&2; exit 2 ;;
    esac
done

if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
    echo "loop_restoration_gate: --bin must point at an av2enc executable" >&2
    exit 2
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

# Film grain is decoder-side output, so the recon comparison always decodes
# with synthesis disabled.  The wrapper is harmless for non-grain streams.
NOGRAIN="$TMP/dav2d_nograin"
printf '#!/bin/sh\nexec "%s" --filmgrain 0 "$@"\n' "$DAV2D" >"$NOGRAIN"
chmod +x "$NOGRAIN"

failures=0
pass=0

# run_case <fixture> <name> <expected_md5> <dump args...> -- <encode args...>
run_case() {
    local fixture=$1 name=$2 expected=$3
    shift 3
    local dump_args=()
    while [ "$1" != "--" ]; do dump_args+=("$1"); shift; done
    shift
    local input="$ROOT/$fixture"
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
run_case tests/data/basic_640x360_1f.y4m still_360_ccso_edge \
    f0f3fb9693d9cb5bf2f86cef305004bb --ccso-edge -- \
    --still --still-ccso-edge
run_case tests/data/basic_640x360_1f.y4m still_360_cdef_ccso_edge \
    a06c1765e37827cbb840ed018b154037 --cdef --ccso-edge -- \
    --still --still-cdef --still-ccso-edge

# --- optional 1080p (testdata is a gitignored symlink) --------------------
run_case testdata/shine_1920x1080_10f.y4m still_hd_wiener_chroma \
    1b266d52a1a77e858a9c9b1e052a1364 --wiener --chroma-lr -- \
    --still --still-wiener --still-chroma-lr
run_case testdata/shine_1920x1080_10f.y4m still_hd_pc_wiener \
    448ce9a6613192fc703efba24274d19c --pc-wiener -- \
    --still --still-pc-wiener
run_case testdata/shine_1920x1080_10f.y4m still_hd_gdf \
    1a0abbc45131695752100f6434b41b3a --gdf -- \
    --still --still-gdf
run_case testdata/shine_1920x1080_10f.y4m still_hd_ccso_edge \
    f5febd934e5e310630d9098e6794b93b --ccso-edge -- \
    --still --still-ccso-edge

# --- film grain (decoder-side output) -------------------------------------
# run_case verifies the IVF MD5 and recon exactness with --filmgrain 0; this
# wrapper additionally proves the decoder actually synthesizes grain.
run_grain_case() {
    local fixture=$1 name=$2 expected=$3
    shift 3
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
run_grain_case testdata/shine_1920x1080_10f.y4m still_hd_grain \
    3b380c0cfed0f5538724465cf02297ed --still --still-grain

echo "loop_restoration_gate: $pass passed, $failures failed"
[ "$failures" -eq 0 ]
