#!/usr/bin/env bash
# Decode gate for the M0 edge-partition work.
#
#   m0_edge_smoke.sh <av2enc-binary> <gen_y4m-binary> <project-root>
#
# Generates deterministic stills at the five reported failing sizes plus the
# pinned regressions, encodes through the normal CLI (no --m1), and requires
# the pinned dav2d build to decode each stream with --strict 1.
#
# Where M0 can represent the geometry it must produce the pinned MD5 and must
# NOT fall back. 136x88 / 518x322 need sub-32-pixel edge blocks that M0 does
# not code; the CLI must print the fallback note and M1 must decode them.
#
# Exit 77 = no dav2d binary (CTest skip).
set -euo pipefail

if [ $# -ne 3 ]; then
    echo "usage: $0 <av2enc> <gen_y4m> <project-root>" >&2
    exit 2
fi

BIN=$1
GEN=$2
ROOT=$3

DECODER=""
for candidate in "${DAV2D:-}" "$ROOT/tools/bin/dav2d" \
                 "$ROOT/tools/bin/dav2d-macos"; do
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then
        DECODER=$candidate
        break
    fi
done
if [ -z "$DECODER" ]; then
    echo "SKIP: dav2d binary not found" >&2
    exit 77
fi

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

fail() {
    echo "error: $*" >&2
    exit 1
}

run_case() {
    local name=$1
    local width=$2
    local height=$3
    local expect_fallback=$4
    local expected_md5=$5
    local y4m="$TMP/${name}.y4m"
    local ivf="$TMP/${name}.ivf"
    local dec="$TMP/${name}_dec.y4m"
    local err="$TMP/${name}.err"

    "$GEN" "$width" "$height" "$y4m" || fail "$name: generator failed"

    set +e
    "$BIN" "$y4m" -o "$ivf" --frames 1 --qp 90 >/dev/null 2>"$err"
    local rc=$?
    set -e
    [ "$rc" -eq 0 ] || fail "$name: encode failed (exit $rc): $(cat "$err")"

    if [ "$expect_fallback" = "1" ]; then
        grep -q "falling back to the M1 still-picture path" "$err" ||
            fail "$name: expected the M1 fallback note, got: $(cat "$err")"
    else
        if grep -q "falling back to the M1 still-picture path" "$err"; then
            fail "$name: unexpected M1 fallback: $(cat "$err")"
        fi
    fi

    "$DECODER" -q --strict 1 -i "$ivf" -o "$dec" --threads 1 ||
        fail "$name: dav2d --strict 1 failed to decode"

    if [ -n "$expected_md5" ]; then
        local got
        got=$("$DECODER" -q -i "$ivf" --muxer md5 -o - --threads 1 2>/dev/null | tail -1)
        [ "$got" = "$expected_md5" ] ||
            fail "$name: decoded MD5 $got != $expected_md5"
    fi
    echo "PASS: $name"
}

# Five reported sizes. The first three are representable by M0 after the edge
# recursion + transform fixes; the last two need sub-32 blocks and fall back.
run_case 1280x720      1280  720 0 62a9bd644486622dea97a135452b41b3
run_case 136x88         136   88 1 ""
run_case 416x208        416  208 0 8e08a5cc791e07da4b562c9295829075
run_case 518x322        518  322 1 ""
run_case 532x368        532  368 0 e9cbf5aa78ebac6aa0f9e14b91559ea7

# Regressions: pinned 64x64 / 640x360 sizes and the native 1080p frame.
run_case 64x64           64   64 0 d4a98234aee1f8384af95b9aaed95d54
run_case 640x360        640  360 0 3df400935025df6e914830349d38a9a9
run_case 1920x1080     1920 1080 0 fbbef81e45fcff162dde6c71a0b56a7d
# 32x8 edge-leaf path (bottom remainder 2 pixels).
run_case 128x66         128   66 0 90e1f9ee08e29e43cbe5051b54f5f475

echo "PASS: m0_edge_smoke"
