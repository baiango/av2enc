#!/usr/bin/env bash
# Encode M0 still frames and require dav2d to decode them bit-exactly.
#
# Cases:
#   64x64     - full superblock, 32x32 coding blocks
#   640x360   - partial bottom superblock (360 = 5*64 + 40)
#
# Usage: m0_encode_smoke.sh <av2enc-binary> <project-root>
set -euo pipefail

if [ $# -ne 2 ]; then
    echo "usage: $0 <av2enc-binary> <project-root>" >&2
    exit 2
fi

BIN=$1
ROOT=$2
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

run_case() {
    local name=$1
    local input=$2
    local expected_md5=$3
    local out="$TMP/m0_${name}.ivf"

    if [ ! -f "$input" ]; then
        echo "SKIP: missing test input $input" >&2
        return 77
    fi

    "$BIN" "$input" -o "$out" --frames 1 --qp 90

    if [ ! -s "$out" ]; then
        echo "error: encoder produced no output for $name" >&2
        return 1
    fi

    set +e
    EXPECTED_MD5="$expected_md5" DAV2D="${DAV2D:-}" \
        "$ROOT/tools/validate_dav2d.sh" "$out" "$input"
    local rc=$?
    set -e

    if [ "$rc" -eq 127 ]; then
        echo "SKIP: dav2d binary not found" >&2
        return 77
    fi
    if [ "$rc" -ne 0 ]; then
        echo "error: dav2d gate failed for $name (exit $rc)" >&2
        return 1
    fi
    echo "PASS: $name"
    return 0
}

set +e
run_case 64x64 "$ROOT/tests/data/m0_64x64_1f.y4m" \
    7efc3162bc0a4785685eba291a8096fd
RC=$?
set -e
if [ "$RC" -eq 77 ]; then exit 77; fi
if [ "$RC" -ne 0 ]; then exit 1; fi

set +e
run_case 640x360 "$ROOT/tests/data/m0_640x360_1f.y4m" \
    56f11d4a63ff07f2356ef5064b84fe0e
RC=$?
set -e
if [ "$RC" -eq 77 ]; then exit 77; fi
if [ "$RC" -ne 0 ]; then exit 1; fi

echo "PASS: m0_encode_smoke"
