#!/usr/bin/env bash
# Encode one 64x64 M0 still frame and require dav2d to decode it bit-exactly.
set -euo pipefail

if [ $# -ne 2 ]; then
    echo "usage: $0 <av2enc-binary> <project-root>" >&2
    exit 2
fi

BIN=$1
ROOT=$2
INPUT="$ROOT/tests/data/m0_64x64_1f.y4m"
EXPECTED_MD5=7efc3162bc0a4785685eba291a8096fd

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
OUT="$TMP/m0_64x64.ivf"

"$BIN" "$INPUT" -o "$OUT" --frames 1 --qp 90

if [ ! -s "$OUT" ]; then
    echo "error: encoder produced no output" >&2
    exit 1
fi

set +e
EXPECTED_MD5="$EXPECTED_MD5" DAV2D="${DAV2D:-}" \
    "$ROOT/tools/validate_dav2d.sh" "$OUT" "$INPUT"
RC=$?
set -e

if [ "$RC" -eq 127 ]; then
    echo "SKIP: dav2d binary not found" >&2
    exit 77
fi
if [ "$RC" -ne 0 ]; then
    echo "error: dav2d gate failed (exit $RC)" >&2
    exit 1
fi

echo "PASS: m0_encode_smoke"
