#!/usr/bin/env bash
# Chroma-format gate: encode 64x64 and 640x360 stills for 4:2:2, 4:4:4 and
# 4:0:0 (plus 4:2:0 as a sanity case), require dav2d --strict 1 to decode each
# stream, require the encoder reconstruction to be bit-exact against the
# decoder output (0 diffs), and compare pinned MD5 goldens.
#
# Usage: ./gate.sh
#
# Environment:
#   DAV2D=/path/to/dav2d     override the pinned decoder
#   CHROMA_UPDATE_GOLDENS=1  rewrite goldens/chroma_goldens.txt from this run
#
# Exit codes: 0 pass, 1 failure, 77 decoder/test media missing.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
BUILD="${TMPDIR:-/tmp}/av2enc_chroma_build"
TOOL="$BUILD/chroma_still"
GOLDENS="$ROOT/src/encoder/format/chroma/goldens/chroma_goldens.txt"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

if [ -z "${DAV2D:-}" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then
            DAV2D=$candidate
            break
        fi
    done
fi
if [ -z "${DAV2D:-}" ] || [ ! -x "$DAV2D" ]; then
    echo "SKIP: dav2d not found (set DAV2D=/path/to/dav2d)" >&2
    exit 77
fi

if [ ! -x "$ROOT/tools/validate_dav2d.sh" ]; then
    echo "error: tools/validate_dav2d.sh is missing" >&2
    exit 1
fi

# Rebuild so the gate always exercises the current sources.
"$ROOT/src/encoder/format/chroma/build_and_test.sh" >/dev/null
if [ ! -x "$TOOL" ]; then
    echo "error: build did not produce $TOOL" >&2
    exit 1
fi

md5_of() {
    if command -v md5 >/dev/null 2>&1; then
        md5 -q "$1"
    else
        md5sum "$1" | awk '{print $1}'
    fi
}

RESULT="$TMP/results.txt"
: > "$RESULT"
FAILED=0
MISSING=0

run_case() {
    local format=$1
    local input=$2
    local width=$3
    local height=$4

    if [ ! -f "$input" ]; then
        echo "SKIP: missing test input $input" >&2
        MISSING=1
        return 0
    fi

    local name="${format}_${width}x${height}"
    local ivf="$TMP/$name.ivf"
    local recon="$TMP/$name.recon.y4m"
    local decoded="$TMP/$name.decoded.y4m"

    "$TOOL" "$input" -o "$ivf" --format "$format" --recon "$recon" \
        > "$TMP/$name.encode.log"
    cat "$TMP/$name.encode.log"

    # Pinned gate: validate_dav2d.sh uses --strict 1 by default and checks the
    # IVF container, resolution and frame count.
    if ! "$ROOT/tools/validate_dav2d.sh" "$ivf"; then
        echo "error: validate_dav2d.sh failed for $name" >&2
        FAILED=1
        return 0
    fi

    # Decode once more explicitly with --strict 1 so the reconstruction
    # comparison and the MD5 golden see exactly the strict-mode output.
    if ! "$DAV2D" --strict 1 -q -i "$ivf" -o "$decoded" --threads 1; then
        echo "error: dav2d --strict 1 failed for $name" >&2
        FAILED=1
        return 0
    fi

    if ! python3 "$ROOT/src/encoder/format/chroma/compare_y4m.py" \
            "$recon" "$decoded"; then
        echo "error: reconstruction mismatch for $name" >&2
        FAILED=1
        return 0
    fi

    local ivf_md5 decoded_md5
    ivf_md5=$(md5_of "$ivf")
    decoded_md5=$("$DAV2D" -q -i "$ivf" --muxer md5 -o - 2>/dev/null | tail -1)
    echo "golden:  $format $width $height ivf=$ivf_md5 decoded=$decoded_md5"
    echo "$format $width $height $ivf_md5 $decoded_md5" >> "$RESULT"
}

for format in 422 444 400 420; do
    run_case "$format" "$ROOT/tests/data/m0_64x64_1f.y4m" 64 64
    run_case "$format" "$ROOT/tests/data/m0_640x360_1f.y4m" 640 360
done

if [ "$MISSING" -eq 1 ] && [ ! -s "$RESULT" ]; then
    echo "SKIP: no test media available" >&2
    exit 77
fi

if [ "$FAILED" -ne 0 ]; then
    echo "FAIL: one or more chroma gate cases failed" >&2
    exit 1
fi

if [ "${CHROMA_UPDATE_GOLDENS:-0}" = "1" ]; then
    mkdir -p "$(dirname "$GOLDENS")"
    {
        echo "# Chroma-format gate goldens (dav2d f9dafbc, --strict 1)."
        echo "# Rebuild with: CHROMA_UPDATE_GOLDENS=1 ./src/encoder/format/chroma/gate.sh"
        echo "# format width height ivf_md5 decoded_md5"
        sort -k1,1 -k2,2n -k3,3n "$RESULT"
    } > "$GOLDENS"
    echo "updated: $GOLDENS"
elif [ -f "$GOLDENS" ]; then
    while read -r format width height ivf_md5 decoded_md5; do
        case "$format" in
            ''|'#'*) continue ;;
        esac
        got=$(awk -v f="$format" -v w="$width" -v h="$height" \
            '$1 == f && $2 == w && $3 == h { print $4 " " $5 }' "$RESULT")
        if [ "$got" != "$ivf_md5 $decoded_md5" ]; then
            echo "error: golden mismatch for $format ${width}x${height}: " \
                 "got '$got', expected '$ivf_md5 $decoded_md5'" >&2
            FAILED=1
        else
            echo "PASS golden: $format ${width}x${height}"
        fi
    done < "$GOLDENS"
    if [ "$FAILED" -ne 0 ]; then
        exit 1
    fi
fi

echo "PASS: chroma format gate (4:2:2, 4:4:4, 4:0:0; 64x64 and 640x360)"
