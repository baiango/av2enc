#!/usr/bin/env bash
# Neutral metadata OBU gate (§5.17.2): --metadata-timecode / --metadata-scan-type.
#
# For one still and one 3-frame video stream:
#   * the metadata stream must differ from the baseline;
#   * pinned dav2d --strict 1 must decode both to the same Y4M MD5 (the pinned
#     decoder ignores OBU types 8/9 but must not change coded output order);
#   * the module scanner must self-parse exactly two supported suffix units;
#   * the pinned still metadata stream MD5 must match.
#
# Usage: gate_neutral_metadata.sh --bin build-int/av2enc [--dav2d PATH]
# Exit: 0 pass, 1 fail, 2 usage, 77 skip.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
BIN=""
DAV2D="${DAV2D:-}"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

while [ $# -gt 0 ]; do
    case "$1" in
        --bin) BIN=$2; shift 2 ;;
        --dav2d) DAV2D=$2; shift 2 ;;
        -h|--help) sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "gate_neutral_metadata: unknown argument $1" >&2; exit 2 ;;
    esac
done
[ -n "$BIN" ] && [ -x "$BIN" ] || { echo "gate_neutral_metadata: --bin required" >&2; exit 2; }
if [ -z "$DAV2D" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
    done
fi
[ -n "$DAV2D" ] && [ -x "$DAV2D" ] || { echo "gate_neutral_metadata: no dav2d" >&2; exit 77; }
[ -f "$ROOT/tests/data/basic_640x360_1f.y4m" ] || {
    echo "gate_neutral_metadata: missing fixture" >&2; exit 77; }

EXPECTED_STILL_MD5=5624dda84b751c7d4485d8afebe1a395
SCANNER=""
scan() { "$SCANNER" --input "$1" --scan "$@"; }
md5_of() { md5 -q "$1" 2>/dev/null || md5sum "$1" | awk '{print $1}'; }
dec_md5() {
    "$DAV2D" -q -i "$1" -o /dev/null --muxer md5 --threads 1 --strict 1 \
        2>/dev/null | tail -1
}

# Build the module scanner (it is not a CMake target).
"$ROOT/src/frame/metadata/types/build.sh" "$TMP/build" >"$TMP/build.log" 2>&1 || {
    echo "FAIL: metadata_types build" >&2; cat "$TMP/build.log" >&2; exit 1; }
SCANNER="$TMP/build/append_metadata_types"

fail=0

# --- still -----------------------------------------------------------------
"$BIN" "$ROOT/tests/data/basic_640x360_1f.y4m" -o "$TMP/s_base.ivf" --still \
    >"$TMP/s_base.log" 2>&1 || { echo "FAIL: still base encode" >&2; fail=1; }
"$BIN" "$ROOT/tests/data/basic_640x360_1f.y4m" -o "$TMP/s_meta.ivf" --still \
    --metadata-timecode --metadata-scan-type >"$TMP/s_meta.log" 2>&1 || {
    echo "FAIL: still metadata encode" >&2; cat "$TMP/s_meta.log" >&2; fail=1; }
if [ "$fail" -eq 0 ]; then
    if cmp -s "$TMP/s_base.ivf" "$TMP/s_meta.ivf"; then
        echo "FAIL: still metadata stream is byte-identical to the baseline" >&2; fail=1
    fi
    base_dec=$(dec_md5 "$TMP/s_base.ivf")
    meta_dec=$(dec_md5 "$TMP/s_meta.ivf")
    [ "$base_dec" = "$meta_dec" ] || {
        echo "FAIL: still decoded MD5 changed ($base_dec vs $meta_dec)" >&2; fail=1; }
    got=$(md5_of "$TMP/s_meta.ivf")
    [ "$got" = "$EXPECTED_STILL_MD5" ] || {
        echo "FAIL: still metadata MD5 $got, expected $EXPECTED_STILL_MD5" >&2; fail=1; }
    scan_out=$($SCANNER --input "$TMP/s_meta.ivf" --scan)
    echo "$scan_out" | grep -q "supported_units=2" || {
        echo "FAIL: scanner did not find two supported units" >&2
        echo "$scan_out" >&2; fail=1; }
    echo "$scan_out" | grep -q "record=timecode suffix=1" || {
        echo "FAIL: timecode unit missing/misplaced" >&2; fail=1; }
    echo "$scan_out" | grep -q "record=scan_type suffix=1" || {
        echo "FAIL: scan_type unit missing/misplaced" >&2; fail=1; }
fi

# --- video -----------------------------------------------------------------
SRC="$ROOT/testdata/shine_1280x720_60f.y4m"
if [ -f "$SRC" ]; then
    "$BIN" "$SRC" -o "$TMP/v_base.ivf" --video --p-frames 2 \
        >"$TMP/v_base.log" 2>&1 || { echo "FAIL: video base encode" >&2; fail=1; }
    "$BIN" "$SRC" -o "$TMP/v_meta.ivf" --video --p-frames 2 \
        --metadata-timecode --metadata-scan-type >"$TMP/v_meta.log" 2>&1 || {
        echo "FAIL: video metadata encode" >&2; cat "$TMP/v_meta.log" >&2; fail=1; }
    if [ "$fail" -eq 0 ]; then
        cmp -s "$TMP/v_base.ivf" "$TMP/v_meta.ivf" && {
            echo "FAIL: video metadata stream is byte-identical" >&2; fail=1; }
        [ "$(dec_md5 "$TMP/v_base.ivf")" = "$(dec_md5 "$TMP/v_meta.ivf")" ] || {
            echo "FAIL: video decoded MD5 changed" >&2; fail=1; }
        $SCANNER --input "$TMP/v_meta.ivf" --scan | grep -q \
            "total_supported_units=2" || {
            echo "FAIL: video scanner did not find the metadata units" >&2; fail=1; }
    fi
else
    echo "SKIP video: missing $SRC" >&2
fi

[ "$fail" -eq 0 ] || exit 1
echo "gate_neutral_metadata: PASS (still md5 $(md5_of "$TMP/s_meta.ivf"))"
