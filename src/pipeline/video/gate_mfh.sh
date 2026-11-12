#!/usr/bin/env bash
# Multi-frame-header gate (§5.7, --video-mfh) on the non-single-picture path.
#
# Encodes a 3-frame P sequence with and without one minimal MFH (id 1).  The
# MFH is emitted in the key temporal unit; every frame header then sets
# cur_mfh_id = 1 and omits seq_header_id_in_frame_header (§5.18.2).  Requires:
#   * the streams to differ;
#   * dav2d --strict 1 to decode both to the same pixel MD5;
#   * the pinned --video-mfh stream MD5.
# The single-picture still path is deliberately not wired: the pinned decoder
# rejects MFH there (docs/negative-results.md).
#
# Usage: gate_mfh.sh --bin build-int/av2enc [--dav2d PATH]
# Exit: 0 pass, 1 fail, 2 usage, 77 skip.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
BIN=""
DAV2D="${DAV2D:-}"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

while [ $# -gt 0 ]; do
    case "$1" in
        --bin) BIN=$2; shift 2 ;;
        --dav2d) DAV2D=$2; shift 2 ;;
        -h|--help) sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "gate_mfh: unknown argument $1" >&2; exit 2 ;;
    esac
done
[ -n "$BIN" ] && [ -x "$BIN" ] || { echo "gate_mfh: --bin required" >&2; exit 2; }
if [ -z "$DAV2D" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
    done
fi
[ -n "$DAV2D" ] && [ -x "$DAV2D" ] || { echo "gate_mfh: no dav2d" >&2; exit 77; }
SRC="$ROOT/testdata/shine_1280x720_60f.y4m"
[ -f "$SRC" ] || { echo "gate_mfh: missing $SRC" >&2; exit 77; }
EXPECTED_MFH_MD5=fcd0a8808a76edff3a2f0f2b9eba1a41

md5_of() { md5 -q "$1" 2>/dev/null || md5sum "$1" | awk '{print $1}'; }
dec_md5() {
    "$DAV2D" -q -i "$1" --muxer md5 -o - --threads 1 --strict 1 2>/dev/null | tail -1
}

"$BIN" "$SRC" -o "$TMP/base.ivf" --video --p-frames 2 \
    >"$TMP/base.log" 2>&1 || {
    echo "FAIL: base encode" >&2; cat "$TMP/base.log" >&2; exit 1; }
"$BIN" "$SRC" -o "$TMP/mfh.ivf" --video --p-frames 2 --video-mfh \
    >"$TMP/mfh.log" 2>&1 || {
    echo "FAIL: --video-mfh encode" >&2; cat "$TMP/mfh.log" >&2; exit 1; }

cmp -s "$TMP/base.ivf" "$TMP/mfh.ivf" && {
    echo "FAIL: --video-mfh stream is byte-identical" >&2; exit 1; }
[ "$(dec_md5 "$TMP/base.ivf")" = "$(dec_md5 "$TMP/mfh.ivf")" ] || {
    echo "FAIL: decoded MD5 changed" >&2; exit 1; }

got=$(md5_of "$TMP/mfh.ivf")
[ "$got" = "$EXPECTED_MFH_MD5" ] || {
    echo "FAIL: --video-mfh MD5 $got, expected $EXPECTED_MFH_MD5" >&2
    exit 1; }

# The still path must stay off: pinned dav2d rejects MFH there.
if "$BIN" "$ROOT/tests/data/basic_64x64_1f.y4m" -o "$TMP/s.ivf" --still \
        --video-mfh >"$TMP/s.log" 2>&1; then
    echo "FAIL: --video-mfh was accepted on the still path" >&2; exit 1
fi

echo "PASS video mfh: $(wc -c <"$TMP/mfh.ivf" | tr -d ' ') B, md5 $got," \
     "decoded pixels identical"
