#!/usr/bin/env bash
# Video timing gate (§5.4.1 / §5.4.12 / §5.4.13, --video-timing).
#
# The decoder-compatible form is:
#   * seq_initial_display_delay_present_flag = 1, delay 0
#   * decoder_model_info_present_flag = 1 + num_units_in_decoding_tick
#   * seq_decoder_model_info_present_flag = 0 (pinned dav2d rejects the nested
#     §5.4.13 block; see src/codec/rate/rc/hrd/README.md)
#   * one CI OBU with timing_info() in the first coded output frame unit
#
# Checks:
#   * the timing stream differs from the pinned baseline;
#   * dav2d --strict 1 decodes both to the same pixel MD5;
#   * --frametimes changes, proving the decoder consumed timing_info();
#   * the pinned timing stream MD5.
#
# Usage: gate_timing.sh --bin build-int/av2enc [--dav2d PATH]
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
        -h|--help) sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "gate_timing: unknown argument $1" >&2; exit 2 ;;
    esac
done
[ -n "$BIN" ] && [ -x "$BIN" ] || { echo "gate_timing: --bin required" >&2; exit 2; }
if [ -z "$DAV2D" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
    done
fi
[ -n "$DAV2D" ] && [ -x "$DAV2D" ] || { echo "gate_timing: no dav2d" >&2; exit 77; }
SRC="$ROOT/testdata/shine_1280x720_60f.y4m"
[ -f "$SRC" ] || { echo "gate_timing: missing $SRC" >&2; exit 77; }
EXPECTED_TIMING_MD5=58181f220426690610686360438f2197

md5_of() { md5 -q "$1" 2>/dev/null || md5sum "$1" | awk '{print $1}'; }
dec_md5() {
    "$DAV2D" -q -i "$1" --muxer md5 -o - --threads 1 --strict 1 2>/dev/null | tail -1
}

"$BIN" "$SRC" -o "$TMP/base.ivf" --video --p-frames 2 \
    >"$TMP/base.log" 2>&1 || {
    echo "FAIL: base encode" >&2; cat "$TMP/base.log" >&2; exit 1; }
"$BIN" "$SRC" -o "$TMP/timing.ivf" --video --p-frames 2 --video-timing \
    >"$TMP/timing.log" 2>&1 || {
    echo "FAIL: --video-timing encode" >&2; cat "$TMP/timing.log" >&2; exit 1; }

cmp -s "$TMP/base.ivf" "$TMP/timing.ivf" && {
    echo "FAIL: --video-timing stream is byte-identical" >&2; exit 1; }

base_dec=$(dec_md5 "$TMP/base.ivf")
timing_dec=$(dec_md5 "$TMP/timing.ivf")
[ "$base_dec" = "$timing_dec" ] || {
    echo "FAIL: decoded MD5 changed ($base_dec vs $timing_dec)" >&2; exit 1; }

"$DAV2D" -q -i "$TMP/base.ivf" -o /dev/null --threads 1 --strict 1 \
    --frametimes "$TMP/base.ft" 2>/dev/null
"$DAV2D" -q -i "$TMP/timing.ivf" -o /dev/null --threads 1 --strict 1 \
    --frametimes "$TMP/timing.ft" 2>/dev/null
if cmp -s "$TMP/base.ft" "$TMP/timing.ft"; then
    echo "FAIL: --frametimes did not change; timing_info() was ignored" >&2
    exit 1
fi

got=$(md5_of "$TMP/timing.ivf")
[ "$got" = "$EXPECTED_TIMING_MD5" ] || {
    echo "FAIL: --video-timing MD5 $got, expected $EXPECTED_TIMING_MD5" >&2
    exit 1; }

# Nested §5.4.13 decoder model info and BRT stay off: the pinned decoder
# rejects both.  The component gate records the evidence.
if "$BIN" "$SRC" -o "$TMP/x.ivf" --video --p-frames 2 --video-timing \
        --video-decoder-model >"$TMP/x.log" 2>&1; then
    echo "FAIL: --video-decoder-model exists; it must stay unwired" >&2
    exit 1
fi

echo "PASS video timing: $(wc -c <"$TMP/timing.ivf" | tr -d ' ') B, md5 $got," \
     "decoded pixels identical, frametimes changed"
