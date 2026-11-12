#!/usr/bin/env bash
# Real per-segment-Q gate (§5.4.9 seg_info, §5.18.7.1, §7.14.2 get_qindex,
# §7.17 deblock qindex).
#
# `--still-per-segment-q` plans a per-superblock segment map (variance split),
# signals SEG_LVL_ALT_Q for segment 1, and uses base_q_idx + delta for forward
# quantisation, dequantisation and the deblock grid.  The gate requires:
#   * the stream to differ from both the pinned --still baseline and the
#     neutral --still-segmentation table;
#   * dav2d --strict 1 to decode it;
#   * the decoded picture to match the encoder reconstruction exactly (this is
#     the real check that the signaled segment id and the qindex-derived dequant
#     agree);
#   * the pinned stream MD5.
#
# Usage:
#   gate_per_segment_q.sh --bin build-int/av2enc [--dump build-int/still_recon_dump]
#                         [--dav2d PATH]
# Exit: 0 pass, 1 fail, 2 usage, 77 skip.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
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
        -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "gate_per_segment_q: unknown argument $1" >&2; exit 2 ;;
    esac
done
[ -n "$BIN" ] && [ -x "$BIN" ] || { echo "gate_per_segment_q: --bin required" >&2; exit 2; }
[ -n "$DUMP" ] || DUMP="$(dirname "$BIN")/still_recon_dump"
[ -x "$DUMP" ] || { echo "gate_per_segment_q: missing still_recon_dump (--dump)" >&2; exit 2; }
if [ -z "$DAV2D" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
    done
fi
[ -n "$DAV2D" ] && [ -x "$DAV2D" ] || { echo "gate_per_segment_q: no dav2d" >&2; exit 77; }
SRC="$ROOT/tests/data/basic_640x360_1f.y4m"
[ -f "$SRC" ] || { echo "gate_per_segment_q: missing $SRC" >&2; exit 77; }
EXPECTED_MD5=24018e6bda4a1bdfbcae646b303eb576

md5_of() { md5 -q "$1" 2>/dev/null || md5sum "$1" | awk '{print $1}'; }

compare_recon() {
    python3 - "$1" "$2" <<'PY'
import sys
def read_y4m(path):
    data = open(path, 'rb').read()
    end = data.index(b'\n')
    parts = data[:end].decode().split()
    w = int(parts[1][1:]); h = int(parts[2][1:]); off = end + 1
    size = (w * h * 3) // 2
    frames = []
    while off < len(data):
        n = data.index(b'\n', off)
        frames.append(data[n + 1:n + 1 + size])
        off = n + 1 + size
    return frames
a = read_y4m(sys.argv[1]); b = read_y4m(sys.argv[2])
if len(a) != len(b) or not a:
    sys.exit(f'frame count mismatch: {len(a)} vs {len(b)}')
for i, (x, y) in enumerate(zip(a, b)):
    bad = sum(1 for p, q in zip(x, y) if p != q)
    if bad:
        sys.exit(f'frame {i}: {bad} samples differ')
PY
}

enc() { "$BIN" "$SRC" -o "$1" "${@:2}" >"$TMP/enc.log" 2>&1; }

enc "$TMP/base.ivf" --still
enc "$TMP/neutral.ivf" --still --still-segmentation
enc "$TMP/psq.ivf" --still --still-per-segment-q
if cmp -s "$TMP/base.ivf" "$TMP/psq.ivf"; then
    echo "FAIL: per-segment stream equals the base stream" >&2; exit 1
fi
if cmp -s "$TMP/neutral.ivf" "$TMP/psq.ivf"; then
    echo "FAIL: per-segment stream equals the neutral segmentation stream" >&2
    exit 1
fi

"$DUMP" -i "$SRC" -o "$TMP/psq.rec.y4m" --still --still-per-segment-q \
    >"$TMP/dump.log" 2>&1 || {
    echo "FAIL: still_recon_dump" >&2; cat "$TMP/dump.log" >&2; exit 1; }
if ! "$DAV2D" -q -i "$TMP/psq.ivf" -o "$TMP/psq.dec.y4m" \
        --threads 1 --strict 1 >"$TMP/dec.log" 2>&1; then
    echo "FAIL: dav2d rejected the per-segment stream" >&2
    cat "$TMP/dec.log" >&2; exit 1
fi
if ! compare_recon "$TMP/psq.rec.y4m" "$TMP/psq.dec.y4m" \
        >"$TMP/cmp.log" 2>&1; then
    echo "FAIL: encoder recon and decoder differ (qindex/segment mismatch)" >&2
    cat "$TMP/cmp.log" >&2; exit 1
fi

got=$(md5_of "$TMP/psq.ivf")
[ "$got" = "$EXPECTED_MD5" ] || {
    echo "FAIL: per-segment MD5 $got, expected $EXPECTED_MD5" >&2; exit 1; }

echo "PASS per-segment Q: base $(wc -c <"$TMP/base.ivf" | tr -d ' ') B," \
     "psq $(wc -c <"$TMP/psq.ivf" | tr -d ' ') B, md5 $got; recon exact"
