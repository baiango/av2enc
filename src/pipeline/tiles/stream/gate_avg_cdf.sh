#!/usr/bin/env bash
# Average-CDF gate for multi-tile stills (§8.2.2 / §8.2.4 / §7.5).
#
# Encodes the same frame twice: the pinned frozen-CDF tiled stream and the
# opt-in --avg-cdf stream (per-tile CDF adaptation under the average-CDF
# bookkeeping).  Requires:
#   * the two streams to differ (adaptation is actually applied);
#   * dav2d --strict 1 to decode both;
#   * the decoder output to match the encoder reconstruction exactly (the
#     §8.2.2 frame->tile seed and the §7.5 frame end are not observable on a
#     single picture, but the tile payload and reconstruction must stay in
#     lockstep);
#   * the pinned --avg-cdf MD5.
#
# Usage: gate_avg_cdf.sh --bin build-int/av2enc [--dav2d PATH]
# Exit: 0 pass, 1 fail, 2 usage, 77 skip (no decoder / missing fixture).
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
        -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "gate_avg_cdf: unknown argument $1" >&2; exit 2 ;;
    esac
done
[ -n "$BIN" ] && [ -x "$BIN" ] || { echo "gate_avg_cdf: --bin required" >&2; exit 2; }
if [ -z "$DAV2D" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
    done
fi
[ -n "$DAV2D" ] && [ -x "$DAV2D" ] || { echo "gate_avg_cdf: no dav2d" >&2; exit 77; }

SRC="$ROOT/tests/data/basic_640x360_1f.y4m"
[ -f "$SRC" ] || { echo "gate_avg_cdf: missing $SRC" >&2; exit 77; }
EXPECTED_AVG_MD5=64c0feb25015b4929157fc65bdf41d80

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
    sys.exit('frame count mismatch')
for i, (x, y) in enumerate(zip(a, b)):
    bad = sum(1 for p, q in zip(x, y) if p != q)
    if bad:
        sys.exit(f'frame {i}: {bad} samples differ')
PY
}

fail=0
for preset in 2x1 2x2; do
    base="$TMP/base_$preset.ivf"
    avg="$TMP/avg_$preset.ivf"
    brec="$TMP/base_$preset.y4m"
    arec="$TMP/avg_$preset.y4m"
    "$BIN" "$SRC" -o "$base" --still --tiles "$preset" --no-deblock \
        --recon "$brec" >"$TMP/base_$preset.log" 2>&1 || {
        echo "FAIL $preset: base encode" >&2; cat "$TMP/base_$preset.log" >&2; fail=1; continue; }
    "$BIN" "$SRC" -o "$avg" --still --tiles "$preset" --no-deblock --avg-cdf \
        --recon "$arec" >"$TMP/avg_$preset.log" 2>&1 || {
        echo "FAIL $preset: --avg-cdf encode" >&2; cat "$TMP/avg_$preset.log" >&2; fail=1; continue; }
    if cmp -s "$base" "$avg"; then
        echo "FAIL $preset: --avg-cdf stream is byte-identical to the frozen stream" >&2
        fail=1; continue
    fi
    for variant in base avg; do
        ivf="$TMP/${variant}_$preset.ivf"
        rec="$TMP/${variant}_$preset.y4m"
        dec="$TMP/${variant}_$preset.dec.y4m"
        if ! "$DAV2D" -q -i "$ivf" -o "$dec" --threads 1 --strict 1 \
                >"$TMP/${variant}_$preset.dec.log" 2>&1; then
            echo "FAIL $preset/$variant: dav2d rejected" >&2
            cat "$TMP/${variant}_$preset.dec.log" >&2; fail=1; continue
        fi
        if ! compare_recon "$rec" "$dec" >"$TMP/${variant}_$preset.cmp.log" 2>&1; then
            echo "FAIL $preset/$variant: recon mismatch" >&2
            cat "$TMP/${variant}_$preset.cmp.log" >&2; fail=1
        fi
    done
    got=$(md5_of "$avg")
    if [ "$preset" = 2x2 ] && [ "$got" != "$EXPECTED_AVG_MD5" ]; then
        echo "FAIL $preset: --avg-cdf MD5 $got, expected $EXPECTED_AVG_MD5" >&2
        fail=1
    fi
    if [ "$fail" -eq 0 ]; then
        echo "PASS tiles $preset (frozen $(wc -c <"$base" | tr -d ' ') B, avg $(wc -c <"$avg" | tr -d ' ') B, md5 $got)"
    fi
done

[ "$fail" -eq 0 ] || exit 1
echo "gate_avg_cdf: PASS"
