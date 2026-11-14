#!/usr/bin/env bash
# Average-CDF gate for the multi-frame video path (§5.4.8 / §8.2.4 / §7.5).
#
# 1. Encodes a 3-frame P sequence with the pinned frozen path and with
#    --avg-cdf --adaptive-cdf (per-tile adaptation + frame-end averaging).
# 2. Requires the two bitstreams to differ.
# 3. Requires dav2d --strict 1 to decode both and the decoded frames to match
#    the encoder reconstruction exactly.
# 4. Pins the --avg-cdf MD5.
# 5. Asserts that --cdf-continuity is rejected by the CLI: the encoder side is
#    wired, but pinned dav2d f9dafbc rejects disable_cross_frame_cdf_init = 0.
# 6. Per-frame QP (--video-cdf-per-frame-qp): a reference-intro plan inside one
#    coefficient qctx slice decodes --strict 1 with exact visible recon; a plan
#    crossing a §8.3.1 boundary, the same plan without the opt-in, and the flag
#    without the CDF chain are all rejected by name.
#
# Usage: gate_avg_cdf.sh --bin build-int/av2enc [--dav2d PATH]
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
        -h|--help) sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
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

SRC="$ROOT/testdata/shine_1280x720_60f.y4m"
[ -f "$SRC" ] || { echo "gate_avg_cdf: missing $SRC" >&2; exit 77; }
EXPECTED_AVG_MD5=b6772cc5c0d7ba3384c7943e653a606e

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

"$BIN" "$SRC" -o "$TMP/base.ivf" --video --p-frames 2 \
    --video-recon "$TMP/base.y4m" >"$TMP/base.log" 2>&1 || {
    echo "FAIL: base encode" >&2; cat "$TMP/base.log" >&2; exit 1; }
"$BIN" "$SRC" -o "$TMP/avg.ivf" --video --p-frames 2 \
    --avg-cdf --adaptive-cdf --video-recon "$TMP/avg.y4m" \
    >"$TMP/avg.log" 2>&1 || {
    echo "FAIL: --avg-cdf encode" >&2; cat "$TMP/avg.log" >&2; exit 1; }

if cmp -s "$TMP/base.ivf" "$TMP/avg.ivf"; then
    echo "FAIL: --avg-cdf stream is byte-identical to the frozen stream" >&2
    exit 1
fi

for variant in base avg; do
    if ! "$DAV2D" -q -i "$TMP/$variant.ivf" -o "$TMP/$variant.dec.y4m" \
            --threads 1 --strict 1 >"$TMP/$variant.dec.log" 2>&1; then
        echo "FAIL $variant: dav2d rejected" >&2
        cat "$TMP/$variant.dec.log" >&2; exit 1
    fi
    if ! compare_recon "$TMP/$variant.y4m" "$TMP/$variant.dec.y4m" \
            >"$TMP/$variant.cmp.log" 2>&1; then
        echo "FAIL $variant: recon mismatch" >&2
        cat "$TMP/$variant.cmp.log" >&2; exit 1
    fi
done

got=$(md5_of "$TMP/avg.ivf")
if [ "$got" != "$EXPECTED_AVG_MD5" ]; then
    echo "FAIL: --avg-cdf MD5 $got, expected $EXPECTED_AVG_MD5" >&2
    exit 1
fi

# Ungated continuity: the pose/reproducer must stay rejected by the CLI.
if "$BIN" "$SRC" -o "$TMP/cont.ivf" --video --p-frames 2 --avg-cdf \
        --adaptive-cdf --cdf-continuity >"$TMP/cont.log" 2>&1; then
    echo "FAIL: --cdf-continuity was accepted; it must stay off until a " \
         "decoder implements disable_cross_frame_cdf_init = 0" >&2
    exit 1
fi
if ! grep -q "not decoder-gated" "$TMP/cont.log"; then
    echo "FAIL: --cdf-continuity rejection message changed" >&2
    cat "$TMP/cont.log" >&2; exit 1
fi

# --- per-frame QP reset (--video-cdf-per-frame-qp) ------------------------
# Nightly-style reference-intro plan: the key frame takes --qp - 16.  At
# --qp 190 that is key 174 / P 190, both in coefficient qctx slice 2 (§8.3.1),
# so the CDF image can reset from each frame's own slice and never carry
# across frames.  The stream must decode under --strict 1 and reproduce the
# visible encoder reconstruction, like the fixed-QP average-CDF stream.
if ! "$BIN" "$SRC" -o "$TMP/pf.ivf" --video --p-frames 2 --qp 190 \
        --video-key-qp-offset -16 --video-cdf-per-frame-qp \
        --avg-cdf --adaptive-cdf --video-recon "$TMP/pf.y4m" \
        >"$TMP/pf.log" 2>&1; then
    echo "FAIL: --video-cdf-per-frame-qp encode" >&2; cat "$TMP/pf.log" >&2
    exit 1
fi
if ! "$DAV2D" -q --strict 1 -i "$TMP/pf.ivf" -o "$TMP/pf.dec.y4m" \
        --threads 1 >"$TMP/pf.dec.log" 2>&1; then
    echo "FAIL per-frame-qp: dav2d rejected" >&2
    cat "$TMP/pf.dec.log" >&2; exit 1
fi
if ! compare_recon "$TMP/pf.y4m" "$TMP/pf.dec.y4m" >"$TMP/pf.cmp.log" 2>&1; then
    echo "FAIL per-frame-qp: recon mismatch" >&2
    cat "$TMP/pf.cmp.log" >&2; exit 1
fi

# A plan that crosses a §8.3.1 slice boundary (key 125 = slice 1, P 141 =
# slice 2) cannot share one default image and must be named by the encoder.
if "$BIN" "$SRC" -o "$TMP/pf-bad.ivf" --video --p-frames 2 --qp 141 \
        --video-key-qp-offset -16 --video-cdf-per-frame-qp \
        --avg-cdf --adaptive-cdf >/dev/null 2>"$TMP/pf-bad.err"; then
    echo "FAIL: a sequence crossing coefficient qctx slices was accepted" >&2
    exit 1
fi
if ! grep -q "one coefficient qctx slice" "$TMP/pf-bad.err"; then
    echo "FAIL: the qctx-slice rejection message changed" >&2
    cat "$TMP/pf-bad.err" >&2; exit 1
fi

# The opt-in stays opt-in: the same per-frame plan without the flag is
# rejected, and the flag alone does not turn on the CDF chain.
if "$BIN" "$SRC" -o "$TMP/pf-nogate.ivf" --video --p-frames 2 --qp 190 \
        --video-key-qp-offset -16 --avg-cdf --adaptive-cdf \
        >/dev/null 2>"$TMP/pf-nogate.err"; then
    echo "FAIL: per-frame QP was accepted without --video-cdf-per-frame-qp" >&2
    exit 1
fi
if ! grep -q -- "--video-cdf-per-frame-qp" "$TMP/pf-nogate.err"; then
    echo "FAIL: the per-frame QP rejection did not name the opt-in" >&2
    cat "$TMP/pf-nogate.err" >&2; exit 1
fi
if "$BIN" "$SRC" -o "$TMP/pf-alone.ivf" --video --p-frames 2 --qp 190 \
        --video-key-qp-offset -16 --video-cdf-per-frame-qp \
        >/dev/null 2>"$TMP/pf-alone.err"; then
    echo "FAIL: --video-cdf-per-frame-qp was accepted without the CDF chain" >&2
    exit 1
fi
if ! grep -q -- "needs --avg-cdf and --adaptive-cdf" "$TMP/pf-alone.err"; then
    echo "FAIL: the per-frame QP CDF-chain rejection message changed" >&2
    cat "$TMP/pf-alone.err" >&2; exit 1
fi

echo "PASS video avg-cdf: base $(wc -c <"$TMP/base.ivf" | tr -d ' ') B," \
     "avg $(wc -c <"$TMP/avg.ivf" | tr -d ' ') B, md5 $got;" \
     "per-frame QP $(wc -c <"$TMP/pf.ivf" | tr -d ' ') B; continuity rejected"
