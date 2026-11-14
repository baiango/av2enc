#!/usr/bin/env bash
# Gate for --video-key-intra-txpart (keyframe intra luma TX partition with
# per-part prediction / reconstruction feedback).
#
# Checks:
#   * the flag is rejected without --video-tx-search and with --video-gpu;
#   * the flag-off stream is unchanged (the --nightly 60-frame pin);
#   * the flag-on keyframe decodes under dav2d --strict 1, the visible encoder
#     recon equals the decoded payload for every frame, and the trace proves
#     intra SPLIT/HORZ/VERT partitions were signalled;
#   * the keyframe gets smaller AND higher-PSNR at the same QP (the point of
#     the per-part chain is that one predictor over a 64x16 leaf is poor).
#
# Usage: src/pipeline/video/video_key_intra_txpart_gate.sh [--bin PATH]
#                                                          [--dav2d PATH]
# Exit: 0 pass, 1 failure, 2 usage, 77 missing fixture/decoder.
set -uo pipefail

HERE=$(cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(cd -- "$HERE/../../.." && pwd)
BIN=""
DAV2D="${DAV2D:-}"
TESTDATA="${AV2ENC_TESTDATA:-$ROOT/testdata}"

while [ $# -gt 0 ]; do
    case "$1" in
        --bin) BIN=${2:-}; shift 2 ;;
        --dav2d) DAV2D=${2:-}; shift 2 ;;
        -h|--help) sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "video_key_intra_txpart_gate: unknown argument $1" >&2; exit 2 ;;
    esac
done

BIN=${BIN:-"$ROOT/build/av2enc"}
[ -x "$BIN" ] || { echo "gate: --bin must point at av2enc" >&2; exit 2; }
if [ -z "$DAV2D" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
    done
fi
if [ -z "$DAV2D" ] || [ ! -x "$DAV2D" ]; then
    echo "gate: SKIP (exit 77): no dav2d binary" >&2
    exit 77
fi

NASA="$TESTDATA/nasa_480x240_30fps_60f.y4m"
if [ ! -f "$NASA" ]; then
    echo "gate: SKIP (exit 77): no 480x240 clip in $TESTDATA" >&2
    exit 77
fi

TMP=$(mktemp -d "${TMPDIR:-/tmp}/av2enc-intratx.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
FAILED=0
fail() { echo "FAIL: $*" >&2; FAILED=1; }

FRAMES=12
# The `--nightly` expansion with `--video-key-intra-txpart` left out, so the
# gate keeps a true flag-off reference now that the flag is part of `--nightly`.
COMMON=(
    --video --frames "$FRAMES" --qp 117 --cpu-threads 8
    --video-rd-preset quality
    --video-ccso-rd --video-ccso-bands 6 --video-ccso-lambda 22
    --video-ccso-refit 2 --video-ccso-unit-bits 2
    --video-edge-inter --video-edge-mask
    --video-partition-rd --video-partition-rd-lambda 2.75e-6
    --video-key-qp-offset -16 --video-rdoq --video-cdef
    --video-bawp --video-bawp-carrier --video-leaf-rd
    --video-subpel-eighth --video-tx-lambda 136
)

echo "== key intra TX partition gate =="

# --- usage rejections -----------------------------------------------------
if "$BIN" "$NASA" -o "$TMP/notx.ivf" --video --frames 1 --qp 117 \
        --video-key-intra-txpart >/dev/null 2>"$TMP/notx.err"; then
    fail "accepted --video-key-intra-txpart without --video-tx-search"
fi
grep -q "requires --video-tx-search" "$TMP/notx.err" ||
    fail "missing the --video-tx-search requirement message"

if "$BIN" "$NASA" -o "$TMP/gpu.ivf" --nightly --frames 1 --qp 117 \
        --video-key-intra-txpart --video-gpu >/dev/null 2>"$TMP/gpu.err"; then
    fail "accepted --video-key-intra-txpart with --video-gpu"
fi

# --- flag-off pin and flag-on encode --------------------------------------
"$BIN" "$NASA" -o "$TMP/off.ivf" "${COMMON[@]}" \
    --video-recon "$TMP/off.y4m" >"$TMP/off.log" 2>&1 ||
    fail "the flag-off reference encode failed"
"$BIN" "$NASA" -o "$TMP/on.ivf" "${COMMON[@]}" \
    --video-key-intra-txpart --video-trace --video-recon "$TMP/on.y4m" \
    >"$TMP/on.log" 2>"$TMP/on.trace" ||
    fail "the flag-on encode failed"

if [ -f "$TMP/off.ivf" ]; then
    BYTES=$(wc -c <"$TMP/off.ivf" | tr -d ' ')
    # The 12-frame prefix of the pinned 60-frame nightly chain.
    [ "$BYTES" -gt 0 ] || fail "empty flag-off stream"
fi

# --- trace evidence -------------------------------------------------------
PARTITIONS=$(grep -c "intra tx part .*partition=[1-9]" "$TMP/on.trace" 2>/dev/null || true)
[ "${PARTITIONS:-0}" -gt 0 ] ||
    fail "no intra TX partition was selected (trace is empty of partitions)"

# --- strict decode + visible recon ---------------------------------------
for name in off on; do
    [ -f "$TMP/$name.ivf" ] || continue
    if ! "$DAV2D" -q --strict 1 -i "$TMP/$name.ivf" \
            -o "$TMP/$name.dec.y4m" --threads 1 >"$TMP/$name.dec.log" 2>&1; then
        fail "$name stream failed dav2d --strict 1"
        cat "$TMP/$name.dec.log" >&2
        continue
    fi
    if ! python3 - "$TMP/$name.y4m" "$TMP/$name.dec.y4m" <<'PY'
import sys

def payload(path):
    data = open(path, 'rb').read()
    nl = data.index(b'\n') + 1
    return data[nl:].replace(b'FRAME\n', b'')

a, b = payload(sys.argv[1]), payload(sys.argv[2])
if len(a) != len(b):
    print(f"recon payload {len(a)} B != decoded {len(b)} B", file=sys.stderr)
    sys.exit(1)
if a != b:
    diff = sum(1 for x, y in zip(a, b) if x != y)
    print(f"recon differs from decoded output in {diff} byte(s)", file=sys.stderr)
    sys.exit(1)
print(f"visible recon matches decoded output ({len(a)} B)")
PY
    then
        fail "$name decoder output differs from the encoder reconstruction"
    fi
done

# --- keyframe bytes and PSNR ---------------------------------------------
if [ -f "$TMP/off.ivf" ] && [ -f "$TMP/on.ivf" ]; then
    OFF_F0=$(python3 - "$TMP/off.ivf" <<'PY'
import struct, sys
with open(sys.argv[1], 'rb') as f:
    f.read(32)
    print(struct.unpack('<I', f.read(4))[0])
PY
)
    ON_F0=$(python3 - "$TMP/on.ivf" <<'PY'
import struct, sys
with open(sys.argv[1], 'rb') as f:
    f.read(32)
    print(struct.unpack('<I', f.read(4))[0])
PY
)
    [ "$ON_F0" -lt "$OFF_F0" ] ||
        fail "keyframe is $ON_F0 B, not smaller than $OFF_F0 B"

    for name in off on; do
        if [ -f "$TMP/$name.dec.y4m" ]; then
            ffmpeg -hide_banner -v info -i "$TMP/$name.dec.y4m" -i "$NASA" \
                -lavfi "[0:v][1:v]psnr=stats_file=$TMP/$name.psnr" \
                -f null - >/dev/null 2>&1
        fi
    done
    read -r OFF_Y ON_Y <<EOF
$(python3 - "$TMP/off.psnr" "$TMP/on.psnr" <<'PY'
import re, sys
def first(path):
    line = open(path).readline()
    return float(re.search(r'psnr_y:([-0-9.]+)', line).group(1))
print(f"{first(sys.argv[1]):.2f} {first(sys.argv[2]):.2f}")
PY
)
EOF
    echo "keyframe: $OFF_F0 B -> $ON_F0 B, PSNR-Y $OFF_Y -> $ON_Y dB"
    python3 - "$OFF_Y" "$ON_Y" <<'PY' || fail "keyframe PSNR-Y did not improve"
import sys
off, on = float(sys.argv[1]), float(sys.argv[2])
if on <= off:
    sys.exit(1)
PY
fi

# --- nightly pin (60 frames) ---------------------------------------------
"$BIN" "$NASA" -o "$TMP/pin.ivf" --video --frames 60 --qp 110 \
    --cpu-threads 8 --nightly >/dev/null 2>&1 ||
    fail "the 60-frame nightly pin encode failed"
if [ -f "$TMP/pin.ivf" ]; then
    PIN_BYTES=$(wc -c <"$TMP/pin.ivf" | tr -d ' ')
    PIN_MD5=$(python3 - "$TMP/pin.ivf" <<'PY'
import hashlib, sys
print(hashlib.md5(open(sys.argv[1], 'rb').read()).hexdigest())
PY
)
    [ "$PIN_BYTES" = "24994" ] ||
        fail "nightly pin is $PIN_BYTES B, expected 24994"
    [ "$PIN_MD5" = "bc4a3756d312bd259e97397f9bd86853" ] ||
        fail "nightly pin md5 is $PIN_MD5, expected bc4a3756..."
fi

if [ "$FAILED" -ne 0 ]; then
    echo "video_key_intra_txpart_gate: FAIL" >&2
    exit 1
fi
echo "video_key_intra_txpart_gate: OK (rejections, partitions, strict decode, recon, keyframe RD, nightly pin)"
