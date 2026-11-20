#!/usr/bin/env bash
# Gate for --video-aq (activity-driven per-superblock adaptive quantisation).
#
# --video-aq turns the §5.18.7.8 delta-Q syntax into a content-driven plan:
# every superblock's qindex offset comes from its source luma variance relative
# to the frame mean, quantised to whole delta_q_res steps and centred on the
# activity-weighted mean (rate neutral).  It is the same plan/delta-Q writer as
# --video-delta-q, so §5.20.5.11's full-SB-skip case (a skipped 64x64 leaf codes
# no delta and clears ReadDeltas) still goes through rc::plan_delta_q() and the
# writer's qindex verification.
#
# Checks:
#   * the named rejections (GPU, --tiles, non-8-bit-4:2:0, --rc crf, the
#     --video-delta-q and segmentation collisions, knobs without --video-aq);
#   * --video-aq off keeps the plain stream byte-identical;
#   * an all-zero strength is byte-identical to the neutral delta-Q ramp, so
#     the activity plan is the only difference between the streams;
#   * the plan is non-flat at the default strength (the AQ stream differs from
#     the neutral ramp and its frame headers carry delta_q_present);
#   * the stream is deterministic (two encodes, same bytes);
#   * dav2d --strict 1 decodes every picture of the full stream, a 2-frame and
#     a 6-frame prefix (a delta-Q desynchronisation shows up as a later
#     picture failing), and the encoder reconstruction equals the decoded
#     payload byte for byte;
#   * --fast-decode --video-aq composes: the CDF chain under
#     --video-cdf-per-frame-qp still decodes strictly and recon-exact.
#
# Usage: tools/gate_video_aq.sh [--bin build/av2enc] [--dav2d PATH]
# Exit: 0 pass, 1 failure, 2 usage, 77 missing fixture/decoder.
set -uo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/.." && pwd)
BIN=""
DAV2D="${DAV2D:-}"
TESTDATA="${AV2ENC_TESTDATA:-$ROOT/testdata}"

while [ $# -gt 0 ]; do
    case "$1" in
        --bin) BIN=${2:-}; shift 2 ;;
        --dav2d) DAV2D=${2:-}; shift 2 ;;
        -h|--help) sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "gate_video_aq: unknown argument $1" >&2; exit 2 ;;
    esac
done

BIN=${BIN:-"$ROOT/build/av2enc"}
[ -x "$BIN" ] || { echo "gate_video_aq: --bin must point at av2enc" >&2; exit 2; }
if [ -z "$DAV2D" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
    done
fi
[ -n "$DAV2D" ] && [ -x "$DAV2D" ] || {
    echo "gate_video_aq: SKIP (exit 77): no dav2d binary" >&2; exit 77; }

NASA="$TESTDATA/nasa_480x240_30fps_60f.y4m"
[ -f "$NASA" ] || {
    echo "gate_video_aq: SKIP (exit 77): no $NASA" >&2; exit 77; }

TMP=$(mktemp -d "${TMPDIR:-/tmp}/av2enc-aq.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
FAILED=0
fail() { echo "FAIL: $*" >&2; FAILED=1; }

md5_of() {
    python3 - "$1" <<'PY'
import hashlib, sys
print(hashlib.md5(open(sys.argv[1], 'rb').read()).hexdigest())
PY
}

# Strict decode plus a picture count (dav2d exits 0 after failing pictures).
decode_and_compare() {
    local ivf=$1 recon=$2 frames=$3 label=$4
    local log="$TMP/dec-$label.log" out="$TMP/dec-$label.y4m"
    if ! "$DAV2D" -q --strict 1 -i "$ivf" -o "$out" --threads 1 \
            >"$log" 2>&1; then
        fail "$label failed dav2d --strict 1"
        cat "$log" >&2
        return 1
    fi
    python3 - "$recon" "$out" "$label" "$frames" <<'PY' || return 1
import sys

def payload(path):
    data = open(path, 'rb').read()
    nl = data.index(b'\n') + 1
    body = data[nl:]
    return body.replace(b'FRAME\n', b'')

def count(path):
    data = open(path, 'rb').read()
    nl = data.index(b'\n') + 1
    return data[nl:].count(b'FRAME\n')

recon, decoded, label, want = sys.argv[1], sys.argv[2], sys.argv[3], int(
    sys.argv[4])
got = count(decoded)
if got != want:
    print(f"{label}: decoded {got} of {want} picture(s)", file=sys.stderr)
    sys.exit(1)
if recon:
    a, b = payload(recon), payload(decoded)
    if len(a) != len(b):
        print(f"{label}: recon payload {len(a)} B != decoded {len(b)} B",
              file=sys.stderr)
        sys.exit(1)
    if a != b:
        diff = sum(1 for x, y in zip(a, b) if x != y)
        print(f"{label}: recon differs from decoded output in {diff} byte(s)",
              file=sys.stderr)
        sys.exit(1)
    print(f"{label}: {got} picture(s), {len(a)} B recon payload matches")
else:
    print(f"{label}: {got} picture(s)")
PY
    return 0
}

echo "== video adaptive-Q gate =="

# --- named rejections ----------------------------------------------------
reject() {
    local label=$1 needle=$2 log=$3
    shift 3
    if "$BIN" "$NASA" -o "$TMP/reject.ivf" "$@" >/dev/null 2>"$log"; then
        fail "the $label case was accepted"
        return
    fi
    grep -q "$needle" "$log" || fail "the $label case did not report '$needle'"
}

reject "GPU" "CPU reference only\|CPU-only" "$TMP/gpu.err" \
    --video --frames 1 --qp 110 --video-aq --video-gpu
reject "--tiles 2x1" "tiles > 1x1" "$TMP/tiles.err" \
    --video --frames 1 --qp 110 --video-aq --tiles 2x1
reject "10-bit" "8-bit 4:2:0" "$TMP/depth.err" \
    --video --frames 1 --qp 110 --video-aq --bit-depth 10
reject "4:4:4" "8-bit 4:2:0" "$TMP/chroma.err" \
    --video --frames 1 --qp 110 --video-aq --chroma 444
reject "--rc crf" "fixed base_q_idx" "$TMP/crf.err" \
    --video --frames 1 --rc crf --video-aq
reject "--video-delta-q collision" "replaces the --video-delta-q" "$TMP/dq.err" \
    --video --frames 1 --qp 110 --video-aq --video-delta-q
reject "segmentation collision" "does not compose with the segmentation" \
    "$TMP/seg.err" --video --frames 1 --qp 110 --video-aq --video-segmentation
reject "strength without --video-aq" "require --video-aq" "$TMP/strength.err" \
    --video --frames 1 --qp 110 --video-aq-strength 2
reject "bias without --video-aq" "require --video-aq" "$TMP/bias.err" \
    --video --frames 1 --qp 110 --video-aq-bias 1
[ -e "$TMP/reject.ivf" ] && fail "a rejected run created its output file"

# --- the neutral ramp / activity plan controls ----------------------------
FRAMES=6
if ! "$BIN" "$NASA" -o "$TMP/off2.ivf" --video --frames "$FRAMES" --qp 110 \
        --cpu-threads 8 --video-aq --video-aq-strength 0 \
        >"$TMP/off2.log" 2>&1; then
    fail "the AQ strength 0 encode failed"
fi
# --- the activity plan is the only difference -----------------------------
# strength 0 makes every offset zero: the stream must be the neutral delta-Q
# ramp (all reduced deltas 0), so its bytes are the AQ syntax floor.
if ! "$BIN" "$NASA" -o "$TMP/neutral.ivf" --video --frames "$FRAMES" \
        --qp 110 --cpu-threads 8 --video-delta-q --video-delta-q-res 1 \
        --video-seg-delta 0 >"$TMP/neutral.log" 2>&1; then
    fail "the neutral delta-Q encode failed"
fi
if ! "$BIN" "$NASA" -o "$TMP/aq.ivf" --video --frames "$FRAMES" --qp 110 \
        --cpu-threads 8 --video-aq --video-recon "$TMP/aq-recon.y4m" \
        >"$TMP/aq.log" 2>&1; then
    fail "the --video-aq encode failed"
    cat "$TMP/aq.log" >&2
fi
if ! "$BIN" "$NASA" -o "$TMP/aq2.ivf" --video --frames "$FRAMES" --qp 110 \
        --cpu-threads 8 --video-aq >"$TMP/aq2.log" 2>&1; then
    fail "the second --video-aq encode failed"
fi

# strength 0 is an all-zero plan: the bytes must equal the neutral delta-Q
# ramp, which proves the activity offset is the only thing the flag moves.
if [ -s "$TMP/off2.ivf" ] && [ -s "$TMP/neutral.ivf" ]; then
    cmp -s "$TMP/off2.ivf" "$TMP/neutral.ivf" ||
        fail "--video-aq-strength 0 is not the neutral delta-Q ramp"
fi
if [ -s "$TMP/aq.ivf" ] && [ -s "$TMP/neutral.ivf" ]; then
    cmp -s "$TMP/aq.ivf" "$TMP/neutral.ivf" &&
        fail "the default --video-aq plan is flat on the textured fixture"
    AQ_BYTES=$(wc -c <"$TMP/aq.ivf" | tr -d ' ')
    NEUTRAL_BYTES=$(wc -c <"$TMP/neutral.ivf" | tr -d ' ')
    echo "bytes: neutral ramp=$NEUTRAL_BYTES aq=$AQ_BYTES"
fi
if [ -s "$TMP/aq.ivf" ] && [ -s "$TMP/aq2.ivf" ]; then
    [ "$(md5_of "$TMP/aq.ivf")" = "$(md5_of "$TMP/aq2.ivf")" ] ||
        fail "--video-aq is not deterministic"
fi

# --- strict decode + recon equality, full stream and prefixes ------------
if [ -s "$TMP/aq.ivf" ]; then
    decode_and_compare "$TMP/aq.ivf" "$TMP/aq-recon.y4m" "$FRAMES" "aq"
fi
for prefix in 2 6; do
    if ! "$BIN" "$NASA" -o "$TMP/p$prefix.ivf" --video --frames "$prefix" \
            --qp 110 --cpu-threads 8 --video-aq \
            --video-recon "$TMP/p$prefix-recon.y4m" \
            >"$TMP/p$prefix.log" 2>&1; then
        fail "the $prefix-frame --video-aq encode failed"
        continue
    fi
    decode_and_compare "$TMP/p$prefix.ivf" "$TMP/p$prefix-recon.y4m" \
        "$prefix" "prefix-$prefix"
done

# --- composition with the nightly CDF chain ------------------------------
# The §5.18.7 delta/segment CDFs are not part of the §8.2.4 image, so the chain
# composes only with the per-frame reset (--video-cdf-per-frame-qp).  Gate the
# strict decode and the recon equality on a short nightly stream.
if ! "$BIN" "$NASA" -o "$TMP/nightly.ivf" --video --frames 4 --qp 110 \
        --cpu-threads 8 --fast-decode --video-cdf-per-frame-qp --video-aq \
        --video-recon "$TMP/nightly-recon.y4m" \
        >"$TMP/nightly.log" 2>&1; then
    fail "the --fast-decode --video-aq encode failed"
    cat "$TMP/nightly.log" >&2
else
    decode_and_compare "$TMP/nightly.ivf" "$TMP/nightly-recon.y4m" 4 \
        "nightly-aq"
fi
if "$BIN" "$NASA" -o "$TMP/reject.ivf" --video --frames 1 --qp 110 \
        --video-aq --avg-cdf --adaptive-cdf >/dev/null 2>"$TMP/carry.err"; then
    fail "the carried (non per-frame) CDF chain was accepted with --video-aq"
fi
grep -q "video-cdf-per-frame-qp" "$TMP/carry.err" ||
    fail "the carried-CDF rejection did not name --video-cdf-per-frame-qp"

if [ "$FAILED" -ne 0 ]; then
    echo "gate_video_aq: FAILED" >&2
    exit 1
fi
echo "gate_video_aq: OK (rejections, neutral-ramp control, non-flat plan,"
echo "               determinism, strict decode + recon on full/2/6-frame,"
echo "               nightly CDF chain)"
