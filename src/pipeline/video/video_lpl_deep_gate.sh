#!/usr/bin/env bash
# Gate for the env-gated deep inter anchor (AV2ENC_PIPE_LPL_DEEP=1).
#
# The lpl36-refresh pipeline chooses the refresh frames perceptually.  This
# gate deepens those anchors to the key picture's quantizer depth (the existing
# `--video-drift-refresh-qp` QP drop, default -27 at the nightly 84/111
# recipe) and keeps the forced inter residual on all three planes.  The carried
# §9.3 image needs no new syntax: --cdf-continuity already signals the
# per-frame carry, and the encoder seeds a reset frame from its *coded*
# base_q_idx, which is what the decoder derives from the frame header.
#
# Checks:
#   * AV2ENC_PIPE_LPL_DEEP unset and =0 keep the pinned nightly stream;
#   * AV2ENC_PIPE_LPL_DEEP=0 under the lpl36 gate is byte-identical to
#     AV2ENC_PIPE_LPL_REFRESH=1, i.e. the deep switch is inert when off;
#   * with the gate on, the trace reports the key-depth anchor, at least one
#     frame is coded at the static inter QP minus the depth, the stream decodes
#     60/60 under dav2d --strict 1, and --video-recon equals the decoded
#     payload byte for byte;
#   * the CDF slice-cross case: a 2-frame 640x360 stream whose first P frame is
#     a deep anchor at QP 71 while the static plan says 111 decodes 2/2
#     recon-exact;
#   * the CDF chain still rejects the mixed form (carry without continuity or
#     per-frame reset) with a named error.
#
# Usage: src/pipeline/video/video_lpl_deep_gate.sh [--bin build/av2enc]
#                                                 [--dav2d PATH]
# Exit: 0 pass, 1 failure, 2 usage, 77 missing fixture/decoder.
set -uo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/../../.." && pwd)
BIN=""
DAV2D="${DAV2D:-}"
TESTDATA="${AV2ENC_TESTDATA:-$ROOT/testdata}"

while [ $# -gt 0 ]; do
    case "$1" in
        --bin) BIN=${2:-}; shift 2 ;;
        --dav2d) DAV2D=${2:-}; shift 2 ;;
        -h|--help) sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "video_lpl_deep_gate: unknown argument $1" >&2; exit 2 ;;
    esac
done

BIN=${BIN:-"$ROOT/build/av2enc"}
[ -x "$BIN" ] || {
    echo "video_lpl_deep_gate: --bin must point at av2enc" >&2; exit 2; }
if [ -z "$DAV2D" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
    done
fi
[ -n "$DAV2D" ] && [ -x "$DAV2D" ] || {
    echo "video_lpl_deep_gate: SKIP (exit 77): no dav2d binary" >&2
    exit 77; }

NASA="$TESTDATA/nasa_480x240_30fps_60f.y4m"
[ -f "$NASA" ] || {
    echo "video_lpl_deep_gate: SKIP (exit 77): no $NASA" >&2
    exit 77; }
PUB="$TESTDATA/pub_640x360_4f.y4m"

TMP=$(mktemp -d "${TMPDIR:-/tmp}/av2enc-lpl-deep.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
FAILED=0
fail() { echo "FAIL: $*" >&2; FAILED=1; }

QP=111
BUDGET=3250
THREADS=8
COMMON=(--video --frames 60 --qp "$QP" --cpu-threads "$THREADS" --fast-decode
        --video-drift-refresh-threshold 1.5
        --video-drift-refresh-budget "$BUDGET")

echo "== video lpl36 deep anchor gate =="

md5_of() { python3 - "$1" <<'PY'
import hashlib, sys
print(hashlib.md5(open(sys.argv[1], 'rb').read()).hexdigest())
PY
}

# Payload of a Y4M: the header line is dropped and the FRAME markers are
# removed, so two files with the same visible pictures compare byte for byte.
payload_bytes() { python3 - "$1" <<'PY'
import sys
data = open(sys.argv[1], 'rb').read()
nl = data.index(b'\n') + 1
print(len(data[nl:].replace(b'FRAME\n', b'')))
PY
}

# --- gate off keeps the pinned streams -----------------------------------
if ! "$BIN" "$NASA" -o "$TMP/off.ivf" --video --frames 60 --qp 110 \
        --cpu-threads "$THREADS" --fast-decode >/dev/null 2>&1; then
    fail "the env-unset nightly control encode failed"
fi
OFF_MD5=$(md5_of "$TMP/off.ivf")
[ "$OFF_MD5" = "1b99127a4cd1d386fa5600fb63c57a83" ] ||
    fail "env-unset nightly md5 is $OFF_MD5, expected 1b99127a..."

if ! AV2ENC_PIPE_LPL_DEEP=0 "$BIN" "$NASA" -o "$TMP/zero.ivf" \
        --video --frames 60 --qp 110 --cpu-threads "$THREADS" --fast-decode \
        >/dev/null 2>&1; then
    fail "the AV2ENC_PIPE_LPL_DEEP=0 control encode failed"
fi
ZERO_MD5=$(md5_of "$TMP/zero.ivf")
[ "$ZERO_MD5" = "$OFF_MD5" ] ||
    fail "AV2ENC_PIPE_LPL_DEEP=0 changed the nightly stream ($ZERO_MD5)"

# --- the deep switch is inert when off ----------------------------------
if ! AV2ENC_PIPE_LPL_REFRESH=1 "$BIN" "$NASA" -o "$TMP/shallow.ivf" \
        "${COMMON[@]}" >/dev/null 2>&1; then
    fail "the shallow lpl36-refresh control encode failed"
fi
if ! AV2ENC_PIPE_LPL_REFRESH=1 \
        AV2ENC_PIPE_LPL_DEEP=0 "$BIN" "$NASA" -o "$TMP/shallow0.ivf" \
        "${COMMON[@]}" >/dev/null 2>&1; then
    fail "the AV2ENC_PIPE_LPL_DEEP=0 shallow encode failed"
fi
cmp -s "$TMP/shallow.ivf" "$TMP/shallow0.ivf" ||
    fail "AV2ENC_PIPE_LPL_DEEP=0 is not byte-identical to the shallow lpl36 stream"

# --- gate on: key-depth anchors, plan, bytes -----------------------------
if ! AV2ENC_PIPE_LPL_DEEP=1 "$BIN" "$NASA" -o "$TMP/deep.ivf" "${COMMON[@]}" \
        --video-recon "$TMP/deep_recon.y4m" --video-trace \
        >"$TMP/deep.log" 2>"$TMP/deep.trace"; then
    fail "the deep anchor encode failed"
    cat "$TMP/deep.log" >&2
fi
[ -s "$TMP/deep.ivf" ] || fail "the deep anchor encode produced no stream"

grep -q "^pipe_lpl_deep anchor_qp_depth=27 " "$TMP/deep.trace" ||
    fail "the trace does not report the default key-depth (-27) anchor"
grep -q "^lpl_refresh_plan frames=.*," "$TMP/deep.trace" ||
    fail "the perceptual plan is empty"
grep -q "^lpl_anchor frame=[0-9]* qp=84 depth=27" "$TMP/deep.trace" ||
    fail "no anchor is coded at the key's QP 84"
if grep -q "^lpl_anchor frame=[0-9]* qp=111 " "$TMP/deep.trace"; then
    fail "an anchor kept the shallow inter QP"
fi

ACTUAL_FRAMES=$(grep -o 'actual_frames=[0-9]*' "$TMP/deep.trace" | tail -1 |
                cut -d= -f2)
ACTUAL_BYTES=$(grep -o 'actual_bytes=[0-9]*' "$TMP/deep.trace" | tail -1 |
               cut -d= -f2)
[ -n "$ACTUAL_FRAMES" ] && [ "${ACTUAL_FRAMES:-0}" -gt 0 ] ||
    fail "the deep stream applied no anchor"
[ -n "$ACTUAL_BYTES" ] && [ "${ACTUAL_BYTES:-0}" -gt 0 ] ||
    fail "the deep stream charged no anchor payload"
# The emit path checks the allowance before an anchor, so one anchor may
# overshoot the remaining budget; bound the total to one deep anchor's worth.
[ "${ACTUAL_BYTES:-0}" -le $((BUDGET + 2200)) ] ||
    fail "the deep stream spent ${ACTUAL_BYTES:-0} B, more than one anchor over the $BUDGET B allowance"

if ! "$DAV2D" -q --strict 1 -i "$TMP/deep.ivf" -o "$TMP/deep_dec.y4m" \
        --threads 1 >"$TMP/deep_strict.log" 2>&1; then
    fail "the deep anchor stream failed dav2d --strict 1"
    cat "$TMP/deep_strict.log" >&2
fi
DEC_BYTES=$(payload_bytes "$TMP/deep_dec.y4m")
[ "$DEC_BYTES" = "$((60 * 480 * 240 * 3 / 2))" ] ||
    fail "the decoder produced $DEC_BYTES B, expected 60 pictures"
if ! python3 - "$TMP/deep_recon.y4m" "$TMP/deep_dec.y4m" <<'PY'
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
    fail "decoder output differs from the encoder reconstruction"
fi

# --- CDF slice cross: anchor QP 71 vs static 111, first P frame ----------
if [ -f "$PUB" ]; then
    if ! AV2ENC_PIPE_LPL_DEEP=1 "$BIN" "$PUB" -o "$TMP/cross.ivf" \
            --video --frames 2 --cpu-threads "$THREADS" --qp "$QP" --fast-decode \
            --video-drift-refresh-budget 20000 --video-drift-refresh-qp 40 \
            --video-recon "$TMP/cross_recon.y4m" --video-trace \
            >"$TMP/cross.log" 2>"$TMP/cross.trace"; then
        fail "the CDF slice-cross encode failed"
    fi
    grep -q "^lpl_anchor frame=1 qp=71 depth=40" "$TMP/cross.trace" ||
        fail "the slice-cross case did not anchor frame 1 at QP 71"
    "$DAV2D" -q --strict 1 -i "$TMP/cross.ivf" -o "$TMP/cross_dec.y4m" \
        --threads 1 >"$TMP/cross_strict.log" 2>&1
    CROSS_BYTES=$(payload_bytes "$TMP/cross_dec.y4m")
    [ "$CROSS_BYTES" = "$((2 * 640 * 360 * 3 / 2))" ] ||
        fail "the slice-cross stream decoded $CROSS_BYTES B, expected 2 pictures (QP-change CDF desync)"
    if ! python3 - "$TMP/cross_recon.y4m" "$TMP/cross_dec.y4m" <<'PY'
import sys

def payload(path):
    data = open(path, 'rb').read()
    nl = data.index(b'\n') + 1
    return data[nl:].replace(b'FRAME\n', b'')

if payload(sys.argv[1]) != payload(sys.argv[2]):
    print("slice-cross recon differs from the decoded payload", file=sys.stderr)
    sys.exit(1)
PY
    then
        fail "the slice-cross stream is not recon-exact"
    fi
else
    echo "video_lpl_deep_gate: no $PUB; slice-cross case skipped"
fi

# --- the mixed CDF form is still rejected by name ------------------------
if "$BIN" "$NASA" -o "$TMP/mixed.ivf" --video --frames 4 --qp "$QP" \
        --cpu-threads "$THREADS" --avg-cdf --adaptive-cdf \
        --video-drift-refresh --video-drift-refresh-qp 16 \
        >/dev/null 2>"$TMP/mixed.err"; then
    fail "the mixed CDF form (carry without continuity) was accepted"
fi
grep -q "needs the §9.3 CDF image re-seeded" "$TMP/mixed.err" ||
    fail "the mixed CDF rejection did not name the coded-QP seed"

# --- the other legal composition: per-frame §9.3 reset -------------------
if [ -f "$PUB" ]; then
    if ! AV2ENC_PIPE_LPL_DEEP=1 "$BIN" "$PUB" -o "$TMP/reset.ivf" \
            --video --frames 2 --cpu-threads "$THREADS" --qp "$QP" --fast-decode \
            --video-cdf-per-frame-qp --video-drift-refresh-budget 20000 \
            --video-drift-refresh-qp 40 --video-recon "$TMP/reset_recon.y4m" \
            --video-trace >"$TMP/reset.log" 2>"$TMP/reset.trace"; then
        fail "the per-frame-reset composition encode failed"
    fi
    grep -q "^lpl_anchor frame=1 qp=71 depth=40" "$TMP/reset.trace" ||
        fail "the per-frame-reset case did not anchor frame 1 at QP 71"
    "$DAV2D" -q --strict 1 -i "$TMP/reset.ivf" -o "$TMP/reset_dec.y4m" \
        --threads 1 >"$TMP/reset_strict.log" 2>&1
    RESET_BYTES=$(payload_bytes "$TMP/reset_dec.y4m")
    [ "$RESET_BYTES" = "$((2 * 640 * 360 * 3 / 2))" ] ||
        fail "the per-frame-reset stream decoded $RESET_BYTES B, expected 2 pictures"
    if ! python3 - "$TMP/reset_recon.y4m" "$TMP/reset_dec.y4m" <<'PY'
import sys

def payload(path):
    data = open(path, 'rb').read()
    nl = data.index(b'\n') + 1
    return data[nl:].replace(b'FRAME\n', b'')

if payload(sys.argv[1]) != payload(sys.argv[2]):
    print("per-frame-reset recon differs from the decoded payload", file=sys.stderr)
    sys.exit(1)
PY
    then
        fail "the per-frame-reset stream is not recon-exact"
    fi
fi

if [ "$FAILED" -ne 0 ]; then
    echo "video_lpl_deep_gate: FAILED" >&2
    exit 1
fi
echo "video_lpl_deep_gate: OK (off pins, inert deep-off, key-depth anchors, strict decode, recon, CDF slice cross)"
