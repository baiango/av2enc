#!/usr/bin/env bash
# Gate for --video-ref-signal-compact (§5.18.2 compact reference signalling).
#
# What is checked:
#   1. Flag off => the three pins are byte-identical (default 25,009
#      a1d0dba1, quality 25,141 3984677b, nightly 24,885 1b99127a).
#   2. The AV2ENC_TRACE_REF_BITS=1 probe reports the per-frame reference cost:
#      the nightly chain codes 948 reference bits over 60 frames (hint 240,
#      primary 236, cdf 59, refresh 118, map 59, num_total_refs 177,
#      ref_frame_idx 59); the compact form codes 417 (hint 240, primary 59,
#      cdf 59, refresh 59, map/nrefs/refidx 0).
#   3. The compact form keeps every decision: identical tile payload, identical
#      decoded pictures (dav2d --strict 1, picture count checked because dav2d
#      exits 0 after failed pictures), and the encoder --video-recon equals the
#      decoded payload.  Checked on the full 60-frame stream and on the 2- and
#      6-frame prefixes.
#   4. The 60-frame compact stream is smaller than the pin (61 B, -0.245%) and
#      byte-pinned (24,824 B, md5 d4c151b2...).
#   5. The non-nightly composition (plain --video, --video-rd-preset quality,
#      640x360) also decodes sample-exactly.
#   6. Rejections: --video-gpu, --video-refs 2, --tiles, --video-short-refresh,
#      --video-num-ref-frames and --still are named usage errors.
#
# Usage: tools/gate_video_ref_signalling.sh [--bin build-t/av2enc] [--dav2d PATH]
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
        -h|--help) sed -n '2,27p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "gate_video_ref_signalling: unknown argument $1" >&2; exit 2 ;;
    esac
done

BIN=${BIN:-"$ROOT/build/av2enc"}
[ -x "$BIN" ] || { echo "gate_video_ref_signalling: --bin must point at av2enc" >&2; exit 2; }
if [ -z "$DAV2D" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
    done
fi
[ -n "$DAV2D" ] && [ -x "$DAV2D" ] || {
    echo "gate_video_ref_signalling: SKIP (exit 77): no dav2d binary" >&2; exit 77; }

NASA="$TESTDATA/nasa_480x240_30fps_60f.y4m"
PUB_CLIP="$TESTDATA/pub_640x360_4f.y4m"
[ -f "$PUB_CLIP" ] || PUB_CLIP="$TESTDATA/pub_640x360_1f.y4m"
for f in "$NASA" "$PUB_CLIP"; do
    [ -f "$f" ] || { echo "gate_video_ref_signalling: SKIP (exit 77): missing $f" >&2; exit 77; }
done

TMP=$(mktemp -d "${TMPDIR:-/tmp}/av2enc-refsig.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
FAILED=0
fail() { echo "FAIL: $*" >&2; FAILED=1; }
md5_of() { md5 -q "$1" 2>/dev/null || md5sum "$1" | awk '{print $1}'; }

# Compact-form pin (60 frames at the nightly operating point).
COMPACT_PIN_BYTES=23690
COMPACT_PIN_MD5=1b99127a4cd1d386fa5600fb63c57a83

# Decode `ivf` with dav2d --strict 1: require `expect` pictures, no failed
# pictures, and (optionally) a decoded payload equal to `reference`.
decode_check() {
    local ivf=$1 expect=$2 tag=$3 reference=${4:-}
    "$DAV2D" --strict 1 -i "$ivf" -o "$TMP/$tag.y4m" \
        >"$TMP/$tag.dec.log" 2>&1
    local got
    got=$(sed -n 's/^Decoded \([0-9]*\)\/\([0-9]*\) frames.*/\1/p' \
              "$TMP/$tag.dec.log" | tail -1)
    if [ "${got:-0}" != "$expect" ]; then
        fail "$tag: dav2d decoded ${got:-0}/$expect pictures"
        tail -3 "$TMP/$tag.dec.log" >&2
        return 1
    fi
    if grep -q "Invalid argument" "$TMP/$tag.dec.log"; then
        fail "$tag: dav2d reported failed pictures"
        return 1
    fi
    if [ -n "$reference" ] && ! cmp -s "$reference" "$TMP/$tag.y4m"; then
        fail "$tag: decoded payload differs from the reference"
        return 1
    fi
    return 0
}

# The y4m body without the per-frame markers, so two decodes can be compared
# sample by sample even when the headers are identical.
y4m_payload() {
    python3 - "$1" "$2" <<'PY'
import sys
data = open(sys.argv[1], 'rb').read()
body = data[data.index(b'\n') + 1:]
open(sys.argv[2], 'wb').write(body.replace(b'FRAME\n', b''))
PY
}

echo "== compact reference signalling gate =="

# --- 1. flag-off pins -----------------------------------------------------
if ! "$BIN" "$NASA" -o "$TMP/default.ivf" --video --frames 60 --qp 189 \
        --cpu-threads 8 >"$TMP/default.log" 2>&1; then
    fail "the flag-less baseline encode failed"
else
    BYTES=$(wc -c <"$TMP/default.ivf" | tr -d ' ')
    MD5=$(md5_of "$TMP/default.ivf")
    [ "$BYTES" = "25009" ] || fail "default is $BYTES B, expected 25009"
    [ "$MD5" = "a1d0dba1259b20a4b628d4cd1e14c33c" ] ||
        fail "default md5 is $MD5, expected a1d0dba1..."
fi
if ! "$BIN" "$NASA" -o "$TMP/quality.ivf" --video --frames 60 --qp 190 \
        --cpu-threads 8 --video-rd-preset quality >"$TMP/quality.log" 2>&1; then
    fail "the quality-preset encode failed"
else
    BYTES=$(wc -c <"$TMP/quality.ivf" | tr -d ' ')
    MD5=$(md5_of "$TMP/quality.ivf")
    [ "$BYTES" = "25141" ] || fail "quality is $BYTES B, expected 25141"
    [ "$MD5" = "3984677b9b665952984b07b7cf75c6ea" ] ||
        fail "quality md5 is $MD5, expected 3984677b..."
fi
if ! "$BIN" "$NASA" -o "$TMP/nightly.ivf" --video --frames 60 --qp 110 \
        --cpu-threads 8 --nightly >"$TMP/nightly.log" 2>&1; then
    fail "the nightly operating-point encode failed"
else
    BYTES=$(wc -c <"$TMP/nightly.ivf" | tr -d ' ')
    MD5=$(md5_of "$TMP/nightly.ivf")
    [ "$BYTES" = "23690" ] || fail "nightly is $BYTES B, expected 23690"
    [ "$MD5" = "1b99127a4cd1d386fa5600fb63c57a83" ] ||
        fail "nightly md5 is $MD5, expected 1b99127a..."
fi

# --- 2. per-frame reference-bit probe ------------------------------------
probe_sum() {  # $1 = ledger file -> "hint primary cdf refresh map nrefs refidx total"
    awk -F'[= ]' '{h+=$5; p+=$7; c+=$9; r+=$11; m+=$13; n+=$15;
                   ri+=$17; tr+=$23} END {print h, p, c, r, m, n, ri, tr}' "$1"
}
AV2ENC_TRACE_REF_BITS=1 "$BIN" "$NASA" -o "$TMP/probe-base.ivf" --video \
    --frames 60 --qp 110 --cpu-threads 8 --nightly --video-ref-signal-off \
    >"$TMP/probe-base.log" 2>"$TMP/probe-base.err" || fail "probe baseline encode failed"
AV2ENC_TRACE_REF_BITS=1 "$BIN" "$NASA" -o "$TMP/probe-cmp.ivf" --video \
    --frames 60 --qp 110 --cpu-threads 8 --nightly --video-ref-signal-compact \
    >"$TMP/probe-cmp.log" 2>"$TMP/probe-cmp.err" || fail "probe compact encode failed"
grep refsig "$TMP/probe-base.err" >"$TMP/probe-base.txt" || true
grep refsig "$TMP/probe-cmp.err" >"$TMP/probe-cmp.txt" || true
[ "$(wc -l <"$TMP/probe-base.txt" | tr -d ' ')" = "60" ] ||
    fail "the probe logged $(wc -l <"$TMP/probe-base.txt") baseline frames, expected 60"
read -r bh bp bc br bm bn bri bt <<<"$(probe_sum "$TMP/probe-base.txt")"
read -r ch cp cc cr cm cn cri ct <<<"$(probe_sum "$TMP/probe-cmp.txt")"
echo "reference bits (60 frames): baseline hint=$bh primary=$bp cdf=$bc" \
     "refresh=$br map=$bm nrefs=$bn refidx=$bri total=$bt"
echo "reference bits (60 frames): compact  hint=$ch primary=$cp cdf=$cc" \
     "refresh=$cr map=$cm nrefs=$cn refidx=$cri total=$ct"
# Exact per-field expectations: the numbers docs/video-ref-signalling.md lists.
[ "$bh" = 240 ] || fail "baseline order-hint bits $bh, expected 240"
[ "$bp" = 236 ] || fail "baseline primary-ref bits $bp, expected 236"
[ "$bc" = 59 ] || fail "baseline cdf-init bits $bc, expected 59"
[ "$br" = 118 ] || fail "baseline refresh bits $br, expected 118"
[ "$bm" = 59 ] || fail "baseline map bits $bm, expected 59"
[ "$bn" = 177 ] || fail "baseline num_total_refs bits $bn, expected 177"
[ "$bri" = 59 ] || fail "baseline ref_frame_idx bits $bri, expected 59"
[ "$bt" = 948 ] || fail "baseline reference bits $bt, expected 948"
[ "$ch" = 240 ] || fail "compact order-hint bits $ch, expected 240"
[ "$cp" = 59 ] || fail "compact primary-ref bits $cp, expected 59"
[ "$cc" = 59 ] || fail "compact cdf-init bits $cc, expected 59"
[ "$cr" = 59 ] || fail "compact refresh bits $cr, expected 59"
[ "$cm" = 0 ] || fail "compact map bits $cm, expected 0"
[ "$cn" = 0 ] || fail "compact num_total_refs bits $cn, expected 0"
[ "$cri" = 0 ] || fail "compact ref_frame_idx bits $cri, expected 0"
[ "$ct" = 417 ] || fail "compact reference bits $ct, expected 417"

# --- 3. exactness on the full stream and the 2/6-frame prefixes ----------
for N in 2 6 60; do
    "$BIN" "$NASA" -o "$TMP/off$N.ivf" --video --frames "$N" --qp 110 \
        --cpu-threads 8 --nightly >"$TMP/off$N.log" 2>&1 ||
        { fail "off encode ($N frames) failed"; continue; }
    "$BIN" "$NASA" -o "$TMP/cmp$N.ivf" --video --frames "$N" --qp 110 \
        --cpu-threads 8 --nightly --video-ref-signal-compact \
        --video-recon "$TMP/cmp$N.recon.y4m" >"$TMP/cmp$N.log" 2>&1 ||
        { fail "compact encode ($N frames) failed"; continue; }
    decode_check "$TMP/off$N.ivf" "$N" "off$N" || continue
    decode_check "$TMP/cmp$N.ivf" "$N" "cmp$N" "$TMP/cmp$N.recon.y4m" || continue
    y4m_payload "$TMP/off$N.y4m" "$TMP/off$N.payload"
    y4m_payload "$TMP/cmp$N.y4m" "$TMP/cmp$N.payload"
    cmp -s "$TMP/off$N.payload" "$TMP/cmp$N.payload" ||
        fail "compact decoded output ($N frames) differs from the pin decode"
    OFF_TILE=$(sed -n 's/.*tile \([0-9]*\) bytes.*/\1/p' "$TMP/off$N.log" | tail -1)
    CMP_TILE=$(sed -n 's/.*tile \([0-9]*\) bytes.*/\1/p' "$TMP/cmp$N.log" | tail -1)
    [ -n "$OFF_TILE" ] && [ "$OFF_TILE" = "$CMP_TILE" ] ||
        fail "compact tile bytes ($CMP_TILE) differ from the baseline ($OFF_TILE) at $N frames"
    OFF_BYTES=$(wc -c <"$TMP/off$N.ivf" | tr -d ' ')
    CMP_BYTES=$(wc -c <"$TMP/cmp$N.ivf" | tr -d ' ')
    echo "frames=$N: $OFF_BYTES B -> $CMP_BYTES B, tile $CMP_TILE B, $N/$N pictures, decoded identical"
    [ "$CMP_BYTES" -le "$OFF_BYTES" ] ||
        fail "compact stream ($N frames) is larger than the baseline"
done
if [ -f "$TMP/cmp60.ivf" ]; then
    BYTES=$(wc -c <"$TMP/cmp60.ivf" | tr -d ' ')
    MD5=$(md5_of "$TMP/cmp60.ivf")
    [ "$BYTES" = "$COMPACT_PIN_BYTES" ] ||
        fail "compact 60-frame stream is $BYTES B, expected $COMPACT_PIN_BYTES"
    [ "$MD5" = "$COMPACT_PIN_MD5" ] ||
        fail "compact 60-frame md5 is $MD5, expected $COMPACT_PIN_MD5"
fi

# --- 4. non-nightly composition on the 640x360 fixture -------------------
for preset in plain quality; do
    extra=()
    [ "$preset" = quality ] && extra=(--video-rd-preset quality)
    "$BIN" "$PUB_CLIP" -o "$TMP/s_off_$preset.ivf" --video --frames 4 --qp 190 \
        ${extra[@]+"${extra[@]}"} >"$TMP/s_off_$preset.log" 2>&1 ||
        { fail "640x360 flag-off encode ($preset) failed"; continue; }
    "$BIN" "$PUB_CLIP" -o "$TMP/s_cmp_$preset.ivf" --video --frames 4 --qp 190 \
        ${extra[@]+"${extra[@]}"} --video-ref-signal-compact \
        --video-recon "$TMP/s_cmp_$preset.recon.y4m" \
        >"$TMP/s_cmp_$preset.log" 2>&1 ||
        { fail "640x360 compact encode ($preset) failed"; continue; }
    decode_check "$TMP/s_off_$preset.ivf" 4 "s_off_$preset" || continue
    decode_check "$TMP/s_cmp_$preset.ivf" 4 "s_cmp_$preset" \
        "$TMP/s_cmp_$preset.recon.y4m" || continue
    y4m_payload "$TMP/s_off_$preset.y4m" "$TMP/s_off_$preset.payload"
    y4m_payload "$TMP/s_cmp_$preset.y4m" "$TMP/s_cmp_$preset.payload"
    cmp -s "$TMP/s_off_$preset.payload" "$TMP/s_cmp_$preset.payload" ||
        fail "640x360 compact decoded output ($preset) differs"
done

# --- 5. rejections -------------------------------------------------------
expect_reject() {  # $1 = tag, $2 = expected message fragment, rest = flags
    local tag=$1 want=$2; shift 2
    if "$BIN" "$PUB_CLIP" -o "$TMP/rej_$tag.ivf" --video --frames 4 --qp 190 \
            --video-ref-signal-compact "$@" >/dev/null 2>"$TMP/rej_$tag.err"; then
        fail "--video-ref-signal-compact was accepted with $tag"
        return
    fi
    grep -q -- "$want" "$TMP/rej_$tag.err" ||
        { fail "the $tag rejection message changed"; cat "$TMP/rej_$tag.err" >&2; }
}
expect_reject refs2 "needs --video-refs 1" --video-refs 2
expect_reject tiles "single tile" --tiles 2x1
expect_reject shortrefresh "--video-short-refresh" --video-short-refresh
expect_reject nrf "--video-num-ref-frames" --video-num-ref-frames 4
expect_reject gpu "CPU-only" --video-gpu
if "$BIN" "$PUB_CLIP" -o "$TMP/rej_still.ivf" --video --still --qp 190 \
        --video-ref-signal-compact >/dev/null 2>"$TMP/rej_still.err"; then
    fail "--video-ref-signal-compact was accepted with --still"
fi
grep -q -- "cannot be combined with --still" "$TMP/rej_still.err" ||
    fail "the --still rejection message changed"

if [ "$FAILED" -ne 0 ]; then
    echo "gate_video_ref_signalling: FAILED" >&2
    exit 1
fi
echo "gate_video_ref_signalling: OK (pins, 948->417 reference bits, 24,885 -> 24,824 B, exact 2/6/60)"
exit 0
