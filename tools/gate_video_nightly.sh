#!/usr/bin/env bash
# Gate for --fast-decode, the CPU-only algorithmic RD aggregate.
#
# Checks:
#   * --fast-decode expands to the documented flags (same IVF bytes as spelling
#     every flag out);
#   * the reference-intro QP offset is live under --fast-decode and an explicit 0
#     restores the fixed-QP decisions;
#   * the flag-less default, the stable quality preset and the nightly
#     operating point keep their pins;
#   * --fast-decode decodes under dav2d --strict 1 and the visible encoder recon
#     equals the decoded payload;
#   * the BAWP carrier selects fitted predictors on interior and bottom-edge
#     leaves, decodes --strict 1 and reproduces the visible recon;
#   * --fast-decode --video-gpu and --fast-decode --still are rejected by name, and the
#     CPU-only BAWP carrier rejects --video-gpu/--tiles/--b-frames.
#
# Usage: tools/gate_video_nightly.sh [--bin build/av2enc] [--dav2d PATH]
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
        -h|--help) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "gate_video_nightly: unknown argument $1" >&2; exit 2 ;;
    esac
done

BIN=${BIN:-"$ROOT/build/av2enc"}
[ -x "$BIN" ] || { echo "gate_video_nightly: --bin must point at av2enc" >&2; exit 2; }
if [ -z "$DAV2D" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
    done
fi
[ -n "$DAV2D" ] && [ -x "$DAV2D" ] || {
    echo "gate_video_nightly: SKIP (exit 77): no dav2d binary" >&2; exit 77; }
export DAV2D

NASA="$TESTDATA/nasa_480x240_30fps_60f.y4m"
PUB_CLIP="$TESTDATA/pub_640x360_4f.y4m"
[ -f "$PUB_CLIP" ] || PUB_CLIP="$TESTDATA/pub_640x360_1f.y4m"
if [ ! -f "$PUB_CLIP" ]; then
    echo "gate_video_nightly: SKIP (exit 77): no short clip in $TESTDATA" >&2
    exit 77
fi

TMP=$(mktemp -d "${TMPDIR:-/tmp}/av2enc-nightly.XXXXXX")
# --video-rd-preset best now carries the joint plan solve (a ~12 min/60-frame
# research gate).  This gate checks the preset expansion against the joint-off
# lead stack so it stays fast; the joint arm is pinned by
# gate_video_joint_plan.sh.  --fast-decode never uses the best preset.
export AV2ENC_RD_JOINT_P=0
trap 'rm -rf "$TMP"' EXIT
FAILED=0
fail() { echo "FAIL: $*" >&2; FAILED=1; }

NIGHTLY_FLAGS=(--video-rd-preset quality --video-ccso-rd --video-edge-inter
               --video-partition-rd
               --video-key-qp-offset -27 --video-rdoq --video-edge-mask
               --video-cdef --video-cdef-rd
               --video-qp-lambda
               --video-tile-flush-min --video-ref-signal-compact
               --video-tile-group-compact --video-ccso-header-rd
               --video-ccso-bands 6 --video-ccso-lambda 22
               --video-ccso-refit 2 --video-ccso-unit-bits 2
               --video-bawp --video-bawp-carrier --video-leaf-rd
               --video-subpel-eighth
               --video-key-intra-txpart
               --avg-cdf --adaptive-cdf --cdf-continuity
               --video-drift-refresh)

echo "== nightly RD gate =="

# --- usage rejections -----------------------------------------------------
if "$BIN" "$PUB_CLIP" -o "$TMP/gpu.ivf" --fast-decode --video-gpu >/dev/null 2>"$TMP/gpu.err"; then
    fail "--fast-decode accepted --video-gpu"
fi
grep -q "fast-decode is CPU-only" "$TMP/gpu.err" ||
    fail "--fast-decode --video-gpu did not report the CPU-only rejection"

if "$BIN" "$PUB_CLIP" -o "$TMP/still.ivf" --fast-decode --still >/dev/null 2>"$TMP/still.err"; then
    fail "--fast-decode accepted --still"
fi
grep -q "fast-decode is a CPU --video mode" "$TMP/still.err" ||
    fail "--fast-decode --still did not report the video-only rejection"

# --- expansion identity ---------------------------------------------------
FRAMES=4
if ! "$BIN" "$PUB_CLIP" -o "$TMP/nightly.ivf" --fast-decode --frames "$FRAMES" \
        --qp 190 >"$TMP/nightly.log" 2>&1; then
    fail "--fast-decode encode failed"
    cat "$TMP/nightly.log" >&2
fi
if ! "$BIN" "$PUB_CLIP" -o "$TMP/explicit.ivf" --video --frames "$FRAMES" \
        --qp 190 "${NIGHTLY_FLAGS[@]}" >"$TMP/explicit.log" 2>&1; then
    fail "the explicit nightly flags failed"
fi
if [ -f "$TMP/nightly.ivf" ] && [ -f "$TMP/explicit.ivf" ]; then
    cmp -s "$TMP/nightly.ivf" "$TMP/explicit.ivf" ||
        fail "--fast-decode differs from its documented flag expansion"
fi

# --- reference-intro QP offset -------------------------------------------
# --fast-decode owns the key/first-P QP offset; an explicit 0 must restore the
# historical fixed-QP decisions, so the two streams must differ.
if ! "$BIN" "$PUB_CLIP" -o "$TMP/nightly-off0.ivf" --fast-decode --frames "$FRAMES" \
        --qp 190 --video-key-qp-offset 0 >"$TMP/nightly-off0.log" 2>&1; then
    fail "the --fast-decode --video-key-qp-offset 0 encode failed"
fi
if [ -f "$TMP/nightly.ivf" ] && [ -f "$TMP/nightly-off0.ivf" ]; then
    cmp -s "$TMP/nightly.ivf" "$TMP/nightly-off0.ivf" &&
        fail "the reference-intro QP offset is not live under --fast-decode"
fi
if "$BIN" "$PUB_CLIP" -o "$TMP/off-still.ivf" --still \
        --video-key-qp-offset -16 >/dev/null 2>"$TMP/off-still.err"; then
    fail "--video-key-qp-offset was accepted without --video"
fi
if "$BIN" "$PUB_CLIP" -o "$TMP/off-rc.ivf" --video --frames "$FRAMES" \
        --rc crf --video-key-qp-offset -16 >/dev/null 2>"$TMP/off-rc.err"; then
    fail "--video-key-qp-offset was accepted with --rc crf"
fi
grep -q "fixed --qp stream" "$TMP/off-rc.err" ||
    fail "--video-key-qp-offset + --rc crf did not name the fixed-QP contract"

# --- default and quality pins --------------------------------------------
if [ -f "$NASA" ]; then
    if ! "$BIN" "$NASA" -o "$TMP/default.ivf" --video --frames 60 \
            --qp 189 --cpu-threads 8 >"$TMP/default.log" 2>&1; then
        fail "the flag-less baseline encode failed"
    fi
    if [ -f "$TMP/default.ivf" ]; then
        BYTES=$(wc -c <"$TMP/default.ivf" | tr -d ' ')
        MD5=$(python3 - "$TMP/default.ivf" <<'PY'
import hashlib, sys
print(hashlib.md5(open(sys.argv[1], 'rb').read()).hexdigest())
PY
)
        [ "$BYTES" = "25009" ] || fail "default stream is $BYTES B, expected 25009"
        [ "$MD5" = "a1d0dba1259b20a4b628d4cd1e14c33c" ] ||
            fail "default stream md5 is $MD5, expected a1d0dba1..."
    fi

    if ! "$BIN" "$NASA" -o "$TMP/quality.ivf" --video --frames 60 \
            --qp 190 --cpu-threads 8 --video-rd-preset quality \
            >"$TMP/quality.log" 2>&1; then
        fail "the quality preset encode failed"
    fi
    if [ -f "$TMP/quality.ivf" ]; then
        BYTES=$(wc -c <"$TMP/quality.ivf" | tr -d ' ')
        MD5=$(python3 - "$TMP/quality.ivf" <<'PY'
import hashlib, sys
print(hashlib.md5(open(sys.argv[1], 'rb').read()).hexdigest())
PY
)
        [ "$BYTES" = "25141" ] || fail "quality stream is $BYTES B, expected 25141"
        [ "$MD5" = "3984677b9b665952984b07b7cf75c6ea" ] ||
            fail "quality stream md5 is $MD5, expected 3984677b..."
    fi

    if ! "$BIN" "$NASA" -o "$TMP/nightly60.ivf" --video --frames 60 \
            --qp 110 --cpu-threads 8 --fast-decode >"$TMP/nightly60.log" 2>&1; then
        fail "the nightly operating-point encode failed"
    fi
    if [ -f "$TMP/nightly60.ivf" ]; then
        BYTES=$(wc -c <"$TMP/nightly60.ivf" | tr -d ' ')
        MD5=$(python3 - "$TMP/nightly60.ivf" <<'PY'
import hashlib, sys
print(hashlib.md5(open(sys.argv[1], 'rb').read()).hexdigest())
PY
)
        [ "$BYTES" = "23690" ] ||
            fail "nightly stream is $BYTES B, expected 23690"
        [ "$MD5" = "1b99127a4cd1d386fa5600fb63c57a83" ] ||
            fail "nightly stream md5 is $MD5, expected 1b99127a..."
    fi

    # --- CDEF params-row pricing (--video-cdef-rd) ---------------------
    # The flag only removes re-signalled §5.18.7.10 rows for frames whose CDEF
    # filter changes no sample, so the decoded pictures must be byte-identical
    # to the same aggregate without it while the stream is strictly smaller.
    if [ -f "$TMP/nightly60.ivf" ]; then
        NO_RD_FLAGS=()
        for f in "${NIGHTLY_FLAGS[@]}"; do
            [ "$f" = "--video-cdef-rd" ] || NO_RD_FLAGS+=("$f")
        done
        if ! "$BIN" "$NASA" -o "$TMP/nightly60-nord.ivf" --video --frames 60 \
                --qp 110 --cpu-threads 8 "${NO_RD_FLAGS[@]}" \
                >"$TMP/nightly60-nord.log" 2>&1; then
            fail "the --video-cdef-rd A/B control encode failed"
        fi
        RD_BYTES=$(wc -c <"$TMP/nightly60.ivf" | tr -d ' ')
        NORD_BYTES=$(wc -c <"$TMP/nightly60-nord.ivf" | tr -d ' ')
        [ "$RD_BYTES" -lt "$NORD_BYTES" ] ||
            fail "--video-cdef-rd did not shrink the stream ($RD_BYTES vs $NORD_BYTES)"
        if ! "$DAV2D" -q --strict 1 -i "$TMP/nightly60.ivf" \
                -o "$TMP/rd_dec.y4m" --threads 1 >"$TMP/rd_dec.log" 2>&1; then
            fail "the --video-cdef-rd stream failed dav2d --strict 1"
        fi
        if ! "$DAV2D" -q --strict 1 -i "$TMP/nightly60-nord.ivf" \
                -o "$TMP/nord_dec.y4m" --threads 1 >"$TMP/nord_dec.log" 2>&1; then
            fail "the --video-cdef-rd A/B control failed dav2d --strict 1"
        fi
        if [ -f "$TMP/rd_dec.y4m" ] && [ -f "$TMP/nord_dec.y4m" ]; then
            cmp -s "$TMP/rd_dec.y4m" "$TMP/nord_dec.y4m" ||
                fail "--video-cdef-rd changed the decoded pictures"
            PICS=$(python3 - "$TMP/rd_dec.y4m" <<'PY'
import sys
print(open(sys.argv[1], 'rb').read().count(b'FRAME'))
PY
)
            [ "$PICS" = "60" ] ||
                fail "--video-cdef-rd decoded $PICS pictures, expected 60"
        fi
        echo "CDEF params-row pricing: $NORD_BYTES -> $RD_BYTES B, decoded identical"
    fi

    # --- best preset byte-identity --------------------------------------
    # --video-rd-preset best is the measured flat lead stack: the --fast-decode
    # aggregate plus the nine env-only RD tools, the drift-refresh allocation
    # point and --video-key-qp-offset -39.  It is a LEAD (K 86.33, +0.097 over
    # the 86.232 flat aggregate), not a KEEP.  The preset must reproduce the
    # explicit command byte for byte at the three measured QPs.
    echo "== best preset: --video-rd-preset best vs the explicit env command"
    BEST_QPS="114 116 118"
    for q in $BEST_QPS; do
        if ! "$BIN" "$NASA" -o "$TMP/best-$q.ivf" --frames 60 --qp "$q" \
                --cpu-threads 8 --video-rd-preset best \
                >"$TMP/best-$q.log" 2>&1; then
            fail "the --video-rd-preset best encode failed at QP$q"
            tail -4 "$TMP/best-$q.log" >&2
            continue
        fi
        if ! env AV2ENC_EFF_PERC=1 AV2ENC_RD_UVFULL=1 \
                AV2ENC_TX_DETAIL_SEARCH_TAIL=1 AV2ENC_TX_DETAIL_MIN=0.50 \
                AV2ENC_RD_KEY_TX=1 AV2ENC_RD_UVLEVEL=1 AV2ENC_RD_DRIVER=1 \
                AV2ENC_RD_HEAD_TX=1 AV2ENC_RD_TAIL_TX=1 \
                "$BIN" "$NASA" -o "$TMP/best-explicit-$q.ivf" --video \
                --frames 60 --qp "$q" --cpu-threads 8 --fast-decode \
                --video-drift-refresh-threshold 1.5 \
                --video-drift-refresh-budget 3250 \
                --video-key-qp-offset -39 \
                >"$TMP/best-explicit-$q.log" 2>&1; then
            fail "the explicit best-stack encode failed at QP$q"
            tail -4 "$TMP/best-explicit-$q.log" >&2
            continue
        fi
        if ! cmp -s "$TMP/best-$q.ivf" "$TMP/best-explicit-$q.ivf"; then
            fail "--video-rd-preset best differs from the explicit stack at QP$q"
        else
            echo "  QP$q: $(wc -c <"$TMP/best-$q.ivf" | tr -d ' ') B, preset identical"
        fi
    done
    # An explicit env value wins over the preset, including 0 for an ablation.
    if ! env AV2ENC_RD_UVLEVEL=0 "$BIN" "$NASA" \
            -o "$TMP/best-116-uvlevel0.ivf" --frames 60 --qp 116 \
            --cpu-threads 8 --video-rd-preset best \
            >"$TMP/best-116-uvlevel0.log" 2>&1; then
        fail "the --video-rd-preset best AV2ENC_RD_UVLEVEL=0 encode failed"
    elif cmp -s "$TMP/best-116.ivf" "$TMP/best-116-uvlevel0.ivf"; then
        fail "the best preset overrode an explicit AV2ENC_RD_UVLEVEL=0"
    else
        echo "  explicit AV2ENC_RD_UVLEVEL=0 wins over the preset"
    fi
    # The lead stack is still a flat key+P stream: the pinned dav2d must
    # decode it (this is not an avmdec-only research stream).
    if ! "$DAV2D" -q --strict 1 -i "$TMP/best-116.ivf" \
            -o "$TMP/best-116-dec.y4m" --threads 1 \
            >"$TMP/best-116-strict.log" 2>&1; then
        fail "--video-rd-preset best failed dav2d --strict 1 at QP116"
    fi
fi

# --- strict decode + visible recon ---------------------------------------
if ! "$DAV2D" -q --strict 1 -i "$TMP/nightly.ivf" -o "$TMP/decoded.y4m" \
        --threads 1 >"$TMP/strict.log" 2>&1; then
    fail "--fast-decode stream failed dav2d --strict 1"
    cat "$TMP/strict.log" >&2
fi
if ! "$BIN" "$PUB_CLIP" -o "$TMP/recon.ivf" --fast-decode --frames "$FRAMES" \
        --qp 190 --video-recon "$TMP/recon.y4m" >"$TMP/recon.log" 2>&1; then
    fail "the --fast-decode --video-recon encode failed"
elif ! python3 - "$TMP/recon.y4m" "$TMP/decoded.y4m" <<'PY'
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

# --- BAWP + inter residual on the edge path ------------------------------
# --fast-decode carries the CPU-only BAWP carrier: a GLOBALMV winner on a fade is
# re-expressed as NEARMV/NEWMV so §5.20.7.6 can code use_bawp, and the residual
# is coded against the fitted predictor.  The 60-frame fade clip must select
# the fitted predictor on both interior and bottom-edge leaves, decode under
# --strict 1 and reproduce the visible encoder reconstruction.
if [ -f "$NASA" ]; then
    if ! "$BIN" "$NASA" -o "$TMP/bawp.ivf" --fast-decode --frames 60 \
            --qp 142 --cpu-threads 8 --video-recon "$TMP/bawp.y4m" \
            --video-trace >"$TMP/bawp.log" 2>"$TMP/bawp.trace"; then
        fail "the --fast-decode BAWP encode failed"
        cat "$TMP/bawp.log" >&2
    else
        APPLIED=$(grep -c "carrier=.*applied=1" "$TMP/bawp.trace" || true)
        EDGE_APPLIED=$(awk '/carrier=.*applied=1/{a=1}
            /m2 inter r=48|m2 intra r=48/{if (a) n++; a=0}
            END{print n+0}' "$TMP/bawp.trace")
        echo "BAWP applied leaves=$APPLIED (bottom rows=$EDGE_APPLIED)"
        [ "$APPLIED" -ge 50 ] ||
            fail "only $APPLIED BAWP leaves selected (expected >= 50)"
        [ "$EDGE_APPLIED" -ge 8 ] ||
            fail "only $EDGE_APPLIED BAWP bottom-row leaves (expected >= 8)"
        if ! "$DAV2D" -q --strict 1 -i "$TMP/bawp.ivf" \
                -o "$TMP/bawp_dec.y4m" --threads 1 >"$TMP/bawp_strict.log" 2>&1; then
            fail "the --fast-decode BAWP stream failed dav2d --strict 1"
            cat "$TMP/bawp_strict.log" >&2
        elif ! python3 - "$TMP/bawp.y4m" "$TMP/bawp_dec.y4m" <<'PY'
import sys

def payload(path):
    data = open(path, 'rb').read()
    nl = data.index(b'\n') + 1
    return data[nl:].replace(b'FRAME\n', b'')

a, b = payload(sys.argv[1]), payload(sys.argv[2])
if len(a) != len(b):
    print(f"BAWP recon payload {len(a)} B != decoded {len(b)} B", file=sys.stderr)
    sys.exit(1)
if a != b:
    diff = sum(1 for x, y in zip(a, b) if x != y)
    print(f"BAWP recon differs from decoded output in {diff} byte(s)",
          file=sys.stderr)
    sys.exit(1)
print(f"BAWP visible recon matches decoded output ({len(a)} B)")
PY
        then
            fail "BAWP decoder output differs from the encoder reconstruction"
        fi
    fi

    # The carrier is CPU-only and keeps the plain-chain surface.  Plain
    # --video-bawp stays skip-only (the CUDA BAWP backend parity path).
    if "$BIN" "$PUB_CLIP" -o "$TMP/bawp_gpu.ivf" --video --frames 1 --qp 190 \
            --video-bawp-carrier --video-gpu >/dev/null 2>"$TMP/bawp_gpu.err"; then
        fail "--video-bawp-carrier was accepted with --video-gpu"
    fi
    grep -q "CPU-only" "$TMP/bawp_gpu.err" ||
        fail "the --video-bawp-carrier GPU rejection did not name the CPU-only path"
    if "$BIN" "$PUB_CLIP" -o "$TMP/bawp_tiles.ivf" --video --frames 1 --qp 190 \
            --video-bawp-carrier --tiles 2x1 >/dev/null 2>"$TMP/bawp_tiles.err"; then
        fail "--video-bawp-carrier was accepted with --tiles"
    fi
    grep -q "single tile" "$TMP/bawp_tiles.err" ||
        fail "the --video-bawp-carrier tile rejection did not name the single tile"
    if "$BIN" "$PUB_CLIP" -o "$TMP/bawp_b.ivf" --video --frames 2 --qp 190 \
            --video-bawp-carrier --b-frames 1 >/dev/null 2>"$TMP/bawp_b.err"; then
        fail "--video-bawp-carrier was accepted with --b-frames"
    fi
    grep -q -- "--video-bawp" "$TMP/bawp_b.err" ||
        fail "the --video-bawp-carrier B-frame rejection did not name --video-bawp"
    if "$BIN" "$PUB_CLIP" -o "$TMP/bawp_res.ivf" --video --frames 1 --qp 190 \
            --video-bawp --video-inter-residual >/dev/null 2>"$TMP/bawp_res.err"; then
        fail "plain --video-bawp was accepted with --video-inter-residual"
    fi
    grep -q -- "--video-bawp-carrier" "$TMP/bawp_res.err" ||
        fail "the BAWP+residual rejection did not name --video-bawp-carrier"
fi

# --- the --nightly-b chroma offset default -------------------------------
# --nightly-b codes both chroma planes 8 qindex steps finer than luma (a
# constant §5.4.8 sequence base).  Pin the preset default and its opt-out so a
# later edit cannot silently drop or double-apply the offset: the default must
# equal an explicit -8, and an explicit 0 must restore the single-step stream
# (strictly smaller, because the §5.4.8 group is not coded at all).
echo "== nightly-b chroma offset =="
if [ -f "$NASA" ]; then
    nb() { "$BIN" "$NASA" -o "$1" --video --frames 1 --qp 110 --cpu-threads 8 \
               --nightly-b "${@:2}" >/dev/null 2>&1; }
    if nb "$TMP/nb_def.ivf"; then
        nb "$TMP/nb_off.ivf" --video-uv-qp-offset 0
        if cmp -s "$TMP/nb_def.ivf" "$TMP/nb_off.ivf"; then
            fail "--nightly-b default equals --video-uv-qp-offset 0: the -8 " \
                 "chroma offset is not being applied"
        fi
        DEF=$(wc -c <"$TMP/nb_def.ivf" | tr -d ' ')
        OFF=$(wc -c <"$TMP/nb_off.ivf" | tr -d ' ')
        [ "$DEF" = "11139" ] ||
            fail "--nightly-b default is $DEF B, expected 11139 (offset -8)"
        [ "$OFF" = "10998" ] ||
            fail "--nightly-b opt-out is $OFF B, expected 10998 (no offset)"
        echo "nightly-b chroma offset: default -8 = $DEF B, opt-out 0 = $OFF B"
    else
        fail "the --nightly-b chroma offset encode failed"
    fi
else
    echo "nightly-b chroma offset: SKIP (no nasa clip)" >&2
fi

# --video-loop-filters-off used to be read only by apply_video_nightly, so
# under --nightly-b the flag was accepted and silently ignored: the streams were
# byte-identical.  Assert it actually changes the stream (and that the default
# still matches its pin, so the switch cannot leak into the default).
echo "== nightly-b loop-filters-off =="
if [ -f "$NASA" ]; then
    if "$BIN" "$NASA" -o "$TMP/lf_def.ivf" --video --frames 1 --qp 110 \
            --cpu-threads 8 --nightly-b >/dev/null 2>"$TMP/lf_def.err"; then
        if ! "$BIN" "$NASA" -o "$TMP/lf_off.ivf" --video --frames 1 --qp 110 \
                --cpu-threads 8 --nightly-b --video-loop-filters-off \
                >/dev/null 2>"$TMP/lf_off.err"; then
            fail "--nightly-b --video-loop-filters-off was rejected"
        elif cmp -s "$TMP/lf_def.ivf" "$TMP/lf_off.ivf"; then
            fail "--video-loop-filters-off is silently ignored under " \
                 "--nightly-b (identical streams)"
        else
            LF_DEF=$(wc -c <"$TMP/lf_def.ivf" | tr -d ' ')
            LF_OFF=$(wc -c <"$TMP/lf_off.ivf" | tr -d ' ')
            [ "$LF_DEF" = "11139" ] ||
                fail "--nightly-b default is $LF_DEF B, expected 11139"
            echo "nightly-b loop-filters-off: $LF_DEF B -> $LF_OFF B (takes effect)"
        fi
    else
        fail "the --nightly-b loop-filters-off encode failed"
    fi
else
    echo "nightly-b loop-filters-off: SKIP (no nasa clip)" >&2
fi

if [ "$FAILED" -ne 0 ]; then
    echo "gate_video_nightly: FAILED" >&2
    exit 1
fi
echo "gate_video_nightly: OK (expansion, pins, strict decode, visible recon, BAWP)"
exit 0
