#!/usr/bin/env bash
# Gate for --nightly, the CPU-only algorithmic RD aggregate.
#
# Checks:
#   * --nightly expands to the documented flags (same IVF bytes as spelling
#     every flag out);
#   * the reference-intro QP offset is live under --nightly and an explicit 0
#     restores the fixed-QP decisions;
#   * the flag-less default, the stable quality preset and the nightly
#     operating point keep their pins;
#   * --nightly decodes under dav2d --strict 1 and the visible encoder recon
#     equals the decoded payload;
#   * the BAWP carrier selects fitted predictors on interior and bottom-edge
#     leaves, decodes --strict 1 and reproduces the visible recon;
#   * --nightly --video-gpu and --nightly --still are rejected by name, and the
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
SHINE="$TESTDATA/shine_640x360_4f.y4m"
[ -f "$SHINE" ] || SHINE="$TESTDATA/shine_640x360_1f.y4m"
if [ ! -f "$SHINE" ]; then
    echo "gate_video_nightly: SKIP (exit 77): no short clip in $TESTDATA" >&2
    exit 77
fi

TMP=$(mktemp -d "${TMPDIR:-/tmp}/av2enc-nightly.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
FAILED=0
fail() { echo "FAIL: $*" >&2; FAILED=1; }

NIGHTLY_FLAGS=(--video-rd-preset quality --video-ccso-rd --video-edge-inter
               --video-partition-rd --video-partition-rd-lambda 2.75e-6
               --video-key-qp-offset -16 --video-rdoq --video-edge-mask
               --video-cdef --video-ccso-bands 6 --video-ccso-lambda 22
               --video-ccso-refit 2 --video-ccso-unit-bits 2
               --video-bawp --video-bawp-carrier --video-leaf-rd
               --video-subpel-eighth --video-tx-lambda 136
               --video-key-intra-txpart)

echo "== nightly RD gate =="

# --- usage rejections -----------------------------------------------------
if "$BIN" "$SHINE" -o "$TMP/gpu.ivf" --nightly --video-gpu >/dev/null 2>"$TMP/gpu.err"; then
    fail "--nightly accepted --video-gpu"
fi
grep -q "nightly is CPU-only" "$TMP/gpu.err" ||
    fail "--nightly --video-gpu did not report the CPU-only rejection"

if "$BIN" "$SHINE" -o "$TMP/still.ivf" --nightly --still >/dev/null 2>"$TMP/still.err"; then
    fail "--nightly accepted --still"
fi
grep -q "nightly is a CPU --video mode" "$TMP/still.err" ||
    fail "--nightly --still did not report the video-only rejection"

# --- expansion identity ---------------------------------------------------
FRAMES=4
if ! "$BIN" "$SHINE" -o "$TMP/nightly.ivf" --nightly --frames "$FRAMES" \
        --qp 190 >"$TMP/nightly.log" 2>&1; then
    fail "--nightly encode failed"
    cat "$TMP/nightly.log" >&2
fi
if ! "$BIN" "$SHINE" -o "$TMP/explicit.ivf" --video --frames "$FRAMES" \
        --qp 190 "${NIGHTLY_FLAGS[@]}" >"$TMP/explicit.log" 2>&1; then
    fail "the explicit nightly flags failed"
fi
if [ -f "$TMP/nightly.ivf" ] && [ -f "$TMP/explicit.ivf" ]; then
    cmp -s "$TMP/nightly.ivf" "$TMP/explicit.ivf" ||
        fail "--nightly differs from its documented flag expansion"
fi

# --- reference-intro QP offset -------------------------------------------
# --nightly owns the key/first-P QP offset; an explicit 0 must restore the
# historical fixed-QP decisions, so the two streams must differ.
if ! "$BIN" "$SHINE" -o "$TMP/nightly-off0.ivf" --nightly --frames "$FRAMES" \
        --qp 190 --video-key-qp-offset 0 >"$TMP/nightly-off0.log" 2>&1; then
    fail "the --nightly --video-key-qp-offset 0 encode failed"
fi
if [ -f "$TMP/nightly.ivf" ] && [ -f "$TMP/nightly-off0.ivf" ]; then
    cmp -s "$TMP/nightly.ivf" "$TMP/nightly-off0.ivf" &&
        fail "the reference-intro QP offset is not live under --nightly"
fi
if "$BIN" "$SHINE" -o "$TMP/off-still.ivf" --still \
        --video-key-qp-offset -16 >/dev/null 2>"$TMP/off-still.err"; then
    fail "--video-key-qp-offset was accepted without --video"
fi
if "$BIN" "$SHINE" -o "$TMP/off-rc.ivf" --video --frames "$FRAMES" \
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
            --qp 110 --cpu-threads 8 --nightly >"$TMP/nightly60.log" 2>&1; then
        fail "the nightly operating-point encode failed"
    fi
    if [ -f "$TMP/nightly60.ivf" ]; then
        BYTES=$(wc -c <"$TMP/nightly60.ivf" | tr -d ' ')
        MD5=$(python3 - "$TMP/nightly60.ivf" <<'PY'
import hashlib, sys
print(hashlib.md5(open(sys.argv[1], 'rb').read()).hexdigest())
PY
)
        [ "$BYTES" = "24994" ] ||
            fail "nightly stream is $BYTES B, expected 24994"
        [ "$MD5" = "bc4a3756d312bd259e97397f9bd86853" ] ||
            fail "nightly stream md5 is $MD5, expected bc4a3756..."
    fi
fi

# --- strict decode + visible recon ---------------------------------------
if ! "$DAV2D" -q --strict 1 -i "$TMP/nightly.ivf" -o "$TMP/decoded.y4m" \
        --threads 1 >"$TMP/strict.log" 2>&1; then
    fail "--nightly stream failed dav2d --strict 1"
    cat "$TMP/strict.log" >&2
fi
if ! "$BIN" "$SHINE" -o "$TMP/recon.ivf" --nightly --frames "$FRAMES" \
        --qp 190 --video-recon "$TMP/recon.y4m" >"$TMP/recon.log" 2>&1; then
    fail "the --nightly --video-recon encode failed"
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
# --nightly carries the CPU-only BAWP carrier: a GLOBALMV winner on a fade is
# re-expressed as NEARMV/NEWMV so §5.20.7.6 can code use_bawp, and the residual
# is coded against the fitted predictor.  The 60-frame fade clip must select
# the fitted predictor on both interior and bottom-edge leaves, decode under
# --strict 1 and reproduce the visible encoder reconstruction.
if [ -f "$NASA" ]; then
    if ! "$BIN" "$NASA" -o "$TMP/bawp.ivf" --nightly --frames 60 \
            --qp 142 --cpu-threads 8 --video-recon "$TMP/bawp.y4m" \
            --video-trace >"$TMP/bawp.log" 2>"$TMP/bawp.trace"; then
        fail "the --nightly BAWP encode failed"
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
            fail "the --nightly BAWP stream failed dav2d --strict 1"
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
    if "$BIN" "$SHINE" -o "$TMP/bawp_gpu.ivf" --video --frames 1 --qp 190 \
            --video-bawp-carrier --video-gpu >/dev/null 2>"$TMP/bawp_gpu.err"; then
        fail "--video-bawp-carrier was accepted with --video-gpu"
    fi
    grep -q "CPU-only" "$TMP/bawp_gpu.err" ||
        fail "the --video-bawp-carrier GPU rejection did not name the CPU-only path"
    if "$BIN" "$SHINE" -o "$TMP/bawp_tiles.ivf" --video --frames 1 --qp 190 \
            --video-bawp-carrier --tiles 2x1 >/dev/null 2>"$TMP/bawp_tiles.err"; then
        fail "--video-bawp-carrier was accepted with --tiles"
    fi
    grep -q "single tile" "$TMP/bawp_tiles.err" ||
        fail "the --video-bawp-carrier tile rejection did not name the single tile"
    if "$BIN" "$SHINE" -o "$TMP/bawp_b.ivf" --video --frames 2 --qp 190 \
            --video-bawp-carrier --b-frames 1 >/dev/null 2>"$TMP/bawp_b.err"; then
        fail "--video-bawp-carrier was accepted with --b-frames"
    fi
    grep -q -- "--video-bawp" "$TMP/bawp_b.err" ||
        fail "the --video-bawp-carrier B-frame rejection did not name --video-bawp"
    if "$BIN" "$SHINE" -o "$TMP/bawp_res.ivf" --video --frames 1 --qp 190 \
            --video-bawp --video-inter-residual >/dev/null 2>"$TMP/bawp_res.err"; then
        fail "plain --video-bawp was accepted with --video-inter-residual"
    fi
    grep -q -- "--video-bawp-carrier" "$TMP/bawp_res.err" ||
        fail "the BAWP+residual rejection did not name --video-bawp-carrier"
fi

if [ "$FAILED" -ne 0 ]; then
    echo "gate_video_nightly: FAILED" >&2
    exit 1
fi
echo "gate_video_nightly: OK (expansion, pins, strict decode, visible recon, BAWP)"
exit 0
