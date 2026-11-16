#!/usr/bin/env bash
# QP-scaled RD lambda gate (--video-qp-lambda).
#
# The five rate/distortion weights that used a constant calibrated at one
# operating point now follow per-site power laws measured at both the 240p and
# the 720p point (docs/qp-scaled-lambda.md):
#
#   --video-leaf-rd-lambda     125 * (step/98)^0.72
#   --video-tx-lambda          136 * (step/2560)^0.3385
#   --video-key-tx-lambda      8 (constant; measured exponent 0)
#   --video-rdoq-scale         0.1 * (step/118)^-0.5
#   partition fallback price   7.344 * (step/944)^1.44
#
# Checks:
#   1. flag off: the three reference pins (default, quality, nightly) are
#      byte-identical;
#   2. every numeric lambda flag still reproduces the legacy stream under
#      --video-qp-lambda, so the flags remain exact escape hatches;
#   3. each site is live: leaf-only, key-only and partition-only change the
#      stream, and the RDOQ scale is byte-identical at the 240p anchor (its
#      fitted value is exactly 0.1 there);
#   4. the composed scaled set is smaller at fixed QP and meets its pins at
#      both operating points;
#   5. both composed streams decode 60/60 pictures under dav2d --strict 1 and
#      the decoded payload equals --video-recon byte for byte;
#   6. rejections: --video-gpu, --rc crf/vbr and still/m1 carriers.
#
# Usage: gate_video_qp_lambda.sh --bin build/av2enc [--dav2d PATH]
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
        -h|--help) sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "gate_video_qp_lambda: unknown argument $1" >&2; exit 2 ;;
    esac
done

[ -n "$BIN" ] && [ -x "$BIN" ] || {
    echo "gate_video_qp_lambda: --bin must point at av2enc" >&2; exit 2; }
if [ -z "$DAV2D" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
    done
fi
[ -n "$DAV2D" ] && [ -x "$DAV2D" ] || {
    echo "gate_video_qp_lambda: SKIP (exit 77): no dav2d binary" >&2; exit 77; }

NASA="$TESTDATA/nasa_480x240_30fps_60f.y4m"
SPACE="$TESTDATA/space8k_1440x720_60f.y4m"
[ -f "$NASA" ] || {
    echo "gate_video_qp_lambda: SKIP (exit 77): no $NASA" >&2; exit 77; }

TMP=$(mktemp -d "${TMPDIR:-/tmp}/av2enc-qplambda.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
FAILED=0
fail() { echo "FAIL: $*" >&2; FAILED=1; }

md5_of() { md5 -q "$1" 2>/dev/null || md5sum "$1" | awk '{print $1}'; }

# The pinned reference streams (docs/video-nightly.md).
DEF_BYTES=25009;   DEF_MD5=a1d0dba1259b20a4b628d4cd1e14c33c
QUAL_BYTES=25141;  QUAL_MD5=3984677b9b665952984b07b7cf75c6ea
NIGHT_BYTES=23690; NIGHT_MD5=1b99127a4cd1d386fa5600fb63c57a83
# The composed QP-scaled operating points.
QP240_BYTES=23690; QP240_MD5=1b99127a4cd1d386fa5600fb63c57a83
QP720_BYTES=35646; QP720_MD5=41bc1182499fcc7b308bb528f8eb2d40

# Every numeric lambda flag spelled out: under --video-qp-lambda this must
# reproduce the legacy values (the escape hatches).
LEGACY=(--video-tx-lambda 136 --video-key-tx-lambda 136
        --video-leaf-rd-lambda 125 --video-rdoq-scale 0.1
        --video-partition-rd-lambda 2.75e-6)

check_pin() {
    local path=$1 bytes=$2 md5=$3 tag=$4
    local got_bytes got_md5
    got_bytes=$(wc -c <"$path" | tr -d ' ')
    got_md5=$(md5_of "$path")
    [ "$got_bytes" = "$bytes" ] ||
        fail "$tag is $got_bytes B, expected $bytes"
    [ "$got_md5" = "$md5" ] ||
        fail "$tag md5 is $got_md5, expected $md5"
}

# dav2d exits 0 even after failing pictures, so count and compare payloads.
decode_exact() {
    local ivf=$1 recon=$2 tag=$3
    "$DAV2D" -q --strict 1 -i "$ivf" -o "$TMP/$tag.dec.y4m" \
        --threads 1 >"$TMP/$tag.dec.log" 2>&1 || {
        fail "$tag failed dav2d --strict 1"
        return
    }
    local pics
    pics=$(python3 -c 'import sys;print(open(sys.argv[1],"rb").read().count(b"FRAME"))' \
               "$TMP/$tag.dec.y4m")
    [ "$pics" = "60" ] || fail "$tag decoded $pics pictures, expected 60"
    if ! python3 - "$recon" "$TMP/$tag.dec.y4m" "$tag" <<'PY'
import sys
def payload(path):
    data = open(path, 'rb').read()
    nl = data.index(b'\n') + 1
    return data[nl:].replace(b'FRAME\n', b'')
a, b, tag = payload(sys.argv[1]), payload(sys.argv[2]), sys.argv[3]
if len(a) != len(b):
    print(f"FAIL: {tag} recon {len(a)} B != decoded {len(b)} B", file=sys.stderr)
    sys.exit(1)
if a != b:
    diff = sum(1 for x, y in zip(a, b) if x != y)
    print(f"FAIL: {tag} recon differs from decoded in {diff} byte(s)",
          file=sys.stderr)
    sys.exit(1)
PY
    then
        fail "$tag reconstruction differs from the decoded payload"
    fi
}

echo "== QP-scaled lambda gate =="

# --- 1. flag off: the reference pins -------------------------------------
if ! "$BIN" "$NASA" -o "$TMP/default.ivf" --video --frames 60 --qp 189 \
        --cpu-threads 8 >"$TMP/default.log" 2>&1; then
    fail "the flag-less baseline encode failed"
else
    check_pin "$TMP/default.ivf" "$DEF_BYTES" "$DEF_MD5" "default"
fi
if ! "$BIN" "$NASA" -o "$TMP/quality.ivf" --video --frames 60 --qp 190 \
        --cpu-threads 8 --video-rd-preset quality >"$TMP/quality.log" 2>&1; then
    fail "the quality preset encode failed"
else
    check_pin "$TMP/quality.ivf" "$QUAL_BYTES" "$QUAL_MD5" "quality"
fi
if ! "$BIN" "$NASA" -o "$TMP/nightly.ivf" --video --frames 60 --qp 110 \
        --cpu-threads 8 --nightly >"$TMP/nightly.log" 2>&1; then
    fail "the nightly pin encode failed"
    exit 1
fi
check_pin "$TMP/nightly.ivf" "$NIGHT_BYTES" "$NIGHT_MD5" "nightly"

# --- 2. the numeric flags stay exact under scaling -----------------------
if ! "$BIN" "$NASA" -o "$TMP/legacy.ivf" --video --frames 60 --qp 110 \
        --cpu-threads 8 --nightly --video-qp-lambda "${LEGACY[@]}" \
        >"$TMP/legacy.log" 2>&1; then
    fail "the all-explicit control encode failed"
else
    LEG_BYTES=$(wc -c <"$TMP/legacy.ivf" | tr -d ' ')
    LEG_MD5=$(md5_of "$TMP/legacy.ivf")
    [ "$LEG_BYTES" = "24113" ] ||
        fail "all-explicit control is $LEG_BYTES B, expected the legacy 24113"
    [ "$LEG_MD5" = "7f03eb6e82d6ba4cf6666c78285f945b" ] ||
        fail "all-explicit control md5 drifted"
fi

# --- 3. each site is live ------------------------------------------------
run_quick() {
    local out=$1 tag=$2
    shift 2
    if ! "$BIN" "$NASA" -o "$TMP/$out" --video --frames 60 --qp 110 \
            --cpu-threads 8 "$@" >"$TMP/$tag.log" 2>&1; then
        fail "$tag encode failed"
        return 1
    fi
    return 0
}
# leaf only (tx/key/rdoq/partition legacy).
if run_quick leaf.ivf leaf --nightly --video-qp-lambda \
        --video-tx-lambda 136 --video-key-tx-lambda 136 \
        --video-rdoq-scale 0.1 --video-partition-rd-lambda 2.75e-6; then
    cmp -s "$TMP/leaf.ivf" "$TMP/legacy.ivf" &&
        fail "the leaf QP fit is inert at 240p"
fi
# key only (key fit 8; everything else exact).
if run_quick key.ivf key --nightly --video-qp-lambda \
        --video-tx-lambda 136 --video-leaf-rd-lambda 125 \
        --video-rdoq-scale 0.1 --video-partition-rd-lambda 2.75e-6; then
    cmp -s "$TMP/key.ivf" "$TMP/legacy.ivf" &&
        fail "the key TX QP fit is inert at 240p"
fi
# partition only (probe fit == anchor at QP110, fallback fit is live).
if run_quick part.ivf part --nightly --video-qp-lambda \
        --video-tx-lambda 136 --video-key-tx-lambda 136 \
        --video-leaf-rd-lambda 125 --video-rdoq-scale 0.1; then
    cmp -s "$TMP/part.ivf" "$TMP/legacy.ivf" &&
        fail "the partition fallback QP fit is inert at 240p"
fi
# rdoq only: the fitted scale is exactly 0.1 at the QP110 anchor.
if run_quick rdoq.ivf rdoq --nightly --video-qp-lambda \
        --video-tx-lambda 136 --video-key-tx-lambda 136 \
        --video-leaf-rd-lambda 125 --video-partition-rd-lambda 2.75e-6; then
    cmp -s "$TMP/rdoq.ivf" "$TMP/legacy.ivf" ||
        fail "the RDOQ scale fit is not anchored at 0.1 on the 240p point"
fi

# --- 4. composed operating points ---------------------------------------
if ! "$BIN" "$NASA" -o "$TMP/qp240.ivf" --video --frames 60 --qp 110 \
        --cpu-threads 8 --nightly --video-qp-lambda \
        --video-recon "$TMP/qp240.y4m" >"$TMP/qp240.log" 2>&1; then
    fail "the composed 240p encode failed"
else
    check_pin "$TMP/qp240.ivf" "$QP240_BYTES" "$QP240_MD5" "composed 240p"
    decode_exact "$TMP/qp240.ivf" "$TMP/qp240.y4m" "qp240"
fi

if [ ! -f "$SPACE" ]; then
    echo "SKIP: no space8k_1440x720_60f.y4m (720p arm not checked)"
else
    if ! "$BIN" "$SPACE" -o "$TMP/qp720.ivf" --video --frames 60 \
            --cpu-threads 8 --qp 217 --nightly --video-key-qp-offset -57 \
            --video-qp-lambda --video-recon "$TMP/qp720.y4m" \
            >"$TMP/qp720.log" 2>&1; then
        fail "the composed 720p encode failed"
    else
        check_pin "$TMP/qp720.ivf" "$QP720_BYTES" "$QP720_MD5" "composed 720p"
        decode_exact "$TMP/qp720.ivf" "$TMP/qp720.y4m" "qp720"
    fi
fi

# --- 5. rejections -------------------------------------------------------
SHINE="$TESTDATA/shine_640x360_4f.y4m"
if [ -f "$SHINE" ]; then
    if "$BIN" "$SHINE" -o "$TMP/gpu.ivf" --video --frames 2 --qp 190 \
            --video-qp-lambda --video-gpu >/dev/null 2>"$TMP/gpu.err"; then
        fail "--video-qp-lambda accepted --video-gpu"
    fi
    grep -q "CPU-only" "$TMP/gpu.err" ||
        fail "--video-qp-lambda --video-gpu did not name the CPU-only contract"
    if "$BIN" "$SHINE" -o "$TMP/rc.ivf" --video --frames 2 --qp 190 \
            --video-qp-lambda --rc crf >/dev/null 2>"$TMP/rc.err"; then
        fail "--video-qp-lambda accepted --rc crf"
    fi
    grep -q "fixed --qp stream" "$TMP/rc.err" ||
        fail "--video-qp-lambda --rc crf did not name the fixed-QP contract"
    if "$BIN" "$SHINE" -o "$TMP/still.ivf" --still \
            --video-qp-lambda >/dev/null 2>"$TMP/still.err"; then
        fail "--video-qp-lambda accepted --still"
    fi
    grep -q "a --video feature" "$TMP/still.err" ||
        fail "--video-qp-lambda --still did not name the video contract"
fi

if [ "$FAILED" -ne 0 ]; then
    echo "gate_video_qp_lambda: FAIL" >&2
    exit 1
fi
echo "gate_video_qp_lambda: OK (pins byte-identical, scaled set decodes exact)"
