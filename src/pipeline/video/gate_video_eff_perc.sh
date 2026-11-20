#!/usr/bin/env bash
# Gate for the extended perceptual RD core (AV2ENC_EFF_PERC).
#
# Builds on the merged pipe-perc path and moves three pieces together:
#   * D is a local error-domain CSF (low/mid-band error above the high-frequency
#     tail), with optional per-pixel multi-scale/edge source weights and a
#     chroma plane weight;
#   * R is the R09 exact inter-syntax probe plus the measured coefficient term
#     (default: the R19 range-coder interval width against the live frame-start
#     CDF image);
#   * one lambda (default scale 0.18) weights the leaf J and the TX candidate J.
#
# Checks:
#   * gate off (unset) keeps the default, quality and nightly pins
#     byte-identical, the 720p pin byte-identical when the fixture exists, and
#     AV2ENC_EFF_PERC=0 is byte-identical to unset;
#   * the merged pipe-perc point still reproduces (24994 B, 650f2e35...);
#   * the tuned point (drift recipe, QP119, defaults) encodes to the pinned
#     byte count/MD5, decodes 60/60 with dav2d --strict 1, and the visible
#     encoder reconstruction equals the decoded payload;
#   * the stderr summary reports the shipped knobs and real decision counts;
#   * the gate is env-only: no CLI flag exists.
#
# Usage: src/pipeline/video/gate_video_eff_perc.sh [--bin build/av2enc]
#                                                  [--dav2d PATH]
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
        -h|--help) sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "gate_video_eff_perc: unknown argument $1" >&2; exit 2 ;;
    esac
done

BIN=${BIN:-"$ROOT/build/av2enc"}
[ -x "$BIN" ] || {
    echo "gate_video_eff_perc: --bin must point at av2enc" >&2; exit 2; }
if [ -z "$DAV2D" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
    done
fi
[ -n "$DAV2D" ] && [ -x "$DAV2D" ] || {
    echo "gate_video_eff_perc: SKIP (exit 77): no dav2d binary" >&2; exit 77; }

NASA="$TESTDATA/nasa_480x240_30fps_60f.y4m"
[ -f "$NASA" ] || {
    echo "gate_video_eff_perc: SKIP (exit 77): no $NASA" >&2; exit 77; }
SPACE8K="$TESTDATA/space8k_1440x720_60f.y4m"

TMP=$(mktemp -d "${TMPDIR:-/tmp}/av2enc-effp.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
FAILED=0
fail() { echo "FAIL: $*" >&2; FAILED=1; }
md5_of() { md5 -q "$1" 2>/dev/null || md5sum "$1" | awk '{print $1}'; }

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

echo "== extended perceptual RD core gate =="

# --- gate off: the three 240p pins ---------------------------------------
"$BIN" "$NASA" -o "$TMP/default.ivf" --video --frames 60 --qp 189 \
    --cpu-threads 8 >"$TMP/default.log" 2>&1 ||
    fail "the default encode failed"
check_pin "$TMP/default.ivf" 25009 a1d0dba1259b20a4b628d4cd1e14c33c \
    "default QP189"

"$BIN" "$NASA" -o "$TMP/quality.ivf" --video --frames 60 --qp 190 \
    --cpu-threads 8 --video-rd-preset quality >"$TMP/quality.log" 2>&1 ||
    fail "the quality encode failed"
check_pin "$TMP/quality.ivf" 25141 3984677b9b665952984b07b7cf75c6ea \
    "quality QP190"

"$BIN" "$NASA" -o "$TMP/nightly.ivf" --video --frames 60 --qp 110 \
    --cpu-threads 8 --fast-decode >"$TMP/nightly.log" 2>&1 ||
    fail "the nightly encode failed"
check_pin "$TMP/nightly.ivf" 23690 1b99127a4cd1d386fa5600fb63c57a83 \
    "nightly QP110 pin"

AV2ENC_EFF_PERC=0 "$BIN" "$NASA" -o "$TMP/off.ivf" --video --frames 60 \
    --qp 110 --cpu-threads 8 --fast-decode >"$TMP/off.log" 2>&1 ||
    fail "the AV2ENC_EFF_PERC=0 encode failed"
cmp -s "$TMP/nightly.ivf" "$TMP/off.ivf" ||
    fail "AV2ENC_EFF_PERC=0 is not byte-identical to unset"

# The merged pipe-perc point must still reproduce under the extended tree.
AV2ENC_PIPE_PERC=1 "$BIN" "$NASA" -o "$TMP/pipe.ivf" --video --frames 60 \
    --qp 119 --cpu-threads 8 --fast-decode \
    --video-drift-refresh-threshold 1.5 \
    --video-drift-refresh-budget 3250 >"$TMP/pipe.log" 2>&1 ||
    fail "the merged pipe-perc point failed"
check_pin "$TMP/pipe.ivf" 24994 650f2e3514978257c78f373f7bf99a50 \
    "pipe-perc drift QP119"

# 720p pin (gate off).  Skip only when the fixture is not staged locally.
if [ -f "$SPACE8K" ]; then
    "$BIN" "$SPACE8K" -o "$TMP/720.ivf" --video --frames 60 \
        --cpu-threads 8 --qp 217 --fast-decode --video-key-qp-offset -57 \
        --video-leaf-rd-lambda 250 --video-ext-partitions \
        >"$TMP/720.log" 2>&1 || fail "the 720p pin encode failed"
    check_pin "$TMP/720.ivf" 36572 c4ba28138d8f6569d5e08721a66d755c \
        "720p QP217 pin"
else
    echo "gate_video_eff_perc: 720p fixture absent, skipped the 720p pin"
fi

# --- gate on: tuned drift point ------------------------------------------
if ! AV2ENC_EFF_PERC=1 "$BIN" "$NASA" -o "$TMP/on.ivf" --video --frames 60 \
        --qp 119 --cpu-threads 8 --fast-decode \
        --video-drift-refresh-threshold 1.5 \
        --video-drift-refresh-budget 3250 --video-recon "$TMP/on.recon.y4m" \
        >"$TMP/on.log" 2>&1; then
    fail "the gate-on tuned encode failed"
fi
check_pin "$TMP/on.ivf" 25230 7e8ab1c19d37a72cfd932e3ec42277a6 \
    "eff-perc drift QP119"

grep -q "eff-perc: gate=1 scale=0.180 shape=0 detail=1.000 chroma=1.000 top=2 rate=1 csf=1.50 csf2=0.00" \
    "$TMP/on.log" ||
    fail "the stderr summary does not report the shipped knobs"
grep -Eq "leaves=[1-9][0-9]*" "$TMP/on.log" ||
    fail "the leaf counter is zero or missing"
grep -Eq "rescored=[1-9][0-9]*" "$TMP/on.log" ||
    fail "the re-scored counter is zero or missing"
grep -Eq "skip_flips=[1-9][0-9]*" "$TMP/on.log" ||
    fail "the skip/residual counter is zero or missing"
grep -q "fallbacks=0" "$TMP/on.log" ||
    fail "the exact-syntax probe fell back on a measured leaf"
grep -Eq "rate_interval=[1-9][0-9]*" "$TMP/on.log" ||
    fail "the interval coefficient term was never used"

"$DAV2D" -q --strict 1 -i "$TMP/on.ivf" -o "$TMP/on.dec.y4m" \
    --threads 1 >"$TMP/on.dec.log" 2>&1 ||
    fail "the gate-on stream failed dav2d --strict 1"
PICS=$(python3 -c 'import sys;print(open(sys.argv[1],"rb").read().count(b"FRAME"))' \
           "$TMP/on.dec.y4m")
[ "$PICS" = "60" ] || fail "gate-on decoded $PICS pictures, expected 60"
if ! python3 - "$TMP/on.recon.y4m" "$TMP/on.dec.y4m" <<'PY'
import sys
def payload(path):
    data = open(path, 'rb').read()
    nl = data.index(b'\n') + 1
    return data[nl:].replace(b'FRAME\n', b'')
a, b = payload(sys.argv[1]), payload(sys.argv[2])
if len(a) != len(b) or a != b:
    print(f"recon {len(a)} B != decoded {len(b)} B or bytes differ",
          file=sys.stderr)
    sys.exit(1)
PY
then
    fail "gate-on visible recon differs from the decoded payload"
fi

# The integration promoted the env gate to a CLI form.  It must be accepted
# and byte-identical to the env form (the env form remains the lane carrier).
if ! "$BIN" "$NASA" -o "$TMP/flag.ivf" --video --frames 1 --qp 119 \
        --video-eff-perc >/dev/null 2>"$TMP/flag.err"; then
    fail "--video-eff-perc CLI flag was rejected"
fi
if ! env AV2ENC_EFF_PERC=1 "$BIN" "$NASA" -o "$TMP/flag_env.ivf" \
        --video --frames 1 --qp 119 >/dev/null 2>&1; then
    fail "AV2ENC_EFF_PERC=1 env form failed"
fi
if ! cmp -s "$TMP/flag.ivf" "$TMP/flag_env.ivf"; then
    fail "--video-eff-perc differs from AV2ENC_EFF_PERC=1"
fi

if [ "$FAILED" = "0" ]; then
    echo "gate_video_eff_perc: OK (pins, pipe-perc, tuned point, 60/60 decode, recon)"
    exit 0
fi
echo "gate_video_eff_perc: FAILED" >&2
exit 1
