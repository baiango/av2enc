#!/usr/bin/env bash
# Gate for --video-drift-refresh (opt-in per-frame drift refresh).
#
# The flat key+P chain codes almost every inter leaf as skip, so a prediction
# error that a frame inherits from its reference is never repaid and the P
# chain's PSNR decays until the content changes.  With the flag on, every
# inter frame measures its primary reference's reconstruction PSNR against the
# source picture it was coded from; when that is more than
# --video-drift-refresh-threshold dB below the key picture's own PSNR, the
# frame is coded as a refresh (optional QP reduction plus residual on every
# inter leaf), charged against one per-sequence byte budget.
#
# Checks:
#   * the named rejections (--video-gpu, --tiles > 1x1, B/reorder/random-access
#     schedules, --rc crf/vbr, --still, 10-bit);
#   * the pinned nightly stream is byte-identical with the flag off;
#   * the flag-on stream decodes under dav2d --strict 1, the visible encoder
#     reconstruction equals the decoded payload, and the trace reports a
#     refresh whose reference was below the allowance;
#   * the refreshed frames 32-40 have a higher mean luma PSNR than the
#     flag-off control at the same QP, in the same number of bytes or more;
#   * --video-drift-refresh-qp 16 (the header-QP path) still decodes
#     recon-exact;
#   * the QP189 and quality/QP190 pins are byte-identical with the flag off.
#
# Usage: src/pipeline/video/video_drift_refresh_gate.sh [--bin build/av2enc]
#                                                    [--dav2d PATH]
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
        -h|--help) sed -n '2,27p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "video_drift_refresh_gate: unknown argument $1" >&2; exit 2 ;;
    esac
done

BIN=${BIN:-"$ROOT/build/av2enc"}
[ -x "$BIN" ] || {
    echo "video_drift_refresh_gate: --bin must point at av2enc" >&2; exit 2; }
if [ -z "$DAV2D" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
    done
fi
[ -n "$DAV2D" ] && [ -x "$DAV2D" ] || {
    echo "video_drift_refresh_gate: SKIP (exit 77): no dav2d binary" >&2
    exit 77; }

NASA="$TESTDATA/nasa_480x240_30fps_60f.y4m"
[ -f "$NASA" ] || {
    echo "video_drift_refresh_gate: SKIP (exit 77): no $NASA" >&2; exit 77; }

TMP=$(mktemp -d "${TMPDIR:-/tmp}/av2enc-drift-refresh.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
FAILED=0
fail() { echo "FAIL: $*" >&2; FAILED=1; }

QP=110
THREADS=8
# The `--nightly` expansion with `--video-drift-refresh` left out, so the gate
# keeps a true flag-off control now that the flag is part of `--nightly`.
COMMON=(--video --frames 60 --qp "$QP" --cpu-threads "$THREADS"
        --video-rd-preset quality
        --video-ccso-rd --video-ccso-bands 6 --video-ccso-lambda 22
        --video-ccso-refit 2 --video-ccso-unit-bits 2
        --video-edge-inter --video-edge-mask
        --video-partition-rd --video-partition-rd-lambda 2.75e-6
        --video-key-qp-offset -27 --video-rdoq --video-cdef
        --video-bawp --video-bawp-carrier --video-leaf-rd
        --video-subpel-eighth --video-tx-lambda 136
        --video-key-intra-txpart
        --avg-cdf --adaptive-cdf --cdf-continuity)

echo "== video drift refresh gate =="

# --- named rejections -----------------------------------------------------
reject() {  # reject <tag> <needle> <args...>
    local tag=$1 needle=$2; shift 2
    if "$BIN" "$NASA" -o "$TMP/$tag.ivf" "$@" >/dev/null 2>"$TMP/$tag.err"; then
        fail "--video-drift-refresh was accepted with $tag"
    fi
    grep -q "$needle" "$TMP/$tag.err" ||
        fail "the $tag rejection did not name the requirement"
}
reject gpu "CPU-only" "${COMMON[@]}" --video-drift-refresh --video-gpu
reject tiles "single tile" "${COMMON[@]}" --video-drift-refresh --tiles 2x1
reject bframes "flat key+P chain" --video --frames 4 --qp "$QP" \
    --b-frames 1 --video-drift-refresh
reject crf "use --rc cqp" --video --frames 4 --qp "$QP" --rc crf --crf 32 \
    --video-drift-refresh
reject still "pass --video" --still --qp "$QP" --video-drift-refresh
reject hbd "8-bit 4:2:0" --video --frames 4 --qp "$QP" --bit-depth 10 \
    --video-drift-refresh
[ -e "$TMP/gpu.ivf" ] && fail "a rejected run created its output file"

# --- flag off keeps the pinned nightly stream ----------------------------
if ! "$BIN" "$NASA" -o "$TMP/off.ivf" "${COMMON[@]}" >"$TMP/off.log" 2>&1; then
    fail "the flag-off nightly control encode failed"
fi
OFF_MD5=$(python3 - "$TMP/off.ivf" <<'PY'
import hashlib, sys
print(hashlib.md5(open(sys.argv[1], 'rb').read()).hexdigest())
PY
)
[ "$OFF_MD5" = "47b85555ce260c5d4db9023548cdc094" ] ||
    fail "flag-off control stream md5 is $OFF_MD5, expected c67818f9..."

# --- flag on: decode, recon, trace evidence, quality ---------------------
if ! "$BIN" "$NASA" -o "$TMP/on.ivf" "${COMMON[@]}" --video-drift-refresh \
        --video-recon "$TMP/on_recon.y4m" --video-trace \
        >"$TMP/on.log" 2>"$TMP/on.trace"; then
    fail "the flag-on encode failed"
    cat "$TMP/on.log" >&2
fi
[ -s "$TMP/on.ivf" ] || fail "the flag-on encode produced no stream"

grep -q "drift refresh frames [1-9]" "$TMP/on.log" ||
    fail "the flag-on stream reports no refresh frame"
grep -q "refresh=1" "$TMP/on.trace" ||
    fail "the trace shows no frame that crossed the drift allowance"

if ! "$DAV2D" -q --strict 1 -i "$TMP/off.ivf" -o "$TMP/off_dec.y4m" \
        --threads 1 >"$TMP/off_strict.log" 2>&1; then
    fail "the flag-off control stream failed dav2d --strict 1"
fi

ON_BYTES=$(wc -c <"$TMP/on.ivf" | tr -d ' ')
OFF_BYTES=$(wc -c <"$TMP/off.ivf" | tr -d ' ')
echo "same-QP bytes: off=$OFF_BYTES on=$ON_BYTES"
[ "$ON_BYTES" -gt "$OFF_BYTES" ] ||
    fail "the refresh stream is not larger than the flag-off control"

if ! "$DAV2D" -q --strict 1 -i "$TMP/on.ivf" -o "$TMP/on_dec.y4m" \
        --threads 1 >"$TMP/strict.log" 2>&1; then
    fail "the flag-on stream failed dav2d --strict 1"
    cat "$TMP/strict.log" >&2
elif ! python3 - "$TMP/on_recon.y4m" "$TMP/on_dec.y4m" <<'PY'
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

# The refreshed region is frames 32-40 (the measured drift trough).  Require a
# real quality gain there, not just a larger file.
if ! python3 - "$NASA" "$TMP/off_dec.y4m" "$TMP/on_dec.y4m" <<'PY'
import math
import sys

LO, HI = 32, 40


def read_luma(path, first, last):
    frames = []
    with open(path, 'rb') as f:
        header = f.readline().split()
        w = int([t for t in header if t.startswith(b'W')][0][1:])
        h = int([t for t in header if t.startswith(b'H')][0][1:])
        idx = 0
        while idx <= last:
            head = f.readline()
            if not head.startswith(b'FRAME'):
                break
            y = f.read(w * h)
            f.read(w * h // 2)
            if idx >= first:
                frames.append(y)
            idx += 1
    return frames


def psnr(a, b):
    sse = 0
    for x, y in zip(a, b):
        d = x - y
        sse += d * d
    if sse == 0:
        return 99.0
    return 10.0 * math.log10(255.0 * 255.0 * len(a) / sse)


ref = read_luma(sys.argv[1], LO, HI)
off = read_luma(sys.argv[2], LO, HI)
on = read_luma(sys.argv[3], LO, HI)
if not ref or len(ref) != len(off) or len(ref) != len(on):
    print("no comparable frames decoded", file=sys.stderr)
    sys.exit(1)
p_off = sum(psnr(r, o) for r, o in zip(ref, off)) / len(ref)
p_on = sum(psnr(r, o) for r, o in zip(ref, on)) / len(on)
print(f"frames {LO}-{HI} mean luma PSNR: off={p_off:.3f} on={p_on:.3f}")
if p_on < p_off + 0.20:
    print(f"refresh gained only {p_on - p_off:.3f} dB", file=sys.stderr)
    sys.exit(1)
PY
then
    fail "the refreshed frames did not gain >= 0.20 dB over the control"
fi

# --- the header-QP path (frame QP - N) stays decoder-exact ---------------
COMMON_NOCDF=()
for a in "${COMMON[@]}"; do
    case "$a" in
        --avg-cdf|--adaptive-cdf|--video-cdf-per-frame-qp|--cdf-continuity) ;;
        *) COMMON_NOCDF+=("$a") ;;
    esac
done
if ! "$BIN" "$NASA" -o "$TMP/drop.ivf" "${COMMON_NOCDF[@]}" \
        --video-drift-refresh --video-drift-refresh-qp 16 --video-recon "$TMP/drop_recon.y4m" \
        >"$TMP/drop.log" 2>&1; then
    fail "the --video-drift-refresh-qp 16 encode failed"
fi
if ! "$DAV2D" -q --strict 1 -i "$TMP/drop.ivf" -o "$TMP/drop_dec.y4m" \
        --threads 1 >"$TMP/drop_strict.log" 2>&1; then
    fail "the --video-drift-refresh-qp 16 stream failed dav2d --strict 1"
    cat "$TMP/drop_strict.log" >&2
elif ! python3 - "$TMP/drop_recon.y4m" "$TMP/drop_dec.y4m" <<'PY'
import sys

def payload(path):
    data = open(path, 'rb').read()
    nl = data.index(b'\n') + 1
    return data[nl:].replace(b'FRAME\n', b'')

a, b = payload(sys.argv[1]), payload(sys.argv[2])
if a != b:
    print("qp-drop recon differs from the decode", file=sys.stderr)
    sys.exit(1)
PY
then
    fail "the --video-drift-refresh-qp 16 recon differs from the decode"
fi

# --- the other pinned streams stay byte-identical with the flag off ------
pin_check() {  # pin_check <tag> <expected md5> <args...>
    local tag=$1 want=$2; shift 2
    if ! "$BIN" "$NASA" -o "$TMP/$tag.ivf" "$@" >/dev/null 2>&1; then
        fail "the $tag pin encode failed"
        return
    fi
    local got
    got=$(python3 - "$TMP/$tag.ivf" <<'PY'
import hashlib, sys
print(hashlib.md5(open(sys.argv[1], 'rb').read()).hexdigest())
PY
)
    [ "$got" = "$want" ] || fail "$tag pin md5 is $got, expected $want"
}
pin_check default a1d0dba1259b20a4b628d4cd1e14c33c \
    --video --frames 60 --qp 189 --cpu-threads "$THREADS"
pin_check quality 3984677b9b665952984b07b7cf75c6ea \
    --video --frames 60 --qp 190 --cpu-threads "$THREADS" \
    --video-rd-preset quality

if [ "$FAILED" -ne 0 ]; then
    echo "video_drift_refresh_gate: FAILED" >&2
    exit 1
fi
echo "video_drift_refresh_gate: OK (rejections, pins, strict decode, recon, refresh quality)"
