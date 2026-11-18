#!/usr/bin/env bash
# Gate for the env-gated lpl36 perceptual refresh pipeline
# (AV2ENC_PIPE_LPL_REFRESH=1).
#
# The drift-refresh chain triggers on the primary reference's luma PSNR.  This
# pipeline replaces the trigger and the selection rule with the measured lpl36
# error: pass 1 measures the no-refresh chain, pass 2 probes the per-frame
# refresh cost and lpl36 floor, the planner allocates the existing
# --video-drift-refresh-budget by predicted error reduction per payload byte,
# and the scheduled stream is emitted through the unchanged drift-refresh
# plumbing (frame QP drop + forced inter residual, no new syntax).
#
# Checks:
#   * with the env unset or 0, the pinned nightly stream is byte-identical;
#   * with the env on, the trace reports a plan, the allocator stays inside a
#     one-refresh margin of the payload budget, and the plan refreshes the late
#     detail band (>= frame 44);
#   * the output stream is 60/60 under dav2d --strict 1 and the visible
#     --video-recon payload equals the decoded payload byte for byte;
#   * the lpl36 arm lowers the mean lpl36 error against the no-refresh control
#     at the same QP by >= 3%, and lowers the frames 44-59 mean against the
#     PSNR-trigger control by >= 2% (the metric buys the detail band).
#
# Usage: src/pipeline/video/video_lpl_refresh_gate.sh [--bin build/av2enc]
#                                                     [--dav2d PATH]
#                                                     [--metric-bin PATH]
# Exit: 0 pass, 1 failure, 2 usage, 77 missing fixture/decoder/metric.
set -uo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/../../.." && pwd)
BIN=""
DAV2D="${DAV2D:-}"
METRIC=""
TESTDATA="${AV2ENC_TESTDATA:-$ROOT/testdata}"

while [ $# -gt 0 ]; do
    case "$1" in
        --bin) BIN=${2:-}; shift 2 ;;
        --dav2d) DAV2D=${2:-}; shift 2 ;;
        --metric-bin) METRIC=${2:-}; shift 2 ;;
        -h|--help) sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "video_lpl_refresh_gate: unknown argument $1" >&2; exit 2 ;;
    esac
done

BIN=${BIN:-"$ROOT/build/av2enc"}
[ -x "$BIN" ] || {
    echo "video_lpl_refresh_gate: --bin must point at av2enc" >&2; exit 2; }
if [ -z "$METRIC" ]; then
    METRIC="$(cd -- "$(dirname -- "$BIN")" && pwd)/lpl36_dump"
fi
[ -x "$METRIC" ] || {
    echo "video_lpl_refresh_gate: SKIP (exit 77): no lpl36_dump next to av2enc" >&2
    exit 77; }
if [ -z "$DAV2D" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
    done
fi
[ -n "$DAV2D" ] && [ -x "$DAV2D" ] || {
    echo "video_lpl_refresh_gate: SKIP (exit 77): no dav2d binary" >&2
    exit 77; }

NASA="$TESTDATA/nasa_480x240_30fps_60f.y4m"
[ -f "$NASA" ] || {
    echo "video_lpl_refresh_gate: SKIP (exit 77): no $NASA" >&2
    exit 77; }

TMP=$(mktemp -d "${TMPDIR:-/tmp}/av2enc-lpl-refresh.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
FAILED=0
fail() { echo "FAIL: $*" >&2; FAILED=1; }

QP=111
BUDGET=3250
THREADS=8
COMMON=(--video --frames 60 --qp "$QP" --cpu-threads "$THREADS" --nightly
        --video-drift-refresh-threshold 1.5
        --video-drift-refresh-budget "$BUDGET")

echo "== video lpl36 refresh gate =="

md5_of() { python3 - "$1" <<'PY'
import hashlib, sys
print(hashlib.md5(open(sys.argv[1], 'rb').read()).hexdigest())
PY
}

# --- gate off keeps the pinned nightly stream ---------------------------
if ! "$BIN" "$NASA" -o "$TMP/off.ivf" --video --frames 60 --qp 110 \
        --cpu-threads "$THREADS" --nightly >/dev/null 2>&1; then
    fail "the env-unset nightly control encode failed"
fi
OFF_MD5=$(md5_of "$TMP/off.ivf")
[ "$OFF_MD5" = "1b99127a4cd1d386fa5600fb63c57a83" ] ||
    fail "env-unset nightly md5 is $OFF_MD5, expected 1b99127a..."

if ! AV2ENC_PIPE_LPL_REFRESH=0 "$BIN" "$NASA" -o "$TMP/zero.ivf" \
        --video --frames 60 --qp 110 --cpu-threads "$THREADS" --nightly \
        >/dev/null 2>&1; then
    fail "the AV2ENC_PIPE_LPL_REFRESH=0 control encode failed"
fi
ZERO_MD5=$(md5_of "$TMP/zero.ivf")
[ "$ZERO_MD5" = "$OFF_MD5" ] ||
    fail "AV2ENC_PIPE_LPL_REFRESH=0 changed the nightly stream ($ZERO_MD5)"

# --- gate on: plan, budget, tail allocation, decode, recon --------------
if ! AV2ENC_PIPE_LPL_REFRESH=1 "$BIN" "$NASA" -o "$TMP/on.ivf" "${COMMON[@]}" \
        --video-recon "$TMP/on_recon.y4m" --video-trace \
        >"$TMP/on.log" 2>"$TMP/on.trace"; then
    fail "the lpl36 refresh encode failed"
    cat "$TMP/on.log" >&2
fi
[ -s "$TMP/on.ivf" ] || fail "the lpl36 refresh encode produced no stream"

grep -q "^lpl_refresh_plan frames=" "$TMP/on.trace" ||
    fail "the trace reports no perceptual plan"
grep -q "^lpl_refresh_plan frames=.*," "$TMP/on.trace" ||
    fail "the perceptual plan is empty"

ACTUAL_BYTES=$(grep -o 'actual_bytes=[0-9]*' "$TMP/on.trace" | tail -1 |
               cut -d= -f2)
[ -n "$ACTUAL_BYTES" ] || fail "the trace reports no actual refresh bytes"
[ "${ACTUAL_BYTES:-0}" -gt 0 ] || fail "the lpl36 stream charged no refresh bytes"
# The emit path checks the allowance before a refresh, so one refresh may
# overshoot the remaining budget; bound the total to one 240p refresh's worth.
[ "${ACTUAL_BYTES:-0}" -le $((BUDGET + 500)) ] ||
    fail "the lpl36 stream spent ${ACTUAL_BYTES:-0} B, more than the $BUDGET B budget + 500"

PLAN_FRAMES=$(sed -n 's/^lpl_refresh_plan frames=\([0-9,]*\).*/\1/p' \
              "$TMP/on.trace" | tail -1)
if ! python3 - "$PLAN_FRAMES" <<'PY'
import sys
frames = [int(x) for x in sys.argv[1].split(",") if x]
if not any(f >= 44 for f in frames):
    print("the lpl36 plan never refreshes the late detail band (>= 44)",
          file=sys.stderr)
    sys.exit(1)
PY
then
    fail "the lpl36 plan has no late detail-band refresh"
fi

if ! "$DAV2D" -q --strict 1 -i "$TMP/on.ivf" -o "$TMP/on_dec.y4m" \
        --threads 1 >"$TMP/strict.log" 2>&1; then
    fail "the lpl36 stream failed dav2d --strict 1"
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

# --- the allocation is real lpl36 quality ------------------------------
# The PSNR-trigger control shares the recipe (same QP, threshold, budget) and
# differs only in the refresh rule.  The no-refresh control isolates what the
# allocator buys at all.
if ! "$BIN" "$NASA" -o "$TMP/psnr.ivf" "${COMMON[@]}" \
        >"$TMP/psnr.log" 2>&1; then
    fail "the PSNR-trigger control encode failed"
fi
if ! "$DAV2D" -q --strict 1 -i "$TMP/psnr.ivf" -o "$TMP/psnr_dec.y4m" \
        --threads 1 >"$TMP/psnr_strict.log" 2>&1; then
    fail "the PSNR-trigger control failed dav2d --strict 1"
fi
if ! "$BIN" "$NASA" -o "$TMP/norefresh.ivf" "${COMMON[@]}" \
        --video-drift-refresh-off >"$TMP/norefresh.log" 2>&1; then
    fail "the no-refresh control encode failed"
fi
if ! "$DAV2D" -q --strict 1 -i "$TMP/norefresh.ivf" \
        -o "$TMP/norefresh_dec.y4m" --threads 1 >"$TMP/nr_strict.log" 2>&1; then
    fail "the no-refresh control failed dav2d --strict 1"
fi

if "$METRIC" --ref "$NASA" --test "$TMP/on_dec.y4m" --frames 60 \
        >"$TMP/on.lpl" 2>"$TMP/on.lpl.err" &&
   "$METRIC" --ref "$NASA" --test "$TMP/psnr_dec.y4m" --frames 60 \
        >"$TMP/psnr.lpl" 2>"$TMP/psnr.lpl.err" &&
   "$METRIC" --ref "$NASA" --test "$TMP/norefresh_dec.y4m" --frames 60 \
        >"$TMP/norefresh.lpl" 2>"$TMP/norefresh.lpl.err"; then
    if ! python3 - "$TMP/on.lpl" "$TMP/psnr.lpl" "$TMP/norefresh.lpl" <<'PY'
import re
import sys

def read(path):
    values = []
    for line in open(path):
        m = re.match(r"frame=(\d+) lpl_error=([-0-9.eE+]+)", line)
        if m:
            values.append((int(m.group(1)), float(m.group(2))))
    return [v for _, v in sorted(values)]

on = read(sys.argv[1])
psnr = read(sys.argv[2])
norefresh = read(sys.argv[3])
if not on or len(on) != len(psnr) or len(on) != len(norefresh):
    print("no comparable lpl36 frame sets", file=sys.stderr)
    sys.exit(1)
on_mean = sum(on) / len(on)
nr_mean = sum(norefresh) / len(norefresh)
late_on = sum(on[44:60]) / len(on[44:60])
late_psnr = sum(psnr[44:60]) / len(psnr[44:60])
print(f"mean lpl36 error: on={on_mean:.4f} psnr={sum(psnr)/len(psnr):.4f} "
      f"no_refresh={nr_mean:.4f}")
print(f"frames 44-59: on={late_on:.4f} psnr={late_psnr:.4f}")
if on_mean > nr_mean * 0.97:
    print(f"lpl36 arm improves the no-refresh mean by only "
          f"{100.0 * (nr_mean - on_mean) / nr_mean:.2f}%", file=sys.stderr)
    sys.exit(1)
if late_on > late_psnr * 0.98:
    print(f"lpl36 arm improves the frames 44-59 mean by only "
          f"{100.0 * (late_psnr - late_on) / late_psnr:.2f}%", file=sys.stderr)
    sys.exit(1)
PY
    then
        fail "the lpl36 allocation did not improve the measured lpl36 error"
    fi
else
    fail "lpl36_dump failed on a gate stream"
    cat "$TMP/on.lpl.err" "$TMP/psnr.lpl.err" "$TMP/norefresh.lpl.err" >&2
fi

if [ "$FAILED" -ne 0 ]; then
    echo "video_lpl_refresh_gate: FAILED" >&2
    exit 1
fi
echo "video_lpl_refresh_gate: OK (pin, plan, budget, tail allocation, lpl36 gain, strict decode, recon)"
