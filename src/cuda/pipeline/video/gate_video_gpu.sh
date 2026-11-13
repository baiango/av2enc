#!/usr/bin/env bash
# Product gate for the opt-in CUDA M2 path (--video-gpu).
#
# For every wired mode it encodes the same 360p source with the CPU reference
# (--video) and the CUDA backend (--video-gpu), then requires:
#   * byte-identical temporal units (stream MD5 equal, cmp clean),
#   * byte-identical encoder reconstruction dumps (--video-recon),
#   * identical dav2d --strict 1 decoded output.
# It also asserts that every unported combination is rejected by name with
# usage exit 2, never silently ignored or CPU-fallback encoded.
#
# Exits 77 (CTest skip) when the pinned decoder or a CUDA-enabled binary is
# unavailable.  The script generates its own deterministic source, so it does
# not depend on gitignored testdata.
#
# Usage: gate_video_gpu.sh --bin <av2enc> [--frames N] [--dav2d <path>]
#                         [--tx-only]
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${HERE}/../../../.." && pwd)"
BIN=""
FRAMES=4
DAV2D="${DAV2D:-}"
INTRA_ONLY=0
TX_ONLY=0
BAWP_ONLY=0
ONLY=""
LONG_ONLY=0
CFL_ONLY=0
MHCCP_ONLY=0
PCW_ONLY=0
QM_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bin) BIN="$2"; shift 2 ;;
    --frames) FRAMES="$2"; shift 2 ;;
    --dav2d) DAV2D="$2"; shift 2 ;;
    # Run only the opt-in full-intra case (--video-intra-full).  Used by the
    # dedicated CTest entry so the new case is visible and re-runnable on its
    # own without re-encoding the whole product matrix.
    --intra-only) INTRA_ONLY=1; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    --tx-only) TX_ONLY=1; shift ;;
    # Run only the §7.13.3.25 BAWP cases (both registered frame counts).  Used
    # by the dedicated CTest entry so the new tool is re-runnable on its own.
    --bawp-only) BAWP_ONLY=1; shift ;;
    # Dedicated entries for the codec-tools cases.
    --cfl-only) CFL_ONLY=1; shift ;;
    --mhccp-only) MHCCP_ONLY=1; shift ;;
    --pcw-only) PCW_ONLY=1; shift ;;
    --qm-only) QM_ONLY=1; shift ;;
    # Run only the cases whose label contains this substring.  Keeps a single
    # regression check cheap instead of re-encoding the whole product matrix.
    --only) ONLY="$2"; shift 2 ;;
    # Long-chain subset: the 8-frame regression only needs the cases that
    # exercise the rotating reference-slot pool.  The rest of the matrix is
    # already covered at 4 frames (and the fixed-count 8-frame cases run in
    # every invocation), so repeating ~100 cases here was pure wall time.
    --long-only) LONG_ONLY=1; shift ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$BIN" || ! -x "$BIN" ]]; then
  echo "gate_video_gpu: missing --bin" >&2
  exit 2
fi

if [[ -z "$DAV2D" ]]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [[ -x "$candidate" ]]; then
      DAV2D="$candidate"
      break
    fi
  done
fi
if [[ -z "$DAV2D" || ! -x "$DAV2D" ]]; then
  echo "gate_video_gpu: SKIP (no pinned dav2d; set DAV2D)"
  exit 77
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "gate_video_gpu: SKIP (python3 not available)"
  exit 77
fi

TMP="$(mktemp -d "${TMPDIR:-/tmp}/gate_video_gpu.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
CLIP="$TMP/clip360.y4m"
# Second deterministic clip: a smooth sinusoid panned by (3,1) per frame.  On the
# textured clip above every P leaf selects intra at QP 90, so it exercises the
# headers but not reference-indexed MC/ME; this clip selects inter leaves and is
# used for the multi-reference / motion-compensation cases.
CLIP_INTER="$TMP/clip360_inter.y4m"
# Third clip for compound prediction: key A, P anchor B, and a third picture
# that is exactly (A+B)/2.  A zero-MV CWP_EQUAL average of the two references
# is then the best predictor, so the gated compound subset is actually selected
# (the smooth clip never selects it).
CLIP_AVG="$TMP/clip360_avg.y4m"
# Fourth/fifth clips for masked compound: frame 2 is the §7.13.3.27 wedge blend
# (index 22, sign 0, one mask per 64x64 block) or the §7.13.3.28 difference
# blend of references 0 and 1, so the opt-in masked candidate wins the SSE
# decision on at least one leaf.  The mask math is the spec transcription from
# src/codec/inter/advanced/advanced_common.h.
CLIP_WEDGE="$TMP/clip360_wedge.y4m"
CLIP_DIFF="$TMP/clip360_diff.y4m"
# Generate enough frames for the longest gated sequence (8-frame temporal GOP /
# 7-reference P chain) even when the caller lowers --frames.
GEN_FRAMES=$(( FRAMES > 8 ? FRAMES : 8 ))

# The clips are deterministic.  Cache them by frame count so a targeted or
# repeated run does not spend seconds re-running the pure-Python sample
# generator (the whole matrix only needs one generation per frame count).
CLIP_CACHE="${AV2ENC_GATE_CLIP_CACHE:-${TMPDIR:-/tmp}/av2enc_gate_clips}/f${GEN_FRAMES}"
mkdir -p "$CLIP_CACHE"
CLIP="$CLIP_CACHE/clip360.y4m"
CLIP_INTER="$CLIP_CACHE/clip360_inter.y4m"
CLIP_AVG="$CLIP_CACHE/clip360_avg.y4m"
CLIP_WEDGE="$CLIP_CACHE/clip360_wedge.y4m"
CLIP_DIFF="$CLIP_CACHE/clip360_diff.y4m"

if [[ ! -s "$CLIP" || ! -s "$CLIP_INTER" || ! -s "$CLIP_AVG" ]]; then
python3 - "$CLIP" "$GEN_FRAMES" <<'PY'
import sys

path = sys.argv[1]
frames = int(sys.argv[2])
w, h = 640, 360
cw, ch = w // 2, h // 2

def sample(x, y, f, plane):
    sx = 8 + 6 * f
    sy = 12 + 3 * f
    inside = sx <= x < sx + 96 and sy <= y < sy + 80
    # A moving textured background plus a drifting square.  The per-frame
    # perturbation keeps some inter residuals non-zero so --video-inter-residual
    # exercises the TX/quant/recon chain instead of selecting every block as
    # skip.
    if plane == 0:
        if inside:
            return 32 + ((x * 29 + y * 17 + f * 41) & 0xBF)
        return 88 + (((x + 3 * f) * 5 + (y + 2 * f) * 3 +
                      ((x * y + f * 37) & 15)) & 0x7F)
    if plane == 1:
        return 104 + (32 if inside else 0) + ((x + 2 * f + y) & 15)
    return 120 + (24 if inside else 0) + ((y + 3 * f + x) & 15)

with open(path, "wb") as out:
    out.write(b"YUV4MPEG2 W%d H%d F30:1 Ip A1:1 C420\n" % (w, h))
    for f in range(frames):
        out.write(b"FRAME\n")
        for y in range(h):
            out.write(bytes(sample(x, y, f, 0) for x in range(w)))
        for plane in (1, 2):
            for y in range(ch):
                out.write(bytes(sample(2 * x, 2 * y, f, plane)
                                for x in range(cw)))
PY

python3 - "$CLIP_INTER" "$GEN_FRAMES" <<'PY'
import math
import sys

path = sys.argv[1]
frames = int(sys.argv[2])
w, h = 640, 360
cw, ch = w // 2, h // 2

def sample(x, y, f, plane):
    xx = x + 3 * f
    yy = y + f
    if plane == 0:
        v = 128 + 60 * math.sin(xx * 0.05) + 40 * math.cos(yy * 0.08) + \
            20 * math.sin((xx + yy) * 0.03)
    elif plane == 1:
        v = 128 + 40 * math.sin((x + 2 * f) * 0.04) + \
            20 * math.cos((y + f) * 0.05)
    else:
        v = 128 + 35 * math.cos((x + 2 * f) * 0.06) + \
            25 * math.sin((y + f) * 0.04)
    return max(0, min(255, int(v + 0.5)))

with open(path, "wb") as out:
    out.write(b"YUV4MPEG2 W%d H%d F30:1 Ip A1:1 C420\n" % (w, h))
    for f in range(frames):
        out.write(b"FRAME\n")
        for y in range(h):
            out.write(bytes(sample(x, y, f, 0) for x in range(w)))
        for plane in (1, 2):
            for y in range(ch):
                out.write(bytes(sample(2 * x, 2 * y, f, plane)
                                for x in range(cw)))
PY

python3 - "$CLIP_AVG" <<'PY'
import math
import sys

path = sys.argv[1]
w, h = 640, 360
cw, ch = w // 2, h // 2

def a(x, y, plane):
    if plane == 0:
        return max(0, min(255, int(128 + 70 * math.sin(x * 0.07 + 1) +
                                     45 * math.cos(y * 0.05))))
    if plane == 1:
        return max(0, min(255, int(120 + 40 * math.sin(x * 0.05))))
    return max(0, min(255, int(130 + 35 * math.cos(y * 0.06))))

def b(x, y, plane):
    if plane == 0:
        return max(0, min(255, int(128 + 70 * math.sin(x * 0.07 + 3.2) +
                                     45 * math.cos(y * 0.05 + 1.7))))
    if plane == 1:
        return max(0, min(255, int(120 + 40 * math.sin(x * 0.05 + 2.1))))
    return max(0, min(255, int(130 + 35 * math.cos(y * 0.06 + 2.4))))

def mid(x, y, plane):
    return (a(x, y, plane) + b(x, y, plane)) // 2

frames = [a, b, mid]
with open(path, "wb") as out:
    out.write(b"YUV4MPEG2 W%d H%d F30:1 Ip A1:1 C420\n" % (w, h))
    for fn in frames:
        out.write(b"FRAME\n")
        for y in range(h):
            out.write(bytes(fn(x, y, 0) for x in range(w)))
        for plane in (1, 2):
            for y in range(ch):
                out.write(bytes(fn(2 * x, 2 * y, plane)
                                for x in range(cw)))
PY
fi

fail() {
  echo "gate_video_gpu: FAIL: $*" >&2
  exit 1
}

if [[ ! -s "$CLIP_WEDGE" || ! -s "$CLIP_DIFF" ]]; then
python3 - "$CLIP_WEDGE" "$CLIP_DIFF" "$GEN_FRAMES" <<'PY'
import math
import sys

wedge_path, diff_path, frames = sys.argv[1], sys.argv[2], int(sys.argv[3])
w, h = 640, 360
cw, ch = w // 2, h // 2

# §7.13.3.27 tables (src/codec/inter/advanced/tables/wedge_tables.h).
COS = [4, 4, 4, 2, 2, 0, -2, -2, -4, -4, -4, -4, -4, -2, -2, 0, 2, 2, 4, 4]
SIN = [0, -1, -2, -2, -4, -4, -4, -2, -2, -1, 0, 1, 2, 2, 4, 4, 4, 2, 2, 1]
POS = [8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14,
       14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16]
NEG = [8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 4, 3, 3, 3, 2, 2,
       2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
CODE = {
    22: (6, 4, 4),
}
WEDGE_INDEX = 22
WEDGE_SIGN = 0

def wedge_mask(i, j, bw=64, bh=64):
    direction, cx, cy = CODE[WEDGE_INDEX]
    xoff = 64 - ((cx * bw) >> 3)
    yoff = 64 - ((cy * bh) >> 3)
    n, m = yoff + i, xoff + j
    d = ((m << 1) - 127) * COS[direction] + ((n << 1) - 127) * SIN[direction]
    dc = max(-31, min(31, d))
    master = (POS[dc] if dc >= 0 else NEG[-dc]) << 2
    return (64 - master) if WEDGE_SIGN else master

def a(x, y, plane):
    if plane == 0:
        return max(0, min(255, int(60 + 30 * math.sin(x * 0.03) +
                                     20 * math.cos(y * 0.04))))
    if plane == 1:
        return max(0, min(255, int(90 + 20 * math.sin(x * 0.04))))
    return max(0, min(255, int(100 + 20 * math.cos(y * 0.05))))

def b(x, y, plane):
    if plane == 0:
        return max(0, min(255, int(190 + 30 * math.sin(x * 0.03 + 1.5) +
                                     20 * math.cos(y * 0.04 + 2))))
    if plane == 1:
        return max(0, min(255, int(180 + 20 * math.sin(x * 0.04 + 1.5))))
    return max(0, min(255, int(170 + 20 * math.cos(y * 0.05 + 2))))

def wedge(x, y, plane):
    m = wedge_mask(y % 64, x % 64)
    av, bv = a(x, y, plane), b(x, y, plane)
    return (m * av + (64 - m) * bv + 32) >> 6

def diffwtd(x, y, plane):
    av, bv = a(x, y, plane), b(x, y, plane)
    m = max(0, min(64, 38 + abs(av - bv) // 16))
    m = 64 - m  # mask_type = 1
    return (m * av + (64 - m) * bv + 32) >> 6

def write_clip(path, pattern):
    with open(path, "wb") as out:
        out.write(b"YUV4MPEG2 W%d H%d F30:1 Ip A1:1 C420\n" % (w, h))
        for f in range(frames):
            fn = pattern[f % len(pattern)]
            out.write(b"FRAME\n")
            for y in range(h):
                out.write(bytes(fn(x, y, 0) for x in range(w)))
            for plane in (1, 2):
                for y in range(ch):
                    out.write(bytes(fn(2 * x, 2 * y, plane) for x in range(cw)))

write_clip(wedge_path, [a, b, wedge, b, a, b, wedge, b])
write_clip(diff_path, [a, b, diffwtd, b, a, b, diffwtd, b])
PY
fi

# Probe: a CPU-only build (or a CUDA build without the M2 target) must reject
# the flag with a clear reason.  That is a skip for this gate, not a failure.
set +e
"$BIN" "$CLIP" -o "$TMP/probe.ivf" --video-gpu --frames 2 \
  >"$TMP/probe.out" 2>"$TMP/probe.err"
probe_rc=$?
set -e
if [[ $probe_rc -ne 0 ]]; then
  if grep -Eq -- "requires a CUDA build|is not available in this build|is not linked" \
      "$TMP/probe.err"; then
    echo "gate_video_gpu: SKIP (binary has no CUDA M2 target)"
    exit 77
  fi
  cat "$TMP/probe.err" >&2
  fail "probe encode failed (exit $probe_rc)"
fi
grep -q -- "backend=cuda" "$TMP/probe.err" ||
  fail "GPU path did not report backend=cuda"
grep -q -- "frame_bytes_from_cpu=0" "$TMP/probe.err" ||
  fail "GPU path did not report frame_bytes_from_cpu=0"

run_case_on() {
  local clip="$1"
  local label="$2"
  shift 2
  if [[ -n "$ONLY" && "$label" != *"$ONLY"* ]]; then return 0; fi
  if [[ "$LONG_ONLY" -eq 1 ]]; then
    case "$label" in
      inter-baseline|refs2|tiles2x2) ;;
      *) return 0 ;;
    esac
  fi
  local cpu="$TMP/${label}.cpu.ivf"
  local gpu="$TMP/${label}.gpu.ivf"
  local cpu_recon="$TMP/${label}.cpu.recon.y4m"
  local gpu_recon="$TMP/${label}.gpu.recon.y4m"
  local gpu_dec="$TMP/${label}.gpu.dec.y4m"

  "$BIN" "$clip" -o "$cpu" --video --frames "$FRAMES" \
    --video-recon "$cpu_recon" "$@" >/dev/null 2>"$TMP/${label}.cpu.log" ||
    fail "$label: CPU reference encode failed"
  "$BIN" "$clip" -o "$gpu" --video --video-gpu --frames "$FRAMES" \
    --video-recon "$gpu_recon" "$@" >/dev/null 2>"$TMP/${label}.gpu.log" ||
    fail "$label: GPU encode failed"
  grep -q -- "frame_bytes_from_cpu=0" "$TMP/${label}.gpu.log" ||
    fail "$label: GPU path reported CPU-produced bytes"

  if [[ -n "${EXPECT_TRACE:-}" ]]; then
    grep -q -- "$EXPECT_TRACE" "$TMP/${label}.cpu.log" ||
      fail "$label: expected trace \"$EXPECT_TRACE\" not found (mode not selected)"
  fi

  cmp -s "$cpu" "$gpu" ||
    fail "$label: temporal units differ (CPU $(md5sum "$cpu" | cut -d' ' -f1), GPU $(md5sum "$gpu" | cut -d' ' -f1))"
  cmp -s "$cpu_recon" "$gpu_recon" ||
    fail "$label: encoder reconstruction differs"

  # The CPU and GPU streams are byte-identical (checked above), so one decode
  # is sufficient.  Require it to reproduce the encoder reconstruction
  # sample-exactly; that is stronger than comparing two decodes of one stream.
  "$DAV2D" --strict 1 -q -i "$gpu" -o "$gpu_dec" >/dev/null 2>&1 ||
    fail "$label: dav2d --strict 1 rejected the stream"
  cmp -s "$gpu_dec" "$gpu_recon" ||
    fail "$label: decoded output differs from the encoder reconstruction"

  echo "PASS video-gpu $label ($(md5sum "$gpu" | cut -d' ' -f1))"
}

run_case() {
  run_case_on "$CLIP" "$@"
}

# CPU-reference-only regression.  The CPU encoder must survive bottom/right-edge
# coding blocks that straddle the MI-aligned reconstruction plane: a 64x64
# PARTITION_NONE leaf is legal while has_rows is true, but its chroma block can
# extend past the coded plane.  --video-force-single forces exactly that leaf
# per superblock, and the bottom superblock has no inter candidate (it does not
# fit the reference extent), so it is coded intra.  Before the fix the CfL /
# MHCCP scratch write ran off the chroma plane and corrupted the heap
# (glibc "double free or corruption", exit 134).
run_case_cpu_on() {
  local clip="$1"
  local label="$2"
  shift 2
  local cpu="$TMP/${label}.cpu.ivf"
  local cpu_recon="$TMP/${label}.cpu.recon.y4m"
  local cpu_dec="$TMP/${label}.cpu.dec.y4m"
  "$BIN" "$clip" -o "$cpu" --video --frames "$FRAMES" \
    --video-recon "$cpu_recon" "$@" >/dev/null 2>"$TMP/${label}.cpu.log" ||
    fail "$label: CPU reference encode failed"
  "$DAV2D" --strict 1 -q -i "$cpu" -o "$cpu_dec" >/dev/null 2>&1 ||
    fail "$label: dav2d --strict 1 rejected the CPU stream"
  cmp -s "$cpu_recon" "$cpu_dec" ||
    fail "$label: CPU recon differs from the decoded output"
  echo "PASS video-cpu $label ($(md5sum "$cpu" | cut -d' ' -f1))"
}

run_case_cpu() {
  run_case_cpu_on "$CLIP" "$@"
}

# Opt-in full luma intra search (--video-intra-full): the still product's
# 61-candidate search must produce the same stream/recon as the CPU reference
# and decode exactly.  The case is exercised on both clips: the textured clip
# makes every leaf intra, the smooth pan keeps inter leaves with intra
# candidates around them.
run_case_intra_full() {
  run_case intra-full --video-intra-full
  run_case_on "$CLIP_INTER" intra-full-inter --video-intra-full
}

if [[ "$INTRA_ONLY" -eq 1 ]]; then
  run_case_intra_full
  echo "gate_video_gpu: PASS"
  exit 0
fi

# Same comparison with an explicit frame count (the wrapper modes require exact
# sequence lengths).
run_case_frames() {
  local count="$1"
  shift
  local saved="$FRAMES"
  FRAMES="$count"
  run_case_on "$CLIP_INTER" "$@"
  FRAMES="$saved"
}

# Opt-in §5.20.7.6 / §7.13.3.25 block adaptive weighted prediction
# (--video-bawp).  The smooth panned clip keeps inter leaves, so the fitted
# predictor is actually selected; both registered frame counts (4 and 8) must
# be byte-identical CPU vs GPU and decode with dav2d --strict 1.
if [[ "$BAWP_ONLY" -eq 1 ]]; then
  run_case_frames 4 bawp-4 --video-bawp
  run_case_frames 8 bawp-8 --video-bawp
  run_case_frames 8 bawp-8-refs2 --video-bawp --video-refs 2
  # The decoder must reproduce the encoder reconstruction sample-exactly (the
  # stream count is checked by dav2d's exit code plus the equal file length).
  for label in bawp-4 bawp-8 bawp-8-refs2; do
    cmp -s "$TMP/${label}.cpu.dec.y4m" "$TMP/${label}.cpu.recon.y4m" ||
      fail "$label: decoded output differs from the encoder reconstruction"
    cmp -s "$TMP/${label}.gpu.dec.y4m" "$TMP/${label}.gpu.recon.y4m" ||
      fail "$label: GPU decoded output differs from the encoder reconstruction"
    echo "PASS video-gpu $label recon-exact"
  done
  # BAWP needs the skip-only single-prediction path (the CUDA residual backend
  # re-runs MC into its own scratch predictor) and whole-frame coordinates.
  for combo in "--video-inter-residual" "--tiles 2x1"; do
    # shellcheck disable=SC2086
    set +e
    "$BIN" "$CLIP_INTER" -o "$TMP/bawp_reject.ivf" --video --frames 4 \
      --video-bawp $combo >/dev/null 2>"$TMP/bawp_reject.err"
    rc=$?
    set -e
    [[ $rc -eq 2 ]] || fail "bawp $combo: expected usage exit 2, got $rc"
    grep -q -- "--video-bawp" "$TMP/bawp_reject.err" ||
      fail "bawp $combo: rejection did not name --video-bawp"
    echo "PASS video-gpu bawp reject $combo"
  done
  echo "gate_video_gpu: PASS"
  exit 0
fi

# Opt-in §7.13.5 CfL chroma decision (--video-cfl): the CPU {DC, CfL} policy
# and the device `still::chroma_decide()` module must pick the same blocks and
# emit identical bytes/recon at 4 and 8 frames.
run_case_cfl() {
  run_case cfl --video-cfl
  run_case_on "$CLIP_INTER" cfl-inter --video-cfl
  run_case_frames 8 cfl8 --video-cfl
  run_case cfl-intra-full --video-cfl --video-intra-full
  # CPU-reference heap-safety regression: straddling bottom-edge leaf.
  run_case_cpu cpu-cfl-straddle --video-force-single --video-cfl
  run_case_cpu_on "$CLIP_INTER" cpu-cfl-straddle-inter \
    --video-force-single --video-cfl
}

if [[ "$CFL_ONLY" -eq 1 ]]; then
  run_case_cfl
  echo "gate_video_gpu: PASS"
  exit 0
fi

# Opt-in §7.13.6-.8 MHCCP (--video-mhccp): enable_mhccp in the sequence header
# plus the three CFL_MULTI directions.  The CPU search and the device
# `still::chroma_decide()` module must choose the same hypothesis and direction.
run_case_mhccp() {
  run_case mhccp --video-mhccp
  run_case_on "$CLIP_INTER" mhccp-inter --video-mhccp
  run_case_frames 8 mhccp8 --video-mhccp
  run_case mhccp-intra-full --video-mhccp --video-intra-full
  run_case_on "$CLIP_INTER" mhccp-tiles2x1 --video-mhccp --tiles 2x1
  # CPU-reference heap-safety regression: straddling bottom-edge leaf.
  run_case_cpu cpu-mhccp-straddle --video-force-single --video-mhccp
}

if [[ "$MHCCP_ONLY" -eq 1 ]]; then
  run_case_mhccp
  echo "gate_video_gpu: PASS"
  exit 0
fi

# Opt-in §7.20.4 luma PC-Wiener (--video-pc-wiener).  The CPU decision/filter
# and the device pc_wiener module must produce identical bytes/recon at 4 and 8
# frames, including the chroma-LR composition and the full §7.18 -> §7.20.5
# chain.
run_case_pcw() {
  run_case pc-wiener --video-pc-wiener
  run_case_cpu cpu-pc-wiener --video-pc-wiener
  run_case_on "$CLIP_INTER" pc-wiener-inter --video-pc-wiener
  run_case_frames 8 pc-wiener8 --video-pc-wiener
  run_case_on "$CLIP_INTER" pc-wiener-chroma-lr --video-pc-wiener \
    --video-chroma-lr
  run_case_on "$CLIP_INTER" pc-wiener-chain --video-cdef --video-ccso \
    --video-pc-wiener --video-gdf
}

if [[ "$PCW_ONLY" -eq 1 ]]; then
  run_case_pcw
  echo "gate_video_gpu: PASS"
  exit 0
fi

# Opt-in §5.18.6.2 default-matrix QM (--video-qm).  using_qmatrix = 1 with the
# §9.4 default matrices; the CPU and device transform chains must dequantize
# with the same per-coefficient q2.
run_case_qm() {
  run_case qm --video-qm
  run_case_cpu cpu-qm --video-qm
  run_case_on "$CLIP_INTER" qm-inter --video-qm
  run_case_frames 8 qm8 --video-qm
  run_case qm-cfl --video-qm --video-cfl
}

if [[ "$QM_ONLY" -eq 1 ]]; then
  run_case_qm  echo "gate_video_gpu: PASS"
  exit 0
fi

if [[ "$TX_ONLY" -eq 0 ]]; then
run_case baseline
run_case quarter --video-mv-precision quarter
run_case half --video-mv-precision half
run_case subpel-eighth --video-subpel-eighth
run_case force-single --video-force-single
run_case depth2 --video-max-depth 2
run_case min-block8 --video-min-block 8
run_case me-range8 --video-me-range 8
# Inter-residual may legitimately pick no residual on this source; the CPU and
# GPU paths must still agree, and the GPU selftest exercises the TX chain.
run_case inter-residual --video-inter-residual
# §7.17 deblocking: the reconstruction changes, so the CPU and GPU must run the
# same filter over the same metadata before the frame is emitted/referenced.
run_case deblock --video-deblock
run_case deblock-inter-residual --video-deblock --video-inter-residual
run_case deblock-min-block8 --video-deblock --video-min-block 8
run_case_on "$CLIP_INTER" deblock-inter --video-deblock
run_case_on "$CLIP_INTER" deblock-refs2 --video-deblock --video-refs 2
# Opt-in 61-candidate luma intra search (both clips, both registered frame
# counts: 4 via gate_video_gpu and 8 via gate_video_gpu_long).
run_case_intra_full
# Opt-in §7.13.5 CfL chroma decision (4 and 8 frames).
run_case_cfl
# Opt-in §7.13.6-.8 MHCCP (4 and 8 frames).
run_case_mhccp
# Opt-in §7.20.4 luma PC-Wiener (4 and 8 frames).
run_case_pcw
# Opt-in §5.18.6.2 default-matrix QM (4 and 8 frames).
run_case_qm
# Reference-indexed MC/ME: run on the smooth panned clip, where P leaves select
# inter.  The textured clip above is intra-only at QP 90 and cannot exercise the
# per-reference path.
run_case_on "$CLIP_INTER" inter-baseline
run_case_on "$CLIP_INTER" refs2 --video-refs 2
run_case_on "$CLIP_INTER" refs3 --video-refs 3
run_case_on "$CLIP_INTER" refs2-quarter --video-refs 2 --video-mv-precision quarter
run_case_on "$CLIP_INTER" refs2-half --video-refs 2 --video-mv-precision half
# §5.18.2 NumTotalRefs 4..7 over the default 8-slot pool, and §5.4.6
# NumRefFrames 4..8 with a full reference list.  Eight frames are needed so the
# P chain actually reaches seven valid recent slots (a 4-frame sequence only
# ever has three).  The N = 4..7 cases pin the f(N) refresh mask, the
# f(CeilLog2(N)) ref_frame_idx width and the rotating write slot per pool size.
run_case_frames 8 refs4 --video-refs 4
run_case_frames 8 refs7 --video-refs 7
run_case_frames 8 nrf4-refs4 --video-refs 4 --video-num-ref-frames 4
run_case_frames 8 nrf5-refs5 --video-refs 5 --video-num-ref-frames 5
run_case_frames 8 nrf6-refs6 --video-refs 6 --video-num-ref-frames 6
run_case_frames 8 nrf7-refs7 --video-refs 7 --video-num-ref-frames 7
run_case_frames 8 nrf8-refs1 --video-refs 1 --video-num-ref-frames 8
# §5.7 multi-frame header on the non-single-picture path.
run_case_on "$CLIP_INTER" mfh --video-mfh
# §5.4.12/.13 decoder-compatible timing + §5.15 content interpretation OBU.
run_case_on "$CLIP_INTER" timing --video-timing
# §8.2.4 average-CDF / §8.2.6 adaptation.
run_case_on "$CLIP_INTER" avg-cdf --avg-cdf
run_case_on "$CLIP_INTER" adaptive-cdf --avg-cdf --adaptive-cdf
# Frame-type wrappers: the tile payloads are the plain GPU sub-encodes; only
# header/OBU assembly changes.
run_case_frames 4 long-term --video-long-term
run_case_frames 4 olk --video-olk
run_case_frames 4 leading --video-leading 1
run_case_frames 4 sef --video-sef
run_case_frames 2 switch --video-switch
run_case_frames 3 ras --video-ras
# B frames and hierarchical-P temporal layers (plan-owned coding order).
run_case_frames 4 bframes2 --b-frames 2
run_case_frames 5 bframes3 --b-frames 3
run_case_frames 4 bframes2-deblock --b-frames 2 --video-deblock
run_case_frames 4 tlayers4 --video-tlayers 4
run_case_frames 8 tlayers8 --video-tlayers 8
run_case_frames 8 tlayers8-drop1 --video-tlayers 8 --video-tlayer-drop 1
run_case_frames 4 tlayers4-deblock --video-tlayers 4 --video-deblock
# §7.18-§7.20 loop filters on the plan-owned coding orders.  Each frame carries
# its own filter plan in its header now that the B/temporal loops use
# apply_video_loop_filter_frame_header().
run_case_frames 4 bframes2-cdef --b-frames 2 --video-cdef
run_case_frames 4 bframes2-cdef-ccso --b-frames 2 --video-cdef --video-ccso
run_case_frames 4 tlayers4-cdef --video-tlayers 4 --video-cdef
run_case_frames 8 tlayers8-cdef-ccso-gdf --video-tlayers 8 --video-cdef \
  --video-ccso --video-gdf
# §5.4.12 CRF/VBR per-frame rate control.  The planner runs on the host for
# both backends; the device quantizer follows the per-frame qlookup step.
run_case_on "$CLIP_INTER" rc-crf --rc crf --crf 32
run_case_on "$CLIP" rc-crf-textured --rc crf --crf 45
run_case_on "$CLIP_INTER" rc-vbr --rc vbr --bitrate 500
# §5.19 multi-tile: tile-local coded geometry/recon, per-tile MV grids and the
# CPU post tile-group writer.
run_case_on "$CLIP_INTER" tiles2x1 --tiles 2x1
run_case_on "$CLIP_INTER" tiles2x2 --tiles 2x2
run_case_on "$CLIP_INTER" tiles2x2-avg --tiles 2x2 --avg-cdf --adaptive-cdf
run_case_on "$CLIP_INTER" tiles2x2-crf --tiles 2x2 --rc crf --crf 32
run_case_frames 8 tiles2x2-8 --tiles 2x2
# §7.18-§7.20 filters with §5.19 tiles: one frame-level filter pass over the
# assembled reconstruction.  CDEF/CCSO/GDF compose; loop restoration keeps its
# named rejection (see docs/negative-results.md).
run_case_on "$CLIP_INTER" tiles2x1-cdef --tiles 2x1 --video-cdef
run_case_on "$CLIP_INTER" tiles2x2-cdef --tiles 2x2 --video-cdef
run_case_on "$CLIP_INTER" tiles2x2-ccso --tiles 2x2 --video-ccso
run_case_on "$CLIP_INTER" tiles2x2-gdf --tiles 2x2 --video-gdf
run_case_frames 4 tiles2x2-cdef-ccso-gdf --tiles 2x2 --video-cdef \
  --video-ccso --video-gdf
run_case_frames 8 tiles2x2-cdef-ccso-gdf-8 --tiles 2x2 --video-cdef \
  --video-ccso --video-gdf
# Compound/CWP AVERAGE is selected on the A/B/mid clip.
_saved_frames="$FRAMES"
FRAMES=3
run_case_on "$CLIP_AVG" compound --video-refs 2 --video-compound
FRAMES="$_saved_frames"
# §7.13.3.27/.28 masked compound: the clip's frame 2 is the wedge/diffwtd blend
# of the two references, so the opt-in masked candidate is selected (asserted
# from the CPU trace) and must be byte-identical on both backends.  These run
# at the caller's frame count (the 4- and 8-frame parity gates).
EXPECT_TRACE="comp=1 .*ct=0" run_case_on "$CLIP_WEDGE" compound-wedge \
  --video-refs 2 --video-compound --video-compound-wedge --video-force-single \
  --video-trace
EXPECT_TRACE="comp=1 .*ct=1" run_case_on "$CLIP_DIFF" compound-diffwtd \
  --video-refs 2 --video-compound --video-compound-diffwtd --video-force-single \
  --video-trace
# §5.18.9 global motion + §7.13.3.19/.20 GLOBALMV warp.  The panned clip gives
# the estimator a coherent model, so GLOBALMV exercises the warped predictor.
run_case_on "$CLIP_INTER" gme --video-gme
run_case_on "$CLIP_INTER" gme-warp --video-gme --video-warp
run_case_on "$CLIP_INTER" gme-depth2 --video-gme --video-max-depth 2
# Multi-reference GME: one §5.18.9 model per explicit reference, projected per
# slot.  The 8-frame case is required: the 4-frame chain only reaches three
# slots, and the tie-break between a later reference's warp candidate and an
# earlier reference's NEWMV was only exercised once the slot pool rotated.
run_case_frames 4 gme-refs2 --video-gme --video-refs 2
run_case_frames 8 gme-refs2-8 --video-gme --video-refs 2
run_case_frames 8 gme-refs3-8 --video-gme --video-refs 3
# Opt-in M2 loop filters (§7.18 CDEF).  Both the textured clip (mostly intra
# P leaves) and the smooth panned clip (inter leaves, some skipped: exercises
# the §7.18.1 Skips[] grid) are gated.  The recon dump must equal the dav2d
# output, which also proves the per-64x64 cdef_idx symbols are in the stream.
run_case_on "$CLIP_INTER" cdef --video-cdef
run_case_on "$CLIP_INTER" cdef-index4 --video-cdef --video-cdef-strength 4
run_case_on "$CLIP_INTER" cdef-index4-damping4 --video-cdef-strength 4 \
  --video-cdef-damping 4
run_case cdef-intra --video-cdef
# §7.19 CCSO (band-offset only) and the normative CDEF -> CCSO order.
run_case_on "$CLIP_INTER" ccso --video-ccso
run_case_on "$CLIP_INTER" ccso-bands6 --video-ccso --video-ccso-bands 6
run_case_on "$CLIP_INTER" cdef-ccso --video-cdef --video-ccso
run_case_on "$CLIP_INTER" cdef-index4-ccso2 --video-cdef-strength 4 \
  --video-ccso --video-ccso-bands 2
# §7.20 loop restoration: luma Wiener NS, chroma LR and the full chain.
run_case_on "$CLIP_INTER" wiener --video-wiener
run_case_on "$CLIP_INTER" chroma-lr --video-chroma-lr
run_case_on "$CLIP_INTER" wiener-chroma-lr --video-wiener --video-chroma-lr
run_case_on "$CLIP_INTER" lr-chain --video-cdef --video-ccso --video-wiener \
  --video-chroma-lr
# §7.20.5 GDF and the complete §7.18 -> §7.20.5 chain.
run_case_on "$CLIP_INTER" gdf --video-gdf
run_case_on "$CLIP_INTER" gdf-lr --video-wiener --video-chroma-lr --video-gdf
run_case_on "$CLIP_INTER" full-chain --video-cdef --video-ccso --video-wiener \
  --video-chroma-lr --video-gdf
fi

# Opt-in §5.20.6/§5.20.8 luma TX size/type/partition search (--video-tx-search).
# The CUDA backend runs the device module under src/cuda/pipeline/video/tx/;
# the CPU reference runs the shared CPU mirror, so both must pick the identical
# plan and emit identical bytes.
run_case tx-search --video-tx-search
run_case tx-search-inter-residual --video-tx-search --video-inter-residual
run_case tx-search-force-single --video-tx-search --video-inter-residual --video-force-single
run_case tx-search-depth2 --video-tx-search --video-inter-residual --video-max-depth 2
run_case_on "$CLIP_INTER" tx-search-inter --video-tx-search --video-inter-residual
run_case_on "$CLIP_INTER" tx-search-refs2 --video-tx-search --video-inter-residual --video-refs 2
run_case_on "$CLIP_INTER" tx-search-quarter --video-tx-search --video-inter-residual --video-mv-precision quarter
run_case_frames 4 tx-search-bframes2 --b-frames 2 --video-tx-search --video-inter-residual
run_case_frames 4 tx-search-tlayers4 --video-tlayers 4 --video-tx-search --video-inter-residual
# §5.20.6.3 TX parts in the §7.17 deblock grid: the decoder filters interior
# transform-part edges, so the encoder grid must carry each part's origin.
run_case tx-search-deblock --video-tx-search --video-inter-residual --video-deblock
run_case_on "$CLIP_INTER" tx-search-deblock-inter --video-tx-search \
  --video-inter-residual --video-deblock
run_case_frames 8 tx-search-deblock-8 --video-tx-search \
  --video-inter-residual --video-deblock
# §8.2.4 bank now carries the §5.20.6.3 TX-partition CDFs.
run_case tx-search-avg-cdf --video-tx-search --video-inter-residual --avg-cdf
run_case tx-search-adaptive-cdf --video-tx-search --video-inter-residual \
  --avg-cdf --adaptive-cdf
# §5.19 tiles + TX_MODE_SELECT frame bit and the per-tile TX-partition CDFs.
run_case tx-search-tiles --video-tx-search --video-inter-residual --tiles 2x1
run_case_on "$CLIP_INTER" tx-search-tiles2x2 --video-tx-search \
  --video-inter-residual --tiles 2x2
run_case_frames 8 tx-search-tiles2x2-8 --video-tx-search \
  --video-inter-residual --tiles 2x2
run_case_on "$CLIP_INTER" tx-search-tiles2x2-avg --video-tx-search \
  --video-inter-residual --tiles 2x2 --avg-cdf --adaptive-cdf
run_case_frames 8 tx-search-tiles2x2-avg-8 --video-tx-search \
  --video-inter-residual --tiles 2x2 --avg-cdf --adaptive-cdf
expect_reject() {
  local label="$1"
  # A targeted --only run must not pay for the whole rejection matrix.
  if [[ -n "$ONLY" && "$label" != *"$ONLY"* ]]; then return 0; fi
  shift
  set +e
  "$BIN" "$CLIP" -o "$TMP/reject.ivf" --video --video-gpu "$@" \
    >"$TMP/reject.out" 2>"$TMP/reject.err"
  local rc=$?
  set -e
  [[ $rc -eq 2 ]] || fail "$label: expected usage exit 2, got $rc"
  grep -q -- "--video-gpu" "$TMP/reject.err" ||
    fail "$label: rejection did not name --video-gpu"
  echo "PASS video-gpu reject $label"
}

expect_reject cdf-continuity --avg-cdf --adaptive-cdf --cdf-continuity
expect_reject svc --svc-layers 0=640x360
# §7.8 past/future/current reference lists need an implicit reference map.
# The M2 writer emits explicit maps on every path, so the carrier is rejected
# by name on both the CPU reference and the GPU mirror.
expect_reject ref-lists --video-implicit-ref-map

# Reference-model bounds are usage errors, not silently clamped values.
set +e
"$BIN" "$CLIP" -o "$TMP/bounds.ivf" --video --video-gpu --frames 4 \
  --video-refs 8 >/dev/null 2>"$TMP/bounds.err"
rc=$?
set -e
[[ $rc -eq 2 ]] || fail "refs 8: expected usage exit 2, got $rc"
grep -q -- "--video-refs" "$TMP/bounds.err" ||
  fail "refs 8: rejection did not name --video-refs"
set +e
"$BIN" "$CLIP" -o "$TMP/bounds.ivf" --video --video-gpu --frames 4 \
  --video-refs 5 --video-num-ref-frames 4 >/dev/null 2>"$TMP/bounds.err"
rc=$?
set -e
[[ $rc -eq 2 ]] || fail "refs > NumRefFrames: expected usage exit 2, got $rc"
grep -q -- "--video-num-ref-frames" "$TMP/bounds.err" ||
  fail "refs > NumRefFrames: rejection did not name --video-num-ref-frames"
echo "PASS video-gpu reject refs-bounds"

# Deblock-specific combinations that cannot be supported: multi-tile filtering
# disagrees with the pinned decoder across interior tile edges, and the
# frame-type wrapper headers do not signal deblocking_filter_params().  Both are
# named usage rejections (exit 2) on the CPU path as well; no silent fallback.
expect_reject_cpu() {
  local label="$1"
  # A targeted --only run must not pay for the whole rejection matrix.
  if [[ -n "$ONLY" && "$label" != *"$ONLY"* ]]; then return 0; fi
  local needle="$2"
  shift 2
  set +e
  "$BIN" "$CLIP" -o "$TMP/reject_cpu.ivf" --video "$@" \
    >"$TMP/reject_cpu.out" 2>"$TMP/reject_cpu.err"
  local rc=$?
  set -e
  [[ $rc -eq 2 ]] || fail "$label: expected usage exit 2, got $rc"
  grep -q -- "$needle" "$TMP/reject_cpu.err" ||
    fail "$label: rejection did not name $needle"
  echo "PASS video reject $label"
}

expect_reject_cpu deblock-tiles --video-deblock --video-deblock --tiles 2x1
expect_reject_cpu deblock-long-term --video-deblock --video-deblock \
  --video-long-term
# Loop restoration keeps its named rejection: the pinned decoder's
# unit/window handling at shared tile edges does not match a frame-level
# filter (docs/negative-results.md).
expect_reject_cpu wiener-tiles --video-wiener --video-wiener --tiles 2x1
expect_reject_cpu chroma-lr-tiles --video-chroma-lr --video-chroma-lr \
  --tiles 2x2
# Loop filters on the frame-type wrappers stay rejected until their component
# header writers signal the filter rows.
expect_reject_cpu cdef-long-term --video-cdef --video-cdef --video-long-term
# GME with more than one reference is gated on the plain skip-only P chain;
# the remaining combinations stay rejected by name.
expect_reject gme-bframes --video-gme --b-frames 2
expect_reject gme-compound --video-gme --video-refs 2 --video-compound
# §5.20.7.14: inter-intra is a single-reference motion mode; a two-reference
# compound block cannot carry interintra_mode/wedge_interintra.
expect_reject compound-interintra --video-refs 2 --video-compound \
  --video-compound-interintra
expect_reject gme-residual --video-gme --video-inter-residual
expect_reject wiener-tiles --video-wiener --tiles 2x1

# --video-tx-search combinations that cannot be supported yet: the §7.17 grid
# models one transform per leaf, the §8.2.4 bank has no §5.20.6.3 partition
# CDFs, the tile writer owns a single tx_mode_select bit, and the frame-type
# wrapper headers do not signal it.  All are named usage rejections.
expect_reject_video() {
  local label="$1"
  # A targeted --only run must not pay for the whole rejection matrix.
  if [[ -n "$ONLY" && "$label" != *"$ONLY"* ]]; then return 0; fi
  local needle="$2"
  shift 2
  set +e
  "$BIN" "$CLIP" -o "$TMP/reject_video.ivf" --video "$@" \
    >"$TMP/reject_video.out" 2>"$TMP/reject_video.err"
  local rc=$?
  set -e
  [[ $rc -eq 2 ]] || fail "$label: expected usage exit 2, got $rc"
  grep -q -- "$needle" "$TMP/reject_video.err" ||
    fail "$label: rejection did not name $needle"
  echo "PASS video reject $label"
}

# The frame-type wrapper headers still own their own tx_mode_select bit.
expect_reject_video tx-search-long-term --video-tx-search --video-tx-search --video-long-term
# OPFL decodes through its component gate but is not reachable from the M2
# inter product path; it stays a named usage error, not a silent no-op.
expect_reject_video opfl --video-opfl --video-opfl

echo "gate_video_gpu: PASS"
