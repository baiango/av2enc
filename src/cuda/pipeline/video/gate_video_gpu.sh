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
# Generate enough frames for the longest gated sequence (8-frame temporal GOP /
# 7-reference P chain) even when the caller lowers --frames.
GEN_FRAMES=$(( FRAMES > 8 ? FRAMES : 8 ))

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

fail() {
  echo "gate_video_gpu: FAIL: $*" >&2
  exit 1
}

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
  local cpu="$TMP/${label}.cpu.ivf"
  local gpu="$TMP/${label}.gpu.ivf"
  local cpu_recon="$TMP/${label}.cpu.recon.y4m"
  local gpu_recon="$TMP/${label}.gpu.recon.y4m"
  local cpu_dec="$TMP/${label}.cpu.dec.y4m"
  local gpu_dec="$TMP/${label}.gpu.dec.y4m"

  "$BIN" "$clip" -o "$cpu" --video --frames "$FRAMES" \
    --video-recon "$cpu_recon" "$@" >/dev/null 2>"$TMP/${label}.cpu.log" ||
    fail "$label: CPU reference encode failed"
  "$BIN" "$clip" -o "$gpu" --video --video-gpu --frames "$FRAMES" \
    --video-recon "$gpu_recon" "$@" >/dev/null 2>"$TMP/${label}.gpu.log" ||
    fail "$label: GPU encode failed"
  grep -q -- "frame_bytes_from_cpu=0" "$TMP/${label}.gpu.log" ||
    fail "$label: GPU path reported CPU-produced bytes"

  cmp -s "$cpu" "$gpu" ||
    fail "$label: temporal units differ (CPU $(md5sum "$cpu" | cut -d' ' -f1), GPU $(md5sum "$gpu" | cut -d' ' -f1))"
  cmp -s "$cpu_recon" "$gpu_recon" ||
    fail "$label: encoder reconstruction differs"

  "$DAV2D" --strict 1 -q -i "$cpu" -o "$cpu_dec" >/dev/null 2>&1 ||
    fail "$label: dav2d rejected the CPU reference stream"
  "$DAV2D" --strict 1 -q -i "$gpu" -o "$gpu_dec" >/dev/null 2>&1 ||
    fail "$label: dav2d --strict 1 rejected the GPU stream"
  cmp -s "$cpu_dec" "$gpu_dec" ||
    fail "$label: decoded output differs"

  echo "PASS video-gpu $label ($(md5sum "$gpu" | cut -d' ' -f1))"
}

run_case() {
  run_case_on "$CLIP" "$@"
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
# Compound/CWP AVERAGE is selected on the A/B/mid clip.
_saved_frames="$FRAMES"
FRAMES=3
run_case_on "$CLIP_AVG" compound --video-refs 2 --video-compound
FRAMES="$_saved_frames"
# §5.18.9 global motion + §7.13.3.19/.20 GLOBALMV warp.  The panned clip gives
# the estimator a coherent model, so GLOBALMV exercises the warped predictor.
run_case_on "$CLIP_INTER" gme --video-gme
run_case_on "$CLIP_INTER" gme-warp --video-gme --video-warp
run_case_on "$CLIP_INTER" gme-depth2 --video-gme --video-max-depth 2
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
expect_reject() {
  local label="$1"
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
  shift
  set +e
  "$BIN" "$CLIP" -o "$TMP/reject_cpu.ivf" --video "$@" \
    >"$TMP/reject_cpu.out" 2>"$TMP/reject_cpu.err"
  local rc=$?
  set -e
  [[ $rc -eq 2 ]] || fail "$label: expected usage exit 2, got $rc"
  grep -q -- "--video-deblock" "$TMP/reject_cpu.err" ||
    fail "$label: rejection did not name --video-deblock"
  echo "PASS video reject $label"
}

expect_reject_cpu deblock-tiles --video-deblock --tiles 2x1
expect_reject_cpu deblock-long-term --video-deblock --video-long-term
# GME currently owns the single-reference P chain; a multi-reference model
# diverges from the GPU mirror on multi-frame clips, so the pair is rejected
# by name until the per-slot model is re-synced.
expect_reject gme-refs2 --video-gme --video-refs 2
expect_reject gme-bframes --video-gme --b-frames 2
expect_reject gme-compound --video-gme --video-refs 2 --video-compound
expect_reject gme-residual --video-gme --video-inter-residual

# --video-tx-search combinations that cannot be supported yet: the §7.17 grid
# models one transform per leaf, the §8.2.4 bank has no §5.20.6.3 partition
# CDFs, the tile writer owns a single tx_mode_select bit, and the frame-type
# wrapper headers do not signal it.  All are named usage rejections.
expect_reject_video() {
  local label="$1"
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

# The first flag is the message the rejection must name; the rest is the
# command line under test.
expect_reject_video tx-search-deblock --video-deblock --video-deblock --video-tx-search
expect_reject_video tx-search-avg-cdf --avg-cdf --avg-cdf --video-tx-search
expect_reject_video tx-search-tiles --tiles --video-tx-search --tiles 2x1
expect_reject_video tx-search-long-term --video-tx-search --video-tx-search --video-long-term

echo "gate_video_gpu: PASS"
