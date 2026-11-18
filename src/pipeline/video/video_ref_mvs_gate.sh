#!/usr/bin/env bash
# CPU gate for --video-ref-mvs (§5.4.6 enable_ref_frame_mvs, §5.18.2
# use_ref_frame_mvs, §7.9 motion-field estimation, §7.22 storage and the
# §7.12.2.7/.8 temporal MV-stack candidates).
#
# Every temporal case is encoded through the product CLI, decoded with the
# pinned dav2d at --strict 1 --threads 1, and the decoder output is compared
# sample-for-sample (Y4M MD5) against the encoder's --video-recon dump.  The
# stream must also report at least one temporal candidate, otherwise the gate
# would pass on an inert feature.
#
# The GME carriers include a pure-zoom clip.  The default GME path must not
# signal a model the decoder warps (the §7.13.3.19-.24 mirror is not
# conformance-proven; docs/gme-720p.md), so the fitted model is flattened to
# its translation-only projection and the decoded payload must equal
# --video-recon byte for byte.  AV2ENC_GME_UNPROVEN_WARP=1 restores the
# historical warped predictor; the gate only uses it to prove the fixture
# really fits a non-identity linear model.
#
# When --base-bin points at a pre-change binary (the commit before the feature),
# the default invocations must be byte-identical; that is the "flag off" check.
#
# Usage:
#   src/pipeline/video/video_ref_mvs_gate.sh --bin build/av2enc \
#       [--input clip.y4m] [--dav2d /path/to/dav2d] [--base-bin old/av2enc]
#
# Exit codes: 0 pass, 1 failure, 2 usage, 77 missing fixture/decoder.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/../../.." && pwd)
BIN=""
INPUT="${AV2ENC_TESTDATA:-$ROOT/testdata}/pub_1280x720_60f.y4m"
DAV2D="${DAV2D:-}"
BASE_BIN=""

while [ $# -gt 0 ]; do
  case "$1" in
    --bin) BIN=$2; shift 2 ;;
    --input) INPUT=$2; shift 2 ;;
    --dav2d) DAV2D=$2; shift 2 ;;
    --base-bin) BASE_BIN=$2; shift 2 ;;
    -h|--help) sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "video_ref_mvs_gate: unknown argument $1" >&2; exit 2 ;;
  esac
done

[ -n "$BIN" ] || { echo "video_ref_mvs_gate: --bin is required" >&2; exit 2; }
[ -x "$BIN" ] || { echo "video_ref_mvs_gate: no such binary $BIN" >&2; exit 2; }
[ -f "$INPUT" ] || { echo "video_ref_mvs_gate: missing fixture $INPUT" >&2; exit 77; }
if [ -z "$DAV2D" ]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
  done
fi
[ -n "$DAV2D" ] && [ -x "$DAV2D" ] || {
  echo "video_ref_mvs_gate: dav2d not found (set DAV2D)" >&2; exit 77; }

OUT="${OUT:-${TMPDIR:-/tmp}/video_ref_mvs_gate}"
mkdir -p "$OUT"
FAIL=0
STREAMS=()

md5() { md5sum "$1" 2>/dev/null | cut -d' ' -f1 || md5 -q "$1"; }

# Input for the next run_case; the fixture cases below switch it.
CASE_INPUT="$INPUT"

# Encodes one case, requires the expected temporal-candidate count, decodes and
# compares the reconstruction.  $1 = label, $2 = expected frame count,
# $3 = want temporal candidates, $4 = require a selected temporal candidate,
# remaining args = encoder options.
run_case() {
  local label=$1 frames=$2 want_temporal=$3 want_selected=$4
  shift 4
  local ivf="$OUT/$label.ivf" recon="$OUT/$label.recon.y4m"
  local dec="$OUT/$label.dec.y4m" log="$OUT/$label.log"
  echo "== $label: $*"
  if ! "$BIN" "$CASE_INPUT" -o "$ivf" --video --frames "$frames" \
      --video-recon "$recon" "$@" >"$log" 2>&1; then
    echo "FAIL($label): encode failed"; tail -5 "$log"; FAIL=1; return
  fi
  if ! grep -q "output frame(s)" "$log"; then
    echo "FAIL($label): no encoded summary"; FAIL=1; return
  fi
  local got_temporal
  got_temporal=$(sed -n 's/.*temporal MV candidates \([0-9]*\).*/\1/p' "$log" | tail -1)
  got_temporal=${got_temporal:-0}
  if [ "$want_temporal" = "yes" ] && [ "$got_temporal" -le 0 ]; then
    echo "FAIL($label): feature inert (0 temporal candidates)"; FAIL=1; return
  fi
  if [ "$want_temporal" = "no" ] && [ "$got_temporal" -ne 0 ]; then
    echo "FAIL($label): unexpected temporal candidates"; FAIL=1; return
  fi
  local got_selected
  got_selected=$(sed -n 's/.*temporal-selected leaves \([0-9]*\).*/\1/p' "$log" | tail -1)
  got_selected=${got_selected:-0}
  if [ "$want_selected" = "yes" ] && [ "$got_selected" -le 0 ]; then
    echo "FAIL($label): no coded leaf selected a temporal candidate"
    FAIL=1; return
  fi
  if ! "$DAV2D" -q -i "$ivf" -o "$dec" --strict 1 --threads 1; then
    echo "FAIL($label): dav2d --strict 1 rejected the stream"; FAIL=1; return
  fi
  local recon_md5 dec_md5
  recon_md5=$(md5 "$recon"); dec_md5=$(md5 "$dec")
  if [ "$recon_md5" != "$dec_md5" ]; then
    echo "FAIL($label): recon mismatch (encoder $recon_md5 decoder $dec_md5)"
    FAIL=1; return
  fi
  # The decoded Y4M frame count is a second, independent check.
  local dec_frames
  # LC_ALL=C: BSD grep otherwise stops at the first non-UTF-8 sample byte.
  dec_frames=$(LC_ALL=C grep -a -o 'FRAME' "$dec" | wc -l | tr -d ' ')
  if [ "$dec_frames" -ne "$frames" ]; then
    echo "FAIL($label): decoded $dec_frames of $frames picture(s)"; FAIL=1; return
  fi
  echo "  PASS: $frames picture(s), temporal=$got_temporal, selected=$got_selected, md5=$recon_md5"
  STREAMS+=("$label")
}

# Strict decode plus byte-for-byte payload equality against --video-recon.
# Strips the Y4M header line and FRAME markers; reports the first differing
# frame and the total differing-byte count so a warp-mirror regression is
# diagnosable (run_case's Y4M MD5 would also differ, but this names the frames).
decode_payload_exact() {
  local ivf=$1 recon=$2 frames=$3 label=$4
  local dec="$OUT/$label.dec.y4m" log="$OUT/$label.dec.log"
  if ! "$DAV2D" -q --strict 1 -i "$ivf" -o "$dec" --threads 1 \
      >"$log" 2>&1; then
    echo "FAIL($label): dav2d --strict 1 rejected the stream"
    tail -5 "$log"
    FAIL=1
    return 1
  fi
  if ! python3 - "$recon" "$dec" "$frames" "$label" <<'PY'
import sys

def split(path):
    data = open(path, 'rb').read()
    end = data.index(b'\n') + 1
    body = data[end:].replace(b'FRAME\n', b'')
    return body

recon_path, dec_path, want, label = sys.argv[1:5]
want = int(want)
a = split(recon_path)
b = split(dec_path)
if len(a) != len(b):
    print(f"FAIL({label}): recon payload {len(a)} B != decoded {len(b)} B",
          file=sys.stderr)
    sys.exit(1)
size = len(a) // want
dec = open(dec_path, 'rb').read()
end = dec.index(b'\n') + 1
got = dec[end:].count(b'FRAME\n')
if got != want:
    print(f"FAIL({label}): decoded {got} of {want} picture(s)", file=sys.stderr)
    sys.exit(1)
diff = 0
bad_frames = []
for i in range(0, len(a), size):
    chunk = a[i:i + size]
    ref = b[i:i + size]
    d = sum(1 for x, y in zip(chunk, ref) if x != y)
    if d:
        bad_frames.append((i // size, d))
    diff += d
if diff:
    frames = ",".join(f"{f}:{d}" for f, d in bad_frames[:6])
    print(f"FAIL({label}): decoded payload differs from --video-recon in "
          f"{diff} of {len(a)} bytes (frame:diff {frames})", file=sys.stderr)
    sys.exit(1)
print(f"  PASS({label}): payload exact, {want} picture(s), {len(a)} bytes")
PY
  then
    FAIL=1
    return 1
  fi
}

# GME warp fallback.  The zoom fixture fits a model with a non-identity linear
# part (proved by the escape-hatch arm below, which produces a different
# stream).  The default must signal only the translation-only projection and
# must decode payload-exactly through the §7.12.2.2 centre-MV path.  The
# escape hatch restores the fitted warp model; if it ever produced the same
# bytes as the default, the fixture would no longer exercise the fallback.
check_gme_zoom_fallback() {
  local clip=$1 frames=$2 label=$3
  local ivf="$OUT/$label.ivf" recon="$OUT/$label.recon.y4m"
  local log="$OUT/$label.log"
  echo "== $label: --video-gme --video-warp (safe translation fallback)"
  if ! "$BIN" "$clip" -o "$ivf" --video --frames "$frames" \
      --video-gme --video-warp --video-recon "$recon" --video-trace \
      >"$log" 2>&1; then
    echo "FAIL($label): encode failed"
    tail -5 "$log"
    FAIL=1
    return
  fi
  if ! grep -q "global mv=" "$log"; then
    echo "FAIL($label): no GLOBALMV candidate evaluated"
    FAIL=1
    return
  fi
  decode_payload_exact "$ivf" "$recon" "$frames" "$label" || return
  # Escape hatch: the fitted model is kept (including its linear part) and the
  # §7.13.3.19/.20 predictor runs.  This arm is conformance-unproven, so it is
  # only checked for liveness and to keep the default arm from passing on a
  # clip whose fit degraded to purely translational content.
  local hatch_ivf="$OUT/$label-hatch.ivf" hatch_log="$OUT/$label-hatch.log"
  if ! env AV2ENC_GME_UNPROVEN_WARP=1 "$BIN" "$clip" -o "$hatch_ivf" \
      --video --frames "$frames" --video-gme --video-warp --video-trace \
      >"$hatch_log" 2>&1; then
    echo "FAIL($label): escape-hatch encode failed"
    tail -5 "$hatch_log"
    FAIL=1
    return
  fi
  if cmp -s "$ivf" "$hatch_ivf"; then
    echo "FAIL($label): escape hatch left the stream unchanged (fixture inert)"
    FAIL=1
    return
  fi
  local hatch_warped
  hatch_warped=$(grep -c "warp=1" "$hatch_log" || true)
  if [ "${hatch_warped:-0}" -le 0 ]; then
    echo "FAIL($label): escape hatch selected no warped candidate"
    FAIL=1
    return
  fi
  echo "  PASS($label): translation fallback active (hatch warps $hatch_warped candidate(s))"
  STREAMS+=("$label")
}

# --- temporal cases --------------------------------------------------------
run_case p8_refs1 8 no no --video-ref-mvs
run_case p8_refs2 8 yes no --video-refs 2 --video-ref-mvs
run_case p8_refs3 8 yes no --video-refs 3 --video-ref-mvs
run_case p16_refs4 16 yes no --video-refs 4 --video-ref-mvs
run_case p8_refs2_depth4 8 yes no --video-refs 2 --video-ref-mvs --video-min-block 8
run_case p8_refs2_deblock 8 yes no --video-refs 2 --video-ref-mvs --video-deblock
run_case p8_refs2_residual 8 yes no --video-refs 2 --video-ref-mvs --video-inter-residual
run_case p8_refs2_stress 8 yes yes --video-refs 2 --video-ref-mvs-stress
run_case p8_refs3_stress 8 yes yes --video-refs 3 --video-ref-mvs-stress
run_case b4_stress 4 yes yes --b-frames 2 --video-ref-mvs-stress
run_case b4 4 yes no --b-frames 2 --video-ref-mvs
run_case b6 6 yes no --b-frames 4 --video-ref-mvs
# Hierarchical-P temporal layers use two references per interior picture, so
# the §7.22 fields exist and §7.9 must project candidates.
run_case tl4 4 yes no --video-tlayers 4 --video-ref-mvs
run_case tl8 8 yes no --video-tlayers 8 --video-ref-mvs

# --- §7.22 warp/GME and wedge-compound field storage -----------------------
#
# These carriers need fixtures whose content selects the tool, so the gate
# generates them deterministically (same bytes on every host) instead of
# depending on an external clip.
PAN64="$OUT/fixture_pan64.y4m"
PAN360="$OUT/fixture_pan360.y4m"
WEDGE64="$OUT/fixture_wedge64.y4m"
WEDGE360="$OUT/fixture_wedge360.y4m"
ZOOM360="$OUT/fixture_zoom360.y4m"

if [ ! -s "$PAN64" ] || [ ! -s "$PAN360" ] || [ ! -s "$WEDGE64" ] ||
   [ ! -s "$WEDGE360" ] || [ ! -s "$ZOOM360" ]; then
  python3 - "$PAN64" "$PAN360" "$WEDGE64" "$WEDGE360" "$ZOOM360" <<'PY'
import math
import sys

pan64, pan360, wedge64, wedge360, zoom360 = sys.argv[1:6]

def write(path, w, h, frames):
    cw, ch = w // 2, h // 2
    with open(path, "wb") as out:
        out.write(b"YUV4MPEG2 W%d H%d F30:1 Ip A1:1 C420\n" % (w, h))
        for f, fn in enumerate(frames):
            out.write(b"FRAME\n")
            for y in range(h):
                out.write(bytes(fn(x, y, f, 0, w, h) for x in range(w)))
            for p in (1, 2):
                for y in range(ch):
                    out.write(bytes(fn(2 * x, 2 * y, f, p, w, h)
                                    for x in range(cw)))

def pan(k):
    # Smooth pan: the §5.18.9 fit lands on a translation-only model, which is
    # the shape the product GLOBALMV carrier decodes exactly.  The texture
    # frequencies are chosen per frame size so the same content family is
    # exercised at 64x64 and 640x360.
    def sample(x, y, f, plane, w, h):
        if w == 64:
            xx = x + k * f
            yy = y + 0.5 * k * f
            if plane == 0:
                v = 128 + 60 * math.sin(xx * 0.5) + \
                    40 * math.cos(yy * 0.8) + 20 * math.sin((xx + yy) * 0.3)
            elif plane == 1:
                v = 128 + 40 * math.sin((x + 0.5 * k * f) * 0.4) + \
                    20 * math.cos((y + 0.3 * k * f) * 0.5)
            else:
                v = 128 + 35 * math.cos((x + 0.4 * k * f) * 0.6) + \
                    25 * math.sin((y + 0.2 * k * f) * 0.4)
        else:
            xx = x + k * f
            yy = y + k * f / 3.0
            if plane == 0:
                v = 128 + 60 * math.sin(xx * 0.05) + \
                    40 * math.cos(yy * 0.08) + 20 * math.sin((xx + yy) * 0.03)
            elif plane == 1:
                v = 128 + 40 * math.sin((x + 2 * k * f / 3.0) * 0.04) + \
                    20 * math.cos((y + k * f / 3.0) * 0.05)
            else:
                v = 128 + 35 * math.cos((x + 2 * k * f / 3.0) * 0.06) + \
                    25 * math.sin((y + k * f / 3.0) * 0.04)
        return max(0, min(255, int(v + 0.5)))
    return sample

COS = [4, 4, 4, 2, 2, 0, -2, -2, -4, -4, -4, -4, -4, -2, -2, 0, 2, 2, 4, 4]
SIN = [0, -1, -2, -2, -4, -4, -4, -2, -2, -1, 0, 1, 2, 2, 4, 4, 4, 2, 2, 1]
POS = [8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14,
       14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16]
NEG = [8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 4, 3, 3, 3, 2, 2,
       2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]

def wedge_mask(i, j, bw, bh):
    # §7.13.3.27 index 22 (direction 6, offset 4/8), sign 0.
    direction, cx, cy = 6, 4, 4
    xoff = bw // 2 - ((cx * bw) >> 3)
    yoff = bh // 2 - ((cy * bh) >> 3)
    n, m = yoff + i, xoff + j
    d = ((m << 1) - (bw - 1)) * COS[direction] + \
        ((n << 1) - (bh - 1)) * SIN[direction]
    d = max(-31, min(31, d))
    return 64 - (((POS[d] if d >= 0 else NEG[-d]) << 2))

def wedge_fixture(w, h):
    def a(x, y, p):
        if p == 0:
            return max(0, min(255, int(60 + 30 * math.sin(x * 0.3) +
                                     20 * math.cos(y * 0.4))))
        if p == 1:
            return max(0, min(255, int(90 + 20 * math.sin(x * 0.4))))
        return max(0, min(255, int(100 + 20 * math.cos(y * 0.5))))

    def b(x, y, p):
        if p == 0:
            return max(0, min(255, int(190 + 30 * math.sin(x * 0.3 + 1.5) +
                                     20 * math.cos(y * 0.4 + 2))))
        if p == 1:
            return max(0, min(255, int(180 + 20 * math.sin(x * 0.4 + 1.5))))
        return max(0, min(255, int(170 + 20 * math.cos(y * 0.5 + 2))))

    def wedge(x, y, p):
        m = wedge_mask(y % 64, x % 64, 64, 64)
        av, bv = a(x, y, p), b(x, y, p)
        return (m * av + (64 - m) * bv + 32) >> 6

    def wrap(fn):
        return lambda x, y, f, p, fw, fh: fn(x, y, p)

    return [wrap(a), wrap(b), wrap(wedge), wrap(b),
            wrap(a), wrap(b), wrap(wedge), wrap(b)]

def zoom_fixture(rate):
    # Pure zoom about the frame origin.  Each picture is sampled from the same
    # continuous texture at rate**f, so the fitted current->reference model has
    # a non-identity linear part (no panning content).  This is the shape whose
    # §7.13.3.19/.20 warp predictor does not mirror the decoder exactly.
    def sample(x, y, f, plane, w, h):
        scale = rate ** f
        xx, yy = x / scale, y / scale
        if plane == 0:
            v = 128 + 55 * math.sin(xx * 0.09 + 0.3) + \
                40 * math.cos(yy * 0.11) + 20 * math.sin((xx + yy) * 0.05)
        elif plane == 1:
            v = 128 + 35 * math.sin(xx * 0.07 + 1.0) + \
                20 * math.cos(yy * 0.09)
        else:
            v = 128 + 30 * math.cos(xx * 0.08) + \
                25 * math.sin(yy * 0.06 + 0.5)
        return max(0, min(255, int(v + 0.5)))
    return [sample] * 4

write(pan64, 64, 64, [pan(1)] * 8)
write(pan360, 640, 360, [pan(3)] * 8)
write(wedge64, 64, 64, wedge_fixture(64, 64))
write(wedge360, 640, 360, wedge_fixture(640, 360))
write(zoom360, 640, 360, zoom_fixture(1.01))
PY
fi

# The wedge carrier reports how many stored 8x8 cells lost a list to the §7.22
# allow-list; the case is only meaningful when that path runs.
check_wedge_drops() {
  local label=$1
  local drops
  drops=$(sed -n 's/.*§7\.22 wedge allow-list cells \([0-9]*\).*/\1/p' \
      "$OUT/$label.log" | tail -1)
  drops=${drops:-0}
  if [ "$drops" -le 0 ]; then
    echo "FAIL($label): §7.22 wedge allow-list never ran"; FAIL=1
  else
    echo "  PASS($label): §7.22 wedge allow-list cells $drops"
  fi
}

CASE_INPUT="$PAN64"
run_case gme64 8 yes no --video-gme --video-refs 2 --video-ref-mvs
run_case gme64_refs3 8 yes no --video-gme --video-refs 3 --video-ref-mvs
CASE_INPUT="$PAN360"
run_case gme360 8 yes no --video-gme --video-refs 2 --video-ref-mvs
run_case gme360_refs3 8 yes no --video-gme --video-refs 3 --video-ref-mvs
run_case warp360 8 yes no --video-warp --video-refs 2 --video-ref-mvs
# Default GME must not emit a warped prediction for a genuinely affine fit;
# the zoom clip decodes byte-exactly through the translation-only fallback.
check_gme_zoom_fallback "$ZOOM360" 4 gme-zoom-fallback
CASE_INPUT="$WEDGE64"
run_case wedge64 8 yes no --video-refs 2 --video-compound \
  --video-compound-wedge --video-ref-mvs
check_wedge_drops wedge64
CASE_INPUT="$WEDGE360"
run_case wedge360 8 yes no --video-refs 2 --video-compound \
  --video-compound-wedge --video-ref-mvs
check_wedge_drops wedge360
CASE_INPUT="$INPUT"

# --- "flag off" byte identity ---------------------------------------------
if [ -n "$BASE_BIN" ] && [ -x "$BASE_BIN" ]; then
  # --video-tlayers is intentionally not in this list: the multi-reference
  # hierarchical plan changes its payload by design (see the §7.9 gate above).
  for spec in "default:8:" "refs2:8:--video-refs 2" "b4:4:--b-frames 2"; do
    label=${spec%%:*}; rest=${spec#*:}; frames=${rest%%:*}; extra=${rest#*:}
    # shellcheck disable=SC2086
    "$BASE_BIN" "$INPUT" -o "$OUT/base_$label.ivf" --video --frames "$frames" \
      $extra >/dev/null 2>&1
    # shellcheck disable=SC2086
    "$BIN" "$INPUT" -o "$OUT/new_$label.ivf" --video --frames "$frames" \
      $extra >/dev/null 2>&1
    if ! cmp -s "$OUT/base_$label.ivf" "$OUT/new_$label.ivf"; then
      echo "FAIL(identity/$label): default stream changed with the flag off"
      FAIL=1
    else
      echo "  PASS(identity/$label): $(md5 "$OUT/new_$label.ivf")"
    fi
  done
  # The flag must actually change the bitstream.
  "$BIN" "$INPUT" -o "$OUT/new_refmvs.ivf" --video --frames 8 \
    --video-refs 2 --video-ref-mvs >/dev/null 2>&1
  if cmp -s "$OUT/base_refs2.ivf" "$OUT/new_refmvs.ivf"; then
    echo "FAIL: --video-ref-mvs left the stream byte-identical (inert header)"
    FAIL=1
  else
    echo "  PASS(marker): --video-ref-mvs changes the stream"
  fi
  # The §7.22 store must not perturb carriers that do not enable it: the GME
  # and wedge-compound streams with --video-ref-mvs off stay byte-identical.
  for spec in "gme:$PAN360:8:--video-gme --video-refs 2" \
              "wedge:$WEDGE64:4:--video-refs 2 --video-compound --video-compound-wedge"; do
    label=${spec%%:*}; rest=${spec#*:}; clip=${rest%%:*}; rest=${rest#*:}
    frames=${rest%%:*}; extra=${rest#*:}
    # shellcheck disable=SC2086
    "$BASE_BIN" "$clip" -o "$OUT/base_$label.ivf" --video --frames "$frames" \
      $extra >/dev/null 2>&1
    # shellcheck disable=SC2086
    "$BIN" "$clip" -o "$OUT/new_$label.ivf" --video --frames "$frames" \
      $extra >/dev/null 2>&1
    if ! cmp -s "$OUT/base_$label.ivf" "$OUT/new_$label.ivf"; then
      echo "FAIL(identity/$label): stream changed with --video-ref-mvs off"
      FAIL=1
    else
      echo "  PASS(identity/$label): $(md5 "$OUT/new_$label.ivf")"
    fi
  done
fi

if [ "$FAIL" -ne 0 ]; then
  echo "video_ref_mvs_gate: FAIL"
  exit 1
fi
echo "video_ref_mvs_gate: PASS (${#STREAMS[@]} temporal stream(s))"
