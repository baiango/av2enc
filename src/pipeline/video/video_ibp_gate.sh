#!/usr/bin/env bash
# Decoder gate for intra bi-prediction on the CPU `--video` writer
# (`--video-ibp`, §5.4.5 `enable_ibp` + §7.13.2).
#
#   ./video_ibp_gate.sh
#   AV2ENC_BIN=build-int/av2enc DAV2D=/path/to/dav2d ./video_ibp_gate.sh
#   KEEP=1 ./video_ibp_gate.sh
#
# What it proves:
#   1. `--video-ibp` writes §5.4.5 enable_ibp = 1 and the decoder derives the
#      §7.13.2 IBP predictor from (mode, angle_delta) with no per-block bit.
#   2. The DC-only writer path (`--video-intra-full` off) applies the
#      §7.13.2.12 Ibp_Weights DC post-filter to every non-4x4 block; the
#      decoded picture therefore differs from the enable_ibp = 0 stream while
#      staying sample-exact against `--video-recon`.
#   3. `--video-ibp --video-intra-full` runs the 61-candidate luma search under
#      IBP semantics: the trace reports blocks with `ibp_blend=1` (the
#      two-predictor §7.13.2.9 weighting) and the stream is still exact.
#   4. Pinned dav2d f9dafbc `--strict 1` decodes every case, the picture count
#      matches, and the decoded samples equal `--video-recon` exactly.
#   5. The default (no flag) streams keep their pinned MD5s: `--video-ibp` is
#      default-off and the sequence bit is still written as 0.
#   6. Named rejections stay named for combinations the writer does not run
#      through the IBP predictor (CfL/MHCCP chroma, MRL, screen content,
#      tiles, a non---video invocation).
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "$0")/../../.." && pwd)"
TMP="${TMPDIR:-/tmp}/av2enc_video_ibp_gate.$$"
mkdir -p "$TMP"
cleanup() {
  if [ "${KEEP:-0}" != "1" ]; then rm -rf "$TMP"; fi
}
trap cleanup EXIT

AV2ENC_BIN="${AV2ENC_BIN:-}"
if [ -z "$AV2ENC_BIN" ]; then
  for candidate in "$ROOT/build-int/av2enc" "$ROOT/build/av2enc" \
                   "$ROOT/build-m1/av2enc"; do
    if [ -x "$candidate" ]; then AV2ENC_BIN="$candidate"; break; fi
  done
fi
if [ -z "$AV2ENC_BIN" ] || [ ! -x "$AV2ENC_BIN" ]; then
  echo "video_ibp_gate: no av2enc binary; set AV2ENC_BIN=..." >&2
  exit 127
fi

DAV2D="${DAV2D:-}"
if [ -z "$DAV2D" ]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$candidate" ]; then DAV2D="$candidate"; break; fi
  done
fi
if [ -z "$DAV2D" ] || [ ! -x "$DAV2D" ]; then
  echo "video_ibp_gate: no dav2d binary (set DAV2D=...)" >&2
  exit 77
fi

INPUT_64="${AV2ENC_INPUT_64:-$ROOT/testdata/m0_64x64_1f.y4m}"
INPUT_360="${AV2ENC_INPUT_360:-$ROOT/testdata/shine_640x360_4f.y4m}"
for path in "$INPUT_64" "$INPUT_360"; do
  if [ ! -f "$path" ]; then
    echo "video_ibp_gate: missing input $path" >&2
    exit 77
  fi
done
if ! command -v python3 >/dev/null 2>&1; then
  echo "video_ibp_gate: python3 is required for the Y4M comparisons" >&2
  exit 77
fi

md5_of() {
  md5 -q "$1" 2>/dev/null || md5sum "$1" | awk '{print $1}'
}

# Y4M helpers: frame count and exact/differ comparisons.
y4m_check() {
  python3 - "$@" <<'PY'
import sys

def frames(path):
    data = open(path, "rb").read()
    out = []
    i = 0
    w = h = 0
    while True:
        e = data.find(b"\n", i)
        if e < 0:
            break
        header = data[i:e].decode()
        if header.startswith("FRAME"):
            j = data.find(b"FRAME\n", e + 1)
            j = len(data) if j < 0 else j
            out.append(data[e + 1:j])
            i = j
        else:
            w = int(header.split("W")[1].split()[0])
            h = int(header.split("H")[1].split()[0])
            i = e + 1
    return w, h, out

mode = sys.argv[1]
a_path, b_path = sys.argv[2], sys.argv[3]
if mode == "count":
    _, _, a = frames(a_path)
    got = len(a)
    want = int(b_path)
    if got != want:
        raise SystemExit(f"decoded {got} picture(s), expected {want}")
    print(f"count {got}")
    raise SystemExit(0)

w, h, a = frames(a_path)
w2, h2, b = frames(b_path)
if (w, h) != (w2, h2) or len(a) != len(b):
    raise SystemExit(f"geometry/frames {w}x{h}/{len(a)} vs {w2}x{h2}/{len(b)}")
diffs = sum(sum(x != y for x, y in zip(fa, fb)) for fa, fb in zip(a, b))
if mode == "exact":
    if diffs:
        raise SystemExit(f"{diffs} sample(s) differ")
    print(f"exact {len(a)} frame(s)")
elif mode == "differ":
    if diffs == 0:
        raise SystemExit("picture data is identical")
    print(f"differ {diffs} sample(s)")
else:
    raise SystemExit("unknown mode " + mode)
PY
}

failures=0
passes=0
pass_case() { echo "PASS $1"; passes=$((passes + 1)); }
fail_case() { echo "FAIL $1: $2" >&2; failures=$((failures + 1)); }

# run_case <name> <input> <frames> <expected md5> <min ibp_blend> [args...]
run_case() {
  local name=$1 input=$2 frames=$3 expected=$4 min_blend=$5
  shift 5
  local out="$TMP/$name.ivf" recon="$TMP/$name.recon.y4m"
  local dec="$TMP/$name.dav.y4m" log="$TMP/$name.log"
  if ! "$AV2ENC_BIN" "$input" -o "$out" --frames "$frames" --video \
        --video-recon "$recon" --video-trace "$@" >"$log" 2>&1; then
    fail_case "$name" "encode failed"
    tail -5 "$log" >&2
    return
  fi
  local got
  got=$(md5_of "$out")
  if [ "$got" != "$expected" ]; then
    fail_case "$name" "IVF md5 $got, expected $expected"
    return
  fi
  local blend
  blend=$(grep -c 'ibp_blend=1' "$log" || true)
  if [ "$blend" -lt "$min_blend" ]; then
    fail_case "$name" "$blend IBP blend block(s), expected at least $min_blend"
    return
  fi
  if ! "$DAV2D" -q -i "$out" -o "$dec" --threads 1 --strict 1 \
        >"$TMP/$name.dec.log" 2>&1; then
    fail_case "$name" "dav2d --strict 1 rejected the stream"
    tail -5 "$TMP/$name.dec.log" >&2
    return
  fi
  if ! y4m_check count "$dec" "$frames" >>"$TMP/$name.cmp.log" 2>&1; then
    fail_case "$name" "picture count mismatch"
    cat "$TMP/$name.cmp.log" >&2
    return
  fi
  if ! y4m_check exact "$recon" "$dec" >>"$TMP/$name.cmp.log" 2>&1; then
    fail_case "$name" "encoder recon differs from dav2d"
    cat "$TMP/$name.cmp.log" >&2
    return
  fi
  pass_case "$name (blend=$blend, $(wc -c <"$out" | tr -d ' ') B, exact)"
}

# Default streams: pinned in this branch.  The IBP feature is default-off, so
# these MD5s are the byte-identical pre-feature streams.
run_case default_64 "$INPUT_64" 1 df6fd676b1c879b6ccc2fd3e575b24a5 0
run_case default_360 "$INPUT_360" 4 83b34e7944346d6b3f64a460bdf1dc96 0

# DC-only path: IBP changes the header bit and the §7.13.2.12 DC post-filter.
run_case ibp_64 "$INPUT_64" 1 57d835c78d39d0e4b13f4a9d9e0e3a8b 0 --video-ibp
run_case ibp_360 "$INPUT_360" 4 d25bc92785ee7a8d15a7fd350345ebc3 0 --video-ibp

# Full search path: directional candidates now blend (§7.13.2.9).
run_case ibp_full_64 "$INPUT_64" 1 dc3b45d97787ebc16531cb92e2444334 1 \
  --video-ibp --video-intra-full
run_case ibp_full_360 "$INPUT_360" 4 880193e82a2eacce061234fd32c0783c 1 \
  --video-ibp --video-intra-full

# The DC-only IBP stream must change the decoded picture, not just the header:
# otherwise only the sequence bit moved and the predictor was not applied.
for pair in "64 1" "360 4"; do
  set -- $pair
  size=$1 frames=$2
  if [ "$size" = "64" ]; then input="$INPUT_64"; else input="$INPUT_360"; fi
  base="$TMP/default_$size.dav.y4m"
  ibp="$TMP/ibp_$size.dav.y4m"
  if [ ! -f "$base" ] || [ ! -f "$ibp" ]; then
    fail_case "dc_live_$size" "decoded pictures missing"
  elif ! y4m_check differ "$base" "$ibp" >"$TMP/dc_live_$size.log" 2>&1; then
    fail_case "dc_live_$size" "IBP decode equals the enable_ibp = 0 decode"
    cat "$TMP/dc_live_$size.log" >&2
  else
    pass_case "dc_live_$size ($(cat "$TMP/dc_live_$size.log"))"
  fi
done

# Named rejections.
expect_reject() {
  local name=$1 pattern=$2
  shift 2
  if "$AV2ENC_BIN" "$@" >/dev/null 2>"$TMP/$name.err"; then
    fail_case "$name" "expected a rejection"
  elif grep -q "$pattern" "$TMP/$name.err"; then
    pass_case "$name (named rejection)"
  else
    fail_case "$name" "rejection message changed"
    cat "$TMP/$name.err" >&2
  fi
}

expect_reject ibp_cfl_rejected "requires the plain DC chroma path" \
  "$INPUT_64" -o "$TMP/reject.ivf" --frames 1 --video --video-ibp --video-cfl
expect_reject ibp_mrl_rejected "does not compose with --video-mrl" \
  "$INPUT_64" -o "$TMP/reject.ivf" --frames 1 --video --video-ibp --video-mrl
expect_reject ibp_palette_rejected "is not wired with --video-ibp" \
  "$INPUT_64" -o "$TMP/reject.ivf" --frames 1 --video --video-ibp \
  --video-palette
expect_reject ibp_tiles_rejected "does not compose with --tiles" \
  "$INPUT_64" -o "$TMP/reject.ivf" --frames 1 --video --video-ibp \
  --tiles 2x1
expect_reject ibp_still_rejected "video options require --video" \
  "$INPUT_64" -o "$TMP/reject.ivf" --frames 1 --video-ibp

echo "video_ibp_gate: $passes passed, $failures failed"
if [ "$failures" -ne 0 ]; then exit 1; fi
