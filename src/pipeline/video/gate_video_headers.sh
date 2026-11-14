#!/usr/bin/env bash
# Product gate for the opt-in CPU `--video` header surface:
#
#   --video-tier high        §5.4.1 seq_tier = 1 (level 4.0+)
#   --video-multi-sequence   §5.4.1 seq_header_id bank + §5.18.2
#                            seq_header_id_in_frame_header
#   --video-lcr-full         §5.8 global/local LCR and §5.10/§5.11
#                            global/local OPS before the key frame
#
# Every case decodes with the pinned dav2d (`--strict 1`) and requires the
# decoder-visible pictures to equal the encoder reconstruction sample by
# sample.  The LCR/OPS carrier additionally runs avmdec, which is the decoder
# that parses and conformance-checks OBU types 16/18 (pinned dav2d prints
# "Unknown OBU type 16/18" and ignores them).
#
# The streams are kept deterministic by a synthetic 320x192 clip; each case
# has a pinned IVF MD5 so the default stream and the three flag carriers cannot
# drift silently.  Structural checks parse the sequence/frame headers from the
# IVF so a flag that emits no bits fails instead of passing on a decode.
#
# Usage: gate_video_headers.sh --bin <av2enc> [--dav2d PATH] [--avmdec PATH]
#                               [--frames N]
# Exits 77 when a required decoder or python3 is unavailable.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${HERE}/../../.." && pwd)"
BIN=""
FRAMES=4
DAV2D="${DAV2D:-}"
AVMDEC="${AVMDEC:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bin) BIN="$2"; shift 2 ;;
    --frames) FRAMES="$2"; shift 2 ;;
    --dav2d) DAV2D="$2"; shift 2 ;;
    --avmdec) AVMDEC="$2"; shift 2 ;;
    -h|--help) sed -n '2,23p' "$0"; exit 0 ;;
    *) echo "gate_video_headers: unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$BIN" && -x "$BIN" ]] || {
  echo "gate_video_headers: missing --bin" >&2
  exit 2
}
if [[ -z "$DAV2D" ]]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [[ -x "$candidate" ]]; then DAV2D="$candidate"; break; fi
  done
fi
if [[ -z "$AVMDEC" ]]; then
  for candidate in "$ROOT/tools/bin/avmdec" \
                   "$ROOT/tools/bin/avm-macos/avmdec" \
                   /Users/arin/work/avm_src/build/avmdec; do
    if [[ -x "$candidate" ]]; then AVMDEC="$candidate"; break; fi
  done
fi
if [[ -z "$DAV2D" || ! -x "$DAV2D" ]]; then
  echo "gate_video_headers: SKIP (no pinned dav2d; set DAV2D)"
  exit 77
fi
if [[ -z "$AVMDEC" || ! -x "$AVMDEC" ]]; then
  echo "gate_video_headers: SKIP (no avmdec for the LCR/OPS carrier; set AVMDEC)"
  exit 77
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "gate_video_headers: SKIP (python3 not available)"
  exit 77
fi

TMP="$(mktemp -d "${TMPDIR:-/tmp}/gate_video_headers.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
CLIP="$TMP/clip320.y4m"

python3 - "$CLIP" "$FRAMES" <<'PY'
import sys

path, frames = sys.argv[1], int(sys.argv[2])
w, h = 320, 192
cw, ch = w // 2, h // 2

def sample(x, y, f, plane):
    sx = 8 + 5 * f
    sy = 10 + 2 * f
    inside = sx <= x < sx + 80 and sy <= y < sy + 64
    if plane == 0:
        if inside:
            return 40 + ((x * 29 + y * 17 + f * 41) & 0xBF)
        return 90 + (((x + 3 * f) * 5 + (y + 2 * f) * 3 +
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

fail() {
  echo "gate_video_headers: FAIL: $*" >&2
  exit 1
}

md5_of() {
  if command -v md5sum >/dev/null 2>&1; then
    md5sum "$1" | cut -d' ' -f1
  else
    md5 -q "$1"
  fi
}

# Pinned streams for this exact synthetic clip at $FRAMES=4.  They cover the
# historical default and the three opt-in carriers.
DEFAULT_PIN="35e075339868593302421bf3cdea8c98"
TIER_PIN="71395070d5ba81cad43c0f09335f29d2"
MULTI_PIN="46ae9dc3f2edb3884f21596c9fd06556"
LCR_PIN="000189a4533aecd825a1eb62aa2412db"

# parse_stream <ivf> <expect...>   Structural assertions in one Python pass.
#
#   default   : exactly one sequence header (id 0, tier 0, lcr 0), every frame
#               header selects id 0
#   tier      : one sequence header with level > 3 and seq_tier == 1
#   multi     : two sequence headers with ids 0 and 1 whose payloads differ,
#               every frame header selects id 1
#   lcr       : sequence header seq_lcr_id == 1; OBU 16/18 present for global
#               (xlayer 31) and local (xlayer 0)
parse_stream() {
  local ivf=$1
  local mode=$2
  python3 - "$ivf" "$mode" <<'PY'
import sys

ivf, mode = sys.argv[1], sys.argv[2]
data = open(ivf, "rb").read()
pos = 32
seq_headers = []
frame_ids = []
config_obus = []

class Bits:
    def __init__(self, payload):
        self.b = payload
        self.p = 0
    def bit(self):
        v = (self.b[self.p >> 3] >> (7 - (self.p & 7))) & 1
        self.p += 1
        return v
    def bits(self, n):
        v = 0
        for _ in range(n):
            v = (v << 1) | self.bit()
        return v
    def uvlc(self):
        zeros = 0
        while self.bit() == 0:
            zeros += 1
        return 0 if zeros == 0 else (1 << zeros) - 1 + self.bits(zeros)

while pos < len(data):
    size = int.from_bytes(data[pos:pos + 4], "little")
    pos += 4
    pos += 8  # timestamp
    payload = data[pos:pos + size]
    pos += size
    i = 0
    while i < len(payload):
        value = 0
        shift = 0
        while True:
            byte = payload[i]
            i += 1
            value |= (byte & 0x7F) << shift
            shift += 7
            if not (byte & 0x80):
                break
        obu = payload[i:i + value]
        i += value
        first = obu[0]
        extension = first >> 7
        obu_type = (first >> 2) & 0x1F
        offset = 1
        xlayer = 31 if obu_type in (2, 20) else 0
        if extension:
            xlayer = obu[1] & 0x1F
            offset = 2
        body = obu[offset:]
        if obu_type == 1:
            br = Bits(body)
            sid = br.uvlc()
            profile = br.bits(5)
            single = br.bit()
            level = br.bits(5)
            tier = br.bit() if level > 3 else 0
            chroma = br.uvlc()
            depth = br.uvlc()
            lcr = br.bits(3)
            still = br.bit()
            seq_headers.append({"id": sid, "level": level, "tier": tier,
                                "lcr": lcr, "payload": body})
            if single != 0:
                raise SystemExit("sequence header is single-picture")
        elif obu_type in (4, 7):
            br = Bits(body)
            first_tile = br.bit()
            mfh = br.uvlc()
            sid = br.uvlc()
            if first_tile != 1:
                raise SystemExit("is_first_tile_group != 1")
            if mfh != 0:
                raise SystemExit("unexpected MFH id")
            frame_ids.append(sid)
        elif obu_type in (16, 18):
            config_obus.append((obu_type, xlayer, len(body)))

if mode == "default":
    if len(seq_headers) != 1:
        raise SystemExit(f"expected 1 sequence header, got {len(seq_headers)}")
    sh = seq_headers[0]
    if (sh["id"], sh["tier"], sh["lcr"]) != (0, 0, 0):
        raise SystemExit(f"default header changed: {sh['id']},{sh['tier']},{sh['lcr']}")
    if frame_ids and set(frame_ids) != {0}:
        raise SystemExit(f"default frame ids {frame_ids}")
elif mode == "tier":
    if len(seq_headers) != 1:
        raise SystemExit(f"expected 1 sequence header, got {len(seq_headers)}")
    sh = seq_headers[0]
    if sh["level"] <= 3 or sh["tier"] != 1:
        raise SystemExit(f"seq_tier not signalled: level {sh['level']} tier {sh['tier']}")
    if frame_ids and set(frame_ids) != {0}:
        raise SystemExit(f"tier frame ids {frame_ids}")
elif mode == "multi":
    if len(seq_headers) != 2:
        raise SystemExit(f"expected 2 sequence headers, got {len(seq_headers)}")
    ids = sorted(sh["id"] for sh in seq_headers)
    if ids != [0, 1]:
        raise SystemExit(f"sequence header bank ids {ids}")
    if seq_headers[0]["payload"] == seq_headers[1]["payload"]:
        raise SystemExit("the two sequence headers are byte-identical; the "
                         "frame-header id selection would be vacuous")
    if not frame_ids or set(frame_ids) != {1}:
        raise SystemExit(f"frame headers select {frame_ids}, expected id 1")
elif mode == "lcr":
    if len(seq_headers) != 1:
        raise SystemExit(f"expected 1 sequence header, got {len(seq_headers)}")
    if seq_headers[0]["lcr"] != 1:
        raise SystemExit(f"seq_lcr_id is {seq_headers[0]['lcr']}, expected 1")
    present = {(t, x) for t, x, _ in config_obus}
    for want in [(16, 31), (18, 31), (16, 0), (18, 0)]:
        if want not in present:
            raise SystemExit(f"missing config OBU {want}; got {sorted(present)}")
    if any(n == 0 for _, _, n in config_obus):
        raise SystemExit("empty LCR/OPS payload")
    if frame_ids and set(frame_ids) != {0}:
        raise SystemExit(f"lcr frame ids {frame_ids}")
else:
    raise SystemExit(f"unknown mode {mode}")

print("  ", mode, "ok:", len(seq_headers), "sequence header(s),",
      len(frame_ids), "frame header(s),", len(config_obus), "LCR/OPS OBU(s)")
PY
}

encode_and_check() {
  local mode=$1
  local pin=$2
  shift 2
  local ivf="$TMP/$mode.ivf"
  local recon="$TMP/$mode.recon.y4m"
  local decoded="$TMP/$mode.decoded.y4m"

  "$BIN" "$CLIP" --video --frames "$FRAMES" --video-recon "$recon" \
    -o "$ivf" "$@" >"$TMP/$mode.enc.log" 2>&1 \
    || { cat "$TMP/$mode.enc.log" >&2; fail "$mode encode"; }

  parse_stream "$ivf" "$mode"

  "$DAV2D" -q -i "$ivf" -o "$decoded" --strict 1 --threads 1 \
    >"$TMP/$mode.dec.log" 2>&1 \
    || { cat "$TMP/$mode.dec.log" >&2; fail "$mode dav2d --strict 1 decode"; }
  python3 - "$recon" "$decoded" <<'PY'
import sys

def read_y4m(path):
    frames = []
    with open(path, "rb") as f:
        header = f.readline()
        if not header.startswith(b"YUV4MPEG2 "):
            raise SystemExit(f"{path}: not a Y4M file")
        width = height = None
        for field in header.split()[1:]:
            if field.startswith(b"W"):
                width = int(field[1:])
            elif field.startswith(b"H"):
                height = int(field[1:])
        cw, ch = (width + 1) // 2, (height + 1) // 2
        size = width * height + 2 * cw * ch
        while True:
            line = f.readline()
            if not line:
                break
            if not line.startswith(b"FRAME"):
                raise SystemExit(f"{path}: bad frame marker")
            chunk = f.read(size)
            if len(chunk) != size:
                raise SystemExit(f"{path}: truncated frame")
            frames.append(chunk)
    return frames

a = read_y4m(sys.argv[1])
b = read_y4m(sys.argv[2])
if len(a) != len(b):
    raise SystemExit(f"frame count {len(a)} != {len(b)}")
if a != b:
    raise SystemExit("decoder output differs from the encoder reconstruction")
PY

  local actual
  actual=$(md5_of "$ivf")
  if [[ "$actual" != "$pin" ]]; then
    fail "$mode stream MD5 $actual != pinned $pin"
  fi
  echo "  $mode: dav2d strict + recon exact, MD5 $actual"
}

echo "== video header surface: tier / multi-sequence / full LCR-OPS =="
echo "dav2d:  $DAV2D"
echo "avmdec: $AVMDEC ($("$AVMDEC" --help 2>&1 | head -1))"

echo
echo "default (main tier, single sequence, no LCR/OPS)"
encode_and_check default "$DEFAULT_PIN"

echo
echo "--video-tier high"
encode_and_check tier "$TIER_PIN" --video-tier high

echo
echo "--video-multi-sequence"
encode_and_check multi "$MULTI_PIN" --video-multi-sequence

echo
echo "--video-lcr-full"
encode_and_check lcr "$LCR_PIN" --video-lcr-full

# avmdec parses and conformance-checks OBU types 16/18; the pinned dav2d
# ignores them.  Compare the decoded pictures against dav2d's yuv output so
# the config OBUs are proven picture-neutral as well.
"$AVMDEC" --i420 -o "$TMP/lcr.avm.yuv" "$TMP/lcr.ivf" \
  >"$TMP/lcr.avm.log" 2>&1 \
  || { cat "$TMP/lcr.avm.log" >&2; fail "avmdec rejected the LCR/OPS carrier"; }
"$DAV2D" -q -i "$TMP/lcr.ivf" -o "$TMP/lcr.dav.yuv" --muxer yuv \
  --strict 1 --threads 1 >/dev/null 2>&1 \
  || fail "dav2d yuv decode of the LCR/OPS carrier"
cmp -s "$TMP/lcr.avm.yuv" "$TMP/lcr.dav.yuv" \
  || fail "avmdec and dav2d decoded different pictures"
echo "  avmdec accepted the full LCR/OPS fields and decoded the same pictures"

echo
echo "usage rejections stay explicit"
reject() {
  local label=$1 needle=$2
  shift 2
  if "$BIN" "$CLIP" -o "$TMP/reject.ivf" "$@" >/dev/null \
     2>"$TMP/reject.err"; then
    fail "$label: expected a rejection"
  fi
  if ! grep -qF -- "$needle" "$TMP/reject.err"; then
    cat "$TMP/reject.err" >&2
    fail "$label: error does not mention '$needle'"
  fi
  echo "  ok: $label"
}
reject "flags need --video" "--video-tier/--video-multi-sequence" \
  --video-tier high
reject "multi-sequence with MFH" "seq_header_id_in_frame_header" \
  --video --video-multi-sequence --video-mfh
reject "multi-sequence with one picture" "at least 2 input pictures" \
  --video --frames 1 --video-multi-sequence
reject "tier outside the plain chain" "plain 8-bit 4:2:0" \
  --video --b-frames 1 --video-tier high

echo
echo "PASS: header surface (tier, multi-sequence, full LCR/OPS) decoder-gated"
