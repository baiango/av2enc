#!/usr/bin/env bash
# dav2d `--strict 1` gate for the 10-bit `src/formats/high_bitdepth/` still encoder.
#
#   ./gate_hbd_still.sh [case ...]        # default: 64x64 640x360 4k
#
# Each case:
#   1. builds a deterministic 10-bit 4:2:0 source (full 10-bit range);
#   2. encodes with hbd_encode to IVF + a C420p10 reconstruction;
#   3. runs the project gate `tools/validate_dav2d.sh`;
#   4. decodes explicitly with `dav2d --strict 1`;
#   5. requires the decoded Y4M to be 420p10 with the IVF dimensions and every
#      visible sample to equal the encoder reconstruction (0 LSB);
#   6. for the 64x64 generator case, checks the pinned IVF MD5.
#
# Env:
#   HBD_ENCODE=/path/to/hbd_encode   else ${TMPDIR}/av2enc_hbd_build/hbd_encode
#   DAV2D=/path/to/dav2d             else tools/bin/dav2d[-macos]
#   THREADS=1                        decoder threads
#   KEEP=1                           keep the temp directory
#
# Exit codes: 0 ok, 1 failure, 2 usage, 127 no decoder/tool.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
BUILD_DIR="${BUILD_DIR:-${TMPDIR:-/tmp}/av2enc_hbd_build}"
HBD_ENCODE="${HBD_ENCODE:-${BUILD_DIR}/hbd_encode}"

if [ ! -x "${HBD_ENCODE}" ]; then
  echo "building hbd_encode (missing ${HBD_ENCODE})"
  "${HERE}/build_and_test.sh" >/dev/null
fi
if [ ! -x "${HBD_ENCODE}" ]; then
  echo "error: no hbd_encode at ${HBD_ENCODE}" >&2
  exit 127
fi

DAV2D="${DAV2D:-}"
if [ -z "${DAV2D}" ]; then
  for candidate in "${ROOT}/tools/bin/dav2d" "${ROOT}/tools/bin/dav2d-macos"; do
    if [ -x "${candidate}" ]; then DAV2D="${candidate}"; break; fi
  done
fi
if [ -z "${DAV2D}" ] || [ ! -x "${DAV2D}" ]; then
  echo "error: dav2d not found; set DAV2D=/path/to/dav2d" >&2
  exit 127
fi

CASES=("$@")
if [ ${#CASES[@]} -eq 0 ]; then
  # The three required depth-gate sizes, plus a q=64 case that proves the
  # qlookup-aware quantizer/dequant pair (step 248) still reconstructs exactly.
  CASES=(64x64 640x360 4k 64x64-q64)
fi

TMP="$(mktemp -d)"
cleanup() {
  if [ "${KEEP:-0}" = "1" ]; then
    echo "kept: ${TMP}"
  else
    rm -rf "${TMP}"
  fi
}
trap cleanup EXIT

# Deterministic 10-bit 4:2:0 source.  The 64x64/640x360 pattern is byte-for-byte
# the M10 generator (so the 64x64 q=90 stream is the pinned M10 golden); the
# 4K case uses NumPy to keep generation fast and still exercises the low two
# bits.  A pure-Python fallback is available if NumPy is absent.
gen_source() {
  local w=$1 h=$2 out=$3
  # Use the project M10 generator for the small cases: it is byte-for-byte
  # the source behind the pinned 64x64 M10 golden.  The 4K case is too slow in
  # scalar Python, so use NumPy there (still full 10-bit range).
  if [ "$((w * h))" -le 409600 ]; then
    "${ROOT}/src/pipeline/still/high_bitdepth/gen_y4m_10bit.py" "$w" "$h" "$out" >/dev/null
    return
  fi
  if python3 -c "import numpy" >/dev/null 2>&1; then
    python3 - "$w" "$h" "$out" <<'PY'
import sys
import numpy as np

w, h, path = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
rng = np.random.default_rng(0x9E3779B9)
yy, xx = np.mgrid[0:h, 0:w].astype(np.uint32)
r = rng.integers(0, 64, size=(h, w), dtype=np.uint32)
y = (xx * np.uint32(5) + yy * np.uint32(3) +
     ((xx ^ yy) << np.uint32(1)) + r) & np.uint32(1023)
cw, ch = (w + 1) // 2, (h + 1) // 2
cy, cx = np.mgrid[0:ch, 0:cw].astype(np.uint32)
ru = rng.integers(0, 32, size=(ch, cw), dtype=np.uint32)
rv = rng.integers(0, 32, size=(ch, cw), dtype=np.uint32)
u = (cx * np.uint32(11) + cy * np.uint32(7) + ru) & np.uint32(1023)
v = (cx * np.uint32(3) + cy * np.uint32(13) + rv) & np.uint32(1023)
with open(path, "wb") as f:
    f.write(f"YUV4MPEG2 W{w} H{h} F1:1 Ip A0:0 C420p10\n".encode())
    f.write(b"FRAME\n")
    f.write(y.astype("<u2").tobytes())
    f.write(u.astype("<u2").tobytes())
    f.write(v.astype("<u2").tobytes())
PY
  else
    "${ROOT}/src/pipeline/still/high_bitdepth/gen_y4m_10bit.py" "$w" "$h" "$out" >/dev/null
  fi
}

# 0-LSB comparison of the decoder output against the encoder reconstruction.
check_recon() {
  python3 - "$1" "$2" "$3" <<'PY'
import re
import struct
import sys

ivf, decoded, recon = sys.argv[1], sys.argv[2], sys.argv[3]

def one_frame(path):
    data = open(path, "rb").read()
    end = data.find(b"\n")
    if end < 0 or not data.startswith(b"YUV4MPEG2 "):
        raise SystemExit(f"error: {path}: not a Y4M file")
    header = data[:end].decode("latin1")
    wm = re.search(r"\bW(\d+)", header)
    hm = re.search(r"\bH(\d+)", header)
    cm = re.search(r"\bC([^\s]+)", header)
    if not (wm and hm and cm):
        raise SystemExit(f"error: {path}: header lacks W/H/C")
    width, height, chroma = int(wm.group(1)), int(hm.group(1)), cm.group(1)
    cw, ch = (width + 1) // 2, (height + 1) // 2
    frame_bytes = 2 * (width * height + 2 * cw * ch)
    pos = end + 1
    if data[pos:pos + 6] != b"FRAME\n":
        raise SystemExit(f"error: {path}: missing FRAME tag")
    pos += 6
    body = data[pos:pos + frame_bytes]
    if len(body) != frame_bytes:
        raise SystemExit(f"error: {path}: truncated frame")
    return width, height, chroma, body

dw, dh, dc, dbody = one_frame(decoded)
rw, rh, rc, rbody = one_frame(recon)
if (dw, dh, dc) != (rw, rh, rc):
    raise SystemExit(
        f"error: decoded {dw}x{dh} {dc} != recon {rw}x{rh} {rc}")
if dc != "420p10":
    raise SystemExit(f"error: decoded chroma {dc}, expected 420p10")
raw = open(ivf, "rb").read(32)
if len(raw) >= 32 and raw[:4] == b"DKIF":
    iw, ih = struct.unpack_from("<HH", raw, 12)
    if (iw, ih) != (dw, dh):
        raise SystemExit(f"error: IVF {iw}x{ih} != decoded {dw}x{dh}")
if dbody != rbody:
    n = min(len(dbody), len(rbody))
    first = next((i for i in range(n) if dbody[i] != rbody[i]), n)
    raise SystemExit(
        f"error: {first // 2}-sample LSB mismatch at sample {first // 2} "
        f"(byte {first})")
print(f"recon:   {dw}x{dh} {dc}, 0-LSB mismatch (all planes)")
PY
}

md5_of() {
  if command -v md5sum >/dev/null 2>&1; then
    md5sum "$1" | awk '{print $1}'
  else
    md5 -q "$1"
  fi
}

echo "decoder: $("${DAV2D}" --version 2>&1 | tail -1)"
echo "encoder: ${HBD_ENCODE}"

for case in "${CASES[@]}"; do
  case "${case}" in
    64x64)  W=64;   H=64;   Q=90; EXPECT="1b5018610a86b8d79aba161e95770ae6" ;;
    64x64-q64) W=64; H=64;  Q=64; EXPECT="" ;;
    640x360) W=640; H=360;  Q=90; EXPECT="" ;;
    4k|3840x2160) W=3840; H=2160; Q=90; EXPECT="" ;;
    *) echo "error: unknown case '${case}' (use 64x64, 64x64-q64, 640x360, 4k)" >&2; exit 2 ;;
  esac
  SRC="${TMP}/${case}.y4m"
  IVF="${TMP}/${case}.ivf"
  RECON="${TMP}/${case}.recon.y4m"
  DEC="${TMP}/${case}.decoded.y4m"

  echo
  echo "=== ${case} ==="
  gen_source "${W}" "${H}" "${SRC}"
  "${HBD_ENCODE}" --q "${Q}" "${SRC}" "${IVF}" "${RECON}"
  got_md5=$(md5_of "${IVF}")
  if [ -n "${EXPECT}" ]; then
    if [ "${got_md5}" != "${EXPECT}" ]; then
      echo "error: ${case} IVF MD5 ${got_md5} != pinned ${EXPECT}" >&2
      exit 1
    fi
    echo "md5:     ${got_md5} (pinned)"
  else
    echo "md5:     ${got_md5}"
  fi

  # Project gate (also runs the dav2d decode with strict defaults).
  THREADS="${THREADS:-1}" "${ROOT}/tools/validate_dav2d.sh" "${IVF}" "${SRC}"

  # Explicit strict decode + exact reconstruction comparison.
  "${DAV2D}" -q --strict 1 -i "${IVF}" -o "${DEC}" --threads "${THREADS:-1}"
  check_recon "${IVF}" "${DEC}" "${RECON}"
done

echo
echo "OK: 10-bit hbd stills passed dav2d --strict 1 with 0-LSB reconstruction"
