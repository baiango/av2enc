#!/usr/bin/env bash
# Standalone build + tests + dav2d gate for src/frame/small_header/.
#
#   ./src/frame/small_header/build_and_test.sh
#   SANITIZE=1 WERROR=1 ./src/frame/small_header/build_and_test.sh
#   SKIP_GATE=1 ./src/frame/small_header/build_and_test.sh
#   AV2ENC=/path/to/av2enc DAV2D=/path/to/dav2d ./...build_and_test.sh
#
# No CMake changes and no CUDA. The gate consumes real M1 still-picture IVF
# streams from the existing `av2enc --still` path; only their sequence/frame
# headers are rewritten, the tile payload is untouched.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
CXX="${CXX:-c++}"
BUILD="${OUT:-${ROOT}/build/small_header}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
SKIP_GATE="${SKIP_GATE:-0}"

mkdir -p "$BUILD"

FLAGS=(-std=c++20 -I"${ROOT}/src" -I"${ROOT}/tests")
if [[ "${SANITIZE}" == "1" ]]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2)
fi
FLAGS+=(-Wall -Wextra -Wpedantic -Wconversion -Wshadow)
if [[ "${WERROR}" == "1" ]]; then
  FLAGS+=(-Werror)
fi

COMMON=(
  "${HERE}/small_header.cpp"
  "${ROOT}/src/core/bitstream/bitwriter.cpp"
)

echo "== build small_header_test"
"${CXX}" "${FLAGS[@]}" "${COMMON[@]}" "${HERE}/small_header_test.cpp" \
  -o "${BUILD}/small_header_test"
echo "== run small_header_test"
"${BUILD}/small_header_test"

echo "== build small_header_gate"
"${CXX}" "${FLAGS[@]}" "${COMMON[@]}" "${HERE}/gate.cpp" \
  -o "${BUILD}/small_header_gate"

if [[ "${SKIP_GATE}" == "1" ]]; then
  echo "== dav2d gate skipped (SKIP_GATE=1)"
  exit 0
fi

# --- real M1 encoder -------------------------------------------------------
if [[ -z "${AV2ENC:-}" ]]; then
  for candidate in "${ROOT}/build/av2enc" "${ROOT}/build-m1/av2enc"; do
    if [[ -x "${candidate}" ]]; then
      AV2ENC="${candidate}"
      break
    fi
  done
fi
if [[ -z "${AV2ENC:-}" || ! -x "${AV2ENC}" ]]; then
  echo "gate: no av2enc binary; build it first or set AV2ENC=..." >&2
  exit 127
fi

# --- pinned decoder --------------------------------------------------------
if [[ -z "${DAV2D:-}" ]]; then
  for candidate in "${ROOT}/tools/bin/dav2d" "${ROOT}/tools/bin/dav2d-macos"; do
    if [[ -x "${candidate}" ]]; then
      DAV2D="${candidate}"
      break
    fi
  done
fi
if [[ -z "${DAV2D:-}" || ! -x "${DAV2D}" ]]; then
  echo "gate: no dav2d binary; set DAV2D=..." >&2
  exit 127
fi

SRC64="${ROOT}/tests/data/basic_64x64_1f.y4m"
SRC360="${ROOT}/tests/data/basic_640x360_1f.y4m"
for src in "$SRC64" "$SRC360"; do
  if [[ ! -f "$src" ]]; then
    echo "gate: missing source $src" >&2
    exit 1
  fi
done

GATE_DIR="${BUILD}/gate"
rm -rf "$GATE_DIR"
mkdir -p "$GATE_DIR"

# Single-block M1 streams keep every prediction block square, so the reduced
# MaxPbAspectRatio feature is exercised on a stream whose existing partition
# tree is already conformant. The full-tree stream is used below as a negative
# control: reducing MaxPbAspectRatio makes dav2d reject its PARTITION_NONE
# syntax, which proves the decoder is really enforcing the feature.
echo "== encode real M1 baselines (--still --no-deblock --still-single)"
"$AV2ENC" "$SRC64" -o "${GATE_DIR}/still_64s.ivf" --still --no-deblock --still-single \
  >/dev/null
"$AV2ENC" "$SRC360" -o "${GATE_DIR}/still_360s.ivf" --still --no-deblock --still-single \
  >/dev/null

echo "== emit feature streams"
"${BUILD}/small_header_gate" --baseline "${GATE_DIR}/still_64s.ivf" \
  --out-dir "$GATE_DIR" --prefix still_64s --width 64 --height 64 --qp 90 \
  --deblock 0 --single-block
"${BUILD}/small_header_gate" --baseline "${GATE_DIR}/still_360s.ivf" \
  --out-dir "$GATE_DIR" --prefix still_360s --width 640 --height 360 --qp 90 \
  --deblock 0 --single-block

echo "== decode with pinned dav2d --strict 1 --threads 1"
"$DAV2D" -q -i "${GATE_DIR}/still_64s_baseline.ivf" \
  -o "${GATE_DIR}/still_64s_baseline.y4m" --strict 1 --threads 1
"$DAV2D" -q -i "${GATE_DIR}/still_360s_baseline.ivf" \
  -o "${GATE_DIR}/still_360s_baseline.y4m" --strict 1 --threads 1
for prefix in still_64s still_360s; do
  while read -r name _file _w _h _cx _cy; do
    [[ "$name" == "baseline" ]] && continue
    "$DAV2D" -q -i "${GATE_DIR}/${prefix}_${name}.ivf" \
      -o "${GATE_DIR}/${prefix}_${name}.y4m" --strict 1 --threads 1
  done < "${GATE_DIR}/${prefix}_manifest.txt"
done

echo "== reconstruction diff check (0 required)"
python3 - "$GATE_DIR" <<'PY'
import sys

gate_dir = sys.argv[1]


def read_y4m(path):
    data = open(path, "rb").read()
    newline = data.index(b"\n")
    parts = data[:newline].decode("ascii").split()
    width = int([x[1:] for x in parts if x.startswith("W")][0])
    height = int([x[1:] for x in parts if x.startswith("H")][0])
    pos = data.index(b"\n", newline + 1) + 1
    y = data[pos:pos + width * height]
    pos += width * height
    cw, ch = width // 2, height // 2
    u = data[pos:pos + cw * ch]
    pos += cw * ch
    v = data[pos:pos + cw * ch]
    return width, height, y, u, v


def crop(plane, stride, x, y, w, h):
    return b"".join(plane[(y + row) * stride + x:(y + row) * stride + x + w]
                    for row in range(h))


failures = 0
for prefix in ("still_64s", "still_360s"):
    bw, bh, by, bu, bv = read_y4m(f"{gate_dir}/{prefix}_baseline.y4m")
    for line in open(f"{gate_dir}/{prefix}_manifest.txt"):
        name, _file, w, h, cx, cy = line.split()
        if name == "baseline":
            continue
        w, h, cx, cy = int(w), int(h), int(cx), int(cy)
        fw, fh, y, u, v = read_y4m(f"{gate_dir}/{prefix}_{name}.y4m")
        if (fw, fh) == (bw, bh):
            # Pinned dav2d currently emits the coded (possibly padded) frame
            # even when seq_cropping_window_present_flag is 1. The cropping
            # window is still fully parsed and constrained; compare the whole
            # frame in that case.
            cx = cy = 0
            w, h = bw, bh
        elif (fw, fh) != (w, h):
            print(f"FAIL {prefix}/{name}: decoder output {fw}x{fh}, "
                  f"expected {w}x{h} or {bw}x{bh}")
            failures += 1
            continue
        dy = sum(a != b for a, b in zip(
            crop(by, bw, cx, cy, w, h), y))
        du = sum(a != b for a, b in zip(
            crop(bu, bw // 2, cx // 2, cy // 2, w // 2, h // 2), u))
        dv = sum(a != b for a, b in zip(
            crop(bv, bw // 2, cx // 2, cy // 2, w // 2, h // 2), v))
        total = dy + du + dv
        print(f"{prefix}/{name}: output {fw}x{fh}, recon diffs "
              f"Y={dy} U={du} V={dv}")
        if total != 0:
            failures += 1

if failures:
    raise SystemExit(f"gate: {failures} feature stream(s) differ")
print("gate: all feature streams decoded with 0 reconstruction diffs")
PY

# --- negative control: the aspect-ratio feature is not a no-op -------------
echo "== negative control: reduced MaxPbAspectRatio rejects a full M1 tree"
"$AV2ENC" "$SRC64" -o "${GATE_DIR}/still_64_full.ivf" --still --no-deblock >/dev/null
"${BUILD}/small_header_gate" --baseline "${GATE_DIR}/still_64_full.ivf" \
  --out-dir "${GATE_DIR}" --prefix still_64_full --width 64 --height 64 --qp 90 \
  --deblock 0 >/dev/null
if "$DAV2D" -q -i "${GATE_DIR}/still_64_full_aspect.ivf" \
    -o "${GATE_DIR}/still_64_full_aspect.y4m" --strict 1 --threads 1 \
    >/dev/null 2>&1; then
  echo "gate: FAIL: dav2d accepted a 4:1 MaxPbAspectRatio stream whose M1" \
       "tree violates the reduced limit" >&2
  exit 1
fi
echo "gate: negative control rejected as expected (reduce_pb_aspect_ratio parsed)"

echo "gate: PASS"
