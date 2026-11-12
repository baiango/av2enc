#!/usr/bin/env bash
# Build and run the 256x256-superblock reference, the use_256 stream, and the
# dav2d gate.
#
#   ./src/codec/partition/sb256/build_and_test.sh
#   CXX=clang++ WERROR=1 SANITIZE=1 ./src/codec/partition/sb256/build_and_test.sh
#   SKIP_GATE=1 ./src/codec/partition/sb256/build_and_test.sh   # no dav2d
#   DAV2D=/path/to/dav2d ./src/codec/partition/sb256/build_and_test.sh
#
# Self-contained: does not touch the root CMake build or any other component.
# The target is Linux + CUDA; this CPU binary is the correctness oracle only.

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
CXX="${CXX:-c++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
SKIP_GATE="${SKIP_GATE:-0}"
OUT="${OUT:-${TMPDIR:-/tmp}/sb256_test}"

FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
# The generated CDF headers use flat initializers; this is a clang-only style
# warning, not a correctness issue.
FLAGS+=(-Wno-missing-braces)
if [[ "${SANITIZE}" == "1" ]]; then
  FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  FLAGS+=(-O2)
fi
if [[ "${WERROR}" == "1" ]]; then
  FLAGS+=(-Werror)
fi

COMMON=(
  "${HERE}/sb256.cpp"
  "${HERE}/sb256_stream.cpp"
  "${ROOT}/src/codec/partition/tree/partition.cpp"
  "${ROOT}/src/codec/partition/sb128/sb128.cpp"
  "${ROOT}/src/codec/partition/sb128/sb128_stream.cpp"
  "${ROOT}/src/core/bitstream/bitwriter.cpp"
  "${ROOT}/src/core/bitstream/basic_headers.cpp"
  "${ROOT}/src/codec/entropy/av2/av2_entropy.cpp"
)

"${CXX}" "${FLAGS[@]}" -I"${ROOT}/src" -I"${ROOT}/tests" -I"${HERE}" \
  "${COMMON[@]}" "${HERE}/sb256_test.cpp" \
  -o "${OUT}"
"${OUT}"

"${CXX}" "${FLAGS[@]}" -I"${ROOT}/src" -I"${HERE}" \
  "${HERE}/sb256.cpp" "${HERE}/sb256_stream.cpp" "${HERE}/sb256_gate.cpp" \
  "${ROOT}/src/core/bitstream/bitwriter.cpp" \
  "${ROOT}/src/core/bitstream/basic_headers.cpp" \
  "${ROOT}/src/core/io/ivf.cpp" \
  "${ROOT}/src/codec/entropy/av2/av2_entropy.cpp" \
  -o "${OUT}_gate"

GATE_DIR="${OUT}.gate"
rm -rf "${GATE_DIR}"
"${OUT}_gate" --out "${GATE_DIR}"

# Cross-process determinism: a second run must emit byte-identical streams.
GATE_DIR2="${OUT}.gate2"
rm -rf "${GATE_DIR2}"
"${OUT}_gate" --out "${GATE_DIR2}" >/dev/null
for ivf in "${GATE_DIR}"/*.ivf; do
  name="$(basename "${ivf}")"
  if ! cmp -s "${ivf}" "${GATE_DIR2}/${name}"; then
    echo "sb256 gate: non-deterministic stream ${name}" >&2
    exit 1
  fi
done
echo "sb256 gate: two independent runs are byte-identical"
rm -rf "${GATE_DIR2}"

if [[ "${SKIP_GATE}" == "1" ]]; then
  echo "sb256 gate: SKIP_GATE=1 (streams left in ${GATE_DIR})"
  exit 0
fi

if [[ -z "${DAV2D:-}" ]]; then
  for candidate in "${ROOT}/tools/bin/dav2d" "${ROOT}/tools/bin/dav2d-macos"; do
    if [[ -x "${candidate}" ]]; then
      DAV2D="${candidate}"
      break
    fi
  done
fi
if [[ -z "${DAV2D:-}" || ! -x "${DAV2D}" ]]; then
  echo "sb256 gate: dav2d not found; set DAV2D=/path/to/dav2d" >&2
  exit 127
fi

# Negative control: keep the 128-SB intra tile, clear use_256x256_superblock.
# The decoder then reads our enable_sdp=0 bit as use_128x128_superblock=0 and
# expects a 64x64-superblock tile, so it must reject the stream.
for name in sb256_256x256 sb256_640x360 sb256_1920x1080 sb256_520x300; do
  patched="${GATE_DIR}/${name}_use256_cleared.ivf"
  python3 "${HERE}/patch_use256_flag.py" "${GATE_DIR}/${name}.ivf" "${patched}"
  if "${DAV2D}" -q -i "${patched}" -o "${GATE_DIR}/${name}_use256_cleared.y4m" \
    --strict 1 --threads 1 >/dev/null 2>&1; then
    echo "sb256 gate: negative control ${name} decoded; the 256 flag is not parsed" >&2
    exit 1
  fi
  echo "sb256 gate: negative control ${name} rejected (use_256x256_superblock is parsed)"
done

check_recon() {
  local name="$1"
  local width="$2"
  local height="$3"
  local ivf="${GATE_DIR}/${name}.ivf"
  local y4m="${GATE_DIR}/${name}.y4m"
  "${DAV2D}" -q -i "${ivf}" -o "${y4m}" --strict 1 --threads 1
  python3 - "${y4m}" "${name}" "${width}" "${height}" <<'PY'
import sys

path, name, width, height = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
with open(path, "rb") as f:
    data = f.read()
try:
    start = data.index(b"\n") + 1
    frame = data.index(b"FRAME", start)
    payload = data.index(b"\n", frame) + 1
except ValueError:
    raise SystemExit(f"{name}: malformed Y4M")
body = data[payload:]
expected_bytes = width * height * 3 // 2
if len(body) != expected_bytes:
    raise SystemExit(
        f"{name}: decoded {len(body)} bytes, expected {expected_bytes}"
    )
diffs = sum(1 for byte in body if byte != 128)
if diffs != 0:
    raise SystemExit(f"{name}: recon differs from the flat-128 source at {diffs} samples")
print(f"{name}: dav2d OK, {len(body)} bytes, recon 0 diffs (all planes 128)")
PY
}

check_recon sb256_256x256 256 256
check_recon sb256_640x360 640 360
check_recon sb256_1920x1080 1920 1080
check_recon sb256_520x300 520 300
check_recon sb256_control_use128_128x128 128 128

echo "sb256 gate: PASS"
