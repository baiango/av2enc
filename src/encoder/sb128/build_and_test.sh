#!/usr/bin/env bash
# Build and run the 128x128-superblock reference, the minimal 128-SB stream,
# and the dav2d gate.
#
#   ./src/encoder/sb128/build_and_test.sh
#   CXX=clang++ WERROR=1 SANITIZE=1 ./src/encoder/sb128/build_and_test.sh
#   SKIP_GATE=1 ./src/encoder/sb128/build_and_test.sh   # no dav2d
#   DAV2D=/path/to/dav2d ./src/encoder/sb128/build_and_test.sh
#
# Self-contained: does not touch the root CMake build or any other component.
# The target is Linux + CUDA; this CPU binary is the correctness oracle only.

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${HERE}/../../.." && pwd)"
CXX="${CXX:-c++}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
SKIP_GATE="${SKIP_GATE:-0}"
OUT="${OUT:-${TMPDIR:-/tmp}/sb128_test}"

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
  "${HERE}/sb128.cpp"
  "${HERE}/sb128_stream.cpp"
  "${ROOT}/src/encoder/partition/partition.cpp"
  "${ROOT}/src/bitstream/bitwriter.cpp"
  "${ROOT}/src/bitstream/m0_headers.cpp"
  "${ROOT}/src/entropy/av2/av2_entropy.cpp"
)

"${CXX}" "${FLAGS[@]}" -I"${ROOT}/src" -I"${ROOT}/tests" -I"${HERE}" \
  "${COMMON[@]}" "${HERE}/sb128_test.cpp" \
  -o "${OUT}"
"${OUT}"

"${CXX}" "${FLAGS[@]}" -I"${ROOT}/src" -I"${HERE}" \
  "${HERE}/sb128.cpp" "${HERE}/sb128_stream.cpp" "${HERE}/sb128_gate.cpp" \
  "${ROOT}/src/bitstream/bitwriter.cpp" \
  "${ROOT}/src/bitstream/m0_headers.cpp" \
  "${ROOT}/src/io/ivf.cpp" \
  "${ROOT}/src/entropy/av2/av2_entropy.cpp" \
  -o "${OUT}_gate"

GATE_DIR="${OUT}.gate"
rm -rf "${GATE_DIR}"
"${OUT}_gate" --out "${GATE_DIR}"

if [[ "${SKIP_GATE}" == "1" ]]; then
  echo "sb128 gate: SKIP_GATE=1 (streams left in ${GATE_DIR})"
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
  echo "sb128 gate: dav2d not found; set DAV2D=/path/to/dav2d" >&2
  exit 127
fi

# Negative control: the same tile under a 64x64-SB header must be rejected.
patched="${GATE_DIR}/sb128_128x128_use64.ivf"
python3 "${HERE}/patch_use128_flag.py" "${GATE_DIR}/sb128_128x128.ivf" \
  "${patched}"
if "${DAV2D}" -q -i "${patched}" -o "${GATE_DIR}/sb128_128x128_use64.y4m" \
  --threads 1 >/dev/null 2>&1; then
  echo "sb128 gate: negative control decoded; the SB flag is not gated" >&2
  exit 1
fi
echo "sb128 gate: negative control rejected (use_128x128_superblock is parsed)"

for name in sb128_128x128 sb128_256x256 sb128_640x360 sb128_1920x1080; do
  ivf="${GATE_DIR}/${name}.ivf"
  y4m="${GATE_DIR}/${name}.y4m"
  "${DAV2D}" -q -i "${ivf}" -o "${y4m}" --threads 1
  python3 - "${y4m}" "${name}" <<'PY'
import sys
path, name = sys.argv[1], sys.argv[2]
with open(path, "rb") as f:
    data = f.read()
try:
    start = data.index(b"\n") + 1
    frame = data.index(b"FRAME", start)
    payload = data.index(b"\n", frame) + 1
except ValueError:
    raise SystemExit(f"{name}: malformed Y4M")
body = data[payload:]
if not body or any(byte != 128 for byte in body):
    first = next((i for i, byte in enumerate(body) if byte != 128), -1)
    raise SystemExit(f"{name}: decoded pixels are not all 128 (first diff {first})")
print(f"{name}: dav2d OK, {len(body)} bytes, all planes 128")
PY
done

echo "sb128 gate: PASS"
