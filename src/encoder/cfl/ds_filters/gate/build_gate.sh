#!/usr/bin/env bash
# Build the non-zero-cfl_ds_filter_index decoder gate.
#
# The production M1 writer hardcodes cfl_ds_filter_index = 0 and
# Block.ds_filter = 0 (src/encoder/m1/m1_headers.cpp, m1_still.cpp).  M1 is
# owned by another agent, so this gate does NOT edit those files.  Instead it
# copies the two translation units into this component's build directory,
# applies a two-line instrumentation patch to the *copies*, and links them in
# front of the existing libav2enc_core.a.  The repository sources are never
# modified.
#
# The patch:
#   * make_cfl_block(): Block.ds_filter reads AV2ENC_CFL_DS_FILTER
#   * write_sequence_header(): cfl_ds_filter_index reads the same variable
#
# Usage:
#   ./src/encoder/cfl/ds_filters/gate/build_gate.sh
#   AV2ENC_BUILD_DIR=build-cuda ./.../build_gate.sh
#
# Requirements: an existing $AV2ENC_BUILD_DIR/libav2enc_core.a (cmake build of
# av2enc_core), a C++20 compiler, and no CUDA.

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../../../../.." && pwd)"
BUILD_DIR="${AV2ENC_BUILD_DIR:-$ROOT/build}"
LIB="$BUILD_DIR/libav2enc_core.a"
OUT="$HERE/build"
CXX="${CXX:-c++}"

if [ ! -f "$LIB" ]; then
  echo "build_gate: missing $LIB (build the av2enc_core target first)" >&2
  exit 2
fi

mkdir -p "$OUT"
cp "$ROOT/src/encoder/m1/m1_still.cpp" "$OUT/m1_still.cpp"
cp "$ROOT/src/encoder/m1/m1_headers.cpp" "$OUT/m1_headers.cpp"

python3 - "$OUT" <<'PY'
import sys
from pathlib import Path

out = Path(sys.argv[1])

still = out / "m1_still.cpp"
text = still.read_text()
old = "    block.ds_filter = 0;"
new = (
    "    block.ds_filter = []() { "
    "const char* e = std::getenv(\"AV2ENC_CFL_DS_FILTER\"); "
    "return e ? std::atoi(e) : 0; }();"
)
if text.count(old) != 1:
    raise SystemExit(f"build_gate: expected one ds_filter assignment in {still}")
still.write_text(text.replace(old, new))

headers = out / "m1_headers.cpp"
text = headers.read_text()
old = "  writer.put_bits(0U, 2U);  // cfl_ds_filter_index"
new = (
    "  { const char* e = std::getenv(\"AV2ENC_CFL_DS_FILTER\"); "
    "const unsigned v = e ? static_cast<unsigned>(std::atoi(e) & 3) : 0U; "
    "writer.put_bits(v, 2U); }  // cfl_ds_filter_index"
)
if text.count(old) != 1:
    raise SystemExit(f"build_gate: expected one cfl_ds_filter_index write in {headers}")
text = text.replace(old, new)
text = text.replace("#include <cassert>", "#include <cassert>\n#include <cstdlib>", 1)
headers.write_text(text)
print("build_gate: instrumentation patch applied to build copies")
PY

ARCH_FLAGS=()
if [ "$(uname -s)" = "Darwin" ]; then
  ARCH_FLAGS=(-arch "$(uname -m)")
fi

FLAGS=(-std=c++20 -O3 -DNDEBUG -ffp-contract=off -I"$ROOT/src"
       -DAV2ENC_HAVE_CUDA=0 -DAV2ENC_HAVE_M1_GPU=0 "${ARCH_FLAGS[@]}")

"$CXX" "${FLAGS[@]}" -c "$OUT/m1_still.cpp" -o "$OUT/m1_still.o"
"$CXX" "${FLAGS[@]}" -c "$OUT/m1_headers.cpp" -o "$OUT/m1_headers.o"

"$CXX" "${FLAGS[@]}" "$ROOT/src/main.cpp" \
  "$OUT/m1_still.o" "$OUT/m1_headers.o" "$LIB" -o "$OUT/av2enc_ds"

"$CXX" "${FLAGS[@]}" -I"$ROOT/src" "$ROOT/tools/m1_verify/m1_dump.cpp" \
  "$OUT/m1_still.o" "$OUT/m1_headers.o" "$LIB" -o "$OUT/m1_dump_ds"

echo "build_gate: wrote $OUT/av2enc_ds and $OUT/m1_dump_ds"
