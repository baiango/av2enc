#!/usr/bin/env bash
# Build and run the chroma-format unit test.
#
# Usage: ./build_and_test.sh
#
# Artifacts go to ${TMPDIR:-/tmp}/av2enc_chroma_build.  This component does not
# modify the root CMakeLists.txt or the M0/M1/M2 builds; it compiles the four
# format files together with the existing bit writer, M0 header framing, format
# conversion, and entropy coder.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
BUILD="${TMPDIR:-/tmp}/av2enc_chroma_build"
mkdir -p "$BUILD"

CXX="${CXX:-}"
if [ -z "$CXX" ]; then
    for candidate in clang++ g++; do
        if command -v "$candidate" >/dev/null 2>&1; then
            CXX=$candidate
            break
        fi
    done
fi
if [ -z "$CXX" ]; then
    echo "error: no C++ compiler found (set CXX)" >&2
    exit 127
fi

COMMON=(
    -std=c++20 -O2 -I "$ROOT/src"
    "$ROOT/src/formats/chroma/chroma_format.cpp"
    "$ROOT/src/formats/chroma/chroma_residual.cpp"
    "$ROOT/src/formats/chroma/chroma_headers.cpp"
    "$ROOT/src/formats/chroma/chroma_still.cpp"
    "$ROOT/src/formats/format.cpp"
    "$ROOT/src/core/bitstream/bitwriter.cpp"
    "$ROOT/src/core/bitstream/basic_headers.cpp"
    "$ROOT/src/codec/entropy/av2/av2_entropy.cpp"
)

"$CXX" "${COMMON[@]}" "$ROOT/src/formats/chroma/chroma_test.cpp" \
    -o "$BUILD/chroma_test"
"$BUILD/chroma_test"

# Build the gate tool as well, so callers can run gate.sh without a second
# compile.
"$CXX" "${COMMON[@]}" "$ROOT/src/formats/chroma/chroma_still_main.cpp" \
    "$ROOT/src/core/io/y4m.cpp" "$ROOT/src/core/io/ivf.cpp" \
    -o "$BUILD/chroma_still"
echo "built: $BUILD/chroma_still"
