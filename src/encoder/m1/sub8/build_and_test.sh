#!/usr/bin/env bash
# Standalone build + tests for src/encoder/m1/sub8/.
#
#   ./src/encoder/m1/sub8/build_and_test.sh
#   SANITIZE=1 WERROR=1 ./src/encoder/m1/sub8/build_and_test.sh
#   SKIP_GATE=1 ./src/encoder/m1/sub8/build_and_test.sh     # no dav2d
#   REPRO=1 ./src/encoder/m1/sub8/build_and_test.sh         # production M1 probe
#
# Builds outside the root CMake project so the CUDA-only target is not needed
# on macOS.  The unit test and gate use exactly the same sources the M1 build
# would: partition.cpp, m1_headers.cpp, bitstream/* and entropy/av2.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
SRC="$ROOT/src"
BUILD="$ROOT/build/sub8"
CXX=${CXX:-c++}
mkdir -p "$BUILD"

COMMON_FLAGS=(-std=c++20 -I"$SRC" -O2 -g)
WARN_FLAGS=(-Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wno-missing-braces)
if [ "${WERROR:-0}" = "1" ]; then
    WARN_FLAGS+=(-Werror)
fi
if [ "${SANITIZE:-0}" = "1" ]; then
    WARN_FLAGS+=(-fsanitize=address,undefined -fno-omit-frame-pointer)
fi

CORE=(
    "$SRC/io/ivf.cpp"
    "$SRC/bitstream/bitwriter.cpp"
    "$SRC/bitstream/obu.cpp"
    "$SRC/bitstream/m0_headers.cpp"
    "$SRC/encoder/partition/partition.cpp"
    "$SRC/encoder/m1/m1_headers.cpp"
    "$SRC/encoder/tx_variants/tx_variants.cpp"
    "$SRC/entropy/av2/av2_entropy.cpp"
    "$SRC/encoder/m1/sub8/sub8.cpp"
    "$SRC/encoder/m1/sub8/sub8_syntax.cpp"
    "$SRC/encoder/m1/sub8/sub8_minimal.cpp"
)

echo "== building sub8_test"
"$CXX" "${COMMON_FLAGS[@]}" "${WARN_FLAGS[@]}" \
    "${CORE[@]}" "$SRC/encoder/m1/sub8/sub8_test.cpp" -o "$BUILD/sub8_test"
echo "== running sub8_test"
"$BUILD/sub8_test"

echo "== building gate_minimal"
"$CXX" "${COMMON_FLAGS[@]}" "${WARN_FLAGS[@]}" \
    "${CORE[@]}" "$SRC/encoder/m1/sub8/gate_minimal.cpp" -o "$BUILD/gate_minimal"

GATE_DIR="$BUILD/gate"
rm -rf "$GATE_DIR"
mkdir -p "$GATE_DIR"
echo "== writing gate streams"
"$BUILD/gate_minimal" "$GATE_DIR"

if [ "${SKIP_GATE:-0}" = "1" ]; then
    echo "== dav2d gate skipped (SKIP_GATE=1)"
else
    echo "== dav2d gate"
    python3 "$SRC/encoder/m1/sub8/gate_dav2d.py" --dir "$GATE_DIR"
fi

if [ "${REPRO:-0}" = "1" ]; then
    echo "== building production M1 sub-8 probe (diagnostic)"
    M1_CORE=(
        "$SRC/io/y4m.cpp" "$SRC/io/ivf.cpp"
        "$SRC/bitstream/bitwriter.cpp" "$SRC/bitstream/obu.cpp"
        "$SRC/bitstream/m0_headers.cpp" "$SRC/encoder/m0_still.cpp"
        "$SRC/encoder/partition/partition.cpp"
        "$SRC/encoder/intra_modes/intra_modes.cpp"
        "$SRC/encoder/rdoq/rdoq.cpp" "$SRC/encoder/rdoq/rdoq_av2_model.cpp"
        "$SRC/encoder/tx_type/tx_type.cpp"
        "$SRC/encoder/tx_variants/tx_variants.cpp"
        "$SRC/encoder/tx_variants/tx_select.cpp"
        "$SRC/encoder/cfl/cfl.cpp"
        "$SRC/encoder/filters/deblock/deblock.cpp"
        "$SRC/encoder/filters/cdef/cdef.cpp"
        "$SRC/encoder/filters/restoration/restoration.cpp"
        "$SRC/encoder/rdo/rdo.cpp" "$SRC/encoder/rdo/intra_decision.cpp"
        "$SRC/encoder/m1/m1_headers.cpp" "$SRC/encoder/m1/m1_syntax.cpp"
        "$SRC/encoder/m1/m1_still.cpp"
        "$SRC/encoder/m1/stream/m1_stream_bridge.cpp"
        "$SRC/entropy/av2/av2_entropy.cpp" "$SRC/profile/profiler.cpp"
        "$SRC/cuda/m1/cdef/m1_cdef.cpp"
    )
    "$CXX" -DAV2ENC_HAVE_CUDA=0 -DAV2ENC_HAVE_M1_GPU=0 -I"$SRC" -O2 -std=c++20 \
        -Wno-missing-braces "${M1_CORE[@]}" \
        "$SRC/encoder/m1/sub8/repro_m1_sub8.cpp" -o "$BUILD/repro_m1_sub8"
    "$BUILD/repro_m1_sub8" "$BUILD/m1_sub8_probe.ivf" 8 8
    if [ "${SKIP_GATE:-0}" != "1" ] && [ -x "$ROOT/tools/bin/dav2d-macos" ]; then
        if "$ROOT/tools/bin/dav2d-macos" -q -i "$BUILD/m1_sub8_probe.ivf" \
                -o "$BUILD/m1_sub8_probe.y4m" --threads 1 >/dev/null 2>&1; then
            echo "production M1 sub-8 stream: decoded"
        else
            echo "production M1 sub-8 stream: dav2d REJECTED (see README root cause)"
        fi
    fi
fi

echo "OK: sub8 tests passed"
