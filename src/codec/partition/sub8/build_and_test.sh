#!/usr/bin/env bash
# Standalone build + tests for src/codec/partition/sub8/.
#
#   ./src/codec/partition/sub8/build_and_test.sh
#   SANITIZE=1 WERROR=1 ./src/codec/partition/sub8/build_and_test.sh
#   SKIP_GATE=1 ./src/codec/partition/sub8/build_and_test.sh     # no dav2d
#   REPRO=1 ./src/codec/partition/sub8/build_and_test.sh         # production M1 probe
#
# Builds outside the root CMake project so the CUDA-only target is not needed
# on macOS.  The unit test and gate use exactly the same sources the M1 build
# would: partition.cpp, still_headers.cpp, bitstream/* and entropy/av2.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
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
    "$SRC/core/io/ivf.cpp"
    "$SRC/core/bitstream/bitwriter.cpp"
    "$SRC/core/bitstream/obu.cpp"
    "$SRC/core/bitstream/basic_headers.cpp"
    "$SRC/codec/partition/tree/partition.cpp"
    "$SRC/pipeline/still/still_headers.cpp"
    "$SRC/codec/transform/variants/tx_variants.cpp"
    "$SRC/codec/entropy/av2/av2_entropy.cpp"
    "$SRC/codec/partition/sub8/sub8.cpp"
    "$SRC/codec/partition/sub8/sub8_syntax.cpp"
    "$SRC/codec/partition/sub8/sub8_minimal.cpp"
)

echo "== building sub8_test"
"$CXX" "${COMMON_FLAGS[@]}" "${WARN_FLAGS[@]}" \
    "${CORE[@]}" "$SRC/codec/partition/sub8/sub8_test.cpp" -o "$BUILD/sub8_test"
echo "== running sub8_test"
"$BUILD/sub8_test"

echo "== building gate_minimal"
"$CXX" "${COMMON_FLAGS[@]}" "${WARN_FLAGS[@]}" \
    "${CORE[@]}" "$SRC/codec/partition/sub8/gate_minimal.cpp" -o "$BUILD/gate_minimal"

GATE_DIR="$BUILD/gate"
rm -rf "$GATE_DIR"
mkdir -p "$GATE_DIR"
echo "== writing gate streams"
"$BUILD/gate_minimal" "$GATE_DIR"

if [ "${SKIP_GATE:-0}" = "1" ]; then
    echo "== dav2d gate skipped (SKIP_GATE=1)"
else
    echo "== dav2d gate"
    python3 "$SRC/codec/partition/sub8/gate_dav2d.py" --dir "$GATE_DIR"
fi

if [ "${REPRO:-0}" = "1" ]; then
    echo "== building production M1 sub-8 probe (diagnostic)"
    STILL_CORE=(
        "$SRC/core/io/y4m.cpp" "$SRC/core/io/ivf.cpp"
        "$SRC/core/bitstream/bitwriter.cpp" "$SRC/core/bitstream/obu.cpp"
        "$SRC/core/bitstream/basic_headers.cpp" "$SRC/pipeline/still/basic_encoder.cpp"
        "$SRC/codec/partition/tree/partition.cpp"
        "$SRC/codec/intra/modes/intra_modes.cpp"
        "$SRC/codec/rate/rdoq/rdoq.cpp" "$SRC/codec/rate/rdoq/rdoq_av2_model.cpp"
        "$SRC/codec/transform/tx_type/tx_type.cpp"
        "$SRC/codec/transform/variants/tx_variants.cpp"
        "$SRC/codec/transform/variants/tx_select.cpp"
        "$SRC/codec/intra/cfl/cfl.cpp"
        "$SRC/codec/filters/deblock/deblock.cpp"
        "$SRC/codec/filters/cdef/cdef.cpp"
        "$SRC/codec/filters/loop_restoration/restoration.cpp"
        "$SRC/codec/rate/rdo/rdo.cpp" "$SRC/codec/rate/rdo/intra_decision.cpp"
        "$SRC/pipeline/still/still_headers.cpp" "$SRC/pipeline/still/still_syntax.cpp"
        "$SRC/pipeline/still/still_encoder.cpp"
        "$SRC/pipeline/still/stream/still_stream_bridge.cpp"
        "$SRC/codec/entropy/av2/av2_entropy.cpp" "$SRC/core/telemetry/profiler.cpp"
        "$SRC/cuda/pipeline/still/cdef/still_cdef.cpp"
    )
    "$CXX" -DAV2ENC_HAVE_CUDA=0 -DAV2ENC_HAVE_STILL_GPU=0 -I"$SRC" -O2 -std=c++20 \
        -Wno-missing-braces "${STILL_CORE[@]}" \
        "$SRC/codec/partition/sub8/repro_still_sub8.cpp" -o "$BUILD/repro_still_sub8"
    "$BUILD/repro_still_sub8" "$BUILD/still_sub8_probe.ivf" 8 8
    if [ "${SKIP_GATE:-0}" != "1" ] && [ -x "$ROOT/tools/bin/dav2d-macos" ]; then
        if "$ROOT/tools/bin/dav2d-macos" -q -i "$BUILD/still_sub8_probe.ivf" \
                -o "$BUILD/still_sub8_probe.y4m" --threads 1 >/dev/null 2>&1; then
            echo "production M1 sub-8 stream: decoded"
        else
            echo "production M1 sub-8 stream: dav2d REJECTED (see README root cause)"
        fi
    fi
fi

echo "OK: sub8 tests passed"
