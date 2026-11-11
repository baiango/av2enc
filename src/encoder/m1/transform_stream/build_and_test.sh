#!/usr/bin/env bash
# Build and run the M1 transform-stream adapter CPU tests.
#
#   ./build_and_test.sh                 # adapter test + real-residual probe
#   ./build_and_test.sh --no-run
#   SANITIZE=1 ./build_and_test.sh      # ASan + UBSan
#   WERROR=1 ./build_and_test.sh
#   QUALITY=0 ./build_and_test.sh       # skip the tile-byte / PSNR probe
#   DATA=/path/to/tests/data ./build_and_test.sh
#
# No CMake.  Reference components are compiled read-only with their own warning
# status; the adapter, tests and probe are built with the component warning set.
set -euo pipefail
cd "$(dirname "$0")"

CXX="${CXX:-c++}"
ROOT="$(cd ../../../.. && pwd)"
BUILD_DIR="${BUILD_DIR:-${TMPDIR:-/tmp}/av2enc_transform_stream_build}"
DATA="${DATA:-${ROOT}/tests/data}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
QUALITY="${QUALITY:-1}"
RUN=1

for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$BUILD_DIR"

OWN_FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
           -ffp-contract=off)
REF_FLAGS=(-std=c++20 -ffp-contract=off -w)
if [ "$WERROR" = "1" ]; then OWN_FLAGS+=(-Werror); fi
if [ "$SANITIZE" = "1" ]; then
  OWN_FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
  REF_FLAGS+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  OWN_FLAGS+=(-O2)
  REF_FLAGS+=(-O2)
fi

CORE_REFS=(
  "$ROOT/src/encoder/tx_type/tx_type.cpp"
  "$ROOT/src/encoder/tx_variants/tx_variants.cpp"
  "$ROOT/src/encoder/rdoq/rdoq.cpp"
  "$ROOT/src/encoder/rdoq/rdoq_av2_model.cpp"
  "$ROOT/src/encoder/intra_modes/intra_modes.cpp"
  "$ROOT/src/encoder/rdo/rdo.cpp"
  "$ROOT/src/io/y4m.cpp"
)
QUALITY_REFS=(
  "$ROOT/src/cuda/m1/plan/plan.cpp"
  "$ROOT/src/encoder/partition/partition.cpp"
  "$ROOT/src/encoder/m1/m1_syntax.cpp"
  "$ROOT/src/encoder/m1/m1_headers.cpp"
  "$ROOT/src/entropy/av2/av2_entropy.cpp"
  "$ROOT/src/bitstream/bitwriter.cpp"
  "$ROOT/src/bitstream/obu.cpp"
  "$ROOT/src/bitstream/m0_headers.cpp"
  "$ROOT/src/io/ivf.cpp"
  "$ROOT/src/profile/profiler.cpp"
)

echo "compiler: $($CXX --version | head -1)"
echo "build:    $BUILD_DIR"

CORE_OBJECTS=()
index=0
for source in "${CORE_REFS[@]}"; do
  object="${BUILD_DIR}/core_${index}.o"
  "$CXX" "${REF_FLAGS[@]}" -I"$ROOT/src" -c "$source" -o "$object"
  CORE_OBJECTS+=("$object")
  index=$((index + 1))
done

echo "== adapter test =="
"$CXX" "${OWN_FLAGS[@]}" -I"$ROOT/src" -c transform_stream.cpp \
  -o "$BUILD_DIR/transform_stream.o"
"$CXX" "${OWN_FLAGS[@]}" -I"$ROOT/src" -c transform_stream_test.cpp \
  -o "$BUILD_DIR/transform_stream_test.o"
"$CXX" "${OWN_FLAGS[@]}" "$BUILD_DIR/transform_stream_test.o" \
  "$BUILD_DIR/transform_stream.o" "${CORE_OBJECTS[@]}" \
  -o "$BUILD_DIR/transform_stream_test"

if [ "$RUN" -eq 1 ]; then
  "$BUILD_DIR/transform_stream_test" --data "$DATA"
fi

if [ "$QUALITY" = "1" ]; then
  echo
  echo "== real-residual quality probe =="
  QUALITY_OBJECTS=()
  index=0
  for source in "${QUALITY_REFS[@]}"; do
    object="${BUILD_DIR}/quality_${index}.o"
    "$CXX" "${REF_FLAGS[@]}" -I"$ROOT/src" -c "$source" -o "$object"
    QUALITY_OBJECTS+=("$object")
    index=$((index + 1))
  done
  "$CXX" "${OWN_FLAGS[@]}" -I"$ROOT/src" -c transform_stream_quality.cpp \
    -o "$BUILD_DIR/transform_stream_quality.o"
  "$CXX" "${OWN_FLAGS[@]}" "$BUILD_DIR/transform_stream_quality.o" \
    "$BUILD_DIR/transform_stream.o" "${CORE_OBJECTS[@]}" \
    "${QUALITY_OBJECTS[@]}" \
    -o "$BUILD_DIR/transform_stream_quality"
  if [ "$RUN" -eq 1 ]; then
    "$BUILD_DIR/transform_stream_quality" \
      --input "$DATA/m0_64x64_1f.y4m" --tx-types
    echo
    "$BUILD_DIR/transform_stream_quality" \
      --input "$DATA/m0_640x360_1f.y4m" --tx-types
  fi
fi
