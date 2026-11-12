#!/usr/bin/env bash
# Build and run the intra identity-transform coverage component.
#
#   ./build_and_test.sh                 # unit tests + dav2d gate
#   ./build_and_test.sh --no-run        # build only
#   SANITIZE=1 ./build_and_test.sh      # ASan + UBSan
#   WERROR=1 ./build_and_test.sh
#   GATE=0 ./build_and_test.sh          # unit tests only
#   DAV2D=/path/to/dav2d ./build_and_test.sh
#
# No CMake: the module is not wired into the core build by this workstream.
# Referenced core sources are compiled read-only.
set -euo pipefail
cd "$(dirname "$0")"

CXX="${CXX:-c++}"
ROOT="$(cd ../../../.. && pwd)"
BUILD_DIR="${BUILD_DIR:-${TMPDIR:-/tmp}/av2enc_idtx_intra_build}"
DATA="${DATA:-${ROOT}/tests/data}"
SANITIZE="${SANITIZE:-0}"
WERROR="${WERROR:-0}"
GATE="${GATE:-1}"
RUN=1

for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$BUILD_DIR"

OWN_FLAGS=(-std=c++20 -Wall -Wextra -Wpedantic -Wshadow -Wconversion
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

REFS=(
  "$ROOT/src/encoder/transform/idtx_intra/idtx_intra.cpp"
  "$ROOT/src/encoder/transform/idtx_intra/type_syntax.cpp"
  "$ROOT/src/encoder/transform/idtx_intra/seq_bits.cpp"
  "$ROOT/src/encoder/tx_type/tx_type.cpp"
  "$ROOT/src/encoder/tx_variants/tx_variants.cpp"
  "$ROOT/src/encoder/intra_modes/intra_modes.cpp"
  "$ROOT/src/encoder/m1/m1_headers.cpp"
  "$ROOT/src/bitstream/m0_headers.cpp"
  "$ROOT/src/bitstream/bitwriter.cpp"
  "$ROOT/src/entropy/av2/av2_entropy.cpp"
)
echo "patcher: §5.20.5.5 fsc_mode -> build-dir copy of m1_syntax.cpp"
python3 "$(pwd)/patch_m1_syntax.py" \
  "$ROOT/src/encoder/m1/m1_syntax.cpp" \
  "$BUILD_DIR/m1_syntax_idtx.cpp"

GATE_REFS=(
  "$ROOT/src/cuda/m1/plan/plan.cpp"
  "$BUILD_DIR/m1_syntax_idtx.cpp"
  "$ROOT/src/encoder/partition/partition.cpp"
  "$ROOT/src/io/ivf.cpp"
  "$ROOT/src/io/y4m.cpp"
)

echo "compiler: $($CXX --version | head -1)"
echo "build:    $BUILD_DIR"

OBJECTS=()
index=0
for source in "${REFS[@]}"; do
  object="${BUILD_DIR}/ref_${index}.o"
  "$CXX" "${REF_FLAGS[@]}" -I"$ROOT/src" -c "$source" -o "$object"
  OBJECTS+=("$object")
  index=$((index + 1))
done

echo "== idtx_intra_test =="
"$CXX" "${OWN_FLAGS[@]}" -I"$ROOT/src" -I. -c idtx_intra_test.cpp \
  -o "$BUILD_DIR/idtx_intra_test.o"
"$CXX" "${OWN_FLAGS[@]}" "$BUILD_DIR/idtx_intra_test.o" "${OBJECTS[@]}" \
  -o "$BUILD_DIR/idtx_intra_test"
if [ "$RUN" -eq 1 ]; then "$BUILD_DIR/idtx_intra_test"; fi

if [ "$GATE" != "1" ]; then
  echo "gate: skipped (GATE=0)"
  exit 0
fi

echo
echo "== idtx_intra_gate =="
GATE_OBJECTS=()
index=0
for source in "${GATE_REFS[@]}"; do
  object="${BUILD_DIR}/gate_${index}.o"
  "$CXX" "${REF_FLAGS[@]}" -I"$ROOT/src" -c "$source" -o "$object"
  GATE_OBJECTS+=("$object")
  index=$((index + 1))
done
"$CXX" "${OWN_FLAGS[@]}" -I"$ROOT/src" -I. -c idtx_intra_gate.cpp \
  -o "$BUILD_DIR/idtx_intra_gate.o"
"$CXX" "${OWN_FLAGS[@]}" "$BUILD_DIR/idtx_intra_gate.o" "${OBJECTS[@]}" \
  "${GATE_OBJECTS[@]}" -o "$BUILD_DIR/idtx_intra_gate"

if [ "$RUN" -ne 1 ]; then exit 0; fi

DAV2D="${DAV2D:-}"
if [ -z "$DAV2D" ]; then
  for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
    if [ -x "$candidate" ]; then DAV2D="$candidate"; break; fi
  done
fi
if [ -z "$DAV2D" ]; then
  echo "SKIP: no dav2d decoder (set DAV2D=...)" >&2
  if [ "${GATE_REQUIRED:-0}" = "1" ]; then exit 1; fi
  exit 0
fi

Y4M="${Y4M:-${DATA}/m0_64x64_1f.y4m}"
GATE_ARGS=(--dav2d "$DAV2D")
[ -f "$Y4M" ] && GATE_ARGS+=(--y4m "$Y4M")
KEEP="${KEEP:-0}"
[ "$KEEP" = "1" ] && GATE_ARGS+=(--keep --prefix "$BUILD_DIR")

"$BUILD_DIR/idtx_intra_gate" "${GATE_ARGS[@]}"
echo "PASS: idtx_intra unit tests + dav2d gate"
