#!/usr/bin/env bash
# Build and run the recovered normative-table references.
#
#   ./build_and_test.sh                # verify tables, build, run
#   ./build_and_test.sh --no-run       # verify tables, build only
#   SANITIZE=1 ./build_and_test.sh     # ASan + UBSan
#   CXX=clang++ ./build_and_test.sh    # alternate compiler
#
# The first step always runs generate_spec_tables.py --check, which verifies
# the raw attachment SHA-256 pins and that every generated header is current.
# Self-contained: does not touch the root CMake build, src/main.cpp, or any
# other component.
set -euo pipefail

cd "$(dirname "$0")"

CXX="${CXX:-c++}"
BUILD_DIR="${BUILD_DIR:-build-cpu}"
SANITIZE="${SANITIZE:-0}"
RUN=1

for arg in "$@"; do
  case "$arg" in
    --no-run)
      RUN=0
      ;;
    -h|--help)
      sed -n '2,10p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

REPO_ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${REPO_ROOT}/CMakeLists.txt" ] && [ "${REPO_ROOT}" != "/" ]; do
    REPO_ROOT="$(dirname -- "${REPO_ROOT}")"
done
mkdir -p "$BUILD_DIR"

echo "CXX: $($CXX --version | head -1)"
echo "verify: generate_spec_tables.py --check"
python3 generate_spec_tables.py --check
if [ -f "$REPO_ROOT/spec/AV2_Spec_v1.0.0.pdf" ]; then
  echo "verify: verify_pdf_tables.py --check"
  python3 verify_pdf_tables.py --check
fi

FLAGS=(-std=c++20 -O2 -g -Wall -Wextra -Wpedantic -Wconversion -Wshadow
       -I "$REPO_ROOT/src")
# The independent test oracles use std::vector indexing with int loop variables;
# keep -Wconversion for product code and silence only the size_t conversion.
TEST_FLAGS=("${FLAGS[@]}" -Wno-sign-conversion)
if [ "$SANITIZE" -eq 1 ]; then
  FLAGS=(-std=c++20 -O1 -g -Wall -Wextra -Wpedantic -Wconversion -Wshadow
         -fsanitize=address,undefined -fno-omit-frame-pointer
         -I "$REPO_ROOT/src")
  TEST_FLAGS=("${FLAGS[@]}" -Wno-sign-conversion)
fi

OBJECTS=()
for src in luma_source gdf pc_wiener film_grain; do
  echo "build: $BUILD_DIR/$src.o"
  "$CXX" "${FLAGS[@]}" -c "$src.cpp" -o "$BUILD_DIR/$src.o"
  OBJECTS+=("$BUILD_DIR/$src.o")
done

TARGETS=(spec_tables gdf pc_wiener film_grain)
for t in "${TARGETS[@]}"; do
  echo "build: $BUILD_DIR/${t}_test"
  "$CXX" "${TEST_FLAGS[@]}" "${t}_test.cpp" "${OBJECTS[@]}" \
    -o "$BUILD_DIR/${t}_test"
done

if [ "$RUN" -eq 1 ]; then
  for t in "${TARGETS[@]}"; do
    "$BUILD_DIR/${t}_test"
  done
fi
