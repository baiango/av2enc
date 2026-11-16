#!/usr/bin/env bash
# Build and run the Y4M header-parser fuzzer for a bounded time.
#
# Usage: tools/fuzz_smoke.sh [seconds] [extra libFuzzer args...]
#   tools/fuzz_smoke.sh 60
#   tools/fuzz_smoke.sh 300 -print_final_stats=1
#
# Needs a clang that ships libFuzzer.  Apple Command Line Tools does not;
# Homebrew LLVM does (/opt/homebrew/opt/llvm/bin/clang++).  Set FUZZ_CXX to
# override.
#
# Exit: 0 clean run, 1 fuzzer found a crash, 2 setup problem.
set -uo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SECONDS_RUN=${1:-60}
shift || true

if [ -n "${FUZZ_CXX:-}" ]; then
    CXX_BIN=$FUZZ_CXX
elif [ -x /opt/homebrew/opt/llvm/bin/clang++ ]; then
    CXX_BIN=/opt/homebrew/opt/llvm/bin/clang++
else
    CXX_BIN=$(command -v clang++ || true)
fi
[ -n "$CXX_BIN" ] || { echo "fuzz_smoke: no clang++ found" >&2; exit 2; }

# libFuzzer runtime is not in Apple CLT; fail early with a useful message.
probe=$(mktemp -d "${TMPDIR:-/tmp}/fuzz-probe.XXXXXX")
trap 'rm -rf "$probe"' EXIT
printf 'extern "C" int LLVMFuzzerTestOneInput(const unsigned char*, unsigned long){return 0;}\n' \
    >"$probe/p.cpp"
if ! "$CXX_BIN" -fsanitize=fuzzer "$probe/p.cpp" -o "$probe/p" >"$probe/log" 2>&1; then
    echo "fuzz_smoke: $CXX_BIN cannot link libFuzzer:" >&2
    tail -3 "$probe/log" >&2
    echo "install Homebrew LLVM and set FUZZ_CXX=/opt/homebrew/opt/llvm/bin/clang++" >&2
    exit 2
fi

BUILD=${FUZZ_BUILD:-$ROOT/build-fuzz}
cmake -S "$ROOT" -B "$BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CXX_COMPILER="$CXX_BIN" \
    -DAV2ENC_ENABLE_CUDA=OFF \
    -DAV2ENC_ALLOW_CPU_ONLY=ON \
    -DAV2ENC_FUZZ=ON >"$probe/cfg.log" 2>&1 || {
        echo "fuzz_smoke: configure failed" >&2; tail -5 "$probe/cfg.log" >&2; exit 2; }
cmake --build "$BUILD" --target fuzz_y4m -j"$(sysctl -n hw.ncpu 2>/dev/null || echo 4)" \
    >"$probe/build.log" 2>&1 || {
        echo "fuzz_smoke: build failed" >&2; tail -10 "$probe/build.log" >&2; exit 2; }

# The target lives in tests/CMakeLists.txt, so its binary sits under tests/.
FUZZ_BIN="$BUILD/fuzz_y4m"
[ -x "$FUZZ_BIN" ] || FUZZ_BIN="$BUILD/tests/fuzz_y4m"
[ -x "$FUZZ_BIN" ] || { echo "fuzz_smoke: fuzz_y4m not built" >&2; exit 2; }

# Run against a build-local copy so libFuzzer's discovered units never land in
# the source tree.
WORK_CORPUS="$BUILD/corpus"
mkdir -p "$WORK_CORPUS"
cp -n "$ROOT/tests/fuzz_corpus/"* "$WORK_CORPUS/" 2>/dev/null || true

echo "fuzz_smoke: running ${SECONDS_RUN}s with $CXX_BIN"
"$FUZZ_BIN" "$WORK_CORPUS" \
    -max_total_time="$SECONDS_RUN" -rss_limit_mb=2048 -print_final_stats=1 "$@"
