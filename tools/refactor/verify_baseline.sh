#!/usr/bin/env bash
# Phase 0 baseline verifier for the folder refactor.
#
#   tools/refactor/verify_baseline.sh --record   # write refactor/baseline.json
#   tools/refactor/verify_baseline.sh --check    # non-zero on any drift
#
# What is captured:
#   * CPU build (and CUDA build when nvcc is available), with timings
#   * ctest results (names, status, counts) from the CPU build
#   * tools/still_verify/feature_gate.sh counts and per-case results
#   * encoder IVF MD5s for the pinned M0 and M1 fixtures
#   * decoded Y4M MD5s via tools/validate_dav2d.sh
#   * toolchain, host, git revision, decoder identity
#
# Safety rules:
#   * no decoder  -> exit 77 (explicit skip, never a silent pass)
#   * --record refuses to re-pin a stream that changed against the constants
#     already published in tools/still_verify/feature_gate.sh or
#     tests/basic_encode_smoke.sh
#   * --check exits 1 on any encoder/decoder MD5, gate-count, or ctest drift
#   * missing baseline -> exit 2
#
# Options:
#   --record | --check        exactly one mode
#   --bin PATH                use an existing av2enc binary (implies --no-build)
#   --no-build                do not configure/build; use --bin or an existing bin
#   --build-dir DIR           build root (default: refactor/build)
#   --baseline PATH           baseline JSON (default: refactor/baseline.json)
#   --decoder PATH            dav2d binary (default: DAV2D or tools/bin/*)
#   --jobs N                  build/ctest parallelism
#   --require-cuda            fail if the CUDA build cannot be attempted
#   --verbose                 keep and show all logs
#   -h | --help
#
# Environment: DAV2D, CC, CXX, CMAKE_CUDA_COMPILER.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MODE=""
BASELINE="$ROOT/refactor/baseline.json"
BUILD_ROOT="$ROOT/refactor/build"
CPU_BUILD="$BUILD_ROOT/cpu"
CUDA_BUILD="$BUILD_ROOT/cuda"
JOBS=""
BIN=""
DECODER="${DAV2D:-}"
DO_BUILD=1
REQUIRE_CUDA=0
VERBOSE=0

usage() {
    sed -n '2,45p' "$0" | sed 's/^# \{0,1\}//'
}

die() { echo "verify_baseline: error: $*" >&2; exit 1; }
skip77() { echo "verify_baseline: SKIP (exit 77): $*" >&2; exit 77; }
note() { echo "verify_baseline: $*"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --record) MODE=record; shift ;;
        --check) MODE=check; shift ;;
        --bin) BIN=${2:-}; DO_BUILD=0; shift 2 ;;
        --no-build) DO_BUILD=0; shift ;;
        --build-dir) BUILD_ROOT=${2:-}; shift 2 ;;
        --baseline) BASELINE=${2:-}; shift 2 ;;
        --decoder) DECODER=${2:-}; shift 2 ;;
        --jobs) JOBS=${2:-}; shift 2 ;;
        --require-cuda) REQUIRE_CUDA=1; shift ;;
        --verbose) VERBOSE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1 (try --help)" ;;
    esac
done

[ -n "$MODE" ] || die "choose --record or --check"
[ -n "$BIN" ] || [ "$DO_BUILD" -eq 1 ] || die "--no-build requires --bin PATH"
[ -n "$BUILD_ROOT" ] || die "--build-dir must not be empty"
CPU_BUILD="$BUILD_ROOT/cpu"
CUDA_BUILD="$BUILD_ROOT/cuda"
BASELINE=$(cd "$(dirname "$BASELINE")" 2>/dev/null && pwd)/$(basename "$BASELINE") \
    || BASELINE="$BASELINE"
mkdir -p "$(dirname "$BASELINE")"

if [ -z "$JOBS" ]; then
    if command -v nproc >/dev/null 2>&1; then
        JOBS=$(nproc)
    elif command -v sysctl >/dev/null 2>&1; then
        JOBS=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
    else
        JOBS=4
    fi
fi

# --- decoder: required for both modes --------------------------------------
if [ -z "$DECODER" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DECODER=$candidate; break; fi
    done
fi
if [ -z "$DECODER" ] || [ ! -x "$DECODER" ]; then
    skip77 "no dav2d decoder (set DAV2D or install tools/bin/dav2d); decoded MD5s cannot be verified"
fi

TMP=$(mktemp -d "${TMPDIR:-/tmp}/av2enc-baseline.XXXXXX")
cleanup() {
    if [ "$VERBOSE" != "1" ]; then rm -rf "$TMP"; else echo "verify_baseline: logs kept in $TMP" >&2; fi
}
trap cleanup EXIT
mkdir -p "$TMP/parts"
STREAMS_JSONL="$TMP/parts/streams.jsonl"
: > "$STREAMS_JSONL"

now() { python3 -c 'import time; print(f"{time.time():.3f}")'; }
elapsed() { python3 -c "print(f'{float($2) - float($1):.3f}')"; }
md5_of() {
    if command -v md5 >/dev/null 2>&1; then md5 -q "$1";
    else md5sum "$1" | awk '{print $1}'; fi
}
tail_log() { tail -n 25 "$1" 2>/dev/null || true; }

# write_flat_json OUT key=value...
write_flat_json() {
    local out=$1; shift
    python3 - "$out" "$@" <<'PY'
import json, sys
out = sys.argv[1]
data = {}
for arg in sys.argv[2:]:
    key, _, value = arg.partition("=")
    try:
        data[key] = json.loads(value)
    except Exception:
        data[key] = value
with open(out, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)
    f.write("\n")
PY
}

# ---------------------------------------------------------------------------
# builds
# ---------------------------------------------------------------------------
CPU_OK=0
CPU_CONFIGURE_SECONDS=0
CPU_BUILD_SECONDS=0
CUDA_AVAILABLE=0
CUDA_OK=0
CUDA_SKIPPED=1
CUDA_REASON="not attempted"
CUDA_CONFIGURE_SECONDS=0
CUDA_BUILD_SECONDS=0

if [ "$DO_BUILD" -eq 1 ]; then
    note "clean CPU build -> $CPU_BUILD"
    rm -rf "$CPU_BUILD"
    mkdir -p "$CPU_BUILD"
    t0=$(now)
    if ! cmake -S "$ROOT" -B "$CPU_BUILD" \
            -DAV2ENC_ENABLE_CUDA=OFF -DAV2ENC_ALLOW_CPU_ONLY=ON \
            -DAV2ENC_PRODUCT_GATES=ON \
            -DCMAKE_BUILD_TYPE=Release >"$TMP/cpu-configure.log" 2>&1; then
        tail_log "$TMP/cpu-configure.log" >&2
        die "CPU configure failed"
    fi
    t1=$(now)
    CPU_CONFIGURE_SECONDS=$(elapsed "$t0" "$t1")
    if ! cmake --build "$CPU_BUILD" --parallel "$JOBS" >"$TMP/cpu-build.log" 2>&1; then
        tail_log "$TMP/cpu-build.log" >&2
        die "CPU build failed"
    fi
    t2=$(now)
    CPU_BUILD_SECONDS=$(elapsed "$t1" "$t2")
    CPU_OK=1
    BIN="$CPU_BUILD/av2enc"
else
    if [ -z "$BIN" ]; then
        for candidate in "$CPU_BUILD/av2enc" "$ROOT/build/av2enc" "$ROOT/build-m1/av2enc"; do
            if [ -x "$candidate" ]; then BIN=$candidate; break; fi
        done
    fi
    [ -n "$BIN" ] && [ -x "$BIN" ] || die "no av2enc binary; pass --bin PATH"
    # A --bin-only run has no build artifacts to time or ctest.
    CPU_OK=1
    CPU_CONFIGURE_SECONDS=0
    CPU_BUILD_SECONDS=0
fi
[ -x "$BIN" ] || die "av2enc binary is not executable: $BIN"

if [ "$DO_BUILD" -eq 1 ]; then
    if command -v nvcc >/dev/null 2>&1 || [ -n "${CMAKE_CUDA_COMPILER:-}" ]; then
        CUDA_AVAILABLE=1
        note "clean CUDA build -> $CUDA_BUILD"
        rm -rf "$CUDA_BUILD"
        mkdir -p "$CUDA_BUILD"
        t0=$(now)
        if ! cmake -S "$ROOT" -B "$CUDA_BUILD" \
                -DAV2ENC_ENABLE_CUDA=ON -DAV2ENC_PRODUCT_GATES=ON \
                -DCMAKE_BUILD_TYPE=Release \
                >"$TMP/cuda-configure.log" 2>&1; then
            tail_log "$TMP/cuda-configure.log" >&2
            if [ "$REQUIRE_CUDA" -eq 1 ]; then die "CUDA configure failed (--require-cuda)"; fi
            CUDA_REASON="configure failed; see log"
            CUDA_AVAILABLE=0
        else
            t1=$(now)
            CUDA_CONFIGURE_SECONDS=$(elapsed "$t0" "$t1")
            if ! cmake --build "$CUDA_BUILD" --parallel "$JOBS" >"$TMP/cuda-build.log" 2>&1; then
                tail_log "$TMP/cuda-build.log" >&2
                if [ "$REQUIRE_CUDA" -eq 1 ]; then die "CUDA build failed (--require-cuda)"; fi
                CUDA_REASON="build failed; see log"
                CUDA_AVAILABLE=0
            else
                t2=$(now)
                CUDA_BUILD_SECONDS=$(elapsed "$t1" "$t2")
                CUDA_OK=1
                CUDA_SKIPPED=0
                CUDA_REASON=""
            fi
        fi
    else
        CUDA_REASON="nvcc/CMAKE_CUDA_COMPILER not available"
        if [ "$REQUIRE_CUDA" -eq 1 ]; then
            die "CUDA requested by --require-cuda but no nvcc/CMAKE_CUDA_COMPILER"
        fi
        note "CUDA build skipped: $CUDA_REASON (CPU-only is the local environment, not a pass claim)"
    fi
else
    CUDA_AVAILABLE=0
    CUDA_REASON="--no-build"
fi

write_flat_json "$TMP/parts/build.json" \
    "cpu_ok=$CPU_OK" \
    "cpu_configure_seconds=$CPU_CONFIGURE_SECONDS" \
    "cpu_build_seconds=$CPU_BUILD_SECONDS" \
    "cpu_binary=$BIN" \
    "cuda_available=$CUDA_AVAILABLE" \
    "cuda_ok=$CUDA_OK" \
    "cuda_skipped=$CUDA_SKIPPED" \
    "cuda_reason=$CUDA_REASON" \
    "cuda_configure_seconds=$CUDA_CONFIGURE_SECONDS" \
    "cuda_build_seconds=$CUDA_BUILD_SECONDS"

# ---------------------------------------------------------------------------
# ctest
# ---------------------------------------------------------------------------
CTEST_SKIPPED=0
if [ "$DO_BUILD" -eq 1 ]; then
    note "ctest -> $CPU_BUILD"
    CTEST_JUNIT="$TMP/ctest-junit.xml"
    CTEST_LOG="$TMP/ctest.log"
    ctest --test-dir "$CPU_BUILD" --output-on-failure -j "$JOBS" \
        --output-junit "$CTEST_JUNIT" >"$CTEST_LOG" 2>&1
    CTEST_RC=$?
    python3 - "$CTEST_JUNIT" "$CTEST_LOG" "$CTEST_RC" "$TMP/parts/ctest.json" <<'PY'
import json, re, sys, xml.etree.ElementTree as ET
junit, log, rc, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
tests = []
suite_attrs = {}
try:
    root = ET.parse(junit).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    if suites:
        suite_attrs = dict(suites[0].attrib)
    for tc in root.iter("testcase"):
        status = "passed"
        if tc.find("failure") is not None or tc.find("error") is not None:
            status = "failed"
        elif tc.find("skipped") is not None:
            status = "skipped"
        tests.append({
            "name": tc.get("name", "?"),
            "status": status,
            "time": tc.get("time", ""),
        })
except Exception as exc:
    tests = []
    suite_attrs = {"parse_error": str(exc)}
total = int(suite_attrs.get("tests", len(tests)) or 0)
failed = int(suite_attrs.get("failures", sum(t["status"] == "failed" for t in tests)) or 0)
skipped = int(suite_attrs.get("skipped", sum(t["status"] == "skipped" for t in tests)) or 0)
data = {
    "returncode": rc,
    "total": total,
    "failed": failed,
    "skipped": skipped,
    "passed": total - failed - skipped,
    "tests": sorted(tests, key=lambda t: t["name"]),
}
try:
    with open(log, encoding="utf-8", errors="replace") as f:
        text = f.read()
    m = re.search(r"Total Test time \(real\) = *([0-9.]+)", text)
    data["seconds"] = float(m.group(1)) if m else None
except OSError:
    data["seconds"] = None
with open(out, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)
    f.write("\n")
PY
    note "ctest rc=$CTEST_RC"
else
    CTEST_RC=0
    CTEST_SKIPPED=1
    write_flat_json "$TMP/parts/ctest.json" \
        "returncode=0" "total=0" "failed=0" "skipped=0" "passed=0" "seconds=0" "tests=[]"
fi

# ---------------------------------------------------------------------------
# feature gate
# ---------------------------------------------------------------------------
FG_LOG="$TMP/feature-gate.log"
note "feature_gate -> $BIN"
"$ROOT/tools/still_verify/feature_gate.sh" --bin "$BIN" >"$FG_LOG" 2>&1
FG_RC=$?
if [ "$FG_RC" -eq 77 ]; then
    tail_log "$FG_LOG" >&2
    skip77 "feature_gate.sh reported no decoder"
fi
python3 - "$FG_LOG" "$FG_RC" "$TMP/parts/feature_gate.json" <<'PY'
import json, re, sys
log, rc, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
text = open(log, encoding="utf-8", errors="replace").read()
passed = failed = None
m = re.search(r"feature_gate:\s*(\d+)\s+passed,\s*(\d+)\s+failed", text)
if m:
    passed, failed = int(m.group(1)), int(m.group(2))
cases = []
for line in text.splitlines():
    pm = re.match(r"PASS\s+(\S+)\s+\(([\d,]+) B, md5 ([0-9a-fA-F]{32})\)", line)
    if pm:
        cases.append({"name": pm.group(1), "status": "passed",
                      "bytes": int(pm.group(2).replace(",", "")),
                      "md5": pm.group(3).lower()})
        continue
    fm = re.match(r"FAIL\s+(\S+):", line)
    if fm:
        cases.append({"name": fm.group(1), "status": "failed"})
        continue
    sm = re.match(r"SKIP\s+(\S+):", line)
    if sm:
        cases.append({"name": sm.group(1), "status": "skipped"})
data = {
    "returncode": rc,
    "passed": passed,
    "failed": failed,
    "cases": sorted(cases, key=lambda c: c["name"]),
    "log_tail": "\n".join(text.splitlines()[-8:]),
}
with open(out, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)
    f.write("\n")
PY

if [ "$MODE" = "record" ] && [ "$FG_RC" -ne 0 ]; then
    tail_log "$FG_LOG" >&2
    die "feature gate failed during --record; refusing to record a broken baseline"
fi
FG_PARSE=$(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));print("ok" if isinstance(d.get("passed"),int) and isinstance(d.get("failed"),int) else "unparsed")' "$TMP/parts/feature_gate.json")
if [ "$FG_PARSE" != "ok" ] && [ "$MODE" = "record" ]; then
    tail_log "$FG_LOG" >&2
    die "could not parse feature-gate counts; refusing to record"
fi
FG_PASSED=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("passed") or 0)' "$TMP/parts/feature_gate.json")
if [ "$MODE" = "record" ] && [ "$FG_PASSED" -eq 0 ]; then
    tail_log "$FG_LOG" >&2
    die "feature gate reported zero passing cases; refusing to record"
fi
if [ "$MODE" = "record" ] && [ "$CTEST_RC" -ne 0 ]; then
    die "ctest failed during --record; refusing to record a broken baseline"
fi
CTEST_TOTAL=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("total",0))' "$TMP/parts/ctest.json")
if [ "$MODE" = "record" ] && [ "$CTEST_TOTAL" -eq 0 ]; then
    die "ctest reported zero tests during --record; refusing to record"
fi

# ---------------------------------------------------------------------------
# pinned streams
# ---------------------------------------------------------------------------
# Fields: name | fixture | encoder args | generator (WxH or empty)
#         | known kind (encoder|decoded|) | known md5
# The known constants come from tools/still_verify/feature_gate.sh and
# tests/basic_encode_smoke.sh.  --record refuses to re-pin them.
STREAM_CASES=(
    "basic_64|tests/data/basic_64x64_1f.y4m|--frames 1 --qp 90||decoded|7efc3162bc0a4785685eba291a8096fd"
    "basic_360|tests/data/basic_640x360_1f.y4m|--frames 1 --qp 90||decoded|56f11d4a63ff07f2356ef5064b84fe0e"
    "basic_64_adaptive|tests/data/basic_64x64_1f.y4m|--frames 1 --qp 90 --adaptive-cdf||decoded|7efc3162bc0a4785685eba291a8096fd"
    "basic_360_adaptive|tests/data/basic_640x360_1f.y4m|--frames 1 --qp 90 --adaptive-cdf||decoded|56f11d4a63ff07f2356ef5064b84fe0e"
    "basic_1080p|generated|--frames 1 --qp 90|1920x1080|decoded|fbbef81e45fcff162dde6c71a0b56a7d"
    "still_64_base|tests/data/basic_64x64_1f.y4m|--frames 1 --qp 90 --still||encoder|0ef9825062a25c4d950ee924113c6f89"
    "still_64_cdef|tests/data/basic_64x64_1f.y4m|--frames 1 --qp 90 --still --still-cdef --still-cdef-strengths 1||encoder|b795a6f998c7037c95b926d6afdfc89e"
    "still_64_rdoq|tests/data/basic_64x64_1f.y4m|--frames 1 --qp 90 --still --still-rdoq||encoder|a7ee3a0ac75781ce2a2c7390c3503019"
    "still_360_base|tests/data/basic_640x360_1f.y4m|--frames 1 --qp 90 --still||encoder|e61de628b0812b716b0efa7dcafcdd06"
    "still_360_cdef|tests/data/basic_640x360_1f.y4m|--frames 1 --qp 90 --still --still-cdef --still-cdef-strengths 1||encoder|dec10527a779e7664ea78e53def9dbf1"
    "still_360_rdoq|tests/data/basic_640x360_1f.y4m|--frames 1 --qp 90 --still --still-rdoq||encoder|ac668d497012333b6aeecb82c2e70337"
    "still_64_qp91|tests/data/basic_64x64_1f.y4m|--frames 1 --qp 91 --still||encoder|f50e18c4e08eab3ca6437ed6283a9216"
    "still_64_qp141|tests/data/basic_64x64_1f.y4m|--frames 1 --qp 141 --still||encoder|c47ea7a5e8f826addb8ef90c527c917a"
    "still_64_qp191|tests/data/basic_64x64_1f.y4m|--frames 1 --qp 191 --still||encoder|3768deaaad9a67f7bece568d743993e5"
    "still_64_qp255|tests/data/basic_64x64_1f.y4m|--frames 1 --qp 255 --still||encoder|d1e1ebebf2885e575c55508f8b7157d1"
    "still_360_qp91|tests/data/basic_640x360_1f.y4m|--frames 1 --qp 91 --still||encoder|f5762980cda6c08ec82e131bac487a77"
    "still_360_qp141|tests/data/basic_640x360_1f.y4m|--frames 1 --qp 141 --still||encoder|6db7ffba200e2b90075fcb826aaef3c1"
    "still_360_qp191|tests/data/basic_640x360_1f.y4m|--frames 1 --qp 191 --still||encoder|df6894744772f7107cad88381ae82e2e"
    "still_360_qp255|tests/data/basic_640x360_1f.y4m|--frames 1 --qp 255 --still||encoder|d0c61808ae8dd6e011046bbab34256af"
)

GEN_Y4M=""
for candidate in "$CPU_BUILD/tests/gen_y4m" "$ROOT/build/tests/gen_y4m" \
                 "$ROOT/build-m1/tests/gen_y4m"; do
    if [ -x "$candidate" ]; then GEN_Y4M=$candidate; break; fi
done

STREAM_FAILURES=0
for spec in "${STREAM_CASES[@]}"; do
    IFS='|' read -r name fixture args gen known_kind known_md5 <<< "$spec"
    if [ "$fixture" = "generated" ]; then
        if [ -z "$GEN_Y4M" ]; then
            note "SKIP stream $name: no gen_y4m binary (build the tests target)"
            printf '%s\n' "{\"name\":\"$name\",\"status\":\"skipped\",\"reason\":\"no gen_y4m\"}" >> "$STREAMS_JSONL"
            continue
        fi
        width=${gen%x*}; height=${gen#*x}
        fixture="$TMP/${name}.y4m"
        if ! "$GEN_Y4M" "$width" "$height" "$fixture" >"$TMP/${name}.gen.log" 2>&1; then
            tail_log "$TMP/${name}.gen.log" >&2
            die "gen_y4m failed for $name"
        fi
        fixture_label="generated:${gen}"
    else
        fixture_label=$fixture
        fixture="$ROOT/$fixture"
        [ -f "$fixture" ] || die "missing pinned fixture: $fixture"
    fi

    out="$TMP/${name}.ivf"
    log="$TMP/${name}.encode.log"
    read -r -a ARGS <<< "$args"
    if ! "$BIN" "$fixture" -o "$out" ${ARGS[@]+"${ARGS[@]}"} >"$log" 2>&1; then
        tail_log "$log" >&2
        echo "FAIL stream $name: encoder failed" >&2
        STREAM_FAILURES=$((STREAM_FAILURES + 1))
        printf '%s\n' "{\"name\":\"$name\",\"status\":\"failed\",\"error\":\"encode\"}" >> "$STREAMS_JSONL"
        continue
    fi
    [ -s "$out" ] || die "encoder produced no output for $name"
    enc_md5=$(md5_of "$out")
    bytes=$(wc -c <"$out" | tr -d ' ')

    dec_md5=""
    validate_rc=99
    vlog="$TMP/${name}.validate.log"
    if ! "$DECODER" -q -i "$out" --muxer md5 -o - --threads 1 \
            >"$TMP/${name}.decode.log" 2>&1; then
        tail_log "$TMP/${name}.decode.log" >&2
        echo "FAIL stream $name: dav2d could not decode" >&2
        STREAM_FAILURES=$((STREAM_FAILURES + 1))
        printf '%s\n' "{\"name\":\"$name\",\"status\":\"failed\",\"error\":\"decode\"}" >> "$STREAMS_JSONL"
        continue
    fi
    dec_md5=$(tail -n 1 "$TMP/${name}.decode.log" | tr -d '\r')
    if ! printf '%s' "$dec_md5" | grep -Eq '^[0-9a-fA-F]{32}$'; then
        tail_log "$TMP/${name}.decode.log" >&2
        die "decoder did not return an MD5 for $name (got: $dec_md5)"
    fi

    # The authoritative gate path, not a reimplementation of it.
    EXPECTED_MD5="$dec_md5" DAV2D="$DECODER" THREADS=1 \
        "$ROOT/tools/validate_dav2d.sh" "$out" >"$vlog" 2>&1
    validate_rc=$?
    if [ "$validate_rc" -ne 0 ]; then
        tail_log "$vlog" >&2
        echo "FAIL stream $name: tools/validate_dav2d.sh exited $validate_rc" >&2
        STREAM_FAILURES=$((STREAM_FAILURES + 1))
        printf '%s\n' "{\"name\":\"$name\",\"status\":\"failed\",\"error\":\"validate\"}" >> "$STREAMS_JSONL"
        continue
    fi

    printf '%s\n' "PASS stream $name (${bytes} B, enc $enc_md5, dec $dec_md5)"

    if [ -n "$known_md5" ]; then
        if [ "$known_kind" = "encoder" ]; then got_known=$enc_md5; else got_known=$dec_md5; fi
        if [ "$got_known" != "$known_md5" ]; then
            echo "verify_baseline: PINNED STREAM CHANGED: $name $known_kind md5 $got_known, published pin $known_md5" >&2
            STREAM_FAILURES=$((STREAM_FAILURES + 1))
        fi
    fi

    python3 - "$STREAMS_JSONL" "$name" "$fixture_label" "$args" \
        "$enc_md5" "$dec_md5" "$bytes" "$validate_rc" <<'PY'
import json, sys
out, name, fixture, args, enc, dec, nbytes, vrc = sys.argv[1:9]
rec = {
    "name": name,
    "status": "passed",
    "fixture": fixture,
    "args": args.split(),
    "encoder_md5": enc,
    "decoded_md5": dec,
    "ivf_bytes": int(nbytes),
    "validate_rc": int(vrc),
}
with open(out, "a", encoding="utf-8") as f:
    f.write(json.dumps(rec, sort_keys=True) + "\n")
PY
done

if [ "$STREAM_FAILURES" -ne 0 ]; then
    if [ "$MODE" = "record" ]; then
        die "$STREAM_FAILURES pinned stream(s) failed or changed; refusing to record"
    fi
    note "$STREAM_FAILURES pinned stream(s) failed or changed"
fi

# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------
python3 - "$ROOT" "$DECODER" "$TMP/parts/meta.json" <<'PY'
import hashlib, json, os, platform, subprocess, sys
from datetime import datetime, timezone
root, decoder, out = sys.argv[1], sys.argv[2], sys.argv[3]

def cmd(args, timeout=20):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        text = (p.stdout or p.stderr or "").strip()
        return text.splitlines()[0] if text else ""
    except Exception:
        return ""

def cmd_all(args, timeout=20):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        text = (p.stdout or p.stderr or "").strip()
        return text
    except Exception:
        return ""

def git(args):
    return cmd(["git", "-C", root] + args)

def sha256(path):
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None

status = git(["status", "--porcelain"])
meta = {
    "created": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    "git": {
        "commit": git(["rev-parse", "HEAD"]),
        "branch": git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(status),
        "dirty_paths": len(status.splitlines()),
        "describe": git(["describe", "--always", "--dirty"]),
    },
    "host": {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu": cmd(["sysctl", "-n", "machdep.cpu.brand_string"]) or cmd(["sh", "-c", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"]),
        "uname": cmd_all(["uname", "-a"]),
        "python": platform.python_version(),
    },
    "toolchain": {
        "cmake": cmd(["cmake", "--version"]),
        "cxx": os.environ.get("CXX", ""),
        "cxx_version": cmd(["c++", "--version"]),
        "clang": cmd(["clang++", "--version"]),
        "gcc": cmd(["g++", "--version"]),
        "nvcc": cmd_all(["nvcc", "--version"]),
        "cuda_arch": os.environ.get("CMAKE_CUDA_ARCHITECTURES", ""),
    },
    "decoder": {
        "path": decoder,
        "version": cmd_all([decoder, "--version"]).splitlines()[-1] if cmd_all([decoder, "--version"]) else "",
        "sha256": sha256(decoder),
    },
}
with open(out, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
    f.write("\n")
PY

# ---------------------------------------------------------------------------
# assemble current result
# ---------------------------------------------------------------------------
CURRENT="$TMP/current.json"
python3 - "$TMP" "$CURRENT" "$CTEST_SKIPPED" "$MODE" <<'PY'
import json, os, sys
tmp, out, ctest_skipped, mode = sys.argv[1], sys.argv[2], sys.argv[3] == "1", sys.argv[4]

def load(name):
    with open(os.path.join(tmp, "parts", name), encoding="utf-8") as f:
        return json.load(f)

meta = load("meta.json")
build_flat = load("build.json")
ctest = load("ctest.json")
fg = load("feature_gate.json")
streams = {}
with open(os.path.join(tmp, "parts", "streams.jsonl"), encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        streams[rec["name"]] = rec

build = {
    "cpu": {
        "ok": bool(build_flat["cpu_ok"]),
        "configure_seconds": build_flat["cpu_configure_seconds"],
        "build_seconds": build_flat["cpu_build_seconds"],
        "binary": build_flat["cpu_binary"],
    },
    "cuda": {
        "available": bool(build_flat["cuda_available"]),
        "ok": bool(build_flat["cuda_ok"]),
        "skipped": bool(build_flat["cuda_skipped"]),
        "reason": build_flat["cuda_reason"],
        "configure_seconds": build_flat["cuda_configure_seconds"],
        "build_seconds": build_flat["cuda_build_seconds"],
    },
}
if ctest_skipped:
    ctest["skipped"] = True
    ctest["reason"] = "--no-build: ctest was not run"

data = {
    "schema": 1,
    "created": meta["created"],
    "tool": "tools/refactor/verify_baseline.sh",
    "mode": mode,
    "git": meta["git"],
    "host": meta["host"],
    "toolchain": meta["toolchain"],
    "decoder": meta["decoder"],
    "build": build,
    "ctest": ctest,
    "feature_gate": fg,
    "streams": streams,
}
with open(out, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, sort_keys=True)
    f.write("\n")
PY

# ---------------------------------------------------------------------------
# record or compare
# ---------------------------------------------------------------------------
if [ "$MODE" = "record" ]; then
    [ "$STREAM_FAILURES" -eq 0 ] || die "cannot record: pinned stream failures"
    cp "$CURRENT" "$BASELINE.tmp.$$"
    mv "$BASELINE.tmp.$$" "$BASELINE"
    note "recorded baseline -> $BASELINE"
    note "  ctest:       $(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]))["ctest"];print("{passed} passed, {failed} failed, {skipped} skipped, {total} total".format(**d))' "$BASELINE")"
    note "  feature gate: $(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]))["feature_gate"];print("{passed} passed, {failed} failed".format(**d))' "$BASELINE")"
    note "  streams:     $(python3 -c 'import json,sys;print(len(json.load(open(sys.argv[1]))["streams"]))' "$BASELINE")"
    note "  CPU build:   $(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]))["build"]["cpu"];print("{}s (configure {}s)".format(d["build_seconds"], d["configure_seconds"]))' "$BASELINE")"
    exit 0
fi

[ -f "$BASELINE" ] || die "baseline not found: $BASELINE (run --record first)"
note "checking against $BASELINE"
python3 - "$BASELINE" "$CURRENT" <<'PY'
import json, sys
baseline = json.load(open(sys.argv[1], encoding="utf-8"))
current = json.load(open(sys.argv[2], encoding="utf-8"))
drift = []

def cmp(path, want, got):
    if want != got:
        drift.append(f"{path}: expected {want!r}, got {got!r}")

# ctest
b, c = baseline.get("ctest", {}), current.get("ctest", {})
bt = {t["name"]: t["status"] for t in b.get("tests", [])}
ct = {t["name"]: t["status"] for t in c.get("tests", [])}

# Wave-4 product gates are additive: a gate_* test that is absent from the
# pinned baseline is tolerated (the assistant re-records after the wave), but a
# new gate that fails is still drift. Existing tests, including gate tests once
# they are in the baseline, are compared exactly.
new_gates = sorted(n for n in ct if n not in bt and n.startswith("gate_"))
for name in new_gates:
    if ct[name] == "failed":
        drift.append(f"ctest.tests[{name}]: new product gate failed")
ct_core = {n: s for n, s in ct.items() if n not in new_gates}


def counts(statuses):
    total = len(statuses)
    failed = sum(1 for s in statuses.values() if s == "failed")
    skipped = sum(1 for s in statuses.values() if s == "skipped")
    return {
        "total": total,
        "failed": failed,
        "skipped": skipped,
        "passed": total - failed - skipped,
    }


for key in ("returncode",):
    cmp(f"ctest.{key}", b.get(key), c.get(key))
bc, cc = counts(bt), counts(ct_core)
for key in ("total", "failed", "skipped", "passed"):
    cmp(f"ctest.{key}", bc[key], cc[key])
for name in sorted(set(bt) | set(ct_core)):
    if name not in bt:
        drift.append(f"ctest.tests: new test {name!r} ({ct_core[name]})")
    elif name not in ct_core:
        drift.append(f"ctest.tests: missing test {name!r}")
    elif bt[name] != ct_core[name]:
        drift.append(f"ctest.tests[{name}]: expected {bt[name]!r}, got {ct_core[name]!r}")
if new_gates:
    print("NOTE: additive product gate(s) not in the pinned baseline: "
          + ", ".join(new_gates))

# feature gate
b, c = baseline.get("feature_gate", {}), current.get("feature_gate", {})
for key in ("returncode", "passed", "failed"):
    cmp(f"feature_gate.{key}", b.get(key), c.get(key))
bc = {x["name"]: x for x in b.get("cases", [])}
cc = {x["name"]: x for x in c.get("cases", [])}
for name in sorted(set(bc) | set(cc)):
    if name not in bc:
        drift.append(f"feature_gate.cases: new case {name!r}")
    elif name not in cc:
        drift.append(f"feature_gate.cases: missing case {name!r}")
    else:
        for key in ("status", "md5", "bytes"):
            if bc[name].get(key) != cc[name].get(key):
                drift.append(f"feature_gate.cases[{name}].{key}: "
                             f"expected {bc[name].get(key)!r}, got {cc[name].get(key)!r}")

# build status (times are informational; CUDA availability is environment)
cmp("build.cpu.ok", baseline.get("build", {}).get("cpu", {}).get("ok"),
    current.get("build", {}).get("cpu", {}).get("ok"))
bcuda = baseline.get("build", {}).get("cuda", {})
ccuda = current.get("build", {}).get("cuda", {})
if bcuda.get("available"):
    cmp("build.cuda.ok", bcuda.get("ok"), ccuda.get("ok"))

# decoder identity is part of the contract for decoded MD5s
cmp("decoder.version", baseline.get("decoder", {}).get("version"),
    current.get("decoder", {}).get("version"))
cmp("decoder.sha256", baseline.get("decoder", {}).get("sha256"),
    current.get("decoder", {}).get("sha256"))

# pinned streams
bs = baseline.get("streams", {})
cs = current.get("streams", {})
known = {
    "basic_64": ("decoded", "7efc3162bc0a4785685eba291a8096fd"),
    "basic_360": ("decoded", "56f11d4a63ff07f2356ef5064b84fe0e"),
    "basic_64_adaptive": ("decoded", "7efc3162bc0a4785685eba291a8096fd"),
    "basic_360_adaptive": ("decoded", "56f11d4a63ff07f2356ef5064b84fe0e"),
    "basic_1080p": ("decoded", "fbbef81e45fcff162dde6c71a0b56a7d"),
    "still_64_base": ("encoder", "0ef9825062a25c4d950ee924113c6f89"),
    "still_64_cdef": ("encoder", "b795a6f998c7037c95b926d6afdfc89e"),
    "still_64_rdoq": ("encoder", "a7ee3a0ac75781ce2a2c7390c3503019"),
    "still_360_base": ("encoder", "e61de628b0812b716b0efa7dcafcdd06"),
    "still_360_cdef": ("encoder", "dec10527a779e7664ea78e53def9dbf1"),
    "still_360_rdoq": ("encoder", "ac668d497012333b6aeecb82c2e70337"),
    "still_64_qp91": ("encoder", "f50e18c4e08eab3ca6437ed6283a9216"),
    "still_64_qp141": ("encoder", "c47ea7a5e8f826addb8ef90c527c917a"),
    "still_64_qp191": ("encoder", "3768deaaad9a67f7bece568d743993e5"),
    "still_64_qp255": ("encoder", "d1e1ebebf2885e575c55508f8b7157d1"),
    "still_360_qp91": ("encoder", "f5762980cda6c08ec82e131bac487a77"),
    "still_360_qp141": ("encoder", "6db7ffba200e2b90075fcb826aaef3c1"),
    "still_360_qp191": ("encoder", "df6894744772f7107cad88381ae82e2e"),
    "still_360_qp255": ("encoder", "d0c61808ae8dd6e011046bbab34256af"),
}
for name in sorted(set(bs) | set(cs)):
    if name not in bs:
        drift.append(f"streams: new case {name!r}")
        continue
    if name not in cs:
        drift.append(f"streams: missing case {name!r}")
        continue
    b, c = bs[name], cs[name]
    for key in ("status", "fixture", "args", "encoder_md5", "decoded_md5",
                "ivf_bytes", "validate_rc"):
        if b.get(key) != c.get(key):
            drift.append(f"streams[{name}].{key}: expected {b.get(key)!r}, got {c.get(key)!r}")
    if name in known:
        kind, pin = known[name]
        got = c.get("encoder_md5" if kind == "encoder" else "decoded_md5")
        if got != pin:
            drift.append(f"streams[{name}].{kind}_md5: got {got!r}, published pin {pin!r}")

# toolchain differences are reported but are not a bitstream drift
warn = []
bt = baseline.get("toolchain", {})
ct = current.get("toolchain", {})
for key in ("cmake", "cxx_version", "clang", "gcc", "nvcc"):
    if bt.get(key) != ct.get(key):
        warn.append(f"toolchain.{key}: baseline {bt.get(key)!r}, current {ct.get(key)!r}")

if warn:
    print("WARN (environment, not a stream drift):")
    for line in warn:
        print("  " + line)
if drift:
    print(f"verify_baseline: DRIFT ({len(drift)} item(s))")
    for line in drift:
        print("  " + line)
    sys.exit(1)
print(f"verify_baseline: OK — baseline matches ({len(cs)} streams, "
      f"ctest {current.get('ctest', {}).get('passed')}/"
      f"{current.get('ctest', {}).get('total')} passed, "
      f"feature gate {current.get('feature_gate', {}).get('passed')} passed)")
PY
RC=$?
if [ "$RC" -ne 0 ]; then
    echo "verify_baseline: FAILED (drift)" >&2
    exit 1
fi
exit 0
