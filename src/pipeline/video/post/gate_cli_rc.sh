#!/usr/bin/env bash
# CLI gate for per-frame rate control (--rc cqp|crf|vbr) on the --video path.
#
# Checks:
#   * the default --video stream and --rc cqp are byte-identical to the pinned
#     720p wave-2 stream (c6d588647aacb2275dfaea41a00fb501);
#   * --rc crf and --rc vbr decode with `dav2d --strict 1` and their encoder
#     reconstruction equals the decoded frames sample for sample (via the
#     existing video CLI gate);
#   * CRF monotonicity, VBR determinism, and the per-frame QP report;
#   * unsupported combinations are rejected by name.
#
# Usage:
#   src/pipeline/video/post/gate_cli_rc.sh --bin build-int/av2enc
#   DAV2D=/path/to/dav2d src/pipeline/video/post/gate_cli_rc.sh --bin ...
#
# Exit codes: 0 pass, 1 failure, 2 usage, 77 missing fixture/decoder.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${HERE}/../../../.." && pwd)"
BIN=""
DAV2D="${DAV2D:-}"
TESTDATA="${AV2ENC_TESTDATA:-$ROOT/testdata}"

while [ $# -gt 0 ]; do
    case "$1" in
        --bin) BIN=$2; shift 2 ;;
        --dav2d) DAV2D=$2; shift 2 ;;
        -h|--help)
            sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "gate_cli_rc: unknown argument $1" >&2; exit 2 ;;
    esac
done

if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
    echo "gate_cli_rc: --bin must point at an av2enc executable" >&2
    exit 2
fi
if [ -z "$DAV2D" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
    done
fi
if [ -z "$DAV2D" ] || [ ! -x "$DAV2D" ]; then
    echo "gate_cli_rc: no dav2d binary (set DAV2D=...)" >&2
    exit 77
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "gate_cli_rc: python3 is required for the recon comparison" >&2
    exit 77
fi

INPUT="$TESTDATA/shine_1280x720_60f.y4m"
if [ ! -f "$INPUT" ]; then
    echo "SKIP: missing $INPUT" >&2
    exit 77
fi

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
export DAV2D

md5_of() {
    if command -v md5 >/dev/null 2>&1; then
        md5 -q "$1"
    else
        md5sum "$1" | awk '{print $1}'
    fi
}

failures=0
pass=0
fail() {
    echo "FAIL: $*" >&2
    failures=$((failures + 1))
}
ok() {
    echo "PASS $*"
    pass=$((pass + 1))
}

COMMON=(--video --frames 6 --video-max-depth 3)

# --- default and CQP are the pinned stream --------------------------------
"$BIN" "$INPUT" -o "$TMP/base.ivf" "${COMMON[@]}" >/dev/null
base_md5=$(md5_of "$TMP/base.ivf")
if [ "$base_md5" != "c6d588647aacb2275dfaea41a00fb501" ]; then
    fail "default --video MD5 moved: $base_md5"
else
    ok "default --video 720p MD5 $base_md5"
fi
"$BIN" "$INPUT" -o "$TMP/cqp.ivf" "${COMMON[@]}" --rc cqp >/dev/null
if ! cmp -s "$TMP/base.ivf" "$TMP/cqp.ivf"; then
    fail "--rc cqp is not byte-identical to the default --video stream"
else
    ok "--rc cqp is byte-identical to default --video"
fi

# run_rc <name> <args...>: encode, strict decode, recon compare, report line
run_rc() {
    local name=$1
    shift
    local ivf="$TMP/${name}.ivf"
    local recon="$TMP/${name}.recon.y4m"
    local log="$TMP/${name}.log"
    if ! "$BIN" "$INPUT" -o "$ivf" "${COMMON[@]}" --video-recon "$recon" \
            "$@" >"$log" 2>&1; then
        cat "$log" >&2
        fail "$name: encode failed"
        return 0
    fi
    if ! grep -q "frame QP:" "$log"; then
        fail "$name: no per-frame QP report"
        return 0
    fi
    if ! "$DAV2D" --strict 1 -q -i "$ivf" -o "$TMP/${name}.dec.y4m" \
            --threads 1 >/dev/null 2>&1; then
        fail "$name: dav2d --strict 1 failed"
        return 0
    fi
    if ! python3 - "$recon" "$TMP/${name}.dec.y4m" <<'PY'
import sys

def read_y4m(path):
    frames = []
    with open(path, "rb") as f:
        header = f.readline()
        if not header.startswith(b"YUV4MPEG2 "):
            raise SystemExit(f"{path}: not a Y4M file")
        width = height = None
        for field in header.split()[1:]:
            if field.startswith(b"W"):
                width = int(field[1:])
            elif field.startswith(b"H"):
                height = int(field[1:])
        cw = (width + 1) // 2
        ch = (height + 1) // 2
        size = width * height + 2 * cw * ch
        while True:
            line = f.readline()
            if not line:
                break
            if not line.startswith(b"FRAME"):
                raise SystemExit(f"{path}: bad frame marker")
            data = f.read(size)
            if len(data) != size:
                raise SystemExit(f"{path}: truncated frame")
            frames.append(data)
    return frames

a = read_y4m(sys.argv[1])
b = read_y4m(sys.argv[2])
if len(a) != len(b):
    raise SystemExit(f"frame count {len(a)} vs {len(b)}")
if a != b:
    diffs = sum(1 for fa, fb in zip(a, b) for x, y in zip(fa, fb) if x != y)
    raise SystemExit(f"{diffs} sample difference(s)")
PY
    then
        fail "$name: decoded frames differ from the encoder reconstruction"
        return 0
    fi
    grep -E "  frame QP:|  rc:" "$log" | sed 's/^/    /'
    ok "$name decodes and is sample-exact"
}

run_rc crf32 --rc crf --crf 32
run_rc crf10 --rc crf --crf 10
run_rc crf50 --rc crf --crf 50
run_rc vbr500 --rc vbr --bitrate 500
run_rc vbr1000 --rc vbr --bitrate 1000
run_rc vbr2000 --rc vbr --bitrate 2000

# --- VBR closed-loop accuracy --------------------------------------------
check_vbr() {
    local name=$1 target=$2
    local log="$TMP/${name}.log"
    if ! grep -q "closed-loop" "$log"; then
        fail "$name: report does not say closed-loop"
        return
    fi
    local achieved
    achieved=$(grep -o 'achieved [0-9.]* kbps' "$log" | head -1 |
        awk '{print $2}')
    if [ -z "$achieved" ]; then
        fail "$name: no achieved-kbps report"
        return
    fi
    if ! python3 - "$achieved" "$target" <<'PY'
import sys
achieved = float(sys.argv[1])
target = float(sys.argv[2])
error = abs(achieved - target) / target
raise SystemExit(0 if error <= 0.10 else 1)
PY
    then
        fail "$name: achieved $achieved kbps is more than 10% off $target kbps"
        return
    fi
    ok "$name: closed-loop achieved $achieved kbps for $target kbps"
}

check_vbr vbr500 500
check_vbr vbr1000 1000
check_vbr vbr2000 2000

# --- CRF monotonicity: lower crf => lower qindex --------------------------
q_of() {
    "$BIN" "$INPUT" -o "$TMP/q.ivf" "${COMMON[@]}" "$@" 2>/dev/null \
        | sed -n 's/^  frame QP: *//p' | awk '{print $1}'
}
q10=$(q_of --rc crf --crf 10)
q50=$(q_of --rc crf --crf 50)
if [ -z "$q10" ] || [ -z "$q50" ] || [ "$q10" -ge "$q50" ]; then
    fail "CRF monotonicity: crf10 qindex $q10 >= crf50 qindex $q50"
else
    ok "CRF monotonicity: crf10 qindex $q10 < crf50 qindex $q50"
fi

# --- VBR determinism ------------------------------------------------------
"$BIN" "$INPUT" -o "$TMP/vbr2.ivf" "${COMMON[@]}" --rc vbr --bitrate 500 \
    >/dev/null
if ! cmp -s "$TMP/vbr500.ivf" "$TMP/vbr2.ivf"; then
    fail "VBR is not deterministic across runs"
else
    ok "VBR is deterministic across runs"
fi

# --- rejected combinations ------------------------------------------------
expect_reject() {
    local needle=$1 label=$2
    shift 2
    if "$BIN" "$INPUT" -o "$TMP/rej.ivf" "$@" >/dev/null 2>"$TMP/err"; then
        fail "$label: expected a rejection"
        return 0
    fi
    if ! grep -qF -e "$needle" "$TMP/err"; then
        cat "$TMP/err" >&2
        fail "$label: error does not mention '$needle'"
        return 0
    fi
    ok "$label (rejected by name)"
}

expect_reject "--rc is wired for --video only" "rc_needs_video" \
    --still --rc crf
expect_reject "--rc vbr requires --bitrate" "vbr_needs_bitrate" \
    --video --rc vbr
expect_reject "--crf requires --rc crf" "crf_needs_mode" \
    --video --rc cqp --crf 20
expect_reject "--bitrate requires --rc vbr" "bitrate_needs_mode" \
    --video --rc crf --bitrate 500

echo
if [ "$failures" -ne 0 ]; then
    echo "gate_cli_rc: $failures failure(s), $pass passed" >&2
    exit 1
fi
echo "gate_cli_rc: $pass/$pass cases passed"
