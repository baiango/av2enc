#!/usr/bin/env bash
# Decoder gate for the video CCTX path (--video-cctx).
#
#   ./video_cctx_gate.sh
#   AV2ENC_BIN=build-int/av2enc DAV2D=tools/bin/dav2d-macos ./video_cctx_gate.sh
#   KEEP=1 ./video_cctx_gate.sh
#
# What it proves:
#   1. `--video-cctx` writes §5.4.8 enable_cctx = 1 and codes §5.20.7.27
#      cctx_type on plane 1 (after the U EOB), including the inter eob == 1
#      form that an intra block infers to 0.
#   2. The §7.14.6 joint U/V dequant rotation is decoder-exact: dav2d
#      --strict 1 reconstructs the encoder samples at 64x64/360p QP 60/90/120.
#   3. At least one block selects a non-zero CCTX type (the gate is not
#      vacuous), and the stream differs from the CCTX-off one.
#   4. CCTX composes with --video-tcq and --video-tx-search.
#   5. The default (CCTX-off) streams stay byte-identical to the pinned MD5s.
#   6. The documented incompatible compositions are rejected by name.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done

BIN="${AV2ENC_BIN:-}"
DAV2D="${DAV2D:-}"
KEEP="${KEEP:-0}"

while [ $# -gt 0 ]; do
    case "$1" in
        --bin) BIN=$2; shift 2 ;;
        --dav2d) DAV2D=$2; shift 2 ;;
        -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "video_cctx_gate: unknown argument $1" >&2; exit 2 ;;
    esac
done

if [ -z "$BIN" ]; then
    for candidate in "$ROOT/build-int/av2enc" "$ROOT/build/av2enc" \
                     "$ROOT/build-m1/av2enc"; do
        if [ -x "$candidate" ]; then BIN="$candidate"; break; fi
    done
fi
if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
    echo "video_cctx_gate: no av2enc binary; set AV2ENC_BIN=..." >&2
    exit 127
fi
if ! "$BIN" --help 2>&1 | grep -q -- '--video-cctx'; then
    echo "video_cctx_gate: $BIN has no --video-cctx (stale build dir);" \
         "set AV2ENC_BIN=..." >&2
    exit 127
fi
if [ -z "$DAV2D" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DAV2D="$candidate"; break; fi
    done
fi
if [ -z "$DAV2D" ] || [ ! -x "$DAV2D" ]; then
    echo "SKIP: no dav2d binary (set DAV2D=...)" >&2
    exit 77
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "SKIP: python3 is required for the recon comparison" >&2
    exit 77
fi

TMP="${TMPDIR:-/tmp}/av2enc_video_cctx_gate.$$"
mkdir -p "$TMP"
cleanup() {
    if [ "$KEEP" != "1" ]; then rm -rf "$TMP"; fi
}
trap cleanup EXIT

FIX_64="$ROOT/tests/data/basic_64x64_1f.y4m"
FIX_360="$ROOT/testdata/shine_640x360_4f.y4m"
[ -f "$FIX_64" ] || { echo "SKIP: missing $FIX_64" >&2; exit 77; }
[ -f "$FIX_360" ] || { echo "SKIP: missing $FIX_360" >&2; exit 77; }

failures=0
pass=0

md5_of() {
    md5 -q "$1" 2>/dev/null || md5sum "$1" | awk '{print $1}'
}

compare_recon() {
    python3 - "$1" "$2" "$3" <<'PY'
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
        if width is None or height is None:
            raise SystemExit(f"{path}: missing dimensions")
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


expected = read_y4m(sys.argv[1])
decoded = read_y4m(sys.argv[2])
if len(expected) != len(decoded):
    raise SystemExit(
        f"{sys.argv[3]}: picture count {len(expected)} != {len(decoded)}")
diffs = 0
first = None
for index, (a, b) in enumerate(zip(expected, decoded)):
    if a == b:
        continue
    for pos, (x, y) in enumerate(zip(a, b)):
        if x != y:
            diffs += 1
            if first is None:
                first = (index, pos, x, y)
if diffs:
    raise SystemExit(
        f"{sys.argv[3]}: {diffs} sample diff(s), first {first}")
print(f"exact {sys.argv[3]}: {len(expected)} picture(s), {len(expected[0])} "
      f"samples/picture")
PY
}

# run_case <name> <fixture> <frames> <expected_md5|-> <encode args...>
run_case() {
    local name=$1 fixture=$2 frames=$3 expected=$4
    shift 4
    local out="$TMP/${name}.ivf"
    local dec="$TMP/${name}.dec.y4m"
    local recon="$TMP/${name}.recon.y4m"
    if ! "$BIN" "$fixture" --video --frames "$frames" \
            --video-recon "$recon" -o "$out" "$@" >"$TMP/${name}.log" 2>&1; then
        echo "FAIL $name: encode failed" >&2
        tail -5 "$TMP/${name}.log" >&2
        failures=$((failures + 1))
        return 0
    fi
    if [ "$expected" != "-" ]; then
        local got
        got=$(md5_of "$out")
        if [ "$got" != "$expected" ]; then
            echo "FAIL $name: IVF md5 $got, expected $expected" >&2
            failures=$((failures + 1))
            return 0
        fi
    fi
    if ! "$DAV2D" -q -i "$out" -o "$dec" --strict 1 --threads 1 \
            >"$TMP/${name}.dec.log" 2>&1; then
        echo "FAIL $name: dav2d --strict 1 rejected the stream" >&2
        tail -5 "$TMP/${name}.dec.log" >&2
        failures=$((failures + 1))
        return 0
    fi
    if ! compare_recon "$recon" "$dec" "$name" >>"$TMP/${name}.cmp.log" 2>&1; then
        echo "FAIL $name: encoder reconstruction is not decoder-exact" >&2
        cat "$TMP/${name}.cmp.log" >&2
        failures=$((failures + 1))
        return 0
    fi
    echo "PASS $name ($(wc -c <"$out" | tr -d ' ') B, md5 $(md5_of "$out"))"
    pass=$((pass + 1))
}

reject_case() {
    local name=$1 needle=$2
    shift 2
    if "$BIN" "$FIX_64" --video --frames 1 -o "$TMP/${name}.ivf" "$@" \
            >"$TMP/${name}.log" 2>&1; then
        echo "FAIL $name: encode unexpectedly succeeded" >&2
        failures=$((failures + 1))
        return 0
    fi
    if grep -qF -- "$needle" "$TMP/${name}.log"; then
        echo "PASS $name"
        pass=$((pass + 1))
    else
        echo "FAIL $name: wrong diagnostic" >&2
        tail -5 "$TMP/${name}.log" >&2
        failures=$((failures + 1))
    fi
}

BASE_ARGS=(--video-inter-residual --video-tx-search)

# Default streams must not move (shared with the TCQ gate).
run_case default_64_qp90 "$FIX_64" 1 582b53aa10c748cefaac54b225867bf0 \
    "${BASE_ARGS[@]}" --qp 90
run_case default_360_qp90 "$FIX_360" 4 230e3d1e3501f8812020e39be418a56d \
    "${BASE_ARGS[@]}" --qp 90

# --video-cctx at QP 60/90/120 on both fixtures.
run_case cctx_64_qp60 "$FIX_64" 1 36d722b69b4f630863017bb501a7f6e7 \
    "${BASE_ARGS[@]}" --video-cctx --qp 60
run_case cctx_64_qp90 "$FIX_64" 1 6b24e3b589220287e1bb682e311a4de0 \
    "${BASE_ARGS[@]}" --video-cctx --qp 90
run_case cctx_64_qp120 "$FIX_64" 1 c07383c338c7873d1c5ffba42079f75e \
    "${BASE_ARGS[@]}" --video-cctx --qp 120
run_case cctx_360_qp60 "$FIX_360" 4 46bbd9a0f16b0e0fa9129c4c4823db85 \
    "${BASE_ARGS[@]}" --video-cctx --qp 60
run_case cctx_360_qp90 "$FIX_360" 4 0704723d485161c60017d7a3115c966c \
    "${BASE_ARGS[@]}" --video-cctx --qp 90
run_case cctx_360_qp120 "$FIX_360" 4 4adf904fec93b0e10210eda1b051ee96 \
    "${BASE_ARGS[@]}" --video-cctx --qp 120

# The non-search chroma path (compute_tx sizing, CCTX search) must be exact.
run_case cctx_360_notx_qp90 "$FIX_360" 4 397f95a320b24c8fafbc3ceb51404e7a \
    --video-inter-residual --video-cctx --qp 90

# Combined with TCQ: independent luma/chroma tools, both exact.
run_case both_64_qp90 "$FIX_64" 1 e16382a49621b40135f728ff3754a6be \
    "${BASE_ARGS[@]}" --video-tcq --video-cctx --qp 90
run_case both_360_qp90 "$FIX_360" 4 d350428d718d97b930507d9f53b8c3a4 \
    "${BASE_ARGS[@]}" --video-tcq --video-cctx --qp 90

# CCTX must change the stream and actually select a rotation (not vacuous).
if [ -f "$TMP/default_360_qp90.ivf" ] && [ -f "$TMP/cctx_360_qp90.ivf" ]; then
    if cmp -s "$TMP/default_360_qp90.ivf" "$TMP/cctx_360_qp90.ivf"; then
        echo "FAIL cctx_active: --video-cctx did not change the stream" >&2
        failures=$((failures + 1))
    else
        echo "PASS cctx_active"
        pass=$((pass + 1))
    fi
fi

if "$BIN" "$FIX_360" --video --frames 4 "${BASE_ARGS[@]}" --video-cctx \
        --qp 90 --video-trace -o "$TMP/cctx_trace.ivf" \
        >"$TMP/cctx_trace.out" 2>"$TMP/cctx_trace.log"; then
    rotations=$(grep -oE 'cctx=[1-6]' "$TMP/cctx_trace.log" | wc -l | tr -d ' ')
    if [ "${rotations:-0}" -lt 1 ]; then
        echo "FAIL cctx_symbols: no non-zero cctx_type was selected" >&2
        failures=$((failures + 1))
    else
        echo "PASS cctx_symbols ($rotations rotated block(s))"
        pass=$((pass + 1))
    fi
else
    echo "FAIL cctx_symbols: trace encode failed" >&2
    failures=$((failures + 1))
fi

# Documented rejections.
reject_case reject_cctx_qm "--video-cctx does not compose with --video-qm" \
    --video-cctx --video-qm
reject_case reject_cctx_lossless \
    "--video-cctx is not applied by the --video-lossless writer" \
    --video-cctx --video-lossless
reject_case reject_cctx_deltaq \
    "--video-cctx does not compose with segmentation/delta-Q" \
    --video-cctx --video-delta-q
reject_case reject_cctx_tiles "--video-cctx does not compose with --tiles" \
    --video-cctx --tiles 2x1
reject_case reject_cctx_gpu "--video-cctx is CPU-only" \
    --video-cctx --video-gpu
reject_case reject_cctx_tip \
    "--video-cctx does not compose with the frame-type wrapper modes" \
    --video-cctx --video-tip

# The named AV2 validator must also accept the representative stream.
if [ -x "$ROOT/tools/validate_dav2d.sh" ] && [ -f "$TMP/cctx_64_qp90.ivf" ]; then
    if "$ROOT/tools/validate_dav2d.sh" "$TMP/cctx_64_qp90.ivf" "$FIX_64" \
            >"$TMP/validate_dav2d.log" 2>&1; then
        echo "PASS validate_dav2d (CCTX)"
        pass=$((pass + 1))
    else
        echo "FAIL validate_dav2d (CCTX)" >&2
        tail -20 "$TMP/validate_dav2d.log" >&2
        failures=$((failures + 1))
    fi
fi

if [ "$failures" -ne 0 ]; then
    echo "video_cctx_gate: $failures failed, $pass passed" >&2
    exit 1
fi
echo "video_cctx_gate: $pass passed, 0 failed"
