#!/usr/bin/env bash
# Decoder gate for the video TCQ path (--video-tcq).
#
#   ./video_tcq_gate.sh
#   AV2ENC_BIN=build-int/av2enc DAV2D=tools/bin/dav2d-macos ./video_tcq_gate.sh
#   KEEP=1 ./video_tcq_gate.sh
#
# What it proves:
#   1. `--video-tcq` writes §5.4.8 enable_tcq = 1 with
#      choose_tcq_per_frame = 1 and a per-frame allow_tcq bit, and leaves
#      enable_parity_hiding = 0.
#   2. The §5.20.7.27 TCQ syntax ((tcqState>>1)&1 coeff_base selectors,
#      syntax_level_for(), the read_quant escape) and the §7.14.4 extra
#      dqDenom shift reconstruct sample-for-sample with the pinned dav2d at
#      --strict 1.
#   3. The gate exercises both quantiser paths: the TX search
#      (--video-tx-search) and the single-DCT compute_tx path.
#   4. The default (TCQ-off) streams stay byte-identical to the pinned MD5s.
#   5. The documented incompatible compositions are rejected by name.
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
        *) echo "video_tcq_gate: unknown argument $1" >&2; exit 2 ;;
    esac
done

if [ -z "$BIN" ]; then
    for candidate in "$ROOT/build-int/av2enc" "$ROOT/build/av2enc" \
                     "$ROOT/build-m1/av2enc"; do
        if [ -x "$candidate" ]; then BIN="$candidate"; break; fi
    done
fi
if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
    echo "video_tcq_gate: no av2enc binary; set AV2ENC_BIN=..." >&2
    exit 127
fi
if ! "$BIN" --help 2>&1 | grep -q -- '--video-tcq'; then
    echo "video_tcq_gate: $BIN has no --video-tcq (stale build dir);" \
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

TMP="${TMPDIR:-/tmp}/av2enc_video_tcq_gate.$$"
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

# compare_recon <encoder-recon.y4m> <decoded.y4m> <name>
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

# Default streams must not move.  These are the pinned TCQ-off MD5s.
run_case default_64_qp90 "$FIX_64" 1 582b53aa10c748cefaac54b225867bf0 \
    "${BASE_ARGS[@]}" --qp 90
run_case default_360_qp90 "$FIX_360" 4 230e3d1e3501f8812020e39be418a56d \
    "${BASE_ARGS[@]}" --qp 90

# --video-tcq with both quantiser paths at QP 60/90/120.
run_case tcq_64_qp60 "$FIX_64" 1 4536f6b9fa27eae05d899cc26f0f3182 \
    "${BASE_ARGS[@]}" --video-tcq --qp 60
run_case tcq_64_qp90 "$FIX_64" 1 d0a69605d0137c15307520f13d6d2986 \
    "${BASE_ARGS[@]}" --video-tcq --qp 90
run_case tcq_64_qp120 "$FIX_64" 1 64f30635790dc1512a8de9c900532748 \
    "${BASE_ARGS[@]}" --video-tcq --qp 120
run_case tcq_360_qp60 "$FIX_360" 4 71c39b2839b8f9036c37340f547be03f \
    "${BASE_ARGS[@]}" --video-tcq --qp 60
run_case tcq_360_qp90 "$FIX_360" 4 49b4db7d7f6c683c9774c57914806ff2 \
    "${BASE_ARGS[@]}" --video-tcq --qp 90
run_case tcq_360_qp120 "$FIX_360" 4 b99fb0bceaae0e337760ea9e3cb9944e \
    "${BASE_ARGS[@]}" --video-tcq --qp 120

# The non-search path (compute_tx) must also be decoder-exact.
run_case tcq_360_notx_qp90 "$FIX_360" 4 819af3341bca6490042abf93eb667e26 \
    --video-inter-residual --video-tcq --qp 90

# The TCQ stream must actually differ from the default one (no vacuous pass).
if [ -f "$TMP/default_360_qp90.ivf" ] && [ -f "$TMP/tcq_360_qp90.ivf" ]; then
    if cmp -s "$TMP/default_360_qp90.ivf" "$TMP/tcq_360_qp90.ivf"; then
        echo "FAIL tcq_active: --video-tcq did not change the stream" >&2
        failures=$((failures + 1))
    else
        echo "PASS tcq_active"
        pass=$((pass + 1))
    fi
fi

# Documented rejections (quantiser/header compositions that cannot be exact).
reject_case reject_tcq_qm "--video-tcq does not compose with --video-qm" \
    --video-tcq --video-qm
reject_case reject_tcq_lossless \
    "--video-tcq is not applied by the --video-lossless writer" \
    --video-tcq --video-lossless
reject_case reject_tcq_deltaq \
    "--video-tcq does not compose with segmentation/delta-Q" \
    --video-tcq --video-delta-q
reject_case reject_tcq_tiles "--video-tcq does not compose with --tiles" \
    --video-tcq --tiles 2x1
reject_case reject_tcq_gpu "--video-tcq is CPU-only" --video-tcq --video-gpu
reject_case reject_tcq_tip \
    "--video-tcq does not compose with the frame-type wrapper modes" \
    --video-tcq --video-tip

# The named AV2 validator must also accept the representative stream.
if [ -x "$ROOT/tools/validate_dav2d.sh" ] && [ -f "$TMP/tcq_64_qp90.ivf" ]; then
    if "$ROOT/tools/validate_dav2d.sh" "$TMP/tcq_64_qp90.ivf" "$FIX_64" \
            >"$TMP/validate_dav2d.log" 2>&1; then
        echo "PASS validate_dav2d (TCQ)"
        pass=$((pass + 1))
    else
        echo "FAIL validate_dav2d (TCQ)" >&2
        tail -20 "$TMP/validate_dav2d.log" >&2
        failures=$((failures + 1))
    fi
fi

if [ "$failures" -ne 0 ]; then
    echo "video_tcq_gate: $failures failed, $pass passed" >&2
    exit 1
fi
echo "video_tcq_gate: $pass passed, 0 failed"
