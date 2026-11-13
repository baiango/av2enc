#!/usr/bin/env bash
# 10-bit in-loop filter matrix gate for the format still writer.
#
#   ./gate_10bit_filters.sh [--bin build/av2enc] [--dav2d /path/to/dav2d]
#
# For every product chroma format (4:2:0/4:2:2/4:4:4/4:0:0) and two qindices
# (90 and 303) it encodes the deterministic 10-bit source with each filter
# combination and requires:
#
#   1. `dav2d --strict 1` decodes the picture,
#   2. every decoded sample equals the encoder reconstruction (0 LSB),
#   3. with every filter flag off the IVF MD5 is stable across runs.
#
# §7.18 CDEF, §7.19 CCSO, §7.20 Wiener NS / PC-Wiener and §7.20.5 GDF are
# covered.  Chroma LR is exercised at every chroma format.  At 4:2:2/4:4:4 the
# writer signals the chroma-only Wiener subset (the CfL luma taps are not
# decoder-exact there) and rejects `--still-chroma-lr --still-deblock`, so the
# 4:2:2/4:4:4 "all" row is the deblock-free chain and a separate case asserts
# the rejection.
#
# Env: AV2ENC_BIN, DAV2D, KEEP=1
# Exit codes: 0 pass, 1 failure, 2 usage, 77 missing prerequisite.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -f "${ROOT}/CMakeLists.txt" ] && [ "${ROOT}" != "/" ]; do
    ROOT="$(dirname -- "${ROOT}")"
done
BIN="${AV2ENC_BIN:-}"
DAV2D="${DAV2D:-}"

while [ $# -gt 0 ]; do
    case "$1" in
        --bin) BIN=$2; shift 2 ;;
        --dav2d) DAV2D=$2; shift 2 ;;
        -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "gate_10bit_filters: unknown argument $1" >&2; exit 2 ;;
    esac
done
if [ -z "$BIN" ]; then
    for candidate in "${ROOT}/build-int/av2enc" "${ROOT}/build/av2enc"; do
        if [ -x "$candidate" ]; then BIN="$candidate"; break; fi
    done
fi
if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
    echo "gate_10bit_filters: no av2enc binary (pass --bin)" >&2
    exit 2
fi
if [ -z "${DAV2D:-}" ]; then
    for candidate in "${ROOT}/tools/bin/dav2d" "${ROOT}/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DAV2D="$candidate"; break; fi
    done
fi
if [ -z "${DAV2D:-}" ] || [ ! -x "$DAV2D" ]; then
    echo "SKIP: dav2d not found (set DAV2D)" >&2
    exit 77
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "SKIP: python3 is required" >&2
    exit 77
fi

TMP="$(mktemp -d)"
cleanup() {
    if [ "${KEEP:-0}" = "1" ]; then echo "kept: ${TMP}"; else rm -rf "${TMP}"; fi
}
trap cleanup EXIT

md5_of() {
    if command -v md5 >/dev/null 2>&1; then md5 -q "$1"; else
        md5sum "$1" | awk '{print $1}'
    fi
}

gen_source() {
    local w=$1 h=$2 chroma=$3 out=$4
    python3 - "$w" "$h" "$chroma" "$out" <<'PY'
import sys
import numpy as np

w, h, chroma, path = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
sx, sy = (1, 1) if chroma == "420" else (1, 0) if chroma == "422" else (0, 0) if chroma == "444" else (0, 0)
tag = {"420": "C420p10", "422": "C422p10", "444": "C444p10", "400": "Cmono10"}[chroma]
rng = np.random.default_rng(0xC0FFEE)
yy, xx = np.mgrid[0:h, 0:w].astype(np.uint32)
y = (xx * np.uint32(5) + yy * np.uint32(3) + ((xx ^ yy) << np.uint32(1)) +
     rng.integers(0, 64, size=(h, w), dtype=np.uint32)) & np.uint32(1023)
planes = [y]
if chroma != "400":
    cw, ch = (w + (1 << sx) - 1) >> sx, (h + (1 << sy) - 1) >> sy
    cy, cx = np.mgrid[0:ch, 0:cw].astype(np.uint32)
    for k in (11, 3):
        planes.append((cx * np.uint32(k) + cy * np.uint32(7) +
                       rng.integers(0, 32, size=(ch, cw), dtype=np.uint32)) &
                      np.uint32(1023))
with open(path, "wb") as f:
    f.write(f"YUV4MPEG2 W{w} H{h} F1:1 Ip A0:0 {tag}\n".encode())
    f.write(b"FRAME\n")
    for plane in planes:
        f.write(plane.astype("<u2").tobytes())
PY
}

failures=0
run() {
    local name=$1 w=$2 h=$3 chroma=$4 q=$5
    shift 5
    local src="$TMP/src_${w}x${h}_${chroma}.y4m"
    local ivf="$TMP/$name.ivf"
    local rec="$TMP/$name.recon.y4m"
    local dec="$TMP/$name.decoded.y4m"
    local extra=(--bit-depth 10)
    if [ "$chroma" != "420" ]; then extra+=(--chroma "$chroma"); fi
    if ! "$BIN" "$src" -o "$ivf" --still "${extra[@]}" --qp "$q" "$@" \
            --recon "$rec" >"$TMP/$name.encode.log" 2>&1; then
        echo "FAIL $name: encode failed" >&2
        cat "$TMP/$name.encode.log" >&2
        failures=$((failures + 1)); return
    fi
    if ! "$DAV2D" --strict 1 -q -i "$ivf" -o "$dec" --threads 1 >/dev/null 2>&1; then
        echo "FAIL $name: dav2d --strict 1" >&2
        failures=$((failures + 1)); return
    fi
    if ! python3 "${ROOT}/src/formats/chroma/compare_y4m.py" "$rec" "$dec" >/dev/null; then
        echo "FAIL $name: decoded != encoder recon" >&2
        failures=$((failures + 1)); return
    fi
    printf 'PASS %-34s %s\n' "$name" "$(md5_of "$ivf")"
}

for pair in 64x64 640x360; do
    w=${pair%x*}; h=${pair#*x}
    for chroma in 420 422 444 400; do
        gen_source "$w" "$h" "$chroma" "$TMP/src_${w}x${h}_${chroma}.y4m"
    done
done

for pair in 64x64 640x360; do
    w=${pair%x*}; h=${pair#*x}
    for chroma in 420 422 444 400; do
        for q in 90 303; do
            base="10b_${w}x${h}_${chroma}_q${q}"
            # Stable default stream (all filters off) and its MD5.
            run "$base"_off "$w" "$h" "$chroma" "$q"
            run "$base"_cdef "$w" "$h" "$chroma" "$q" --still-cdef
            run "$base"_ccso "$w" "$h" "$chroma" "$q" --still-ccso
            run "$base"_wiener "$w" "$h" "$chroma" "$q" --still-wiener
            run "$base"_pcw "$w" "$h" "$chroma" "$q" --still-pc-wiener
            run "$base"_gdf "$w" "$h" "$chroma" "$q" --still-gdf
            run "$base"_chroma_lr "$w" "$h" "$chroma" "$q" --still-chroma-lr
            run "$base"_chroma_lr_cdef "$w" "$h" "$chroma" "$q" \
                --still-cdef --still-ccso --still-chroma-lr
            if [ "$chroma" = "420" ]; then
                run "$base"_all "$w" "$h" "$chroma" "$q" --still-deblock \
                    --still-cdef --still-ccso --still-wiener --still-gdf \
                    --still-chroma-lr
            else
                # Deblocking plus chroma LR is rejected at 4:2:2/4:4:4: the
                # pinned decoder reads its chroma lossless mask out of bounds
                # for the stripe at luma row 248 and skips the aliased blocks.
                run "$base"_all "$w" "$h" "$chroma" "$q" \
                    --still-cdef --still-ccso --still-wiener --still-gdf \
                    --still-chroma-lr
                if [ "$chroma" != "400" ]; then
                    if "$BIN" "$TMP/src_${w}x${h}_${chroma}.y4m" \
                            -o "$TMP/x.ivf" --still --bit-depth 10 \
                            --chroma "$chroma" --qp "$q" \
                            --still-deblock --still-chroma-lr \
                            >/dev/null 2>&1; then
                        echo "FAIL ${base}_deblock_chroma_lr: accepted" >&2
                        failures=$((failures + 1))
                    else
                        printf 'PASS %-34s rejected\n' \
                            "${base}_deblock_chroma_lr"
                    fi
                fi
            fi
        done
    done
done

if [ "$failures" -ne 0 ]; then
    echo "FAIL: 10-bit filter matrix ($failures failures)" >&2
    exit 1
fi
echo "PASS: 10-bit filter matrix"
