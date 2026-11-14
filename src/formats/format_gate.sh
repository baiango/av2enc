#!/usr/bin/env bash
# Product-CLI decoder gate for the optical-format still paths.
#
#   --bit-depth 10       10-bit 4:2:0  (C420p10 input)
#   --chroma 422|444|400 8-bit 4:2:2 / 4:4:4 / 4:0:0
#
# Every case:
#   1. builds a deterministic source in the requested format;
#   2. encodes with `av2enc --still --bit-depth/--chroma ... --recon`;
#   3. runs tools/validate_dav2d.sh (pinned dav2d, --strict 1);
#   4. decodes again with `dav2d --strict 1` and requires the encoder
#      reconstruction to be sample-for-sample identical (0 diffs);
#   5. compares the IVF/decoded MD5 against a pinned golden when the golden file
#      exists, and rewrites it with FORMAT_UPDATE_GOLDENS=1.
#
# Usage:
#   src/formats/format_gate.sh [--bin build-int/av2enc] [--dav2d /path/to/dav2d]
#
# Env: AV2ENC_BIN, DAV2D, FORMAT_UPDATE_GOLDENS=1, KEEP=1
# Exit codes: 0 pass, 1 failure, 2 usage, 77 decoder/test media missing.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd -- "${HERE}/../.." && pwd)"
BIN="${AV2ENC_BIN:-}"
DAV2D="${DAV2D:-}"
GOLDENS="${ROOT}/src/formats/format_goldens.txt"

while [ $# -gt 0 ]; do
    case "$1" in
        --bin) BIN=$2; shift 2 ;;
        --dav2d) DAV2D=$2; shift 2 ;;
        -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "format_gate: unknown argument $1" >&2; exit 2 ;;
    esac
done
if [ -z "$BIN" ]; then
    for candidate in "${ROOT}/build-int/av2enc" "${ROOT}/build/av2enc"; do
        if [ -x "$candidate" ]; then BIN="$candidate"; break; fi
    done
fi
if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
    echo "format_gate: no av2enc binary (pass --bin)" >&2
    exit 2
fi

if [ -z "${DAV2D:-}" ]; then
    for candidate in "${ROOT}/tools/bin/dav2d" "${ROOT}/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DAV2D="$candidate"; break; fi
    done
fi
if [ -z "${DAV2D:-}" ] || [ ! -x "$DAV2D" ]; then
    echo "SKIP: dav2d not found (set DAV2D or install tools/bin/dav2d[-macos])" >&2
    exit 77
fi
# The gate scripts share the pinned decoder with tools/validate_dav2d.sh.
export DAV2D
if ! command -v python3 >/dev/null 2>&1; then
    echo "SKIP: python3 is required for source generation and recon compare" >&2
    exit 77
fi
for fixture in "${ROOT}/tests/data/basic_64x64_1f.y4m" \
               "${ROOT}/tests/data/basic_640x360_1f.y4m" \
               "${ROOT}/src/pipeline/still/high_bitdepth/gen_y4m_10bit.py"; do
    if [ ! -f "$fixture" ]; then
        echo "SKIP: missing prerequisite $fixture" >&2
        exit 77
    fi
done

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

# Deterministic 8-bit source in the requested chroma format.  Chroma is
# upsampled from the 4:2:0 fixture with the same nearest-sample rule as
# fmt::convert_420_8_to_picture(); the frame rate is rewritten to F1:1 so the
# IVF header is stable.
gen_8bit_source() {
    local src=$1 format=$2 out=$3
    python3 - "$src" "$format" "$out" <<'PY'
import sys

src, fmt, out = sys.argv[1], sys.argv[2], sys.argv[3]
data = open(src, "rb").read()
end = data.find(b"\n")
header = data[:end].decode("latin1")
tokens = header.split()
width = int(next(t[1:] for t in tokens if t.startswith("W")))
height = int(next(t[1:] for t in tokens if t.startswith("H")))
colorspace = next((t[1:] for t in tokens if t.startswith("C")), "420")
if not colorspace.startswith("420"):
    raise SystemExit(f"{src}: expected a 4:2:0 source, got C{colorspace}")
pos = data.find(b"FRAME", end)
pos = data.find(b"\n", pos) + 1
cw, ch = (width + 1) // 2, (height + 1) // 2
size = width * height + 2 * cw * ch
body = data[pos:pos + size]
if len(body) != size:
    raise SystemExit(f"{src}: truncated frame")
y = body[:width * height]
u = body[width * height:width * height + cw * ch]
v = body[width * height + cw * ch:]

if fmt == "422":
    sub_x, sub_y, tag = 1, 0, "C422"
elif fmt == "444":
    sub_x, sub_y, tag = 0, 0, "C444"
elif fmt == "400":
    tag = "Cmono"
else:
    raise SystemExit(f"unsupported format {fmt}")

planes = [y]
if fmt != "400":
    dw = (width + (1 << sub_x) - 1) >> sub_x
    dh = (height + (1 << sub_y) - 1) >> sub_y
    for plane in (u, v):
        data_out = bytearray(dw * dh)
        for yy in range(dh):
            sy = min(((yy << sub_y) >> 1), ch - 1)
            for xx in range(dw):
                sx = min(((xx << sub_x) >> 1), cw - 1)
                data_out[yy * dw + xx] = plane[sy * cw + sx]
        planes.append(bytes(data_out))

with open(out, "wb") as f:
    f.write(f"YUV4MPEG2 W{width} H{height} F1:1 Ip A0:0 {tag}\n".encode())
    f.write(b"FRAME\n")
    for plane in planes:
        f.write(plane)
PY
}

# gen_8bit_textured <chroma> <out>: deterministic 640x360 8-bit source with
# strong block edges, so the restoration tools find units that improve SSE at
# the pinned qp 90.  (On the natural fixtures at qp 90 the Wiener/LR search
# keeps every unit unfiltered, which leaves the header row untested.)
gen_8bit_textured() {
    local format=$1 out=$2
    python3 - "$format" "$out" <<'PY'
import sys
import numpy as np

fmt, out = sys.argv[1], sys.argv[2]
w, h = 640, 360
yy, xx = np.mgrid[0:h, 0:w].astype(np.uint32)
y = ((xx // 3 + yy // 3) % 2 * 120 + 64 + (xx * 7 + yy * 13) % 40).astype(np.uint8)
if fmt == "400":
    tag = "Cmono"
    planes = [y]
else:
    sx, sy = (1, 0) if fmt == "422" else (0, 0)
    tag = "C422" if fmt == "422" else "C444"
    cw, ch = (w + (1 << sx) - 1) >> sx, (h + (1 << sy) - 1) >> sy
    cy, cx = np.mgrid[0:ch, 0:cw].astype(np.uint32)
    u = ((cx // 4 + cy // 4) % 2 * 140 + 40 + (cx * 3) % 20).astype(np.uint8)
    v = ((cx // 16 + cy // 16) % 2 * 120 + 60).astype(np.uint8)
    planes = [y, u, v]
with open(out, "wb") as f:
    f.write(f"YUV4MPEG2 W{w} H{h} F1:1 Ip A0:0 {tag}\n".encode())
    f.write(b"FRAME\n")
    for plane in planes:
        f.write(plane.tobytes())
PY
}

# gen_10bit_source <w> <h> <out> [chroma]
gen_10bit_source() {
    "${ROOT}/src/pipeline/still/high_bitdepth/gen_y4m_10bit.py" "$@" >/dev/null
}

RESULT="$TMP/results.txt"
: > "$RESULT"
FAILED=0

# run_case <name> <depth> <chroma> <source> [qp] [extra flags...]
run_case() {
    local name=$1 depth=$2 chroma=$3 source=$4 qp=${5:-90}
    local extra_filters=("${@:6}")
    local ivf="$TMP/$name.ivf"
    local recon="$TMP/$name.recon.y4m"
    local decoded="$TMP/$name.decoded.y4m"
    local extra=()
    if [ "$depth" = "10" ]; then
        extra=(--bit-depth 10)
        if [ "$chroma" != "420" ]; then
            extra+=(--chroma "$chroma")
        fi
    else
        extra=(--chroma "$chroma")
    fi

    echo
    echo "=== $name (${depth}-bit $chroma, qp $qp) ==="
    if ! "$BIN" "$source" -o "$ivf" --still "${extra[@]}" --qp "$qp" \
            ${extra_filters[@]+"${extra_filters[@]}"} \
            --recon "$recon" >"$TMP/$name.encode.log" 2>&1; then
        echo "FAIL $name: encode failed" >&2
        cat "$TMP/$name.encode.log" >&2
        FAILED=1
        return
    fi
    grep -E 'format:|frames:|recon:' "$TMP/$name.encode.log" || true

    if ! "${ROOT}/tools/validate_dav2d.sh" "$ivf" >"$TMP/$name.validate.log" 2>&1; then
        echo "FAIL $name: validate_dav2d.sh" >&2
        cat "$TMP/$name.validate.log" >&2
        FAILED=1
        return
    fi
    tail -1 "$TMP/$name.validate.log"

    if ! "$DAV2D" --strict 1 -q -i "$ivf" -o "$decoded" --threads 1; then
        echo "FAIL $name: dav2d --strict 1" >&2
        FAILED=1
        return
    fi
    if ! python3 "${ROOT}/src/formats/chroma/compare_y4m.py" "$recon" "$decoded"; then
        echo "FAIL $name: reconstruction != decoder output" >&2
        FAILED=1
        return
    fi

    local ivf_md5 decoded_md5
    ivf_md5=$(md5_of "$ivf")
    decoded_md5=$("$DAV2D" -q -i "$ivf" --muxer md5 -o - 2>/dev/null | tail -1)
    echo "golden:  $name ivf=$ivf_md5 decoded=$decoded_md5"
    echo "$name $depth $chroma $ivf_md5 $decoded_md5" >> "$RESULT"
}

# expect_reject <name> <source> <needle> [encoder args...]
#
# The strict reader must reject a body that the C tag under-declares or any
# bytes after the single frame.  Accepting them silently re-encodes a prefix as
# the smaller format (SCOUT-FORMATS-HBD-02 / SCOUT-CHROMA-02).
expect_reject() {
    local name=$1 source=$2 needle=$3
    shift 3
    echo
    echo "=== negative: $name ==="
    local log="$TMP/neg_$name.log"
    if "$BIN" "$source" -o "$TMP/neg_$name.ivf" "$@" >"$log" 2>&1; then
        echo "FAIL $name: encoder accepted an invalid frame body" >&2
        cat "$log" >&2
        FAILED=1
        return
    fi
    if ! grep -q "$needle" "$log"; then
        echo "FAIL $name: rejection did not mention '$needle'" >&2
        cat "$log" >&2
        FAILED=1
        return
    fi
    echo "PASS $name: $(grep -m1 "$needle" "$log")"
}

gen_10bit_source 64 64 "$TMP/src_10_64x64.y4m"
gen_10bit_source 640 360 "$TMP/src_10_640x360.y4m"
gen_10bit_source 64 64 "$TMP/src_10_422_64x64.y4m" 422
gen_10bit_source 64 64 "$TMP/src_10_444_64x64.y4m" 444
gen_10bit_source 64 64 "$TMP/src_10_400_64x64.y4m" 400
gen_10bit_source 640 360 "$TMP/src_10_422_640x360.y4m" 422
gen_10bit_source 640 360 "$TMP/src_10_444_640x360.y4m" 444
gen_10bit_source 640 360 "$TMP/src_10_400_640x360.y4m" 400
gen_8bit_source "${ROOT}/tests/data/basic_64x64_1f.y4m" 422 "$TMP/src_422_64x64.y4m"
gen_8bit_source "${ROOT}/tests/data/basic_64x64_1f.y4m" 444 "$TMP/src_444_64x64.y4m"
gen_8bit_source "${ROOT}/tests/data/basic_64x64_1f.y4m" 400 "$TMP/src_400_64x64.y4m"
gen_8bit_source "${ROOT}/tests/data/basic_640x360_1f.y4m" 422 "$TMP/src_422_640x360.y4m"
gen_8bit_source "${ROOT}/tests/data/basic_640x360_1f.y4m" 444 "$TMP/src_444_640x360.y4m"
gen_8bit_source "${ROOT}/tests/data/basic_640x360_1f.y4m" 400 "$TMP/src_400_640x360.y4m"
gen_8bit_textured 422 "$TMP/tex_422_640x360.y4m"
gen_8bit_textured 444 "$TMP/tex_444_640x360.y4m"
gen_8bit_textured 400 "$TMP/tex_400_640x360.y4m"

run_case 10bit_420_64x64    10 420 "$TMP/src_10_64x64.y4m"
run_case 10bit_420_640x360  10 420 "$TMP/src_10_640x360.y4m"
run_case 10bit_422_64x64    10 422 "$TMP/src_10_422_64x64.y4m"
run_case 10bit_444_64x64    10 444 "$TMP/src_10_444_64x64.y4m"
run_case 10bit_400_64x64    10 400 "$TMP/src_10_400_64x64.y4m"
run_case 10bit_422_640x360  10 422 "$TMP/src_10_422_640x360.y4m"
run_case 10bit_444_640x360  10 444 "$TMP/src_10_444_640x360.y4m"
run_case 10bit_400_640x360  10 400 "$TMP/src_10_400_640x360.y4m"
# High-QP 10-bit: qctx slices 1..3 (§8.3.1) plus a chroma-format combination.
run_case 10bit_420_q91_64x64   10 420 "$TMP/src_10_64x64.y4m" 91
run_case 10bit_420_q141_64x64  10 420 "$TMP/src_10_64x64.y4m" 141
run_case 10bit_420_q191_64x64  10 420 "$TMP/src_10_64x64.y4m" 191
run_case 10bit_420_q303_64x64  10 420 "$TMP/src_10_64x64.y4m" 303
run_case 10bit_422_q191_64x64  10 422 "$TMP/src_10_422_64x64.y4m" 191
# §7.18 CDEF at 10-bit on every product chroma format.  The gate requires the
# decoded picture to equal the encoder reconstruction, which pins the strength
# scaling (coeffShift = 2), tap parity and 10-bit clipping.
run_case 10bit_420_cdef_64x64    10 420 "$TMP/src_10_64x64.y4m" 90 --still-cdef --still-cdef-strengths 4
run_case 10bit_422_cdef_64x64    10 422 "$TMP/src_10_422_64x64.y4m" 90 --still-cdef --still-cdef-strengths 4
run_case 10bit_444_cdef_64x64    10 444 "$TMP/src_10_444_64x64.y4m" 90 --still-cdef --still-cdef-strengths 4
run_case 10bit_400_cdef_64x64    10 400 "$TMP/src_10_400_64x64.y4m" 90 --still-cdef --still-cdef-strengths 4
run_case 10bit_420_cdef_640x360  10 420 "$TMP/src_10_640x360.y4m" 90 --still-cdef --still-cdef-strengths 4
run_case 10bit_422_cdef_640x360  10 422 "$TMP/src_10_422_640x360.y4m" 90 --still-cdef --still-cdef-strengths 4
run_case 10bit_444_cdef_640x360  10 444 "$TMP/src_10_444_640x360.y4m" 90 --still-cdef --still-cdef-strengths 4
run_case 10bit_400_cdef_640x360  10 400 "$TMP/src_10_400_640x360.y4m" 90 --still-cdef --still-cdef-strengths 4
# Header-only CdefStrengths = 1 (no cdef_idx symbol) and the 10-bit qindex
# slices where coeffShift interacts with a coarser quantiser.
run_case 10bit_420_cdef1_64x64   10 420 "$TMP/src_10_64x64.y4m" 90 --still-cdef --still-cdef-strengths 1
run_case 10bit_420_cdef_q191_64x64 10 420 "$TMP/src_10_64x64.y4m" 191 --still-cdef --still-cdef-strengths 8
# CDEF after deblock at 10-bit (normative order: reconstruct -> deblock -> CDEF).
run_case 10bit_420_deblock_cdef_640x360 10 420 "$TMP/src_10_640x360.y4m" 90 --still-deblock --still-cdef --still-cdef-strengths 4
# §7.19 CCSO at 10-bit on every product chroma format, plus composition with
# CDEF and a higher qindex where the offsets are large.
run_case 10bit_420_ccso_64x64    10 420 "$TMP/src_10_64x64.y4m" 90 --still-ccso
run_case 10bit_422_ccso_64x64    10 422 "$TMP/src_10_422_64x64.y4m" 90 --still-ccso
run_case 10bit_444_ccso_64x64    10 444 "$TMP/src_10_444_64x64.y4m" 90 --still-ccso
run_case 10bit_400_ccso_64x64    10 400 "$TMP/src_10_400_64x64.y4m" 90 --still-ccso
run_case 10bit_420_ccso_640x360  10 420 "$TMP/src_10_640x360.y4m" 90 --still-ccso
run_case 10bit_422_ccso_640x360  10 422 "$TMP/src_10_422_640x360.y4m" 90 --still-ccso
run_case 10bit_444_ccso_640x360  10 444 "$TMP/src_10_444_640x360.y4m" 90 --still-ccso
run_case 10bit_400_ccso_640x360  10 400 "$TMP/src_10_400_640x360.y4m" 90 --still-ccso
run_case 10bit_420_ccso_q191_64x64 10 420 "$TMP/src_10_64x64.y4m" 191 --still-ccso
run_case 10bit_420_cdef_ccso_640x360 10 420 "$TMP/src_10_640x360.y4m" 90 --still-cdef --still-cdef-strengths 4 --still-ccso
run_case 10bit_444_cdef_ccso_640x360 10 444 "$TMP/src_10_444_640x360.y4m" 90 --still-cdef --still-cdef-strengths 4 --still-ccso
# Forced edge-classified form (ccso_bo_only = 0) so the class table is coded
# even where the band-offset-only form would win.
run_case 10bit_420_ccso_edge_64x64  10 420 "$TMP/src_10_64x64.y4m" 90 --still-ccso-edge
run_case 10bit_444_ccso_edge_640x360 10 444 "$TMP/src_10_444_640x360.y4m" 90 --still-ccso-edge
# §7.20 loop restoration / §7.20.5 GDF at 10-bit.  q303 makes the filters win
# units; luma Wiener, PC-Wiener and GDF are gated on every product format.
run_case 10bit_420_wiener_640x360  10 420 "$TMP/src_10_640x360.y4m" 303 --still-wiener
run_case 10bit_420_chroma_lr_640x360 10 420 "$TMP/src_10_640x360.y4m" 303 --still-chroma-lr
run_case 10bit_420_pc_wiener_640x360 10 420 "$TMP/src_10_640x360.y4m" 303 --still-pc-wiener
run_case 10bit_420_gdf_640x360     10 420 "$TMP/src_10_640x360.y4m" 303 --still-gdf
run_case 10bit_420_wiener_gdf_640x360 10 420 "$TMP/src_10_640x360.y4m" 303 --still-wiener --still-gdf
run_case 10bit_420_all_filters_640x360 10 420 "$TMP/src_10_640x360.y4m" 303 --still-deblock --still-cdef --still-ccso --still-wiener --still-gdf
run_case 10bit_422_chroma_lr_640x360 10 422 "$TMP/src_10_422_640x360.y4m" 303 --still-chroma-lr
run_case 10bit_444_chroma_lr_640x360 10 444 "$TMP/src_10_444_640x360.y4m" 303 --still-chroma-lr
run_case 10bit_422_wiener_640x360  10 422 "$TMP/src_10_422_640x360.y4m" 303 --still-wiener
run_case 10bit_422_pc_wiener_640x360 10 422 "$TMP/src_10_422_640x360.y4m" 303 --still-pc-wiener
run_case 10bit_422_gdf_640x360     10 422 "$TMP/src_10_422_640x360.y4m" 303 --still-gdf
run_case 10bit_444_wiener_640x360  10 444 "$TMP/src_10_444_640x360.y4m" 303 --still-wiener
run_case 10bit_444_pc_wiener_640x360 10 444 "$TMP/src_10_444_640x360.y4m" 303 --still-pc-wiener
run_case 10bit_444_gdf_640x360     10 444 "$TMP/src_10_444_640x360.y4m" 303 --still-gdf
run_case 10bit_400_wiener_640x360  10 400 "$TMP/src_10_400_640x360.y4m" 303 --still-wiener
run_case 10bit_400_pc_wiener_640x360 10 400 "$TMP/src_10_400_640x360.y4m" 303 --still-pc-wiener
run_case 10bit_400_gdf_640x360     10 400 "$TMP/src_10_400_640x360.y4m" 303 --still-gdf
run_case 10bit_420_all_filters_64x64 10 420 "$TMP/src_10_64x64.y4m" 90 --still-cdef --still-ccso --still-pc-wiener --still-gdf
run_case 8bit_422_64x64      8 422 "$TMP/src_422_64x64.y4m"
run_case 8bit_444_64x64      8 444 "$TMP/src_444_64x64.y4m"
run_case 8bit_400_64x64      8 400 "$TMP/src_400_64x64.y4m"
run_case 8bit_422_640x360    8 422 "$TMP/src_422_640x360.y4m"
run_case 8bit_444_640x360    8 444 "$TMP/src_444_640x360.y4m"
run_case 8bit_400_640x360    8 400 "$TMP/src_400_640x360.y4m"

# 8-bit non-4:2:0 in-loop filters (§7.17-§7.20.5).  The 8-bit chroma writer
# carries the same plan/header/tile hooks as the 10-bit writer and runs the
# depth-parameterised cores at BitDepth = 8.  Each filter alone and the full
# chain must decode to the encoder reconstruction exactly.
for fmt in 422 444 400; do
    for size in 64x64 640x360; do
        src="$TMP/src_${fmt}_${size}.y4m"
        run_case "8bit_${fmt}_${size}_deblock" 8 "$fmt" "$src" 90 --still-deblock
        run_case "8bit_${fmt}_${size}_cdef" 8 "$fmt" "$src" 90 --still-cdef
        run_case "8bit_${fmt}_${size}_ccso" 8 "$fmt" "$src" 90 --still-ccso
        run_case "8bit_${fmt}_${size}_wiener" 8 "$fmt" "$src" 90 --still-wiener
        run_case "8bit_${fmt}_${size}_pc_wiener" 8 "$fmt" "$src" 90 --still-pc-wiener
        run_case "8bit_${fmt}_${size}_gdf" 8 "$fmt" "$src" 90 --still-gdf
        if [ "$fmt" != "400" ]; then
            run_case "8bit_${fmt}_${size}_chroma_lr" 8 "$fmt" "$src" 90 --still-chroma-lr
            run_case "8bit_${fmt}_${size}_lr_chain" 8 "$fmt" "$src" 90 \
                --still-cdef --still-ccso --still-wiener --still-chroma-lr --still-gdf
        else
            run_case "8bit_${fmt}_${size}_all_filters" 8 "$fmt" "$src" 90 \
                --still-deblock --still-cdef --still-ccso --still-wiener --still-gdf
        fi
        run_case "8bit_${fmt}_${size}_deblock_chain" 8 "$fmt" "$src" 90 \
            --still-deblock --still-cdef --still-ccso --still-wiener --still-gdf
    done
    # Restoration search actually wins units on the textured source, so the
    # §5.18.7.11 row and the §5.20.10.4 unit symbols are exercised.
    run_case "8bit_${fmt}_640x360_wiener_tex" 8 "$fmt" \
        "$TMP/tex_${fmt}_640x360.y4m" 90 --still-wiener
    run_case "8bit_${fmt}_640x360_lr_chain_tex" 8 "$fmt" \
        "$TMP/tex_${fmt}_640x360.y4m" 90 \
        --still-cdef --still-ccso --still-wiener --still-chroma-lr --still-gdf
done

# Frame-body validation: a tag that under-declares the real body, and a valid
# frame followed by extra bytes.
python3 - "$TMP" <<'PY'
import sys

tmp = sys.argv[1]


def retag(src, dst, old, new):
    data = open(src, "rb").read()
    if old not in data:
        raise SystemExit(f"{src}: missing {old}")
    open(dst, "wb").write(data.replace(old, new, 1))


retag(f"{tmp}/src_444_64x64.y4m", f"{tmp}/bad_8bit_444_as422.y4m",
      b"C444", b"C422")
retag(f"{tmp}/src_10_444_64x64.y4m", f"{tmp}/bad_10bit_444_as420.y4m",
      b"C444p10", b"C420p10")
with open(f"{tmp}/src_10_64x64.y4m", "rb") as f:
    body = f.read()
with open(f"{tmp}/bad_trailing.y4m", "wb") as f:
    f.write(body)
    f.write(b"FRAME\n" + bytes(64))
PY

expect_reject 8bit_444_as422 "$TMP/bad_8bit_444_as422.y4m" \
    "oversized Y4M frame" --still --chroma 422
expect_reject 10bit_444_as420 "$TMP/bad_10bit_444_as420.y4m" \
    "oversized Y4M frame" --still --bit-depth 10 --qp 90
expect_reject trailing_bytes "$TMP/bad_trailing.y4m" \
    "oversized Y4M frame" --still --bit-depth 10 --qp 90
# §7.20 chroma LR composes with §7.17 deblocking only at 4:2:0: at
# 4:2:2/4:4:4 pinned dav2d reads its chroma lossless mask past its extent for
# the stripe at luma row 248 and skips the aliased blocks, so the writer must
# reject the combination by name.
expect_reject 10bit_444_deblock_chroma_lr "$TMP/src_10_444_640x360.y4m" \
    "do not decode exactly at 4:2:2/4:4:4 with pinned dav2d" --still \
    --bit-depth 10 --chroma 444 --still-deblock --still-chroma-lr --qp 303

# Accepted-and-ignored Still tree flags: the format paths use their own minimal
# writer, so a flag that they cannot apply must be a usage error, not a
# byte-identical stream (SCOUT-FORMATS-HBD-01 / SCOUT-CHROMA-01).
for flag in --still-sub8 --still-sdp; do
    expect_reject "10bit${flag}" "$TMP/src_10_64x64.y4m" \
        "format paths use their own minimal still writer" \
        --still --bit-depth 10 --qp 90 "$flag"
    expect_reject "8bit${flag}" "$TMP/src_444_64x64.y4m" \
        "format paths use their own minimal still writer" \
        --still --chroma 444 "$flag"
done
expect_reject tcq_scale_without_tcq "$TMP/src_444_64x64.y4m" \
    "requires --still-tcq" --still --chroma 444 --still-tcq-scale 0.5

if [ "$FAILED" -ne 0 ]; then
    echo "FAIL: one or more format gate cases failed" >&2
    exit 1
fi

if [ "${FORMAT_UPDATE_GOLDENS:-0}" = "1" ]; then
    {
        echo "# Optical-format product-CLI gate goldens (dav2d f9dafbc, --strict 1)."
        echo "# Regenerate: FORMAT_UPDATE_GOLDENS=1 src/formats/format_gate.sh"
        echo "# name depth chroma ivf_md5 decoded_md5"
        sort "$RESULT"
    } > "$GOLDENS"
    echo "updated: $GOLDENS"
elif [ -f "$GOLDENS" ]; then
    while read -r name depth chroma ivf_md5 decoded_md5; do
        case "$name" in ''|'#'*) continue ;; esac
        got=$(awk -v n="$name" '$1 == n { print $4 " " $5 }' "$RESULT")
        if [ "$got" != "$ivf_md5 $decoded_md5" ]; then
            echo "FAIL golden $name: got '$got', expected '$ivf_md5 $decoded_md5'" >&2
            FAILED=1
        else
            echo "PASS golden: $name"
        fi
    done < "$GOLDENS"
fi

if [ "$FAILED" -ne 0 ]; then
    exit 1
fi
echo
echo "PASS: optical-format product gate (10-bit 4:2:0/4:2:2/4:4:4/4:0:0 + qctx 91..303; 8-bit 4:2:2/4:4:4/4:0:0)"
