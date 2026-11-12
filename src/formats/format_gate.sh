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
#   src/formats/format_gate.sh [--bin build-int/av2enc]
#
# Env: AV2ENC_BIN, DAV2D, FORMAT_UPDATE_GOLDENS=1, KEEP=1
# Exit codes: 0 pass, 1 failure, 2 usage, 77 decoder/test media missing.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd -- "${HERE}/../.." && pwd)"
BIN="${AV2ENC_BIN:-}"
GOLDENS="${ROOT}/src/formats/format_goldens.txt"

while [ $# -gt 0 ]; do
    case "$1" in
        --bin) BIN=$2; shift 2 ;;
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
    echo "SKIP: dav2d not found (set DAV2D)" >&2
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

# gen_10bit_source <w> <h> <out>
gen_10bit_source() {
    "${ROOT}/src/pipeline/still/high_bitdepth/gen_y4m_10bit.py" "$1" "$2" "$3" >/dev/null
}

RESULT="$TMP/results.txt"
: > "$RESULT"
FAILED=0

# run_case <name> <depth> <chroma> <source>
run_case() {
    local name=$1 depth=$2 chroma=$3 source=$4
    local ivf="$TMP/$name.ivf"
    local recon="$TMP/$name.recon.y4m"
    local decoded="$TMP/$name.decoded.y4m"
    local extra=()
    if [ "$depth" = "10" ]; then
        extra=(--bit-depth 10)
    else
        extra=(--chroma "$chroma")
    fi

    echo
    echo "=== $name (${depth}-bit $chroma) ==="
    if ! "$BIN" "$source" -o "$ivf" --still "${extra[@]}" --qp 90 \
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

gen_10bit_source 64 64 "$TMP/src_10_64x64.y4m"
gen_10bit_source 640 360 "$TMP/src_10_640x360.y4m"
gen_8bit_source "${ROOT}/tests/data/basic_64x64_1f.y4m" 422 "$TMP/src_422_64x64.y4m"
gen_8bit_source "${ROOT}/tests/data/basic_64x64_1f.y4m" 444 "$TMP/src_444_64x64.y4m"
gen_8bit_source "${ROOT}/tests/data/basic_64x64_1f.y4m" 400 "$TMP/src_400_64x64.y4m"
gen_8bit_source "${ROOT}/tests/data/basic_640x360_1f.y4m" 422 "$TMP/src_422_640x360.y4m"
gen_8bit_source "${ROOT}/tests/data/basic_640x360_1f.y4m" 444 "$TMP/src_444_640x360.y4m"
gen_8bit_source "${ROOT}/tests/data/basic_640x360_1f.y4m" 400 "$TMP/src_400_640x360.y4m"

run_case 10bit_420_64x64    10 420 "$TMP/src_10_64x64.y4m"
run_case 10bit_420_640x360  10 420 "$TMP/src_10_640x360.y4m"
run_case 8bit_422_64x64      8 422 "$TMP/src_422_64x64.y4m"
run_case 8bit_444_64x64      8 444 "$TMP/src_444_64x64.y4m"
run_case 8bit_400_64x64      8 400 "$TMP/src_400_64x64.y4m"
run_case 8bit_422_640x360    8 422 "$TMP/src_422_640x360.y4m"
run_case 8bit_444_640x360    8 444 "$TMP/src_444_640x360.y4m"
run_case 8bit_400_640x360    8 400 "$TMP/src_400_640x360.y4m"

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
echo "PASS: optical-format product gate (10-bit 4:2:0; 8-bit 4:2:2/4:4:4/4:0:0)"
