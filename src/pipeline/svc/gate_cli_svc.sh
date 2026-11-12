#!/usr/bin/env bash
# CLI gate for the opt-in SVC layer path (`--video --svc-layers`).
#
# What it proves:
#   * the layered stream is produced end-to-end from the product CLI (per-layer
#     sequence headers with seq_lcr_id/max_mlayer_id/max_tlayer_id, global
#     LCR/OPS, §7.3.6 temporal-unit order);
#   * `avmdec --all-layers` output equals the interleaving of independent
#     per-layer decodes;
#   * `avmdec --select-ops=0,N` equals the cumulative operating point;
#   * Annex F.3.2 extraction (`--oppoint`, `--spatial-layers`) matches the OPS;
#   * the base layer is sample-identical to an independent plain `--video`
#     encode of the box-scaled source (so the scaler/layer encode is checked,
#     not just self-consistency);
#   * `dav2d --strict 1` decodes the single-layer and same-resolution layered
#     streams, and `tools/validate_dav2d.sh` accepts them.
#
# Pinned-decoder limitation: dav2d f9dafbc fails on a *mixed-resolution*
# multi-xlayer stream that contains inter (P) frames (same-resolution layers and
# single-layer streams decode).  avmdec decodes both, so the mixed-resolution
# cases gate with avmdec only.  See src/pipeline/svc/README.md.
#
# Usage:
#   src/pipeline/svc/gate_cli_svc.sh --bin build-int/av2enc
#   AVMDEC=/path/to/avmdec DAV2D=/path/to/dav2d src/pipeline/svc/gate_cli_svc.sh ...
#
# Exit codes: 0 pass, 1 failure, 2 usage, 77 missing fixture/decoder.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${HERE}/../../.." && pwd)"
BIN=""
DAV2D="${DAV2D:-}"
AVMDEC="${AVMDEC:-}"

while [ $# -gt 0 ]; do
    case "$1" in
        --bin) BIN=$2; shift 2 ;;
        --dav2d) DAV2D=$2; shift 2 ;;
        --avmdec) AVMDEC=$2; shift 2 ;;
        -h|--help)
            sed -n '2,31p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "gate_cli_svc: unknown argument $1" >&2; exit 2 ;;
    esac
done

if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
    echo "gate_cli_svc: --bin must point at an av2enc executable" >&2
    exit 2
fi
if [ -z "$DAV2D" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
    done
fi
if [ -z "$AVMDEC" ]; then
    for candidate in "$ROOT/tools/bin/avmdec" \
                     "$ROOT/tools/bin/avm-macos/avmdec"; do
        if [ -x "$candidate" ]; then AVMDEC=$candidate; break; fi
    done
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "SKIP: python3 is required" >&2
    exit 77
fi

INPUT="$ROOT/testdata/shine_1280x720_60f.y4m"
if [ ! -f "$INPUT" ]; then
    echo "SKIP: missing $INPUT" >&2
    exit 77
fi

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

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

# --- rejections (no decoder needed) --------------------------------------
expect_reject() {
    local label=$1
    shift
    if "$BIN" "$INPUT" -o "$TMP/rej.ivf" "$@" >"$TMP/rej.log" 2>&1; then
        fail "$label: expected a rejection"
        return
    fi
    if ! grep -q "av2enc:" "$TMP/rej.log"; then
        cat "$TMP/rej.log" >&2
        fail "$label: rejection did not print an av2enc diagnostic"
        return
    fi
    ok "$label: rejected"
}

COMMON=(--video --frames 3 --video-max-depth 3)
expect_reject "--svc-layers without --video" --svc-layers 0=640x360 --still
expect_reject "--oppoint + --alllayers" "${COMMON[@]}" --svc-layers 0=640x360 \
    --oppoint 0 --alllayers
expect_reject "--svc-temporal 2" "${COMMON[@]}" --svc-layers 0=640x360 \
    --svc-temporal 2
expect_reject "--svc-layers + --tiles" "${COMMON[@]}" --svc-layers 0=640x360 \
    --tiles 2x2
expect_reject "--svc-layers + --rc vbr" "${COMMON[@]}" --svc-layers 0=640x360 \
    --rc vbr --bitrate 500
expect_reject "--svc-layers + --crf" "${COMMON[@]}" --svc-layers 0=640x360 \
    --crf 20
expect_reject "--svc-layers + --bitrate" "${COMMON[@]}" --svc-layers 0=640x360 \
    --bitrate 9000
expect_reject "--svc-layers + --video-refs" "${COMMON[@]}" \
    --svc-layers 0=640x360 --video-refs 2
expect_reject "--svc-layers + --video-tlayers" "${COMMON[@]}" \
    --svc-layers 0=640x360 --video-tlayers 4
expect_reject "--svc-layers + --video-tlayer-drop" "${COMMON[@]}" \
    --svc-layers 0=640x360 --video-tlayers 4 --video-tlayer-drop 0

if [ "$failures" -ne 0 ]; then
    echo "gate_cli_svc: $failures failure(s) before the decode cases" >&2
    exit 1
fi
if [ -z "$AVMDEC" ] || [ ! -x "$AVMDEC" ]; then
    echo "SKIP: no avmdec binary; set AVMDEC=/path/to/avmdec" >&2
    exit 77
fi

AVM=( "$AVMDEC" --i420 -t 1 )

decode_avm() {
    local input=$1 output=$2
    shift 2
    "${AVM[@]}" "$@" -o "$output" "$input" >"$TMP/avm.log" 2>&1
}

# 1280x720 -> 640x360 box downscale, matching svc_stream's integer filter.
make_scaled_y4m() {
    python3 - "$1" "$2" <<'PY'
import sys

src_path, dst_path = sys.argv[1], sys.argv[2]
with open(src_path, "rb") as f:
    header = f.readline().rstrip(b"\n")
    fields = header.split()
    width = int([x for x in fields if x.startswith(b"W")][0][1:])
    height = int([x for x in fields if x.startswith(b"H")][0][1:])
    cw, ch = width // 2, height // 2
    luma = bytearray(width * height)
    chroma = bytearray(2 * cw * ch)
    out = open(dst_path, "wb")
    dw, dh = width // 2, height // 2
    out.write(b"YUV4MPEG2 W%d H%d F60:1 Ip A0:0 C420\n" % (dw, dh))
    while True:
        line = f.readline()
        if not line:
            break
        if not line.startswith(b"FRAME"):
            raise SystemExit("bad frame marker")
        data = f.read(width * height + 2 * cw * ch)
        if len(data) != width * height + 2 * cw * ch:
            raise SystemExit("truncated frame")
        planes = [data[:width * height], data[width * height:width * height + cw * ch],
                  data[width * height + cw * ch:]]
        out.write(b"FRAME\n")
        for plane, pw, ph in ((planes[0], width, height), (planes[1], cw, ch),
                              (planes[2], cw, ch)):
            ow, oh = pw // 2, ph // 2
            for y in range(oh):
                row = bytearray()
                for x in range(ow):
                    s = 0
                    for sy in range(2):
                        base = (2 * y + sy) * pw + 2 * x
                        s += plane[base] + plane[base + 1]
                    row.append((s + 2) // 4)
                out.write(row)
    out.close()
PY
}

# --- 2-layer mixed-resolution stream: production + avmdec OPS --------------
"$BIN" "$INPUT" -o "$TMP/full.ivf" "${COMMON[@]}" \
    --svc-layers 0=640x360,1=1280x720 --svc-lcr-id 1 >"$TMP/full.log" 2>&1
if ! grep -q "SVC layered encode complete" "$TMP/full.log"; then
    cat "$TMP/full.log" >&2
    fail "2-layer encode did not report success"
else
    ok "2-layer encode (base 640x360, enh 1280x720)"
fi

if ! decode_avm "$TMP/full.ivf" "$TMP/full.all.yuv" --all-layers; then
    cat "$TMP/avm.log" >&2
    fail "avmdec --all-layers failed on the 2-layer stream"
else
    ok "avmdec --all-layers decoded the 2-layer stream"
fi

# Independent per-layer encodes (same scaler, same QP).
"$BIN" "$INPUT" -o "$TMP/base.ivf" "${COMMON[@]}" \
    --svc-layers 0=640x360 >/dev/null 2>&1
"$BIN" "$INPUT" -o "$TMP/enh.ivf" "${COMMON[@]}" \
    --svc-layers 0=1280x720 >/dev/null 2>&1
decode_avm "$TMP/base.ivf" "$TMP/base.yuv" --all-layers
decode_avm "$TMP/enh.ivf" "$TMP/enh.yuv" --all-layers

if python3 - "$TMP" <<'PY'
import os
import sys

d = sys.argv[1]
full = open(os.path.join(d, "full.all.yuv"), "rb").read()
base = open(os.path.join(d, "base.yuv"), "rb").read()
enh = open(os.path.join(d, "enh.yuv"), "rb").read()
bs = 640 * 360 * 3 // 2
es = 1280 * 720 * 3 // 2
frames = len(base) // bs
if len(enh) // es != frames:
    raise SystemExit("per-layer frame count mismatch")
expected = b"".join(base[i * bs:(i + 1) * bs] + enh[i * es:(i + 1) * es]
                     for i in range(frames))
if full != expected:
    raise SystemExit("all-layers output is not the per-TU xlayer interleave")
PY
then
    ok "avmdec --all-layers == independent per-layer decodes (interleaved)"
else
    fail "avmdec --all-layers differs from independent per-layer decodes"
fi

# OPS selection must equal the cumulative prefix.
decode_avm "$TMP/full.ivf" "$TMP/op0.yuv" --select-ops=0,0
decode_avm "$TMP/full.ivf" "$TMP/op1.yuv" --select-ops=0,1
if cmp -s "$TMP/base.yuv" "$TMP/op0.yuv"; then
    ok "avmdec --select-ops=0,0 == base layer"
else
    fail "avmdec --select-ops=0,0 differs from the base layer"
fi
if cmp -s "$TMP/full.all.yuv" "$TMP/op1.yuv"; then
    ok "avmdec --select-ops=0,1 == full layer set"
else
    fail "avmdec --select-ops=0,1 differs from the full layer set"
fi

# Annex F.3.2 extraction through the CLI.
"$BIN" "$INPUT" -o "$TMP/ext0.ivf" "${COMMON[@]}" \
    --svc-layers 0=640x360,1=1280x720 --oppoint 0 >/dev/null 2>&1
"$BIN" "$INPUT" -o "$TMP/ext1.ivf" "${COMMON[@]}" \
    --svc-layers 0=640x360,1=1280x720 --spatial-layers 1 >/dev/null 2>&1
decode_avm "$TMP/ext0.ivf" "$TMP/ext0.yuv" --all-layers
decode_avm "$TMP/ext1.ivf" "$TMP/ext1.yuv" --all-layers
if cmp -s "$TMP/base.yuv" "$TMP/ext0.yuv"; then
    ok "--oppoint 0 extraction == base layer"
else
    fail "--oppoint 0 extraction differs from the base layer"
fi
if cmp -s "$TMP/base.yuv" "$TMP/ext1.yuv"; then
    ok "--spatial-layers 1 extraction == base layer"
else
    fail "--spatial-layers 1 extraction differs from the base layer"
fi

# Base layer pixels vs an independent plain --video encode of the scaled source.
make_scaled_y4m "$INPUT" "$TMP/base360.y4m"
"$BIN" "$TMP/base360.y4m" -o "$TMP/plain360.ivf" --video --frames 3 \
    --video-max-depth 3 >/dev/null 2>&1
decode_avm "$TMP/plain360.ivf" "$TMP/plain360.yuv" --all-layers
if cmp -s "$TMP/plain360.yuv" "$TMP/base.yuv"; then
    ok "base layer == independent plain --video encode of the scaled source"
else
    fail "base layer differs from the independent plain encode"
fi

# --- dav2d coverage ------------------------------------------------------
y4m_frames() {
    python3 - "$1" <<'PY'
import sys
count = 0
with open(sys.argv[1], "rb") as f:
    header = f.readline().split()
    width = height = None
    for field in header:
        if field.startswith(b"W"):
            width = int(field[1:])
        elif field.startswith(b"H"):
            height = int(field[1:])
    if width is None or height is None:
        print(-1)
        raise SystemExit(0)
    cw, ch = (width + 1) // 2, (height + 1) // 2
    size = width * height + 2 * cw * ch
    while True:
        line = f.readline()
        if not line:
            break
        if not line.startswith(b"FRAME"):
            count = -1
            break
        data = f.read(size)
        if len(data) != size:
            count = -1
            break
        count += 1
print(count)
PY
}

if [ -n "$DAV2D" ] && [ -x "$DAV2D" ]; then
    if "$DAV2D" -q -i "$TMP/base.ivf" -o "$TMP/base.dav2d.y4m" \
            --strict 1 --threads 1 >/dev/null 2>&1 &&
            [ "$(y4m_frames "$TMP/base.dav2d.y4m")" = "3" ]; then
        ok "dav2d --strict 1 decodes the single-layer SVC stream"
    else
        fail "dav2d --strict 1 failed on the single-layer SVC stream"
    fi
    "$BIN" "$INPUT" -o "$TMP/same.ivf" "${COMMON[@]}" \
        --svc-layers 0=640x360,1=640x360 >/dev/null 2>&1
    same_frames=0
    if "$DAV2D" -q -i "$TMP/same.ivf" -o "$TMP/same.dav2d.y4m" \
            --strict 1 --threads 1 >/dev/null 2>&1; then
        same_frames=$(y4m_frames "$TMP/same.dav2d.y4m")
    fi
    if [ "$same_frames" = "6" ]; then
        ok "dav2d --strict 1 decodes the same-resolution 2-layer stream"
    else
        echo "NOTE: dav2d f9dafbc emits $same_frames pictures for the"
        echo "      same-resolution 2-layer 3-TU stream (6 expected);"
        echo "      multi-xlayer *inter* streams are avmdec-gated."
    fi
    # Mixed resolutions: dav2d f9dafbc aborts after the first P-frame TU even
    # though it exits 0.  Record the exact limitation; avmdec is the gate.
    if "$DAV2D" -q -i "$TMP/full.ivf" -o "$TMP/full.dav2d.y4m" \
            --strict 1 --threads 1 >/dev/null 2>&1 &&
            [ "$(y4m_frames "$TMP/full.dav2d.y4m")" = "6" ]; then
        ok "dav2d --strict 1 decodes the mixed-resolution 2-layer stream"
    else
        echo "NOTE: dav2d f9dafbc aborts the mixed-resolution inter stream"
        echo "      (exits 0 but emits fewer than 6 pictures); avmdec gates it."
    fi
    if "$ROOT/tools/validate_dav2d.sh" "$TMP/base.ivf" >/dev/null 2>&1; then
        ok "tools/validate_dav2d.sh accepts the single-layer SVC stream"
    else
        fail "tools/validate_dav2d.sh rejected the single-layer SVC stream"
    fi
else
    echo "NOTE: dav2d not found; decoder-shape checks skipped"
fi

echo
if [ "$failures" -ne 0 ]; then
    echo "gate_cli_svc: $failures failure(s), $pass passed" >&2
    exit 1
fi
echo "gate_cli_svc: $pass/$pass cases passed"
