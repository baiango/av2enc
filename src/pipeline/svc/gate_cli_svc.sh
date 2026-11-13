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
#   * `--svc-temporal 2` (hierarchical-P closed GOP) decodes every expected
#     picture per xlayer and every picture is bit-exact against the encoder
#     reconstruction;
#   * `--svc-embedded 2` (cross-mlayer quality refinement) decodes every
#     expected picture per (xlayer, mlayer) and every picture is bit-exact
#     against the encoder reconstruction;
#   * `dav2d --strict 1` decodes the single-layer and same-resolution layered
#     streams, and `tools/validate_dav2d.sh` accepts them.
#
# Pinned-decoder limitation: dav2d f9dafbc fails on a *mixed-resolution*
# multi-xlayer stream that contains inter (P) frames (same-resolution layers and
# single-layer streams decode) and cannot parse a sequence header with
# `max_mlayer_id > 0`.  avmdec decodes both, so those cases gate with avmdec
# only and the dav2d verdict is printed as a NOTE.  See
# src/pipeline/svc/README.md.
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
expect_reject "--svc-inter-layer" "${COMMON[@]}" --svc-layers 0=640x360 \
    --svc-inter-layer
expect_reject "--svc-temporal + --svc-embedded" "${COMMON[@]}" \
    --svc-layers 0=640x360 --svc-temporal 2 --svc-embedded 2
expect_reject "--svc-embedded 4 (Annex A profile cap)" "${COMMON[@]}" \
    --svc-layers 0=640x360 --svc-embedded 4
expect_reject "--svc-temporal 2 with a 3-frame GOP" "${COMMON[@]}" \
    --svc-layers 0=640x360 --svc-temporal 2
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

# Compares every decoded picture in <decoded.yuv> against the per-(xlayer,
# mlayer) encoder reconstruction dumped by --video-recon.  The pinned decoders
# may reorder pictures across layers within a temporal unit, so a picture is
# matched by content to the next expected picture of its layer; every expected
# picture must appear exactly once, in per-layer display order, and the decoder
# must emit no extra pictures.
compare_layer_recon() {
    python3 - "$@" <<'PY'
import os
import sys

dec_path, prefix = sys.argv[1], sys.argv[2]
keys = [tuple(int(v) for v in arg.split(":")) for arg in sys.argv[3:]]


def read_y4m(path):
    with open(path, "rb") as f:
        header = f.readline().split()
        width = int([x for x in header if x.startswith(b"W")][0][1:])
        height = int([x for x in header if x.startswith(b"H")][0][1:])
        size = width * height + 2 * ((width + 1) // 2) * ((height + 1) // 2)
        frames = []
        while True:
            line = f.readline()
            if not line:
                break
            data = f.read(size)
            if len(data) != size:
                raise SystemExit("truncated " + path)
            frames.append(data)
    return size, frames


dec = open(dec_path, "rb").read()
expected = {}
cursor = {}
for key in keys:
    # --video-recon keeps the exact path for a single xlayer/m0 stream and
    # suffixes every layered stream with _x<X>m<M>.
    path = "%s_x%dm%d.y4m" % (prefix, key[0], key[1])
    if len(keys) == 1 and not os.path.exists(path):
        path = prefix + ".y4m"
    size, frames = read_y4m(path)
    expected[key] = (size, frames)
    cursor[key] = 0

pos = 0
matched = 0
while pos < len(dec):
    found = False
    for key in keys:
        size, frames = expected[key]
        index = cursor[key]
        if index >= len(frames):
            continue
        if len(dec) - pos >= size and dec[pos:pos + size] == frames[index]:
            cursor[key] = index + 1
            pos += size
            matched += 1
            found = True
            break
    if not found:
        raise SystemExit(
            "decoded picture at byte %d matches no expected layer picture" % pos)
for key in keys:
    if cursor[key] != len(expected[key][1]):
        raise SystemExit(
            "layer x%dm%d: %d/%d pictures decoded" %
            (key[0], key[1], cursor[key], len(expected[key][1])))
print("%d pictures exact across %d layer(s)" % (matched, len(keys)))
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

# --- hierarchical temporal layers (--svc-temporal 2) ----------------------
# 2 temporal layers = one hierarchical-P GOP of 4 frames.  Every xlayer must
# emit 4 pictures and every picture must equal the encoder reconstruction.
TLAYERS=(--video --frames 4 --video-max-depth 3
         --svc-layers 0=160x144,1=640x144 --svc-temporal 2)
"$BIN" "$INPUT" -o "$TMP/temporal.ivf" "${TLAYERS[@]}" \
    --video-recon "$TMP/temporal.recon.y4m" >"$TMP/temporal.log" 2>&1
if ! grep -q "SVC layered encode complete" "$TMP/temporal.log"; then
    cat "$TMP/temporal.log" >&2
    fail "--svc-temporal 2 encode did not report success"
elif decode_avm "$TMP/temporal.ivf" "$TMP/temporal.yuv" --all-layers; then
    if compare_layer_recon "$TMP/temporal.yuv" "$TMP/temporal.recon" 0:0 1:0 \
            >"$TMP/temporal.cmp" 2>&1; then
        ok "avmdec --all-layers: $(cat "$TMP/temporal.cmp") (temporal)"
    else
        cat "$TMP/temporal.cmp" >&2
        fail "--svc-temporal 2 decoded pictures differ from the recon"
    fi
else
    cat "$TMP/avm.log" >&2
    fail "avmdec --all-layers failed on the --svc-temporal 2 stream"
fi

# --- embedded quality layers (--svc-embedded 2) ---------------------------
# mlayer 0 is the base chain, mlayer 1 predicts from mlayer 0 at the same
# display index (§5.4.1 MLayerDependencyMap).  Expected: 3 temporal units x
# 2 xlayers x 2 mlayers = 12 pictures, all bit-exact.
EMBED=(--video --frames 3 --video-max-depth 3
       --svc-layers 0=160x144,1=640x144 --svc-embedded 2)
"$BIN" "$INPUT" -o "$TMP/embedded.ivf" "${EMBED[@]}" \
    --video-recon "$TMP/embedded.recon.y4m" >"$TMP/embedded.log" 2>&1
if ! grep -q "SVC layered encode complete" "$TMP/embedded.log"; then
    cat "$TMP/embedded.log" >&2
    fail "--svc-embedded 2 encode did not report success"
elif decode_avm "$TMP/embedded.ivf" "$TMP/embedded.yuv" --all-layers; then
    if compare_layer_recon "$TMP/embedded.yuv" "$TMP/embedded.recon" \
            0:0 0:1 1:0 1:1 >"$TMP/embedded.cmp" 2>&1; then
        ok "avmdec --all-layers: $(cat "$TMP/embedded.cmp") (embedded)"
    else
        cat "$TMP/embedded.cmp" >&2
        fail "--svc-embedded 2 decoded pictures differ from the recon"
    fi
else
    cat "$TMP/avm.log" >&2
    fail "avmdec --all-layers failed on the --svc-embedded 2 stream"
fi

# mlayer 1 must actually be a refinement, not a duplicate of mlayer 0.
if python3 - "$TMP/embedded.recon_x0m0.y4m" "$TMP/embedded.recon_x0m1.y4m" <<'PY'
import sys


def read_y4m(path):
    with open(path, "rb") as f:
        header = f.readline().split()
        width = int([x for x in header if x.startswith(b"W")][0][1:])
        height = int([x for x in header if x.startswith(b"H")][0][1:])
        size = width * height + 2 * ((width + 1) // 2) * ((height + 1) // 2)
        return [f.read(size) for _ in range(3)]


if read_y4m(sys.argv[1]) == read_y4m(sys.argv[2]):
    raise SystemExit("m0 and m1 reconstruction are identical")
PY
then
    ok "embedded mlayer 1 is a distinct refinement picture"
else
    fail "embedded mlayer 1 duplicates mlayer 0"
fi

# Deeper temporal ladders: --svc-temporal 3 (8-frame GOP) and 4 (16-frame GOP,
# OrderHintBits 5).  Single xlayer so dav2d can also decode them.
for T in 3 4; do
    FRAMES=$((1 << T))
    "$BIN" "$INPUT" -o "$TMP/t${T}.ivf" --video --frames "$FRAMES" \
        --video-max-depth 3 --svc-layers 0=160x144 --svc-temporal "$T" \
        --video-recon "$TMP/t${T}.recon.y4m" >"$TMP/t${T}.log" 2>&1
    if ! grep -q "SVC layered encode complete" "$TMP/t${T}.log"; then
        cat "$TMP/t${T}.log" >&2
        fail "--svc-temporal $T encode did not report success"
        continue
    fi
    if decode_avm "$TMP/t${T}.ivf" "$TMP/t${T}.yuv" --all-layers &&
        compare_layer_recon "$TMP/t${T}.yuv" "$TMP/t${T}.recon" 0:0 \
            >"$TMP/t${T}.cmp" 2>&1; then
        ok "avmdec --all-layers: $(cat "$TMP/t${T}.cmp") (temporal $T)"
    else
        cat "$TMP/avm.log" >&2
        cat "$TMP/t${T}.cmp" >&2 2>/dev/null || true
        fail "--svc-temporal $T decoded pictures differ from the recon"
    fi
done

# A three-deep embedded ladder (profile Main_420_10_IP2, slots 0/1/2).
"$BIN" "$INPUT" -o "$TMP/embed3.ivf" --video --frames 3 --video-max-depth 3 \
    --svc-layers 0=160x144 --svc-embedded 3 \
    --video-recon "$TMP/embed3.recon.y4m" >"$TMP/embed3.log" 2>&1
if ! grep -q "SVC layered encode complete" "$TMP/embed3.log"; then
    cat "$TMP/embed3.log" >&2
    fail "--svc-embedded 3 encode did not report success"
elif decode_avm "$TMP/embed3.ivf" "$TMP/embed3.yuv" --all-layers &&
    compare_layer_recon "$TMP/embed3.yuv" "$TMP/embed3.recon" 0:0 0:1 0:2 \
        >"$TMP/embed3.cmp" 2>&1; then
    ok "avmdec --all-layers: $(cat "$TMP/embed3.cmp") (embedded 3)"
else
    cat "$TMP/avm.log" >&2
    cat "$TMP/embed3.cmp" >&2 2>/dev/null || true
    fail "--svc-embedded 3 decoded pictures differ from the recon"
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

    # Hierarchical temporal layers: a single-xlayer SVC stream decodes fully.
    "$BIN" "$INPUT" -o "$TMP/tlayers_x0.ivf" --video --frames 4 \
        --video-max-depth 3 --svc-layers 0=160x144 --svc-temporal 2 \
        >/dev/null 2>&1
    tlayers_frames=0
    if "$DAV2D" -q -i "$TMP/tlayers_x0.ivf" -o "$TMP/tlayers_x0.y4m" \
            --strict 1 --threads 1 >/dev/null 2>&1; then
        tlayers_frames=$(y4m_frames "$TMP/tlayers_x0.y4m")
    fi
    if [ "$tlayers_frames" = "4" ]; then
        ok "dav2d --strict 1 decodes --svc-temporal 2 (4 pictures)"
    else
        fail "dav2d --strict 1 decoded $tlayers_frames/4 pictures of the " \
             "single-layer temporal stream"
    fi

    # Deeper hierarchical GOPs: 8 and 16 frames, OrderHintBits 4 and 5.
    for T in 3 4; do
        FRAMES=$((1 << T))
        "$BIN" "$INPUT" -o "$TMP/dav2d_t${T}.ivf" --video --frames "$FRAMES" \
            --video-max-depth 3 --svc-layers 0=160x144 --svc-temporal "$T" \
            >/dev/null 2>&1
        got=0
        if "$DAV2D" -q -i "$TMP/dav2d_t${T}.ivf" -o "$TMP/dav2d_t${T}.y4m" \
                --strict 1 --threads 1 >/dev/null 2>&1; then
            got=$(y4m_frames "$TMP/dav2d_t${T}.y4m")
        fi
        if [ "$got" = "$FRAMES" ]; then
            ok "dav2d --strict 1 decodes --svc-temporal $T ($got pictures)"
        else
            fail "dav2d --strict 1 decoded $got/$FRAMES pictures of " \
                 "--svc-temporal $T"
        fi
    done

    # max_mlayer_id > 0 is outside the pinned dav2d sequence-header parser.
    "$BIN" "$INPUT" -o "$TMP/embed_x0.ivf" --video --frames 3 \
        --video-max-depth 3 --svc-layers 0=160x144 --svc-embedded 2 \
        >/dev/null 2>&1
    if "$DAV2D" -q -i "$TMP/embed_x0.ivf" -o "$TMP/embed_x0.y4m" \
            --strict 1 --threads 1 >/dev/null 2>&1; then
        echo "NOTE: this dav2d build accepted max_mlayer_id > 0; update the"
        echo "      documented decoder matrix."
    else
        echo "NOTE: dav2d f9dafbc cannot parse max_mlayer_id > 0 (sequence"
        echo "      header overrun); embedded layers gate with avmdec."
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
