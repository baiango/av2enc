#!/usr/bin/env bash
# Decoder gate for the §7.7/§7.8 implicit reference map (--video-implicit-ref-map).
#
# The stream sets sequence explicit_ref_frame_map = 0, so no frame codes
# frame_explicit_ref_frame_map / num_total_refs / ref_frame_idx.  The decoder
# derives both from the §7.23 slot state, and the encoder must use the same
# mapping for read_single_ref().
#
# Checks, for the flat P chain, the B-frame schedule and the hierarchical-P
# temporal layers:
#   * --video-implicit-ref-map changes the stream (no map bits are written);
#   * --video-trace reports the derived NumTotalRefs (must exceed 1 somewhere,
#     and must not be the constant explicit --video-refs value);
#   * dav2d --strict 1 decodes the IVF header's picture count;
#   * the decoded samples equal the encoder reconstruction sample for sample.
#
# Usage: gate_implicit_ref_map.sh --bin build-cpu/av2enc [--dav2d PATH]
#        [--input testdata/shine_1280x720_60f.y4m]
# Exit: 0 pass, 1 fail, 2 usage, 77 skip.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
BIN=""
DAV2D="${DAV2D:-}"
SRC="${AV2ENC_TESTDATA:-$ROOT/testdata}/shine_1280x720_60f.y4m"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

while [ $# -gt 0 ]; do
    case "$1" in
        --bin) BIN=$2; shift 2 ;;
        --dav2d) DAV2D=$2; shift 2 ;;
        --input) SRC=$2; shift 2 ;;
        -h|--help) sed -n '2,23p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "gate_implicit_ref_map: unknown argument $1" >&2; exit 2 ;;
    esac
done
[ -n "$BIN" ] && [ -x "$BIN" ] || {
    echo "gate_implicit_ref_map: --bin required" >&2; exit 2; }
if [ -z "$DAV2D" ]; then
    for candidate in "$ROOT/tools/bin/dav2d" "$ROOT/tools/bin/dav2d-macos"; do
        if [ -x "$candidate" ]; then DAV2D=$candidate; break; fi
    done
fi
[ -n "$DAV2D" ] && [ -x "$DAV2D" ] || {
    echo "gate_implicit_ref_map: no dav2d" >&2; exit 77; }
[ -f "$SRC" ] || { echo "gate_implicit_ref_map: missing $SRC" >&2; exit 77; }

md5_of() { md5 -q "$1" 2>/dev/null || md5sum "$1" | awk '{print $1}'; }

compare_recon() {
    python3 - "$1" "$2" "$3" "$4" <<'PY'
import struct
import sys

rec, dec, ivf, name = sys.argv[1:5]


def read_y4m(path):
    frames = []
    with open(path, "rb") as f:
        header = f.readline()
        if not header.startswith(b"YUV4MPEG2"):
            return frames
        tokens = header.split()
        width = int(next(t for t in tokens if t.startswith(b"W"))[1:])
        height = int(next(t for t in tokens if t.startswith(b"H"))[1:])
        chroma = next(t for t in tokens if t.startswith(b"C"))[1:]
        if chroma.startswith(b"420"):
            size = width * height + 2 * ((width + 1) // 2) * ((height + 1) // 2)
        elif chroma.startswith(b"422"):
            size = width * height + 2 * ((width + 1) // 2) * height
        elif chroma.startswith(b"444"):
            size = 3 * width * height
        else:
            size = width * height
        while True:
            line = f.readline()
            if not line:
                break
            if not line.startswith(b"FRAME"):
                break
            frame = f.read(size)
            if len(frame) != size:
                break
            frames.append(frame)
    return frames


with open(ivf, "rb") as f:
    data = f.read()
declared = struct.unpack_from("<I", data, 24)[0]
expected = read_y4m(rec)
decoded = read_y4m(dec)
if declared != len(decoded):
    print(f"{name}: FAIL IVF declares {declared} picture(s), decoder wrote "
          f"{len(decoded)}")
    sys.exit(1)
if len(expected) != len(decoded):
    print(f"{name}: FAIL recon has {len(expected)} picture(s), decoder wrote "
          f"{len(decoded)}")
    sys.exit(1)
diffs = 0
worst = 0
for a, b in zip(expected, decoded):
    for x, y in zip(a, b):
        if x != y:
            diffs += 1
            worst = max(worst, abs(x - y))
if diffs:
    print(f"{name}: FAIL {diffs} sample(s) differ, max {worst}")
    sys.exit(1)
print(f"{name}: PASS {declared} picture(s), recon exact")
PY
}

STATUS=0
run_case() {
    local name=$1 frames=$2
    shift 2
    local extra=("$@")

    if ! "$BIN" "$SRC" -o "$TMP/$name.implicit.ivf" --video --frames "$frames" \
            --video-implicit-ref-map --video-recon "$TMP/$name.recon.y4m" \
            --video-trace ${extra[@]+"${extra[@]}"} \
            >"$TMP/$name.log" 2>&1; then
        echo "FAIL $name: implicit encode" >&2
        tail -5 "$TMP/$name.log" >&2
        STATUS=1
        return
    fi
    if ! "$BIN" "$SRC" -o "$TMP/$name.explicit.ivf" --video --frames "$frames" \
            ${extra[@]+"${extra[@]}"} >/dev/null 2>&1; then
        echo "FAIL $name: explicit encode" >&2
        STATUS=1
        return
    fi
    if cmp -s "$TMP/$name.implicit.ivf" "$TMP/$name.explicit.ivf"; then
        echo "FAIL $name: implicit stream is byte-identical to the explicit map" \
             >&2
        STATUS=1
        return
    fi
    local imp_size exp_size
    imp_size=$(wc -c <"$TMP/$name.implicit.ivf" | tr -d ' ')
    exp_size=$(wc -c <"$TMP/$name.explicit.ivf" | tr -d ' ')

    # The trace proves the encoder used the derived map (variable
    # NumTotalRefs), not the constant --video-refs default.
    if ! grep -q "implicit ref map .* num_total_refs=" "$TMP/$name.log"; then
        echo "FAIL $name: no implicit map trace" >&2
        STATUS=1
        return
    fi
    local max_refs
    max_refs=$(grep -o "num_total_refs=[0-9]*" "$TMP/$name.log" |
               sed 's/.*=//' | sort -n | tail -1)
    if [ "${max_refs:-0}" -lt 2 ]; then
        echo "FAIL $name: derived NumTotalRefs never exceeds 1" >&2
        STATUS=1
        return
    fi

    "$DAV2D" -q -i "$TMP/$name.implicit.ivf" -o "$TMP/$name.dec.y4m" \
        --threads 1 --strict 1 >"$TMP/$name.dec.log" 2>&1 || {
        echo "FAIL $name: dav2d --strict 1 rejected the stream" >&2
        tail -5 "$TMP/$name.dec.log" >&2
        STATUS=1
        return
    }
    if ! compare_recon "$TMP/$name.recon.y4m" "$TMP/$name.dec.y4m" \
            "$TMP/$name.implicit.ivf" "$name"; then
        STATUS=1
        return
    fi
    echo "  $name: max NumTotalRefs=$max_refs," \
         "md5 $(md5_of "$TMP/$name.implicit.ivf")," \
         "$imp_size B (explicit $exp_size B)"
}

# Flat P chain: NumTotalRefs grows 1,2,3,... as the slot pool fills.
run_case flat 6
# A four-slot pool caps the derivation at four references.
run_case flat_nrf4 6 --video-num-ref-frames 4
# B schedules (coding order != display order: mixed past/future refs).
run_case b2 4 --b-frames 2
run_case b4 6 --b-frames 4
# Hierarchical-P temporal layers (obu_tlayer_id + the default dependency map).
run_case tlayers4 4 --video-tlayers 4
run_case tlayers8 8 --video-tlayers 8

if [ "$STATUS" -ne 0 ]; then
    exit 1
fi
echo "PASS implicit reference map: every carrier decoded with dav2d --strict 1"
