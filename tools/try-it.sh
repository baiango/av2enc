#!/usr/bin/env bash
# Hands-on: encode a stream with one of the three measured presets and decode
# it back with whatever decoders are installed.
#
# Nothing here is pinned: no expected byte counts, no MD5s, no commit hashes.
# Two checks per decode: did every picture come back, and is the result
# pixel-identical to the encoder's own reconstruction (--video-recon)? The input
# is generated every run (ffmpeg testsrc2, or a tiny Python Y4M writer when
# ffmpeg is absent), so there is no test media to fetch.
#
# The three presets are the ones in README "Presets"; each is only an argument
# bundle for the flags listed below it, so this script stays correct when the
# presets change. Add --verbose to print the exact command it ran.
#
#   --fast-decode            shipped CPU RD path        (seconds)
#   --video-rd-preset best   flat best stack            (research; slow)
#   --nightly-b              chained single-key B       (research; --video-gops N)
#
# Usage: tools/try-it.sh [options]
#   --preset NAME|=fast-decode|best|nightly-b   (default: fast-decode)
#   --av2enc PATH      binary (default: build-release, build-pub, build)
#   --frames N         video length                      (default: 12)
#   --size WxH         frame size                        (default: 128x96)
#   --qp Q             base_q_idx                        (default: per preset)
#   --gops N           chained blocks for nightly-b       (default: 2)
#   --all-presets      run all three back to back
#   --verbose          print the encoder command lines
#   --keep             keep the streams and decoded Y4M
# Exit: 0 everything that could be checked passed, 1 an encode failed,
#       2 setup problem. A missing or short decoder is reported, not failed on.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=""
PRESET=fast-decode
ALL_PRESETS=0
FRAMES=12
SIZE=128x96
QP=""
GOPS=2
KEEP=0
VERBOSE=0

while [ $# -gt 0 ]; do
    case "$1" in
        --preset) PRESET=${2:-}; shift 2 ;;
        --av2enc) BIN=${2:-}; shift 2 ;;
        --frames) FRAMES=${2:-}; shift 2 ;;
        --size) SIZE=${2:-}; shift 2 ;;
        --qp) QP=${2:-}; shift 2 ;;
        --gops) GOPS=${2:-}; shift 2 ;;
        --all-presets) ALL_PRESETS=1; shift ;;
        --verbose) VERBOSE=1; shift ;;
        --keep) KEEP=1; shift ;;
        -h|--help) sed -n '2,33p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "try-it: unknown argument: $1" >&2; exit 2 ;;
    esac
done

# preset_args <name>: sets PA_QPS (default QP), PA_FLAGS.
# Keep this in step with README "Presets"; it is deliberately nothing more
# than the flag bundle, so there is no pinned stream to drift.
preset_args() {
    case "$1" in
        fast-decode)
            PA_QPS=106
            PA_FLAGS=(--fast-decode)
            PA_NOTE="shipped CPU RD path" ;;
        best)
            PA_QPS=112
            PA_FLAGS=(--video-rd-preset best)
            PA_NOTE="flat best stack, research (slow)" ;;
        nightly-b)
            PA_QPS=118
            PA_FLAGS=(--video-gops "$GOPS" --nightly-b)
            PA_NOTE="chained single-key hierarchical B, research" ;;
        *)
            echo "try-it: unknown preset '$1' (fast-decode, best, nightly-b)" >&2
            exit 2 ;;
    esac
    [ -n "$QP" ] && PA_QPS=$QP
}

if [ -z "$BIN" ]; then
    for cand in "$ROOT/build-release/av2enc" "$ROOT/build-pub/av2enc" "$ROOT/build/av2enc"; do
        [ -x "$cand" ] && { BIN=$cand; break; }
    done
fi
if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
    echo "try-it: no av2enc binary found." >&2
    echo "  cmake --preset release && cmake --build --preset release -j8" >&2
    echo "  then: tools/try-it.sh --av2enc build-release/av2enc" >&2
    exit 2
fi

TMP=$(mktemp -d "${TMPDIR:-/tmp}/av2enc-try.XXXXXX")
cleanup() { [ "$KEEP" -eq 1 ] || rm -rf "$TMP"; }
trap cleanup EXIT

fail=0
note() { printf '%s\n' "$*"; }
bad() { printf 'FAIL: %s\n' "$*"; fail=1; }

# --- input ----------------------------------------------------------------
gen_input() {  # gen_input <path> <WxH> <frames>
    local path=$1 size=$2 frames=$3
    local w=${size%x*} h=${size#*x}
    if command -v ffmpeg >/dev/null 2>&1; then
        ffmpeg -hide_banner -loglevel error -f lavfi \
            -i "testsrc2=size=$size:rate=30:duration=2" \
            -frames:v "$frames" -pix_fmt yuv420p -f yuv4mpegpipe "$path" 2>/dev/null
    fi
    if [ -s "$path" ]; then return 0; fi
    if ! command -v python3 >/dev/null 2>&1; then
        echo "try-it: need ffmpeg or python3 to generate input" >&2
        exit 2
    fi
    python3 - "$path" "$w" "$h" "$frames" <<'PY'
import sys
path, w, h, frames = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
with open(path, "wb") as f:
    f.write(f"YUV4MPEG2 W{w} H{h} F30:1 Ip A1:1 C420jpeg\n".encode())
    chroma = bytes([128]) * (w // 2 * h // 2)
    for n in range(frames):
        y = bytearray(w * h)
        for r in range(h):
            base = (r * 255) // (h - 1)
            row = r * w
            for c in range(w):
                y[row + c] = (base + (c * 3) // 4 + n * 3) & 0xFF
        bw, bh = max(1, w // 6), max(1, h // 5)
        x0 = (n * 7) % max(1, w - bw)
        y0 = (n * 5) % max(1, h - bh)
        for r in range(y0, y0 + bh):
            row = r * w
            for c in range(x0, x0 + bw):
                y[row + c] = 235
        f.write(b"FRAME\n")
        f.write(bytes(y))
        f.write(chroma)
        f.write(chroma)
PY
    [ -s "$path" ] || { echo "try-it: could not generate $path" >&2; exit 2; }
}

frame_count() {
    if command -v python3 >/dev/null 2>&1; then
        python3 -c 'import sys; print(open(sys.argv[1], "rb").read().count(b"FRAME"))' "$1"
    else
        grep -a -c FRAME "$1"
    fi
}

decoders=()
command -v avmdec >/dev/null 2>&1 && decoders+=("avmdec")
command -v dav2d >/dev/null 2>&1 && decoders+=("dav2d")
command -v splot >/dev/null 2>&1 && decoders+=("splot")
[ -x "$ROOT/tools/bin/dav2d" ] && decoders+=("$ROOT/tools/bin/dav2d")
[ -x "$ROOT/tools/bin/dav2d-macos" ] && decoders+=("$ROOT/tools/bin/dav2d-macos")

clock() { python3 -c 'import time;print(time.time())' 2>/dev/null || date +%s; }

# md5 of everything from the first FRAME marker on: the Y4M header carries the
# frame rate, which a decoder may normalise, so the header is not compared.
payload_md5() {
    if command -v python3 >/dev/null 2>&1; then
        python3 -c '
import hashlib, sys
d = open(sys.argv[1], "rb").read()
i = d.find(b"FRAME")
print(hashlib.md5(d[i:] if i >= 0 else d).hexdigest())' "$1"
    else
        tail -c +1 "$1" | grep -a -b -m1 FRAME | cut -d: -f1 | xargs -I{} tail -c +{} "$1" | md5 2>/dev/null \
            || md5 -q "$1"
    fi
}

# --- one preset ------------------------------------------------------------
run_one() {
    local name=$1
    preset_args "$name"
    local in="$TMP/in_${name}.y4m"
    local ivf="$TMP/${name}.ivf"
    local flags=("${PA_FLAGS[@]}")
    local args=(--video --frames "$FRAMES" --qp "$PA_QPS" --cpu-threads 8 "${flags[@]}")

    gen_input "$in" "$SIZE" "$FRAMES"
    note "preset: $name  ($PA_NOTE)"
    [ "$VERBOSE" -eq 1 ] && note "  av2enc $in -o $ivf ${args[*]}"

    local s e secs
    s=$(clock)
    local recon="$TMP/${name}.recon.y4m"
    # --video-recon writes the encoder's own reconstruction, so the decode is
    # compared picture-for-picture rather than only counted. It costs one
    # extra file write and no extra encode time.
    if ! "$BIN" "$in" -o "$ivf" --video-recon "$recon" "${args[@]}" \
            >"$TMP/enc_$name.log" 2>&1; then
        bad "$name: encode failed"; tail -n 5 "$TMP/enc_$name.log"; return 1
    fi
    e=$(clock)
    [ -s "$ivf" ] || { bad "$name: encoder wrote nothing"; return 1; }
    secs=$(python3 -c "print(f'{$e-$s:.2f}s')" 2>/dev/null || echo "?")

    note "encode: $SIZE ${FRAMES}f qp$PA_QPS -> $(wc -c <"$ivf" | tr -d ' ') bytes in $secs"

    if [ ${#decoders[@]} -eq 0 ]; then
        note "decode: no decoder found (avmdec / dav2d / splot on PATH, or tools/bin/dav2d)"
        KEEP=1
        return 0
    fi
    note "decode:"
    local d
    for d in "${decoders[@]}"; do
        local dname
        dname=$(basename "$d")
        local out="$TMP/${name}.${dname}.y4m"
        case "$dname" in
            avmdec) "$d" "$ivf" -o "$out" -t 1 >"$TMP/dec.log" 2>&1 ;;
            splot)  "$d" decode "$ivf" -o "$out" >"$TMP/dec.log" 2>&1 ;;
            *)      "$d" -q -i "$ivf" -o "$out" --threads 1 >"$TMP/dec.log" 2>&1 ;;
        esac
        local rc=$?
        if [ $rc -ne 0 ] || [ ! -s "$out" ]; then
            note "  $dname: decoder exit $rc (no output)"
            continue
        fi
        local got
        got=$(frame_count "$out")
        # Recon equality is only meaningful when every picture came back, and
        # it is only compared picture-for-picture at the same count; otherwise
        # the byte stream differs trivially (a missing picture shifts the rest).
        local recon_state=""
        if [ -s "$recon" ]; then
            if [ "$got" = "$FRAMES" ]; then
                if [ "$(payload_md5 "$out")" = "$(payload_md5 "$recon")" ]; then
                    recon_state="recon-equal"
                else
                    recon_state="recon-DIFFERS"
                fi
            else
                recon_state="recon-n/a (short)"
            fi
        fi
        if [ "$got" = "$FRAMES" ]; then
            note "  $dname: $got/$FRAMES pictures  ${recon_state}"
        else
            note "  $dname: $got/$FRAMES pictures  <-- short  ${recon_state}"
        fi
    done
    note ""
}

note "av2enc: $BIN"
note ""
if [ "$ALL_PRESETS" -eq 1 ]; then
    for p in fast-decode best nightly-b; do run_one "$p"; done
else
    run_one "$PRESET"
fi

if [ "$KEEP" -eq 1 ]; then
    note "kept: $TMP"
else
    note "done. --keep keeps the streams and decoded Y4M."
fi
exit $fail
