#!/usr/bin/env bash
# M1 reconstruction verification matrix.
#
# For each resolution and intra-mode configuration:
#   1. encode one Y4M frame with the real CLI  -> IVF
#   2. reproduce the encode with m1_dump        -> encoder recon
#   3. decode the IVF with the pinned dav2d     -> decoder recon
#   4. compare bit-exactly with verify.py
#
# Descriptions:
#   dc         --m1                      (luma_dc_only = true)
#   force:N    --m1 --m1-full-intra      + AV2ENC_M1_FORCE_Y=N:0
#   full       --m1 --m1-full-intra      (mode search)
#
# Usage (builds and runs on the remote 3060 box):
#   tools/m1_verify/run_matrix.sh --remote
#
# Usage (directly on the box, from the repo root):
#   tools/m1_verify/run_matrix.sh
#
# Options:
#   --remote             tar tools/m1_verify to $AV2ENC_REMOTE and run there
#   --sizes "LIST"       default "64x64 640x360 1920x1080"
#   --modes "LIST"       default "dc force:0 ... force:12 full"
#   --depth N            --m1-depth for CLI and harness (default: encoder 4)
#   --no-deblock         pass --no-deblock to both
#   --no-cfl             pass --no-cfl to both
#   --rebuild            rebuild build-cuda (av2enc target) before running
#   --keep               keep IVF/dump/decoded Y4M (default: keep)
#   --clean              delete per-case artifacts after verifying
#   --timeout SECONDS    per-command timeout (default 600, 0 = none)
#
# Env:
#   AV2ENC_REMOTE=<user>@<host>   AV2ENC_PORT=<port>
#   AV2ENC_DEST=/workspace/av2enc       DAV2D=/path/to/dav2d
#   CXX=g++                             CUDART_LIB_DIR=/usr/local/cuda/lib64
#
# Exit: 0 only if every case is bit-exact; 1 if any case differs; 2 on setup
# error. A mismatch is data, not a crash, so the matrix keeps going.

set -uo pipefail

SELF=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SELF/../.." && pwd)

REMOTE=0
SIZES="64x64 640x360 1920x1080"
MODES=""
DEPTH=""
DEBLOCK=1
CFL=1
REBUILD=0
CLEAN=0
TIMEOUT=600

for ((i = 0; i <= 12; i++)); do
    MODES+="force:$i "
done
MODES="dc ${MODES}full"

while [ $# -gt 0 ]; do
    case "$1" in
        --local) ;;
        --remote) REMOTE=1 ;;
        --sizes) SIZES=$2; shift ;;
        --modes) MODES=$2; shift ;;
        --depth) DEPTH=$2; shift ;;
        --no-deblock) DEBLOCK=0 ;;
        --no-cfl) CFL=0 ;;
        --rebuild) REBUILD=1 ;;
        --keep) CLEAN=0 ;;
        --clean) CLEAN=1 ;;
        --timeout) TIMEOUT=$2; shift ;;
        -h|--help) sed -n '2,45p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "run_matrix: unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

AV2ENC_REMOTE=${AV2ENC_REMOTE:?set AV2ENC_REMOTE (user@host)}
AV2ENC_PORT=${AV2ENC_PORT:-<port>}
AV2ENC_DEST=${AV2ENC_DEST:-/workspace/av2enc}

if [ "$REMOTE" = 1 ]; then
    echo "== syncing tools/m1_verify to $AV2ENC_REMOTE:$AV2ENC_DEST =="
    COPYFILE_DISABLE=1 tar czf - -C "$ROOT" tools/m1_verify |
        ssh -p "$AV2ENC_PORT" "$AV2ENC_REMOTE" \
            "mkdir -p '$AV2ENC_DEST' && tar xzf - -C '$AV2ENC_DEST'" || {
        echo "run_matrix: remote sync failed" >&2
        exit 2
    }
    remote_args=(--local --sizes "$SIZES" --modes "$MODES" --timeout "$TIMEOUT")
    [ -n "$DEPTH" ] && remote_args+=(--depth "$DEPTH")
    [ "$DEBLOCK" = 0 ] && remote_args+=(--no-deblock)
    [ "$CFL" = 0 ] && remote_args+=(--no-cfl)
    [ "$REBUILD" = 1 ] && remote_args+=(--rebuild)
    [ "$CLEAN" = 1 ] && remote_args+=(--clean)
    quoted=""
    for a in "${remote_args[@]}"; do
        quoted+=" $(printf '%q' "$a")"
    done
    # Forward the M1 debug knobs so `AV2ENC_M1_UV_DC=1 ... --remote` works.
    env_prefix=""
    for name in AV2ENC_M1_FORCE_Y AV2ENC_M1_FORCE_Y2 AV2ENC_M1_BASE_ONLY \
                AV2ENC_M1_NO_DIR AV2ENC_M1_NO_EDGE AV2ENC_M1_UV_DC \
                AV2ENC_M1_LUMA_DC AV2ENC_M1_TRACE AV2ENC_M1_SQUARE_ONLY \
                AV2ENC_M1_LEAF32 AV2ENC_M1_ZERO_DC AV2ENC_M1_YMODE_CTX0 \
                AV2ENC_M1_YMODE_CTX2 AV2ENC_M1_DCSIGN_CTX \
                AV2ENC_M1_NO_TXTYPE DAV2D; do
        value=${!name-}
        if [ -n "$value" ]; then
            env_prefix+="$name=$(printf '%q' "$value") "
        fi
    done
    exec ssh -p "$AV2ENC_PORT" "$AV2ENC_REMOTE" \
        "cd '$AV2ENC_DEST' && ${env_prefix}bash tools/m1_verify/run_matrix.sh$quoted"
fi

CLI="$ROOT/build-cuda/av2enc"
LIB="$ROOT/build-cuda/libav2enc_core.a"
WORK="$ROOT/build-m1-verify"
HARNESS="$WORK/m1_dump"
VERIFY="$SELF/verify.py"
CXX=${CXX:-g++}

if [ ! -x "$CLI" ] || [ ! -f "$LIB" ]; then
    echo "run_matrix: missing $CLI or $LIB; build build-cuda first" >&2
    exit 2
fi

if [ "$REBUILD" = 1 ]; then
    echo "== rebuilding build-cuda (av2enc) =="
    if command -v flock >/dev/null 2>&1; then
        flock /tmp/av2enc-build-cuda.lock \
            cmake --build "$ROOT/build-cuda" --target av2enc -j "$(nproc 2>/dev/null || echo 8)" ||
            { echo "run_matrix: build failed" >&2; exit 2; }
    else
        cmake --build "$ROOT/build-cuda" --target av2enc -j \
            "$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)" ||
            { echo "run_matrix: build failed" >&2; exit 2; }
    fi
elif [ -n "$(find "$ROOT/src" -newer "$LIB" -print -quit 2>/dev/null)" ]; then
    echo "run_matrix: WARNING: build-cuda/libav2enc_core.a is older than src/;" \
         "the CLI and harness may not match the tree (use --rebuild)" >&2
fi

if [ -z "${DAV2D:-}" ] && [ ! -x "$ROOT/tools/bin/dav2d" ]; then
    echo "run_matrix: tools/bin/dav2d missing; set DAV2D or install the pinned decoder" >&2
    exit 2
fi

mkdir -p "$WORK"
echo "== building harness =="
CUDART_DIR=${CUDART_LIB_DIR:-}
if [ -z "$CUDART_DIR" ]; then
    for candidate in /usr/local/cuda/lib64 /usr/local/cuda/targets/x86_64-linux/lib; do
        if [ -e "$candidate/libcudart.so" ]; then
            CUDART_DIR=$candidate
            break
        fi
    done
fi
link_args=("$LIB")
[ -n "$CUDART_DIR" ] && link_args+=(-L"$CUDART_DIR" -lcudart)
if ! "$CXX" -O2 -std=c++20 -I"$ROOT/src" "$SELF/m1_dump.cpp" \
    -o "$HARNESS" "${link_args[@]}"; then
    echo "run_matrix: harness build failed" >&2
    exit 2
fi
echo "harness: $HARNESS"

# --- inputs -----------------------------------------------------------------
gen_input() {
    local size=$1 out=$2
    case "$size" in
        1920x1080)
            ffmpeg -y -v error -i "$WORK/src_640x360.y4m" \
                -vf "scale=1920:1080:flags=lanczos" -frames:v 1 \
                -pix_fmt yuv420p "$out"
            ;;
        *)
            ffmpeg -y -v error -f lavfi \
                -i "testsrc2=size=$size:rate=1:duration=1" \
                -frames:v 1 -pix_fmt yuv420p "$out"
            ;;
    esac
}

if [ ! -s "$WORK/src_640x360.y4m" ]; then
    ffmpeg -y -v error -f lavfi -i "testsrc2=size=640x360:rate=1:duration=1" \
        -frames:v 1 -pix_fmt yuv420p "$WORK/src_640x360.y4m" || {
        echo "run_matrix: ffmpeg failed to create the base 640x360 source" >&2
        exit 2
    }
fi

run_cmd() {
    if [ "$TIMEOUT" -gt 0 ] && command -v timeout >/dev/null 2>&1; then
        timeout "$TIMEOUT" "$@"
    else
        "$@"
    fi
}

mode_flags() {
    case "$1" in
        dc) CLI_FLAGS=(--m1); HARNESS_FLAGS=() ;;
        full) CLI_FLAGS=(--m1 --m1-full-intra); HARNESS_FLAGS=(--full-intra) ;;
        force:*) CLI_FLAGS=(--m1 --m1-full-intra); HARNESS_FLAGS=(--full-intra) ;;
        *) echo "run_matrix: bad mode $1" >&2; return 1 ;;
    esac
    if [ -n "$DEPTH" ]; then
        CLI_FLAGS+=(--m1-depth "$DEPTH")
        HARNESS_FLAGS+=(--depth "$DEPTH")
    fi
    [ "$DEBLOCK" = 0 ] && { CLI_FLAGS+=(--no-deblock); HARNESS_FLAGS+=(--no-deblock); }
    [ "$CFL" = 0 ] && { CLI_FLAGS+=(--no-cfl); HARNESS_FLAGS+=(--no-cfl); }
    return 0
}

EXPECT_DEPTH=${DEPTH:-4}
EXPECT_FULL_INTRA=0
TMP_TABLE="$WORK/table.tsv"
: > "$TMP_TABLE"
exact=0
total=0
setup_fail=0

printf '\n%-10s %-7s %8s %6s %6s %6s %4s %-11s %-10s %s\n' \
    "size" "mode" "bytes" "Y" "U" "V" "max" "first(4x4)" "result" "time"
printf '%s\n' "------------------------------------------------------------------------------------------"

for size in $SIZES; do
    SRC="$WORK/src_${size}.y4m"
    if [ "$size" = "640x360" ]; then
        SRC="$WORK/src_640x360.y4m"
    elif [ ! -s "$SRC" ]; then
        if ! gen_input "$size" "$SRC"; then
            echo "run_matrix: cannot create $size source" >&2
            setup_fail=1
            continue
        fi
    fi

    for mode in $MODES; do
        total=$((total + 1))
        case "$mode" in force:*) short="F${mode#force:}" ;; *) short="$mode" ;; esac
        base="$WORK/${size}_${short}"
        ivf="$base.ivf"
        dump="$base.dump"
        dec="${base}_dec.y4m"
        ok=1

        mode_flags "$mode" || { setup_fail=1; continue; }
        unset AV2ENC_M1_FORCE_Y
        case "$mode" in
            force:*) export AV2ENC_M1_FORCE_Y="${mode#force:}:0" ;;
        esac
        case "$mode" in full|force:*) EXPECT_FULL_INTRA=1 ;; *) EXPECT_FULL_INTRA=0 ;; esac

        start=$SECONDS
        if ! run_cmd "$CLI" "$SRC" -o "$ivf" "${CLI_FLAGS[@]}" >"$base.cli.log" 2>&1; then
            printf '%-10s %-7s %8s %6s %6s %6s %4s %-11s %-10s %s\n' \
                "$size" "$short" "-" "-" "-" "-" "-" "-" "ENC-ERR" "-"
            echo "  see $base.cli.log" >&2
            ok=0
            setup_fail=1
        elif ! run_cmd "$HARNESS" -i "$SRC" -o "$dump" "${HARNESS_FLAGS[@]}" \
                >"$base.dump.log" 2>&1; then
            printf '%-10s %-7s %8s %6s %6s %6s %4s %-11s %-10s %s\n' \
                "$size" "$short" "-" "-" "-" "-" "-" "-" "DUMP-ERR" "-"
            echo "  see $base.dump.log" >&2
            ok=0
            setup_fail=1
        fi

        verify_args=(--ivf "$ivf" --dump "$dump" --tsv
                     --expect-depth "$EXPECT_DEPTH"
                     --expect-deblock "$DEBLOCK"
                     --expect-full-intra "$EXPECT_FULL_INTRA")
        elapsed=$((SECONDS - start))

        if [ "$ok" = 1 ]; then
            if run_cmd python3 "$VERIFY" "${verify_args[@]}" \
                    --keep-dec "$dec" >"$base.tsv" 2>"$base.verify.err"; then
                true
            else
                rc=$?
                if [ "$rc" != 1 ]; then
                    printf '%-10s %-7s %8s %6s %6s %6s %4s %-11s %-10s %s\n' \
                        "$size" "$short" "-" "-" "-" "-" "-" "-" "VERIFY-ERR" "-"
                    echo "  see $base.verify.err" >&2
                    setup_fail=1
                    ok=0
                fi
            fi
        fi

        if [ "$ok" = 1 ]; then
            IFS=$'\t' read -r status dy du dv mx fx fy fp <"$base.tsv"
            bytes=$(stat -c %s "$ivf" 2>/dev/null || stat -f %z "$ivf")
            if [ "$status" = "BIT-EXACT" ]; then
                exact=$((exact + 1))
                block="-"
                result="EXACT"
            else
                block="$((fx / 4)),$((fy / 4))"
                result="DIFF($fp)"
            fi
            printf '%-10s %-7s %8s %6s %6s %6s %4s %-11s %-10s %s\n' \
                "$size" "$short" "$bytes" "$dy" "$du" "$dv" "$mx" "$block" \
                "$result" "${elapsed}s"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$size" "$short" "$bytes" "$dy" "$du" "$dv" "$mx" "$block" \
                "$status" "$elapsed" >>"$TMP_TABLE"
            [ "$CLEAN" = 1 ] && rm -f "$ivf" "$dump" "$dec"
        fi
    done
done

echo
printf 'summary: %d/%d cases bit-exact' "$exact" "$total"
if [ "$exact" != "$total" ]; then
    printf '  (%d differ)' "$((total - exact))"
fi
printf '\nartifacts: %s\n' "$WORK"
if [ "$setup_fail" != 0 ]; then
    echo "run_matrix: setup/encode errors occurred" >&2
    exit 2
fi
[ "$exact" = "$total" ]
