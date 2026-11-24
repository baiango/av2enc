#!/usr/bin/env bash
# Build-seam check: every migrated translation unit must compile in BOTH
# configurations of the pixel-kernel seam.
#
#   1. product:       -DAV2ENC_PRIMITIVES, i.e. the call site takes
#                     primitives::kernels().  The preprocessed TU must contain
#                     a kernel call, and compiling with -c must succeed (no
#                     link is performed; in the product av2enc_core links
#                     src/primitives/kernels.cpp).
#   2. component:     no AV2ENC_PRIMITIVES and no kernels.cpp in the link.  The
#                     preprocessed TU must contain no kernel call, and the -c
#                     object must have no undefined dispatch symbol: the inlined
#                     #else fallback is what keeps the per-component
#                     build_and_test.sh gates buildable.
#
# A TU that fails (2) has been migrated without a guard, or with a guard whose
# fallback does not compile.  That is the exact bug this script exists to catch:
# a migrated TU that only builds in one configuration.
#
# This is a compile-only check: it needs a C++20 compiler and nm, never a CMake
# configure/build, and writes only under $BUILD_DIR (default: a fresh mktemp
# directory, removed on exit).
#
# Usage:
#   tools/refactor/build_seam_check.sh [--verbose] [--keep]
#   CXX=clang++ tools/refactor/build_seam_check.sh
#   BUILD_DIR=/tmp/seam tools/refactor/build_seam_check.sh
#
# Exit: 0 every TU passes both ways, 1 a failure, 2 setup/usage error.
set -uo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
CXX="${CXX:-c++}"
VERBOSE=0
KEEP=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --verbose) VERBOSE=1; shift ;;
        --keep) KEEP=1; shift ;;
        # Accepted for tools/run_gates.sh compatibility: it invokes every gate
        # as `gate.sh --bin <binary> [--dav2d <path>]`.  This check is
        # compile-only and never runs an encoder, so the values are unused.
        --bin|--dav2d) shift 2 ;;
        -h|--help) sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "build_seam_check: unknown argument: $1" >&2; exit 2 ;;
    esac
done

command -v "$CXX" >/dev/null 2>&1 || {
    echo "build_seam_check: compiler not found: $CXX" >&2; exit 2; }
command -v nm >/dev/null 2>&1 || {
    echo "build_seam_check: nm not found" >&2; exit 2; }

OWN_BUILD=0
if [ -z "${BUILD_DIR:-}" ]; then
    BUILD_DIR=$(mktemp -d "${TMPDIR:-/tmp}/av2enc-build-seam.XXXXXX") || exit 2
    OWN_BUILD=1
fi
mkdir -p "$BUILD_DIR" || exit 2
cleanup() { if [ "$OWN_BUILD" -eq 1 ] && [ "$KEEP" -eq 0 ]; then rm -rf "$BUILD_DIR"; fi; }
trap cleanup EXIT

# The component gates' real flags: C++20, the repository's only include root,
# and the bit-exactness flag the CPU reference requires.  -O2 so the object
# actually carries the call (at -O0 it would too, but this matches the gates).
BASE=(-std=c++20 -O2 -ffp-contract=off -I "$ROOT/src")
# video_frame.cpp only compiles with the product feature set it is written
# against; those switches select unrelated modules and are held constant in
# both columns so the only variable is AV2ENC_PRIMITIVES.
VIDEO_DEFINES=(
    -DAV2ENC_VIDEO_FRAMES=1
    -DAV2ENC_VIDEO_CDF_CHAIN=1
    -DAV2ENC_VIDEO_GME=1
    -DAV2ENC_VIDEO_FORMATS=1
    -DAV2ENC_VIDEO_DELTAQ=1
    -DAV2ENC_VIDEO_EXT_PARTITIONS=1
)

# TU|extra flags.  Every TU with a primitives::kernels() call site, directly or
# through an inline header, belongs here.  Keep in sync with the table in
# docs/build-seam.md.
TUS=(
    "src/codec/inter/me/me.cpp|"
    "src/codec/inter/mc/mc.cpp|"
    "src/codec/rate/rdo/rdo.cpp|"
    "src/codec/inter/temporal/refinemv.cpp|"
    "src/pipeline/video/video_frame.cpp|${VIDEO_DEFINES[*]}"
    "src/codec/intra/modes/intra_modes.cpp|"
    "src/codec/intra/dip/dip.cpp|"
    "src/codec/intra/ibc/core/ibc.cpp|"
    "src/codec/transform/variants/tx_variants.cpp|"
)

has_kernel_ref() {
    # The only dispatch symbol these TUs can produce is av2enc::primitives::
    # kernels(); match it in both Itanium-mangled forms (Linux/macOS).
    nm -u "$1" 2>/dev/null | grep -q 'primitives.*kernels'
}

# The preprocessed text is the authoritative "which branch was selected" probe.
# Object symbols alone are not enough: at -O2 a static/class-local function the
# optimiser drops leaves no undefined reference even when its body calls the
# kernel, which would hide an unguarded call site.  Preprocessing applies the
# same `#if` the compile does, so a guarded fallback has no call text left.
preprocessed_has_kernel_call() {  # <tu> <flags...>
    local tu=$1
    shift
    # shellcheck disable=SC2086  # extra flags are intentionally word-split
    "$CXX" "${BASE[@]}" -E -P $* "$ROOT/$tu" 2>/dev/null |
        tr -d ' \n\t' | grep -q 'primitives::kernels()'
}

compile() {  # <tu> <object> <flags...>
    local tu=$1 obj=$2
    shift 2
    # shellcheck disable=SC2086  # extra flags are intentionally word-split
    "$CXX" "${BASE[@]}" $* -c "$ROOT/$tu" -o "$obj" 2>"$obj.log"
}

failures=0
checked=0
printf '  %-46s %-10s %-10s %s\n' TU with-define without-define detail
printf '  %-46s %-10s %-10s %s\n' '----------------------------------------------' \
    '-----------' '------------' '------'

for entry in "${TUS[@]}"; do
    tu=${entry%%|*}
    extra=${entry#*|}
    checked=$((checked + 1))
    slug=$(printf '%s' "$tu" | tr '/.' '__')
    win="$BUILD_DIR/$slug.with.o"
    wout="$BUILD_DIR/$slug.without.o"
    wstat=fail
    ostat=fail
    wnote=""
    detail=""

    if compile "$tu" "$win" "$extra -DAV2ENC_PRIMITIVES=1" &&
        preprocessed_has_kernel_call "$tu" "$extra -DAV2ENC_PRIMITIVES=1"; then
        if has_kernel_ref "$win"; then
            wstat=ok
        else
            # Optimised away in this TU shape; the selected branch is still the
            # kernel path (preprocessor probe above).
            wstat=ok
            wnote="(DCE)"
        fi
    else
        detail="with-define: kernel path not shown"
        [ "$VERBOSE" -eq 1 ] && sed -n '1,6p' "$win.log" >&2
    fi

    if ! preprocessed_has_kernel_call "$tu" "$extra"; then
        if compile "$tu" "$wout" "$extra"; then
            if has_kernel_ref "$wout"; then
                detail="${detail:+$detail; }fallback still calls kernels() (symbol)"
            else
                ostat=ok
            fi
        else
            detail="${detail:+$detail; }without-define compile failed"
            [ "$VERBOSE" -eq 1 ] && sed -n '1,6p' "$wout.log" >&2
        fi
    else
        detail="${detail:+$detail; }fallback still calls kernels() (text)"
    fi

    if [ "$wstat" = ok ] && [ "$ostat" = ok ]; then
        detail=${detail:-kernel path live; fallback self-contained}
        detail="$detail${wnote:+, $wnote}"
    else
        failures=$((failures + 1))
        [ "$VERBOSE" -eq 1 ] && detail="$detail (logs in $BUILD_DIR)"
    fi
    printf '  %-46s %-10s %-10s %s\n' "$tu" "$wstat" "$ostat" "$detail"
done

echo
if [ "$failures" -ne 0 ]; then
    echo "build_seam_check: $failures of $checked TUs FAILED (logs in $BUILD_DIR)" >&2
    exit 1
fi
echo "build_seam_check: OK ($checked TUs, both configurations)"
exit 0
