#!/usr/bin/env bash
# Build the pinned dav2d decoder for Linux and install it as tools/bin/dav2d.
#
# Why: tools/bin only ships a macOS binary, so the GPU product gates
# (src/cuda/pipeline/video/gate_video_gpu.sh and gate_gpu_throughput.sh) cannot
# run their decode half on a Linux GPU host. Those gates exit 77 rather than
# fail, which is correct but leaves a hole; this script closes it.
#
# Usage:
#   tools/build_dav2d_linux.sh [--dry-run] [--prefix DIR]
#
# Env:
#   DAV2D_REF      commit/tag to build (default: the pinned f9dafbc)
#   DAV2D_SOURCE   existing checkout to use instead of cloning
#   BUILD_DIR      scratch directory (default: ${TMPDIR:-/tmp}/dav2d-build)
#
# Requires: git, meson, ninja, nasm, a C compiler.
#   Debian/Ubuntu: apt-get install -y git meson ninja-build nasm build-essential
#
# The source pin is the commit; the produced binary is not added to
# tools/tool-pins.sha256, because that manifest binds checked-in binaries only.
set -uo pipefail

DAV2D_REF=${DAV2D_REF:-f9dafbc747e82a9c51ca648bf2108d9fa0f8de7d}
REPO_URL=${REPO_URL:-https://code.videolan.org/videolan/dav2d.git}
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
BUILD_DIR=${BUILD_DIR:-${TMPDIR:-/tmp}/dav2d-build}
DRY=0
PREFIX="$ROOT/tools/bin"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run) DRY=1; shift ;;
        --prefix) PREFIX=${2:-}; shift 2 ;;
        -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "build_dav2d_linux: unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [ "$DRY" -eq 0 ] && [ "$(uname -s)" != "Linux" ]; then
    echo "build_dav2d_linux: this builds the Linux decoder; host is $(uname -s)." >&2
    echo "  For macOS, use the pinned tools/bin/dav2d-macos." >&2
    exit 2
fi

missing=""
for tool in git meson ninja nasm; do
    command -v "$tool" >/dev/null 2>&1 || missing="$missing $tool"
done
if [ -n "$missing" ] && [ "$DRY" -eq 0 ]; then
    echo "build_dav2d_linux: missing:$missing" >&2
    echo "  Debian/Ubuntu: apt-get install -y git meson ninja-build nasm build-essential" >&2
    exit 2
fi

say() { printf '+ %s\n' "$*"; }
run() {
    if [ "$DRY" -eq 1 ]; then say "$@"; else "$@"; fi
}

SRC=${DAV2D_SOURCE:-$BUILD_DIR/src}
if [ -z "${DAV2D_SOURCE:-}" ]; then
    if [ ! -d "$SRC/.git" ]; then
        run mkdir -p "$BUILD_DIR"
        run git clone "$REPO_URL" "$SRC"
    fi
    run git -C "$SRC" fetch --quiet origin "$DAV2D_REF" || true
    run git -C "$SRC" checkout --quiet "$DAV2D_REF"
else
    [ -d "$SRC" ] || { echo "build_dav2d_linux: no checkout at $SRC" >&2; exit 2; }
fi

# Build only the CLI tools: examples/tests/docs are not needed and cost time.
# Static libdav2d: the default shared build produces a binary that needs
# libdav2d.so at run time, which is not installed next to tools/bin/dav2d.
BUILDDIR="$SRC/build-linux-static"
run rm -rf "$BUILDDIR"
run meson setup "$BUILDDIR" "$SRC" --buildtype release \
    -Ddefault_library=static \
    -Denable_tools=true -Denable_tests=false -Denable_examples=false \
    -Denable_docs=false
run ninja -C "$BUILDDIR"

BIN="$BUILDDIR/tools/dav2d"
[ -x "$BIN" ] || BIN=$(find "$BUILDDIR" -type f -name dav2d -perm -u+x | head -1)
if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
    [ "$DRY" -eq 1 ] || { echo "build_dav2d_linux: built binary not found" >&2; exit 1; }
    BIN="$BUILDDIR/tools/dav2d"
fi

run mkdir -p "$PREFIX"
run install -m 0755 "$BIN" "$PREFIX/dav2d"

if [ "$DRY" -eq 0 ]; then
    if ! "$PREFIX/dav2d" --version >/dev/null 2>&1; then
        echo "build_dav2d_linux: installed binary does not run; check the linker" >&2
        exit 1
    fi
    "$PREFIX/dav2d" --version 2>/dev/null | head -2 || true
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$PREFIX/dav2d"
    fi
    echo "build_dav2d_linux: installed $PREFIX/dav2d (ref $DAV2D_REF)"
fi
