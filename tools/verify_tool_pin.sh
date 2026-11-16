#!/usr/bin/env bash
# Verify a prebuilt tool binary against tools/tool-pins.sha256.
#
# tools/bin/ is gitignored, so the decoders it holds are not covered by any
# commit: the docs say "dav2d pinned at f9dafbc", but nothing binds those bytes
# to that commit.  This check does, for the binaries we ship as trust anchors.
#
# Usage: tools/verify_tool_pin.sh <tool-binary>
# Exit: 0 verified or no pin recorded, 1 checksum mismatch, 2 usage.
#
# A tool with no manifest entry (for example a distro dav2d on Linux) is not an
# error: the pin only covers the binaries this repo checks in.
set -uo pipefail

path=${1:-}
if [ -z "$path" ]; then
    echo "usage: $0 <tool-binary>" >&2
    exit 2
fi
if [ ! -f "$path" ]; then
    echo "verify_tool_pin: no such file: $path" >&2
    exit 2
fi

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
manifest="$here/tool-pins.sha256"
[ -f "$manifest" ] || exit 0

name=$(basename -- "$path")
want=$(awk -v n="$name" '$2 == n { print $1; exit }' "$manifest")
[ -n "$want" ] || exit 0

if command -v shasum >/dev/null 2>&1; then
    got=$(shasum -a 256 -- "$path" | awk '{ print $1 }')
elif command -v sha256sum >/dev/null 2>&1; then
    got=$(sha256sum -- "$path" | awk '{ print $1 }')
else
    echo "verify_tool_pin: no sha256 tool available; cannot verify $name" >&2
    exit 2
fi

if [ "$want" != "$got" ]; then
    echo "verify_tool_pin: CHECKSUM MISMATCH for $name" >&2
    echo "  expected $want" >&2
    echo "  actual   $got" >&2
    echo "  Replace the binary only if you also update tools/tool-pins.sha256" >&2
    exit 1
fi

echo "tool pin: $name $got (ok)"
exit 0
