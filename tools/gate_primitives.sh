#!/usr/bin/env bash
# Primitives kernel gate.
#
# The kernel layer is the SIMD landing zone (docs/simd-plan.md), and its
# bit-exactness test is a ctest test rather than a shell gate.  tools/run_gates.sh
# runs unit tests with `ctest -E '^gate_'` because shell gates are registered
# separately, which silently excluded gate_primitives: the one test that proves
# every kernel matches its inlined oracle never ran in the default dev loop.
#
# This wrapper gives it a normal `run_gate` entry.  It passes the same
# --bin/--dav2d arguments as every other gate and ignores them, since the test
# binary needs neither.
#
# Usage: tools/gate_primitives.sh [--bin PATH] [--dav2d PATH]
set -euo pipefail

BIN=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --bin) BIN="$2"; shift 2 ;;
        --dav2d) shift 2 ;;
        -h|--help) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "gate_primitives: unknown argument: $1" >&2; exit 2 ;;
    esac
done

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -n "$BIN" ]; then
    BUILD="$(cd -- "$(dirname -- "$BIN")" && pwd)"
else
    BUILD="$ROOT/build"
fi

if [ ! -d "$BUILD" ]; then
    echo "gate_primitives: build directory not found: $BUILD" >&2
    exit 2
fi

ctest --test-dir "$BUILD" -R '^gate_primitives$' --output-on-failure
