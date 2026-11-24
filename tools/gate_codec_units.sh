#!/usr/bin/env bash
# Fixture-free codec unit gates: kernel layer, RDOQ trellis, transform variants.
#
# These bit-exactness tests are ctest tests named `gate_*` rather than shell
# gates.  tools/run_gates.sh runs unit tests with `ctest -E '^gate_'` because
# shell gates are registered separately, which silently excluded
# gate_primitives: the one test that proves every kernel matches its inlined
# oracle never ran in the default dev loop.  This wrapper gives the fixture-free
# gate_ codec tests a normal `run_gate` entry.
#
# `rdoq` and `tx_variants` joined it after two regressions landed that only the
# per-component build_and_test.sh scripts had been running: a per-magnitude rate
# cache that ignored the arrival state, and an unguarded primitives::kernels()
# call in the transform row pass.
#
# The wrapper passes the same --bin/--dav2d arguments as every other gate and
# ignores them, since none of these test binaries need an encoder or a decoder.
#
# Usage: tools/gate_codec_units.sh [--bin PATH] [--dav2d PATH]
set -euo pipefail

BIN=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --bin) BIN="$2"; shift 2 ;;
        --dav2d) shift 2 ;;
        -h|--help) sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "gate_codec_units: unknown argument: $1" >&2; exit 2 ;;
    esac
done

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -n "$BIN" ]; then
    BUILD="$(cd -- "$(dirname -- "$BIN")" && pwd)"
else
    BUILD="$ROOT/build"
fi

if [ ! -d "$BUILD" ]; then
    echo "gate_codec_units: build directory not found: $BUILD" >&2
    exit 2
fi

ctest --test-dir "$BUILD" -R '^gate_(primitives|rdoq|tx_variants)$' --output-on-failure
