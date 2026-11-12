#!/usr/bin/env bash
# Effort-ranking gate: budget invariant, selected prefix, rejection rules,
# dav2d --strict 1 decode, default-stream byte identity, and a representative
# still + video measurement.
#
# Usage:
#   tools/effort/gate_effort.sh [--bin build-int/av2enc] [--dav2d PATH]
#       [--table tools/effort/effort_table.json] [--quick] [--verbose]
#
#   --quick   skip the 720p video measurement (the reject/byte checks remain)
#
# Exits 77 when dav2d or the 640x360 fixture is missing, 1 on a failed check.
# The gate never recalibrates and never edits the table.
set -uo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
if ! command -v python3 >/dev/null 2>&1; then
    echo "gate_effort: SKIP (exit 77): python3 is required" >&2
    exit 77
fi

exec python3 "$ROOT/tools/effort/gate_effort.py" "$@"
