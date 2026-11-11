#!/usr/bin/env bash
# Sync this tool directory to the 3060 box and run a gate there.
#
#   tools/m1_gpu/remote.sh verify [args...]   -> tools/m1_gpu/verify_gpu.sh
#   tools/m1_gpu/remote.sh bench  [args...]   -> tools/m1_gpu/bench.py
#   tools/m1_gpu/remote.sh shell              -> interactive shell
#
# Only tools/m1_gpu is sent by default. That deliberately avoids a full-tree
# sync while other agents are editing src/: a stale local src/main.cpp would
# overwrite their work on the shared box. Set FULL_SYNC=1 to use
# tools/sync_remote.sh instead.
#
# Env: AV2ENC_HOST, AV2ENC_PORT, AV2ENC_DEST (default /workspace/av2enc)
set -euo pipefail

SELF=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SELF/../.." && pwd)
HOST=${AV2ENC_HOST:?set AV2ENC_HOST (user@host)}
PORT=${AV2ENC_PORT:-22}
DEST=${AV2ENC_DEST:-/workspace/av2enc}

TOOL=${1:-shell}
[ $# -gt 0 ] && shift

if [ "${FULL_SYNC:-0}" = "1" ]; then
    AV2ENC_HOST="$HOST" AV2ENC_PORT="$PORT" AV2ENC_DEST="$DEST" \
        "$ROOT/tools/sync_remote.sh"
else
    echo "== syncing tools/m1_gpu to $HOST:$DEST =="
    COPYFILE_DISABLE=1 tar czf - -C "$ROOT" tools/m1_gpu |
        ssh -p "$PORT" -o BatchMode=yes "$HOST" \
            "mkdir -p '$DEST' && tar xzf - -C '$DEST'"
fi

if [ "$TOOL" = "shell" ]; then
    exec ssh -p "$PORT" -o BatchMode=yes "$HOST" "cd '$DEST' && exec bash -l"
fi

case "$TOOL" in
    verify) RUNNER=(tools/m1_gpu/verify_gpu.sh) ;;
    bench)  RUNNER=(python3 tools/m1_gpu/bench.py) ;;
    *) echo "remote.sh: unknown tool '$TOOL' (verify|bench|shell)" >&2; exit 2 ;;
esac

quoted=""
for arg in "$@"; do
    quoted+=" $(printf '%q' "$arg")"
done

env_prefix=""
for name in ALLOW_SKIP DAV2D AV2ENC PYTHON NSYS TIMEOUT; do
    value=${!name-}
    if [ -n "$value" ]; then
        env_prefix+="$name=$(printf '%q' "$value") "
    fi
done

exec ssh -p "$PORT" -o BatchMode=yes "$HOST" \
    "cd '$DEST' && exec ${env_prefix}${RUNNER[*]}$quoted"
