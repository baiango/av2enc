#!/usr/bin/env bash
# Sync tools/m3_gpu to the CUDA box and run the M3 feature gate there.
#
#   tools/m3_gpu/remote.sh verify [args...]   -> verify_m3_gpu.sh (single case)
#   tools/m3_gpu/remote.sh matrix [args...]   -> verify_m3_gpu.sh --matrix
#   tools/m3_gpu/remote.sh shell              -> interactive shell
#
# Only tools/m3_gpu is sent by default, so this cannot overwrite another
# agent's in-flight src/ changes on the shared box.  Set FULL_SYNC=1 to use
# tools/sync_remote.sh instead (needed when the encoder itself must be rebuilt).
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
    echo "== syncing tools/m3_gpu to $HOST:$DEST =="
    COPYFILE_DISABLE=1 tar czf - -C "$ROOT" tools/m3_gpu |
        ssh -p "$PORT" -o BatchMode=yes "$HOST" \
            "mkdir -p '$DEST' && tar xzf - -C '$DEST'"
fi

if [ "$TOOL" = "shell" ]; then
    exec ssh -p "$PORT" -o BatchMode=yes "$HOST" "cd '$DEST' && exec bash -l"
fi

case "$TOOL" in
    verify) RUNNER=(tools/m3_gpu/verify_m3_gpu.sh) ;;
    matrix) RUNNER=(tools/m3_gpu/verify_m3_gpu.sh --matrix) ;;
    *) echo "remote.sh: unknown tool '$TOOL' (verify|matrix|shell)" >&2; exit 2 ;;
esac

quoted=""
for arg in "$@"; do
    quoted+=" $(printf '%q' "$arg")"
done

env_prefix=""
for name in ALLOW_SKIP DAV2D AV2ENC PYTHON NSYS COMPUTE_SANITIZER \
            TIMEOUT DAV2D_THREADS MATRIX_BUDGET; do
    value=${!name-}
    if [ -n "$value" ]; then
        env_prefix+="$name=$(printf '%q' "$value") "
    fi
done

exec ssh -p "$PORT" -o BatchMode=yes "$HOST" \
    "cd '$DEST' && exec ${env_prefix}${RUNNER[*]}$quoted"
