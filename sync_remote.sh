#!/usr/bin/env bash
# Sync the local source tree to the Vast.ai box.
# The remote image does not have rsync, so this uses a tar pipe.
set -euo pipefail

HOST="${AV2ENC_HOST:?set AV2ENC_HOST (user@host)}"
PORT="${AV2ENC_PORT:-22}"
DEST="${AV2ENC_DEST:-/workspace/av2enc}"

cd "$(dirname "$0")/.."

tar czf - \
  --exclude './build' \
  --exclude './build-*' \
  --exclude './.git' \
  --exclude './.venv-cvvdp' \
  --exclude './testdata' \
  --exclude './tools/bin' \
  --exclude './bench/hello_cuda/build' \
  . | ssh -p "$PORT" -o BatchMode=yes "$HOST" \
      "mkdir -p '$DEST' && tar xzf - -C '$DEST'"

echo "synced $(pwd) -> $HOST:$DEST"
