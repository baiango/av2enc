#!/usr/bin/env bash
# Back up av2enc to this Mac.
#
# Captures:
#   - full git history as a bundle
#   - working tree tarball (source/docs/tests, excluding builds/media/binaries)
#   - git status/log
#   - remote Vast.ai snapshot (source/docs only)
#
# Usage: tools/backup_to_mac.sh
# Env:   AV2ENC_BACKUP_DIR (default: ~/av2enc-backups)
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
STAMP=$(date +%Y%m%d-%H%M%S)
DEST="${AV2ENC_BACKUP_DIR:-$HOME/av2enc-backups}/$STAMP"
REMOTE="${AV2ENC_HOST:?set AV2ENC_HOST (user@host)}"
PORT="${AV2ENC_PORT:-22}"

mkdir -p "$DEST"

echo "==> git bundle"
git -C "$ROOT" bundle create "$DEST/repo.bundle" --all

echo "==> git metadata"
git -C "$ROOT" status --short > "$DEST/status.txt" || true
git -C "$ROOT" log --oneline -20 > "$DEST/log.txt" || true
git -C "$ROOT" rev-parse HEAD > "$DEST/HEAD.txt" || true

echo "==> working tree"
COPYFILE_DISABLE=1 tar czf "$DEST/worktree.tar.gz" \
  -C "$ROOT" \
  --exclude='./.git' \
  --exclude='./build*' \
  --exclude='./testdata' \
  --exclude='./tools/bin' \
  --exclude='./src/core/telemetry/build' \
  --exclude='./bench/hello_cuda/build*' \
  --exclude='*.y4m' --exclude='*.ivf' --exclude='*.webm' --exclude='*.mp4' \
  --exclude='*.log' --exclude='*.nsys-rep' --exclude='*.sqlite' \
  .

echo "==> remote snapshot"
set +e
COPYFILE_DISABLE=1 ssh -p "$PORT" -o BatchMode=yes "$REMOTE" \
  "cd /workspace/av2enc && tar czf - \
     --exclude='./build*' --exclude='./testdata' --exclude='./tools/bin' \
     --exclude='./src/core/telemetry/build' --exclude='./bench/hello_cuda/build*' \
     --exclude='*.y4m' --exclude='*.ivf' --exclude='*.webm' --exclude='*.mp4' \
     --exclude='*.log' --exclude='*.nsys-rep' --exclude='*.sqlite' ." \
  > "$DEST/remote-worktree.tar.gz"
RC=$?
set -e
if [ "$RC" -ne 0 ]; then
  echo "warning: remote snapshot failed (exit $RC); local backup is intact" >&2
  rm -f "$DEST/remote-worktree.tar.gz"
fi

echo "==> checksums"
( cd "$DEST" && shasum -a 256 ./* > SHA256SUMS 2>/dev/null || true )

cat > "$DEST/README.md" <<EOF
# av2enc backup $STAMP

- repo.bundle: full git history
- worktree.tar.gz: working tree (excluding builds, testdata, tools/bin)
- remote-worktree.tar.gz: Vast.ai /workspace/av2enc source/docs snapshot
- status.txt / log.txt / HEAD.txt / SHA256SUMS
EOF

echo "backup: $DEST"
du -sh "$DEST"
