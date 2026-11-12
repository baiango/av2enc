#!/usr/bin/env bash
# Show every worktree, fix branch and active bug claim.
#
#   tools/agents/status.sh
#
# Read-only. Safe to run from any worktree of the repository.
set -euo pipefail

ROOT=$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel)
PRIMARY=$(git -C "$ROOT" worktree list --porcelain |
    awk '/^worktree / { print $2; exit }')
CLAIMS="$PRIMARY/.agents/claims"

echo "primary: $PRIMARY"
echo
echo "== worktrees =="
git -C "$ROOT" worktree list
echo
echo "== fix branches =="
git -C "$ROOT" for-each-ref --sort=-committerdate \
    --format='%(refname:short)  %(objectname:short)  %(subject)' \
    refs/heads/fix/ 2>/dev/null || true
echo
echo "== claims =="
if [ -d "$CLAIMS" ]; then
    found=0
    for d in "$CLAIMS"/*; do
        [ -d "$d" ] || continue
        found=1
        id=$(basename "$d")
        owner=$(cat "$d/owner" 2>/dev/null || echo "?")
        branch=$(cat "$d/branch" 2>/dev/null || echo "?")
        lane=$(cat "$d/lane" 2>/dev/null || echo "?")
        printf '%s  lane=%s  owner=%s  branch=%s\n' "$id" "$lane" "$owner" "$branch"
    done
    [ "$found" -eq 1 ] || echo "(none)"
else
    echo "(none)"
fi
echo
echo "== board: open/blocked =="
if [ -f "$ROOT/refactor/bugs.csv" ]; then
    python3 - "$ROOT/refactor/bugs.csv" <<'PY'
import csv, sys
with open(sys.argv[1], newline="") as f:
    for row in csv.DictReader(f):
        if row.get("status") in {"open", "claimed", "reproduced", "fixing", "blocked"}:
            print(f'{row.get("id","")}  {row.get("lane","")}  {row.get("status","")}  '
                  f'{row.get("owner","")}  {row.get("branch","")}')
PY
else
    echo "(no refactor/bugs.csv)"
fi
