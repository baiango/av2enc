#!/usr/bin/env bash
# Create an isolated worktree + branch for one bug fix.
#
#   tools/agents/new_fix.sh BUG-INTRA-014 palette-row [--lane INTRA] [--base main]
#
# Worktree:  ${AV2ENC_FIX_ROOT:-<repo-parent>/av2enc-fix}/<bug-id>
# Branch:    fix/<lane>/<bug-id>-<slug>
#
# Also creates the testdata/tools-bin/spec symlinks so the pinned gates run.
set -euo pipefail

ROOT=$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel)
PARENT=$(dirname "$ROOT")
FIX_ROOT=${AV2ENC_FIX_ROOT:-"$PARENT/av2enc-fix"}

die() { echo "new_fix: $*" >&2; exit 1; }

id=${1:-}
slug=${2:-}
[ -n "$id" ] || die "usage: new_fix.sh BUG-ID slug [--lane LANE] [--base REF]"
[ -n "$slug" ] || die "usage: new_fix.sh BUG-ID slug [--lane LANE] [--base REF]"
shift 2
lane="misc"
base="main"
while [ $# -gt 0 ]; do
    case "$1" in
        --lane) lane=${2:-}; shift 2 ;;
        --base) base=${2:-}; shift 2 ;;
        *) die "unknown option: $1" ;;
    esac
done

[[ "$id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "invalid bug id: $id"
[[ "$slug" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "invalid slug: $slug"
[[ "$lane" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid lane: $lane"

branch="fix/$lane/$id-$slug"
worktree="$FIX_ROOT/$id"

[ -e "$worktree" ] && die "worktree already exists: $worktree"
git -C "$ROOT" rev-parse --verify "$base" >/dev/null 2>&1 ||
    die "unknown base ref: $base"
git -C "$ROOT" rev-parse --verify "refs/heads/$branch" >/dev/null 2>&1 &&
    die "branch already exists: $branch"

mkdir -p "$FIX_ROOT"
git -C "$ROOT" worktree add -q -b "$branch" "$worktree" "$base"

ln -sfn "$ROOT/tools/bin" "$worktree/tools/bin"
ln -sfn "$ROOT/testdata" "$worktree/testdata"
if [ -f "$ROOT/spec/AV2_Spec_v1.0.0.pdf" ]; then
    ln -sfn "$ROOT/spec/AV2_Spec_v1.0.0.pdf" "$worktree/spec/AV2_Spec_v1.0.0.pdf"
fi

echo "worktree: $worktree"
echo "branch:   $branch"
echo "base:     $base"
echo
echo "next: claim the bug, reproduce it, add a failing check, then fix."
