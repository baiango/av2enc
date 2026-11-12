#!/usr/bin/env bash
# Atomically claim/release a bug so two agents cannot fix the same thing.
#
#   tools/agents/claim.sh claim  BUG-INTRA-014 --owner alice \
#       [--lane INTRA] [--branch fix/INTRA/BUG-INTRA-014-palette-row]
#   tools/agents/claim.sh release BUG-INTRA-014 [--owner alice] [--force]
#   tools/agents/claim.sh list
#
# Claims live in the primary worktree under .agents/claims/<id>/ (gitignored).
set -euo pipefail

ROOT=$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel)
PRIMARY=$(git -C "$ROOT" worktree list --porcelain |
    awk '/^worktree / { print $2; exit }')
CLAIMS="$PRIMARY/.agents/claims"

die() { echo "claim: $*" >&2; exit 1; }

cmd=${1:-}
[ -n "$cmd" ] || die "usage: claim.sh claim|release|list ..."
shift || true

valid_id() { [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; }

case "$cmd" in
claim)
    id=${1:-}
    [ -n "$id" ] || die "claim needs a bug id"
    shift || true
    owner=${AGENT_NAME:-}
    lane="misc"
    branch=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --owner) owner=${2:-}; shift 2 ;;
            --lane) lane=${2:-}; shift 2 ;;
            --branch) branch=${2:-}; shift 2 ;;
            *) die "unknown claim option: $1" ;;
        esac
    done
    [ -n "$owner" ] || die "claim needs --owner (or AGENT_NAME)"
    valid_id "$id" || die "invalid bug id: $id"
    mkdir -p "$CLAIMS"
    if mkdir "$CLAIMS/$id" 2>/dev/null; then
        printf '%s\n' "$owner" >"$CLAIMS/$id/owner"
        printf '%s\n' "$lane" >"$CLAIMS/$id/lane"
        printf '%s\n' "$branch" >"$CLAIMS/$id/branch"
        date -u '+%Y-%m-%dT%H:%M:%SZ' >"$CLAIMS/$id/claimed_at"
        echo "claimed $id by $owner"
    else
        holder=$(cat "$CLAIMS/$id/owner" 2>/dev/null || echo "?")
        die "$id is already claimed by $holder (run: claim.sh list)"
    fi
    ;;
release)
    id=${1:-}
    [ -n "$id" ] || die "release needs a bug id"
    shift || true
    owner=${AGENT_NAME:-}
    force=0
    while [ $# -gt 0 ]; do
        case "$1" in
            --owner) owner=${2:-}; shift 2 ;;
            --force) force=1; shift ;;
            *) die "unknown release option: $1" ;;
        esac
    done
    [ -d "$CLAIMS/$id" ] || die "$id is not claimed"
    holder=$(cat "$CLAIMS/$id/owner" 2>/dev/null || echo "?")
    if [ "$force" -ne 1 ] && [ -n "$owner" ] && [ "$owner" != "$holder" ]; then
        die "$id is held by $holder, not $owner (use --force to override)"
    fi
    rm -rf "$CLAIMS/$id"
    echo "released $id (was $holder)"
    ;;
list)
    [ -d "$CLAIMS" ] || { echo "(no claims)"; exit 0; }
    found=0
    for d in "$CLAIMS"/*; do
        [ -d "$d" ] || continue
        found=1
        printf '%s  owner=%s  lane=%s  branch=%s  since=%s\n' \
            "$(basename "$d")" \
            "$(cat "$d/owner" 2>/dev/null || echo '?')" \
            "$(cat "$d/lane" 2>/dev/null || echo '?')" \
            "$(cat "$d/branch" 2>/dev/null || echo '?')" \
            "$(cat "$d/claimed_at" 2>/dev/null || echo '?')"
    done
    [ "$found" -eq 1 ] || echo "(no claims)"
    ;;
*)
    die "unknown command: $cmd"
    ;;
esac
