#!/bin/sh
# Run five isolated tool-improvement lanes in parallel.
#
# Each lane gets its own worktree, branch, ledger and target. They never share
# a checkout, so a code edit in one lane cannot affect another. The script
# starts the lanes, waits for them, and leaves the ledgers/branches for review.
#
# Usage:
#   tools/agent_loop_multi.sh
#   LANES=5 ITERATIONS=10 tools/agent_loop_multi.sh
#   DRY_RUN=1 tools/agent_loop_multi.sh
#
# Stop: touch the stop file (default ../agent-loop/STOP).

set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
LANES=${LANES:-5}
ITERATIONS=${ITERATIONS:-10}
MODEL=${OPENCODE_MODEL:-opencode-go/deepseek-v4.1-flash#max}
AGENT=${OPENCODE_AGENT:-general}
MAX_STALE=${MAX_STALE:-5}
TIMEOUT=${TIMEOUT:-3600}
STOP=${STOP:-$ROOT/../agent-loop/STOP}
LOG=$ROOT/build/agent-loop
PROMPT=$ROOT/docs/agent-loop-lane-prompt.md

mkdir -p "$LOG"

lane_target() {
    case "$1" in
        1) echo "P-head inter residual/prediction coding (f0-32), excluding the closed skip variants" ;;
        2) echo "P-tail residual/allocation (f33-59), preserving the kept skip/exact-rate composition" ;;
        3) echo "inter transform/quantisation/coefficient coding (non-trellis, non-key)" ;;
        4) echo "inter entropy/context/rate implementation (non-trellis)" ;;
        5) echo "head prediction/ME/reference-selection implementation (existing syntax only)" ;;
        *) echo "P-tail residual coding without a closed mechanism" ;;
    esac
}

i=1
pids=""
while [ "$i" -le "$LANES" ]; do
    target=$(lane_target "$i")
    worktree="$ROOT/../agent-loop-$i"
    branch="exp/agent-loop-$i"
    ledger="docs/agent-loop-ledger-$i.md"
    prompt="$LOG/lane-$i-prompt.txt"
    sed -e "s|{{lane}}|$i|g" -e "s|{{target}}|$target|g" "$PROMPT" > "$prompt"

    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "lane $i: $target"
        echo "  worktree=$worktree branch=$branch ledger=$ledger"
        echo "  prompt=$prompt"
    else
        python3 "$ROOT/tools/agent_loop.py" \
            --iterations "$ITERATIONS" \
            --worktree "$worktree" \
            --branch "$branch" \
            --ledger "$ledger" \
            --prompt-file "$prompt" \
            --model "$MODEL" \
            --agent "$AGENT" \
            --max-stale "$MAX_STALE" \
            --iteration-timeout "$TIMEOUT" \
            --stop-file "$STOP" \
            --sleep 2 >> "$LOG/lane-$i.log" 2>&1 &
        pids="$pids $!"
        echo "lane $i pid $! target: $target"
        sleep 20
    fi
    i=$((i + 1))
done

if [ "${DRY_RUN:-0}" != "1" ]; then
    wait $pids
    echo "all lanes finished"
fi
