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
LANES=${AV2ENC_LANES:-${LANES:-5}}
ITERATIONS=${AV2ENC_ITERATIONS:-${ITERATIONS:-10}}
MODEL=${OPENCODE_MODEL:-opencode-go/deepseek-v4.1-flash#max}
AGENT=${OPENCODE_AGENT:-general}
MAX_STALE=${AV2ENC_MAX_STALE:-5}
TIMEOUT=${AV2ENC_TIMEOUT:-10800}
STOP=${AV2ENC_STOP:-$ROOT/../agent-loop/STOP}
PREFIX=${AV2ENC_LANE_PREFIX:-agent-loop}
LOG=$ROOT/build/agent-loop
PROMPT=$ROOT/docs/agent-loop-lane-prompt.md

DRY=${AV2ENC_DRY_RUN:-${DRY_RUN:-0}}
mkdir -p "$LOG"

lane_base_env() {
    case "$1" in
        2) echo '{"AV2ENC_SKIP_DRIFT":"1","AV2ENC_SKIP_DRIFT_FADE":"1","AV2ENC_SKIP_DRIFT_ILLUM":"1","AV2ENC_SKIP_DRIFT_REFDEF":"1","AV2ENC_DECISION_EXACT_RATE":"1"}' ;;
        *) echo '{}' ;;
    esac
}

lane_ceiling() {
    case "$1" in
        2|4) echo "f44-59 88.995" ;;
        *) echo "f0-32 90.197" ;;
    esac
}

lane_target() {
    case "$1" in
        1) echo "P-head inter residual/prediction coding (f0-32), excluding the closed skip variants" ;;
        2) echo "P-tail residual/allocation (f33-59), preserving the kept skip/exact-rate composition" ;;
        3) echo "inter transform/quantisation/coefficient coding (non-trellis, non-key)" ;;
        4) echo "loop-filter strength: deblock df_delta_q, CDEF strength/index, per-frame LR/CCSO choice" ;;
        5) echo "per-block switchable interp filter and MC prediction (frame-global form is closed)" ;;
        *) echo "P-tail residual coding without a closed mechanism" ;;
    esac
}

i=1
pids=""
while [ "$i" -le "$LANES" ]; do
    target=$(lane_target "$i")
    worktree="$ROOT/../$PREFIX-$i"
    branch="exp/$PREFIX-$i"
    ledger="docs/agent-loop-ledger-$i.md"
    prompt="$LOG/lane-$i-prompt.txt"
    sed -e "s|{{lane}}|$i|g" -e "s|{{target}}|$target|g" "$PROMPT" > "$prompt"

    if [ "$DRY" = "1" ]; then
        echo "lane $i: $target"
        echo "  worktree=$worktree branch=$branch ledger=$ledger"
        echo "  prompt=$prompt agent=$AGENT model=$MODEL iterations=$ITERATIONS"
    else
        echo "=== run $(date '+%Y-%m-%d %H:%M:%S') prefix=$PREFIX lane=$i iterations=$ITERATIONS ===" >> "$LOG/lane-$i.log"
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
            --envelope-gate \
            --envelope-threshold "${AV2ENC_ENVELOPE_DELTA:-0.15}" \
            --envelope-base-env "$(lane_base_env "$i")" \
            --envelope-ceiling-segment "$(set -- $(lane_ceiling "$i"); echo "$1")" \
            --envelope-ceiling-floor "$(set -- $(lane_ceiling "$i"); echo "$2")" \
            --envelope-revert \
            --stop-file "$STOP" \
            --sleep 2 >> "$LOG/lane-$i.log" 2>&1 &
        pids="$pids $!"
        echo "lane $i pid $! target: $target"
        sleep 20
    fi
    i=$((i + 1))
done

if [ "$DRY" != "1" ]; then
    wait $pids
    echo "all lanes finished"
fi
