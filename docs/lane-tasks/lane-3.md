# Lane 3 — PIPE-TRELLIS: non-greedy partition/mode pipeline

Program (multi-iteration): replace the greedy leaf decision with a **per-frame
dynamic program** under `AV2ENC_PIPE_TRELLIS`, default off: cost every legal
partition/leaf mode combination with the exact syntax rate and perceptual D,
then choose the minimum-cost tree for the frame. Reuse the primitives and bug
fixes; do not extend eff-perc's branch stack.

Each iteration: one stage (cost table, DP over the tree, reconstruct), build,
gate, record.

## Read only
- this file; last 15 lines of `docs/agent-loop-ledger-3.md`;
- `docs/video-partition-rd.md`.

Do not run `git log`/`branch`/`remote`. You may **add** first-party analysis tooling under `tools/lane-3/` (stdlib only). Do **not** edit the gate or scoring tools: `tools/envelope.py`, `tools/oracle.py`, `tools/quality.py`, `tools/kpi.py`, `tools/extra_metrics.py`, or any pin.

## Allowed files
new `src/pipeline/video/video_pipe_trellis.h/.cpp`,
`video_frame_decision.cpp`, `video_frame_internal.h`,
`docs/agent-loop-ledger-3.md`.

## Tooling you may improve
Add only analysis/measurement helpers under `tools/lane-3/`:
- per-frame/per-segment K, JOD, BA, SS2 attribution for a stream;
- decision counters for the new pipeline (leaves, mode changes, cost);
- a rate/distortion scatter or horizon-error profiler.
Each tool must be first-party, import only the Python stdlib, and be reported in the ledger with what it measured. Never modify the gate.

## Gate
Arms lambda 0.15/0.20/0.30 under `AV2ENC_PIPE_TRELLIS`,
`AV2ENC_PIPE_TRELLIS_LAMBDA`. Same gate command, base env eff, 3 arms,
`--ceiling-segment f0-32 --ceiling-floor 90.197`.
PASS only if `PASS`: envelope ≥ 85.249, ≥3 arms, ceiling delta ≥ +0.15.
