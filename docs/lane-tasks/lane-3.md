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

Do not run `git log`/`branch`/`remote`. Do not edit `tools/`.

## Allowed files
new `src/pipeline/video/video_pipe_trellis.h/.cpp`,
`video_frame_decision.cpp`, `video_frame_internal.h`,
`docs/agent-loop-ledger-3.md`.

## Gate
Arms lambda 0.15/0.20/0.30 under `AV2ENC_PIPE_TRELLIS`,
`AV2ENC_PIPE_TRELLIS_LAMBDA`. Same gate command, base env eff, 3 arms,
`--ceiling-segment f0-32 --ceiling-floor 90.197`.
PASS only if `PASS`: envelope ≥ 85.249, ≥3 arms, ceiling delta ≥ +0.15.
