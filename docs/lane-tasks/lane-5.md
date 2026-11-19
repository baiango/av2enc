# Lane 5 — PIPE-2PASS: measured two-pass frame pipeline

Program (multi-iteration): build a **two-pass per-frame pipeline** under
`AV2ENC_PIPE_2PASS`, default off. Pass 1 runs the current core and records each
leaf's exact rate, distortion and the frame's measured reference error. Pass 2
re-decides and re-encodes the frame with those measured costs and a corrected
lambda, then keeps whichever pass wins J. Reuse the exact-rate probe, CDF carry,
per-block interp and bug fixes.

Each iteration: one stage (statistics capture, re-decision, re-encode), build,
gate, record.

## Read only
- this file; last 15 lines of `docs/agent-loop-ledger-5.md`;
- `docs/rate-model-results.md`.

Do not run `git log`/`branch`/`remote`. You may **add** first-party analysis tooling under `tools/lane-5/` (stdlib only). Do **not** edit the gate or scoring tools: `tools/envelope.py`, `tools/oracle.py`, `tools/quality.py`, `tools/kpi.py`, `tools/extra_metrics.py`, or any pin.

## Allowed files
new `src/pipeline/video/video_pipe_2pass.h/.cpp`,
`video_frame_decision.cpp`, `video_frame_internal.h`,
`video_frame_sequence.cpp`, `docs/agent-loop-ledger-5.md`.

## Tooling you may improve
Add only analysis/measurement helpers under `tools/lane-5/`:
- per-frame/per-segment K, JOD, BA, SS2 attribution for a stream;
- decision counters for the new pipeline (leaves, mode changes, cost);
- a rate/distortion scatter or horizon-error profiler.
Each tool must be first-party, import only the Python stdlib, and be reported in the ledger with what it measured. Never modify the gate.

## Gate
Arms lambda correction 0.8/1.0/1.25 under `AV2ENC_PIPE_2PASS`,
`AV2ENC_PIPE_2PASS_LAMBDA`. Same gate command, base env eff, 3 arms,
`--ceiling-segment f0-32 --ceiling-floor 90.197`.
PASS only if `PASS`: envelope ≥ 85.249, ≥3 arms, ceiling delta ≥ +0.15.
