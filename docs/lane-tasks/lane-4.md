# Lane 4 — PIPE-BA: Butteraugli-targeted decision pipeline

Program (multi-iteration): build a decision pipeline under `AV2ENC_PIPE_BA`,
default off, whose distortion term is a **Butteraugli-shaped error model**
(local masking, low-frequency emphasis, chroma coupling), not the current CSF.
Reuse the exact-rate probe, CDF carry, per-block interp and bug fixes; write a
new decision path. The target is the BA term: 2.25 now vs AVM 1.29.

Each iteration: one stage (error model, weight map, decision wiring), build,
gate, record.

## Read only
- this file; last 15 lines of `docs/agent-loop-ledger-4.md`;
- `docs/eff-perc.md` (the D term), `docs/negative-results.md` dead-zone row.

Do not run `git log`/`branch`/`remote`. You may **add** first-party analysis tooling under `tools/lane-4/` (stdlib only). Do **not** edit the gate or scoring tools: `tools/envelope.py`, `tools/oracle.py`, `tools/quality.py`, `tools/kpi.py`, `tools/extra_metrics.py`, or any pin.

## Allowed files
new `src/pipeline/video/video_pipe_ba.h/.cpp`,
`video_frame_decision.cpp`, `video_frame_internal.h`, `video_eff_perc.h`,
`docs/agent-loop-ledger-4.md`.

## Tooling you may improve
Add only analysis/measurement helpers under `tools/lane-4/`:
- per-frame/per-segment K, JOD, BA, SS2 attribution for a stream;
- decision counters for the new pipeline (leaves, mode changes, cost);
- a rate/distortion scatter or horizon-error profiler.
Each tool must be first-party, import only the Python stdlib, and be reported in the ledger with what it measured. Never modify the gate.

## Gate
Arms BA weight 0.5/1.0/2.0 under `AV2ENC_PIPE_BA`,
`AV2ENC_PIPE_BA_WEIGHT`. Same gate command, base env eff, 3 arms,
`--ceiling-segment f0-32 --ceiling-floor 90.197`.
PASS only if `PASS`: envelope ≥ 85.249, ≥3 arms, ceiling delta ≥ +0.15.
