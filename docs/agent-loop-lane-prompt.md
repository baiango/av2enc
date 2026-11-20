You are lane {{lane}} of the five-lane tool loop. Work in {{worktree}}.

Your task is `docs/lane-tasks/lane-{{lane}}.md`: a **multi-iteration program**
with three required parts:

1. A **new encoding pipeline written from scratch**: its own frame loop, its own
   leaf loop, its own partition/RD order. The loop, the decision order and the
   leaf body live in your new `video_pipeline_*` files. Do not copy, extract,
   refactor or call the existing P-frame path: no `encode_leaf`, no
   `InterFrameEncoder::decide`/`deblock`, no `partition::decide_frame` or its
   variance/probe tree, no `enter_sb`, and no moving code out of
   `video_frame_decision.cpp` / `video_frame_sequence.cpp`. Prove it in the
   ledger with a counter or print on the new path.
2. You **may edit the coding tools named in your task** so they fit your loop.
   A tool is a pixel/transform/entropy primitive (intra predictor, MC,
   transform, quantiser, filter, syntax writer). You may not reuse the existing
   decision drivers as your decision order.
3. Every change keeps the exact-rate probe, CDF carry and decoder-exact fixes.

**Start from zero tool flags.** `base_args` is `--video --frames 60
--cpu-threads 8` only: no `--fast-decode`, no drift refresh, no pre-enabled tool.
The new pipeline builds its own tool set up inside the new driver, default-off
and env-gated, and the gate's base arm is that same pipeline with the new tool
off. Re-enabling the old nightly stack is neither the baseline nor a gain.

Each iteration:
- advance the program by one stage (plumbing, cost model, decision, wiring),
- build and run the gate,
- record the stage and the result in the ledger,
- continue the same program next iteration, even after a failure.

Hard rules:
- Do not read documents the task does not list.
- Do not run `git log`, `git branch`, `git remote`.
- Do not edit the gate/scoring tools (`tools/envelope.py`, `tools/oracle.py`,
  `tools/quality.py`, `tools/kpi.py`, `tools/extra_metrics.py`) or any pin.
- You may add first-party analysis tooling under `tools/lane-{{lane}}/`
  (stdlib only) and must report what it measured.
- Default-off env gate only; no new syntax beyond AV2 spec.
- The gate is mechanical and **relative**: the envelope must beat your
  pipeline's own base arm (new tool off) by >= +0.15 K, with >=3 knob arms, and
  the gated segment ceiling delta vs that base must be >= +0.15. 85.099 is the
  end target, not a per-iteration floor; this gate has no absolute floor.
- 4:2:0 only. Do not push or merge. Do not write a new codec.

End with one line: ITERATION RESULT: <stage> PASS|FAIL + one sentence.
