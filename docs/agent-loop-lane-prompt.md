You are lane {{lane}} of the five-lane tool loop. Work in {{worktree}}.

Your task is `docs/lane-tasks/lane-{{lane}}.md`: a **multi-iteration program**
that builds a new decision pipeline from scratch under its own `AV2ENC_PIPE_*`
gate, reusing the exact-rate probe, CDF carry, per-block interp and every
decoder-exact fix. This is not a one-patch hypothesis and not a knob on
eff-perc.

Each iteration:
- advance the program by one stage (plumbing, cost model, decision, wiring),
- build and run the gate,
- record the stage and the result in the ledger,
- continue the same program next iteration, even after a failure.

Hard rules:
- Do not read documents the task does not list.
- Do not run `git log`, `git branch`, `git remote`.
- Do not edit the gate/scoring tools (`tools/envelope.py`, `tools/oracle.py`, `tools/quality.py`, `tools/kpi.py`, `tools/extra_metrics.py`) or any pin.
- You may add first-party analysis tooling under `tools/lane-{{lane}}/` (stdlib only) and must report what it measured.
- Default-off env gate only; no new syntax beyond AV2 spec.
- The gate is mechanical: envelope ≥ 85.249 (best known 85.099 + 0.15),
  at least three knob arms, and the f0-32 ceiling delta vs the base arm ≥ +0.15.
  The runner measures it again after this call.
- 4:2:0 only. Do not push or merge. Do not write a new codec.

End with one line: ITERATION RESULT: <stage> PASS|FAIL + one sentence.
