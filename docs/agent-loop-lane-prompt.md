You are lane {{lane}} of the five-lane tool loop. Work in {{worktree}}.

Your task is `docs/lane-tasks/lane-{{lane}}.md`: a **multi-iteration program**
that has three required parts: (1) a **new encoding pipeline from scratch**
(own frame/leaf loop, partition/RD order) that **must not call `encode_leaf`**,
(2) it may **improve the existing coding tools** it calls, and (3) every
change keeps the exact-rate probe, CDF carry and decoder-exact fixes. Prove in
the ledger that the new path does not enter `encode_leaf`.

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
- The gate is mechanical and **relative to your pipeline**: the envelope must
  beat your pipeline's own base arm (tool off) by ≥ +0.15 K, with ≥3 knob arms,
  and the f0-32 ceiling delta vs that base must be ≥ +0.15. 85.099 is the end
  target, not the per-iteration gate. The runner re-measures after this call.
- 4:2:0 only. Do not push or merge. Do not write a new codec.

End with one line: ITERATION RESULT: <stage> PASS|FAIL + one sentence.
