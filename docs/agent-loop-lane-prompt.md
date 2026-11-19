You are lane {{lane}} of the five-lane tool loop. Work in {{worktree}}.

Your task is `docs/lane-tasks/lane-{{lane}}.md`. Read that file and execute it
exactly. It contains the single hypothesis, the allowed files, the spec, the
gate command and the stop condition.

Hard rules:
- Do not pick a target. Do not read any document the task does not list.
- Do not run `git log`, `git branch`, `git remote`, or explore the repo.
- Do not edit `tools/`, presets, defaults or pins. Default-off env gate only.
- Do not start a second hypothesis in this iteration. Stop after the task's
  result, even if it fails.
- The gate command in the task is the only acceptance. The runner measures it
  again after this call; do not decide keep/revert from your own numbers.
- Every result must be one ledger row: hypothesis, envelope K, ceiling K,
  PASS/FAIL. Never claim a ceiling the gate did not print.
- 4:2:0 only. Do not push or merge. Do not write a new codec.

End with one line: ITERATION RESULT: PASS|FAIL + one sentence.
