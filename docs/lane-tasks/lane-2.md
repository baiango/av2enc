# Lane 2 task — P-tail residual on top of eff-perc

Target: P-tail residual/allocation, frames 33-59.

## Read only
- this file;
- the last 15 lines of `docs/agent-loop-ledger-2.md`;
- `docs/negative-results.md`, the rows containing `inter leaf residual`,
  `non-causal reference`, and `multi-reference P`.

Do not read other docs. Do not run `git log`, `git branch`, `git remote`, or
explore the repository. Do not edit `tools/`.

## One hypothesis
The kept skip/refdef latches were measured on nightly and the composition with
eff-perc was negative. This worktree still carries the nightly-path
`video_skip_drift.h`. Wire **one** of its terms (the reference-deficit latch)
into the eff-perc leaf decision on frames 33-59, default off, and measure.
No other change.

## Allowed files
`src/pipeline/video/video_skip_drift.h`,
`src/pipeline/video/video_frame_decision.cpp`,
`docs/agent-loop-ledger-2.md`.

## Steps
1. Edit only the allowed files. Default off, env-gated, no preset or pin change,
   no new syntax, no third-party code.
2. Build: `cmake --build build-release -j8`.
3. Write `build/agent-loop/spec.json`:
```json
{"ref":"testdata/nasa_480x240_30fps_60f.y4m","bin":"build-release/av2enc",
 "frames":60,
 "base_args":["--video","--frames","60","--cpu-threads","8","--nightly",
  "--video-drift-refresh-threshold","1.5","--video-drift-refresh-budget","3250"],
 "qp":119,"target_kbps":100.0,
 "arms":[{"name":"base","env":{}},
         {"name":"cand","env":{"AV2ENC_SKIP_DRIFT_REFDEF":"1"}}]}
```
4. Run from the worktree root:
```
python3 tools/envelope.py --spec build/agent-loop/spec.json --worktree "$PWD" \
  --base-env '{"AV2ENC_EFF_PERC":"1"}' \
  --ceiling-segment f44-59 --ceiling-floor 88.995
```
5. PASS only if the last line says `PASS` **and** the printed f44-59 ceiling is
   at least 88.995. Anything else is FAIL.
6. On FAIL: `git checkout -- src && git clean -fd src`. On PASS: leave `src/`
   uncommitted. Never commit source.
7. Append one row to the ledger: hypothesis, envelope, ceiling, PASS/FAIL.
8. Stop. Do not try a second hypothesis.

## Closed
Re-enabling the stack as an arm, linear pacing, AQ/delta-Q, multi-reference P,
TIP, BAWP, key/intra.

## Measured closed: allocation
Static head/tail QP plans, key-offset spends and larger refresh budgets were measured and do not raise the f0-32 ceiling at 100 kbps (`docs/ceiling-allocation.md`). Do not retry them; the ceiling is coding-efficiency-bound.
