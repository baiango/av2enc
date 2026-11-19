# Lane 3 task — inter TX-plan RDOQ on top of eff-perc

Target: inter transform/quantisation/coefficient coding. Not key, not trellis.

## Read only
- this file;
- the last 15 lines of `docs/agent-loop-ledger-3.md`;
- `docs/negative-results.md`, the rows containing `inter leaf residual` and
  `RDOQ`.

Do not read other docs. Do not run `git log`, `git branch`, `git remote`, or
explore the repository. Do not edit `tools/`.

## One hypothesis
First run `git grep -n AV2ENC_INTER_TX_PLAN_RDOQ`. If it exists, wire that
top-2 plan re-quantisation into the eff-perc TX search path. If it does not
exist in this tree, implement a minimal top-2 inter TX-plan re-quantisation in
the allowed files. Either way, one mechanism only.

## Allowed files
`src/pipeline/video/video_tx_search.cpp`,
`src/pipeline/video/video_tx_search.h`,
`src/pipeline/video/video_frame_decision.cpp`,
`docs/agent-loop-ledger-3.md`.

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
         {"name":"cand","env":{"AV2ENC_INTER_TX_PLAN_RDOQ":"1",
          "AV2ENC_INTER_TX_PLAN_RDOQ_TOP":"2",
          "AV2ENC_INTER_TX_PLAN_RDOQ_TYPES":"1"}}]}
```
4. Run from the worktree root:
```
python3 tools/envelope.py --spec build/agent-loop/spec.json --worktree "$PWD" \
  --base-env '{"AV2ENC_EFF_PERC":"1"}' \
  --ceiling-segment f0-32 --ceiling-floor 90.197
```
5. PASS only if the last line says `PASS`: envelope delta at least +0.15 K
   **and** ceiling delta vs the base arm at least +0.15 K. Anything else is FAIL.
6. On FAIL: `git checkout -- src && git clean -fd src`. On PASS: leave `src/`
   uncommitted. Never commit source.
7. Append one row to the ledger: hypothesis, envelope, ceiling, PASS/FAIL.
8. Stop. Do not try a second hypothesis.

## Closed
HF RDOQ/trellis, key/intra, dead-zone quantisers, TX-size menus, multi-ref, TIP.

## Measured closed: allocation
Static head/tail QP plans, key-offset spends and larger refresh budgets were measured and do not raise the f0-32 ceiling at 100 kbps (`docs/ceiling-allocation.md`). Do not retry them; the ceiling is coding-efficiency-bound.
