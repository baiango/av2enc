# Lane 1 task — head f0-32 on top of eff-perc

Target: P-head inter residual/prediction, frames 1-32.

## Read only
- this file;
- the last 15 lines of `docs/agent-loop-ledger-1.md`;
- `docs/negative-results.md`, the rows containing `inter leaf residual`,
  `frame-level subpel`, and `intra`.

Do not read other docs. Do not run `git log`, `git branch`, `git remote`, or
explore the repository. Do not edit `tools/`.

## One hypothesis
This worktree already has the head allocator (`AV2ENC_HEAD_REPAIR_ALLOC`), and
it is **inert under eff-perc**: `effh` and `eff` encode byte-identically
(25,230 / 24,674 B). Wire it into the eff-perc decision path so a head leaf
whose prediction SSE is high can spend the rank it would otherwise skip, then
measure. No other change.

## Allowed files
`src/pipeline/video/video_head_repair_alloc.h`,
`src/pipeline/video/video_frame_decision.cpp`,
`src/pipeline/video/video_eff_perc.h`,
`src/pipeline/video/video_frame_internal.h`,
`docs/agent-loop-ledger-1.md`.

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
         {"name":"cand","env":{"AV2ENC_HEAD_REPAIR_ALLOC":"1"}}]}
```
4. Run from the worktree root:
```
python3 tools/envelope.py --spec build/agent-loop/spec.json --worktree "$PWD" \
  --base-env '{"AV2ENC_EFF_PERC":"1"}' \
  --ceiling-segment f0-32 --ceiling-floor 90.197
```
5. PASS only if the last line says `PASS` **and** the printed f0-32 ceiling is
   at least 90.197. Anything else is FAIL.
6. On FAIL: `git checkout -- src && git clean -fd src`. On PASS: leave `src/`
   uncommitted. Never commit source.
7. Append one row to the ledger: hypothesis, envelope, ceiling, PASS/FAIL.
8. Stop. Do not try a second hypothesis.

## Closed
Key/intra, RDOQ/trellis, multi-reference P, TIP, BAWP, frame-global subpel
filter, AQ, delta-Q, skip-drift variants on nightly.
