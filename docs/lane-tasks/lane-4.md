# Lane 4 task — deblock delta signalling on top of eff-perc

Target: loop-filter strength, tail frames 33-59.

## Read only
- this file;
- the last 15 lines of `docs/agent-loop-ledger-4.md`;
- `docs/negative-results.md`, the rows containing `GDF` and `CDEF`.

Do not read other docs. Do not run `git log`, `git branch`, `git remote`, or
explore the repository. Do not edit `tools/`.

## One hypothesis
The encoder never signals `df_delta_q`; every frame uses only the spec-derived
deblock level. Add the §5.18.7.9 `df_delta_q_present` + delta for luma only,
default off, and choose the delta over a small set (e.g. -2..+2) by measuring
the post-deblock luma SSE against the source on frames 33-59. No other change.

If the delta cannot be made decoder-exact in this iteration, record that and
stop; do not add a different filter tool.

## Allowed files
`src/pipeline/video/post/loop_filters.h`,
`src/pipeline/video/post/loop_filters.cpp`,
`src/pipeline/video/video_headers.cpp`,
`src/pipeline/video/video_frame_filters.cpp`,
`src/pipeline/video/post/tiled_frame.cpp`,
`docs/agent-loop-ledger-4.md`.

## Steps
1. Edit only the allowed files. Default off, env-gated, no preset or pin change,
   no new syntax beyond the existing §5.18.7.9 row, no third-party code.
2. Build: `cmake --build build-release -j8`.
3. Validate: encode 4 frames, decode with `tools/bin/dav2d-macos --strict 1`,
   and require the decoded payload to equal `--video-recon` byte for byte.
4. Write `build/agent-loop/spec.json`:
```json
{"ref":"testdata/nasa_480x240_30fps_60f.y4m","bin":"build-release/av2enc",
 "frames":60,
 "base_args":["--video","--frames","60","--cpu-threads","8","--nightly",
  "--video-drift-refresh-threshold","1.5","--video-drift-refresh-budget","3250"],
 "qp":119,"target_kbps":100.0,
 "arms":[{"name":"base","env":{}},
         {"name":"cand","env":{"AV2ENC_DEBLOCK_DELTA":"1"}}]}
```
5. Run from the worktree root:
```
python3 tools/envelope.py --spec build/agent-loop/spec.json --worktree "$PWD" \
  --base-env '{"AV2ENC_EFF_PERC":"1"}' \
  --ceiling-segment f44-59 --ceiling-floor 88.995
```
6. PASS only if the last line says `PASS`: envelope delta at least +0.15 K
   **and** ceiling delta vs the base arm at least +0.15 K. Anything else is FAIL.
7. On FAIL: `git checkout -- src && git clean -fd src`. On PASS: leave `src/`
   uncommitted. Never commit source.
8. Append one row to the ledger: hypothesis, envelope, ceiling, PASS/FAIL.
9. Stop. Do not try a second hypothesis.

## Closed
GDF, CDEF row omitting, chroma lambda, AQ/delta-Q, key/intra.

## Measured closed: allocation
Static head/tail QP plans, key-offset spends and larger refresh budgets were measured and do not raise the f0-32 ceiling at 100 kbps (`docs/ceiling-allocation.md`). Do not retry them; the ceiling is coding-efficiency-bound.
