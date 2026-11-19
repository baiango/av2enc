# Lane 5 task — per-block switchable interp filter on top of eff-perc

Target: per-block switchable interp filter and MC prediction. The frame-global
form is closed.

## Read only
- this file;
- the last 15 lines of `docs/agent-loop-ledger-5.md`;
- `docs/negative-results.md`, the row containing `frame-level subpel`.

Do not read other docs. Do not run `git log`, `git branch`, `git remote`, or
explore the repository. Do not edit `tools/`.

## One hypothesis
This worktree already has the per-block switchable filter (`AV2ENC_INTERP_BLOCK`,
`form_b2`, +0.205 envelope on nightly). It was measured on nightly. Wire it into
the eff-perc prediction path and measure whether it survives there. No other
change.

## Allowed files
`src/pipeline/video/video_interp_filter.h`,
`src/pipeline/video/video_frame_me.cpp`,
`src/pipeline/video/video_frame_decision.cpp`,
`src/pipeline/video/video_headers.cpp`,
`docs/agent-loop-ledger-5.md`.

## Steps
1. Edit only the allowed files. Default off, env-gated, no preset or pin change,
   no new syntax, no third-party code.
2. Build: `cmake --build build-release -j8`.
3. Validate one stream: filter on and off must decode `dav2d --strict 1` and
   match `--video-recon` byte for byte.
4. Write `build/agent-loop/spec.json`:
```json
{"ref":"testdata/nasa_480x240_30fps_60f.y4m","bin":"build-release/av2enc",
 "frames":60,
 "base_args":["--video","--frames","60","--cpu-threads","8","--nightly",
  "--video-drift-refresh-threshold","1.5","--video-drift-refresh-budget","3250"],
 "qp":119,"target_kbps":100.0,
 "arms":[{"name":"base","env":{}},
         {"name":"cand","env":{"AV2ENC_INTERP_BLOCK":"1",
          "AV2ENC_INTERP_BLOCK_ALPHABET":"0",
          "AV2ENC_INTERP_BLOCK_BITS":"2",
          "AV2ENC_INTERP_BLOCK_CHROMA":"1",
          "AV2ENC_INTERP_BLOCK_CSF":"0",
          "AV2ENC_INTERP_BLOCK_EMIT_BITS":"1"}}]}
```
5. Run from the worktree root:
```
python3 tools/envelope.py --spec build/agent-loop/spec.json --worktree "$PWD" \
  --base-env '{"AV2ENC_EFF_PERC":"1"}' \
  --ceiling-segment f0-32 --ceiling-floor 90.197
```
6. PASS only if the last line says `PASS` **and** the printed f0-32 ceiling is
   at least 90.197. Anything else is FAIL.
7. On FAIL: `git checkout -- src && git clean -fd src`. On PASS: leave `src/`
   uncommitted. Never commit source.
8. Append one row to the ledger: hypothesis, envelope, ceiling, PASS/FAIL.
9. Stop. Do not try a second hypothesis.

## Closed
Frame-global EIGHTTAP/SMOOTH/SHARP/BILINEAR (measured negative), OBMC (not in
AV2), TIP, multi-ref, warp.
