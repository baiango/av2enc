# Lane 3 — per-block switchable interp with a perceptual RD rule

Target: raise the 100 kbps K envelope. Bar: **envelope ≥ 85.249**.

## Read only
- this file; last 15 lines of `docs/agent-loop-ledger-3.md`;
- `docs/interp-filter.md` (frame-global negative; per-block unmeasured).

Do not run `git log`/`branch`/`remote`. Do not edit `tools/`.

## Hypothesis
The frame-global filter is negative, but the per-block switchable form has a
strictly higher bound. Wire `is_filter_switchable=1` + `TileInterpFilterCdf`
(large/small rules per `needs_interp_filter`) and select per leaf with a
**perceptual** RD term, gated `AV2ENC_INTERP_BLOCK`. A PSNR/SSE rule picks the
wrong filter, so sweep the CSF weight.

## Allowed files
`src/pipeline/video/video_interp_filter.h`, `video_frame_me.cpp`,
`video_frame_decision.cpp`, `video_headers.cpp`, `video_syntax_inter.cpp`,
`docs/agent-loop-ledger-3.md`.

## Steps
1. Default off, env-gated, no preset/pin change, no new syntax (use the
   existing §5.18.5.1 switchable path), no third-party code.
2. Build and validate: filter on/off decode `dav2d --strict 1` and match
   `--video-recon` byte for byte.
3. Spec:
```json
{"ref":"testdata/nasa_480x240_30fps_60f.y4m","bin":"build-release/av2enc",
 "frames":60,
 "base_args":["--video","--frames","60","--cpu-threads","8","--nightly",
  "--video-drift-refresh-threshold","1.5","--video-drift-refresh-budget","3250"],
 "qp":119,"target_kbps":100.0,
 "arms":[{"name":"base","env":{}},
   {"name":"c05","env":{"AV2ENC_INTERP_BLOCK":"1","AV2ENC_INTERP_BLOCK_CSF":"0.5"}},
   {"name":"c10","env":{"AV2ENC_INTERP_BLOCK":"1","AV2ENC_INTERP_BLOCK_CSF":"1.0"}},
   {"name":"c20","env":{"AV2ENC_INTERP_BLOCK":"1","AV2ENC_INTERP_BLOCK_CSF":"2.0"}}]}
```
4. Run:
```
python3 tools/envelope.py --spec build/agent-loop/spec.json --worktree "$PWD" \
  --base-env '{"AV2ENC_EFF_PERC":"1"}' --best-known 85.099 --min-arms 3
```
5. PASS only if `PASS`: envelope ≥ 85.249 with ≥3 arms and exact decode.
6. On FAIL: `git checkout -- src && git clean -fd src`. On PASS: leave `src/`.
7. One ledger row; stop.

## Closed
Frame-global filters, OBMC (not AV2), warp, TIP, multi-ref, allocation.
