# Lane 1 — PIPE-EXACT + BA-targeted interp tool

Two-part program, both required:
1. **Pipeline**: a new exhaustive per-leaf RD path under `AV2ENC_PIPE_EXACT`
   (skip / inter / intra, exact syntax rate, perceptual D), not eff-perc's
   branch stack.
2. **Tool**: per-block switchable interp selection scored by a
   Butteraugli-shaped error (local masking, low-frequency emphasis, chroma)
   instead of CSF, under `AV2ENC_INTERP_BLOCK_BA`; sweep 0.5/1.0/2.0.

Each iteration advances one part. Keep the exact-rate model, CDF carry,
per-block interp and decoder-exact fixes.

## Read only
this file; last 15 lines of `docs/agent-loop-ledger-1.md`; `docs/interp-filter.md`,
`docs/pipe-perc.md`.

Do not run `git log`/`branch`/`remote`. You may add analysis tooling under `tools/lane-N/`; never edit the gate/scoring tools or pins.

## Allowed files
new `video_pipe_exact.h/.cpp`, `video_interp_filter.h`, `video_frame_me.cpp`,
`video_frame_decision.cpp`, `video_frame_internal.h`, `video_headers.cpp`,
`docs/agent-loop-ledger-1.md`.

## Gate
Three knob arms (pipeline and/or tool). Spec:
```json
{"ref":"testdata/nasa_480x240_30fps_60f.y4m","bin":"build-release/av2enc",
 "frames":60,
 "base_args":["--video","--frames","60","--cpu-threads","8","--nightly",
  "--video-drift-refresh-threshold","1.5","--video-drift-refresh-budget","3250"],
 "qp":119,"target_kbps":100.0,
 "arms":[{"name":"base","env":{}},{"name":"a1","env":{}},{"name":"a2","env":{}},{"name":"a3","env":{}}]}
```
Fill a1/a2/a3 with the knob values named above. Run:
```
python3 tools/envelope.py --spec build/agent-loop/spec.json --worktree "$PWD" \\
  --base-env '{"AV2ENC_PIPE_EXACT":"1"}' --min-arms 3 \\
  --ceiling-segment f0-32 --ceiling-floor 90.197
```
PASS only if `PASS`: the envelope beats the **pipeline's own base arm**
(tool off) by ≥ +0.15 K, with ≥3 arms, and the f0-32 ceiling delta vs that
base is ≥ +0.15. 85.099 is the end target, not this gate.
Continue the program after a failure; append one ledger row per iteration.

