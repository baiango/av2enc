# Lane 2 — PIPE-HORIZON + true lookahead skip tool

Two-part program, both required:
1. **Pipeline**: a frame-horizon decision path under `AV2ENC_PIPE_HORIZON`
   whose J includes the error installed in the next K source frames.
2. **Tool**: the horizon skip charge itself — project the leaf MV onto future
   source frames, measure the error, charge it (`AV2ENC_PIPE_HORIZON_K` 2/4/8).
   Plumb future frames from the sequence encoder; do not use a decoded proxy.

Each iteration advances one part. Keep the fixes.

## Read only
this file; last 15 lines of `docs/agent-loop-ledger-2.md`; `docs/agent-loop-ledger.md`
future-error rows.

Do not run `git log`/`branch`/`remote`. You may add analysis tooling under `tools/lane-N/`; never edit the gate/scoring tools or pins.

## Allowed files
new `video_pipe_horizon.h/.cpp`, `video_frame_decision.cpp`,
`video_frame_internal.h`, `video_frame.cpp`, `video_frame_sequence.cpp`,
`video_skip_drift.h`, `docs/agent-loop-ledger-2.md`.

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
  --base-env '{"AV2ENC_EFF_PERC":"1"}' --best-known 85.099 --min-arms 3 \\
  --ceiling-segment f0-32 --ceiling-floor 90.197
```
PASS only if `PASS`: envelope ≥ 85.249, ≥3 arms, f0-32 ceiling delta ≥ +0.15.
Continue the program after a failure; append one ledger row per iteration.

