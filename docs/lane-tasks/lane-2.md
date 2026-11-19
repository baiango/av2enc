# Lane 2 — PIPE-HORIZON: future-source decision pipeline

Program (multi-iteration): build a new pipeline under `AV2ENC_PIPE_HORIZON`,
default off, whose J includes the error a decision installs in the **next K
source frames** (frame horizon), not a causal decay. Reuse the exact-rate
probe, perceptual D, CDF carry and decoder-exact primitives. Keep the bug fixes.
Write a new decision path; do not extend eff-perc's branches.

Each iteration: advance one stage (plumb future frames, project MV, charge,
re-decide), build, gate, record.

## Read only
- this file; last 15 lines of `docs/agent-loop-ledger-2.md`;
- `docs/agent-loop-ledger.md` future-error rows.

Do not run `git log`/`branch`/`remote`. Do not edit `tools/`.

## Allowed files
new `src/pipeline/video/video_pipe_horizon.h/.cpp`,
`video_frame_decision.cpp`, `video_frame_internal.h`, `video_frame.cpp`,
`video_frame_sequence.cpp`, `docs/agent-loop-ledger-2.md`.

## Gate
Arms K = 2/4/8:
```json
{"ref":"testdata/nasa_480x240_30fps_60f.y4m","bin":"build-release/av2enc",
 "frames":60,
 "base_args":["--video","--frames","60","--cpu-threads","8","--nightly",
  "--video-drift-refresh-threshold","1.5","--video-drift-refresh-budget","3250"],
 "qp":119,"target_kbps":100.0,
 "arms":[{"name":"base","env":{}},
  {"name":"k2","env":{"AV2ENC_PIPE_HORIZON":"1","AV2ENC_PIPE_HORIZON_K":"2"}},
  {"name":"k4","env":{"AV2ENC_PIPE_HORIZON":"1","AV2ENC_PIPE_HORIZON_K":"4"}},
  {"name":"k8","env":{"AV2ENC_PIPE_HORIZON":"1","AV2ENC_PIPE_HORIZON_K":"8"}}]}
```
Run the same gate command as lane 1 (`--base-env eff`, `--best-known 85.099`,
`--min-arms 3`, `--ceiling-segment f0-32 --ceiling-floor 90.197`).
PASS only if `PASS`: envelope ≥ 85.249, ≥3 arms, ceiling delta ≥ +0.15.
