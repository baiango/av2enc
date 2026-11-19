# Lane 4 — PIPE-BA + Butteraugli-shaped D tool

Two-part program, both required:
1. **Pipeline**: a decision path under `AV2ENC_PIPE_BA` whose distortion is a
   Butteraugli-shaped error (local masking, low-frequency emphasis, chroma).
2. **Tool**: that error model itself, replacing the current CSF in the leaf and
   TX decision; sweep the weight 0.5/1.0/2.0.

Each iteration advances one part. Keep the fixes.

## Read only
this file; last 15 lines of `docs/agent-loop-ledger-4.md`; `docs/eff-perc.md`.

Do not run `git log`/`branch`/`remote`. You may add analysis tooling under `tools/lane-N/`; never edit the gate/scoring tools or pins.

## Allowed files
new `video_pipe_ba.h/.cpp`, `video_frame_decision.cpp`,
`video_frame_internal.h`, `video_eff_perc.h`, `docs/agent-loop-ledger-4.md`.

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

