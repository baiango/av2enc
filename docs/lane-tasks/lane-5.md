# Lane 5 — PIPE-2PASS + adaptive MVD tool

Two-part program, both required:
1. **Pipeline**: a measured two-pass per-frame path under `AV2ENC_PIPE_2PASS`
   (pass 1 measures exact rates/errors, pass 2 re-decides and re-encodes).
2. **Tool**: adaptive MVD on the eff-perc path (`AV2ENC_ADAPTIVE_MVD`), never
   composed yet; sweep the precision/strength.

Each iteration advances one part. Keep the fixes.

## Read only
this file; last 15 lines of `docs/agent-loop-ledger-5.md`; `docs/combo-pred.md`
rows with `amvd`.

Do not run `git log`/`branch`/`remote`. You may add analysis tooling under `tools/lane-N/`; never edit the gate/scoring tools or pins.

## Allowed files
new `video_pipe_2pass.h/.cpp`, `video_frame_decision.cpp`,
`video_frame_internal.h`, `video_frame_me.cpp`, `video_syntax_inter.cpp`,
`docs/agent-loop-ledger-5.md`.

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

