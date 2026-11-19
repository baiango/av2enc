# Lane 2 — tool: adaptive MVD composed with eff-perc

Adaptive MVD measured +0.20/+0.39 K on nightly and has never been composed with
eff-perc. Implement/wire the tool on the eff decision path and sweep its
strength. This is a coding tool, not a decision-wrapper change.

## Read only
this file; last 15 lines of `docs/agent-loop-ledger-2.md`; `docs/combo-pred.md`
rows with `amvd`.

Do not run `git log`/`branch`/`remote`. Analysis tooling under `tools/lane-2/`;
never edit the gate/scoring tools or pins.

## Allowed files
`video_frame_decision.cpp`, `video_frame_internal.h`, `video_frame_me.cpp`,
`video_syntax_inter.cpp`, `docs/agent-loop-ledger-2.md`.

## Gate
Three knob arms of the tool on the eff-perc base env. Spec:
```json
{"ref":"testdata/nasa_480x240_30fps_60f.y4m","bin":"build-release/av2enc",
 "frames":60,
 "base_args":["--video","--frames","60","--cpu-threads","8","--nightly",
  "--video-drift-refresh-threshold","1.5","--video-drift-refresh-budget","3250"],
 "qp":119,"target_kbps":100.0,
 "arms":[{"name":"base","env":{}},{"name":"a1","env":{}},{"name":"a2","env":{}},{"name":"a3","env":{}}]}
```
Fill a1/a2/a3 with the tool knob values named above. Run:
```
python3 tools/envelope.py --spec build/agent-loop/spec.json --worktree "$PWD" \\
  --base-env '{"AV2ENC_EFF_PERC":"1"}' --best-known 85.099 --min-arms 3 \\
  --ceiling-segment f0-32 --ceiling-floor 90.197
```
PASS only if `PASS`: envelope ≥ 85.249, ≥3 arms, f0-32 ceiling delta ≥ +0.15.
On FAIL keep the tool code and continue next iteration; the runner reverts only
on regression. Append one ledger row per iteration.

