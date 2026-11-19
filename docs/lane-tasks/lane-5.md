# Lane 5 — tool: true lookahead skip cost

The tail cost is drift: skips install error that later frames repay. The kept
`AV2ENC_SKIP_DRIFT` charge is a causal decay. Implement the tool with actual
future source frames: project the leaf MV onto the next K source frames, charge
the measured error, default off, `AV2ENC_SKIP_LOOKAHEAD`, swept K=2/4/8. The
per-frame encoder may need the future frames plumbed from the sequence encoder.

## Read only
this file; last 15 lines of `docs/agent-loop-ledger-5.md`; `docs/agent-loop-ledger.md`
future-error rows.

Do not run `git log`/`branch`/`remote`. Analysis tooling under `tools/lane-5/`;
never edit the gate/scoring tools or pins.

## Allowed files
`video_frame_decision.cpp`, `video_frame_internal.h`, `video_frame.cpp`,
`video_frame_sequence.cpp`, `video_skip_drift.h`, `docs/agent-loop-ledger-5.md`.

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

