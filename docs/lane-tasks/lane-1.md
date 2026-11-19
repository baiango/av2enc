# Lane 1 — tool: BA-targeted per-block interp selection

The per-block switchable filter already passes (K 85.230, ceiling 90.500) with
a CSF rule. It improves BA only 2.273 -> 2.251; AVM is 1.29. Improve the tool's
selection rule: score each candidate filter with a Butteraugli-shaped error
(local masking, low-frequency emphasis, chroma) instead of the current CSF, so
the block filter is chosen for the BA term. Gate `AV2ENC_INTERP_BLOCK` +
`AV2ENC_INTERP_BLOCK_BA` swept at 0.5/1.0/2.0.

## Read only
this file; last 15 lines of `docs/agent-loop-ledger-1.md`; `docs/interp-filter.md`.

Do not run `git log`/`branch`/`remote`. You may add analysis tooling under
`tools/lane-1/`; never edit the gate/scoring tools or pins.

## Allowed files
`video_interp_filter.h`, `video_frame_me.cpp`, `video_frame_decision.cpp`,
`video_headers.cpp`, `video_syntax_inter.cpp`, `docs/agent-loop-ledger-1.md`.

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

