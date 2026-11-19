# Lane 4 — new pipeline + distortion/quant tool

**New pipeline (from scratch)**: Butteraugli-shaped distortion driver
`AV2ENC_PIPE_BA`. Own loop; no `encode_leaf`.
**Tool it may improve**: the distortion/quantisation path
(`video_eff_perc.h`, `video_tx_search.cpp`): BA-shaped error in the leaf and
TX decisions, weight 0.5/1.0/2.0.

## Mandate
Build a **new encoding pipeline from scratch**: its own frame/leaf loop, its
own partition/RD order, calling the existing coding tools. The new path must
**not call `encode_leaf`**; prove it (counter/print) in the ledger. You may also
**improve the existing coding tools** it calls (listed under Allowed files):
first-party only, decoder-exact, keep every bug fix.

## Rules
Do not run `git log`/`branch`/`remote`. You may add analysis tooling under
`tools/lane-N/`; never edit the gate/scoring tools or pins. Default-off gate
only; no new syntax beyond AV2 spec; no third-party code.

## Allowed files
new `video_pipeline_ba.h/.cpp`, `video_frame_sequence.cpp`,
`video_frame_internal.h`, `video_frame_decision.cpp`,
`video_eff_perc.h`, `video_tx_search.cpp`, `video_tx_search.h`,
`docs/agent-loop-ledger-4.md`.

## Gate
Three knob arms. Spec:
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
  --base-env '{"AV2ENC_PIPE_X":"1"}' --min-arms 3 \\
  --ceiling-segment f0-32 --ceiling-floor 90.197
```
PASS only if `PASS`: envelope beats the pipeline base arm by +0.15, >=3 arms,
f0-32 ceiling delta >= +0.15. Append one ledger row per iteration.

