# Lane 5 — new pipeline + motion tool

**New pipeline (from scratch)**: measured two-pass driver
`AV2ENC_PIPE_2PASS`: pass 1 decides and measures, pass 2 re-decides and
re-encodes; own loop; no `encode_leaf`.
**Tool it may improve**: motion (`video_frame_me.cpp`,
`video_syntax_inter.cpp`): adaptive MVD / per-block precision, plus the
pass-2 lambda.

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
new `video_pipeline_2pass.h/.cpp`, `video_frame_sequence.cpp`,
`video_frame_internal.h`, `video_frame_decision.cpp`,
`video_frame_me.cpp`, `video_syntax_inter.cpp`,
`docs/agent-loop-ledger-5.md`.

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

