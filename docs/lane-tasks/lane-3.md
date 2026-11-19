# Lane 3 — new pipeline + loop-filter tool

**New pipeline (from scratch)**: per-frame trellis/DP over partition and mode
`AV2ENC_PIPE_TRELLIS`. Own loop; no `encode_leaf`.
**Tool it may improve**: loop filters (`post/loop_filters.*`,
`video_frame_filters.cpp`): per-frame CDEF strength search and deblock delta.
Knob: `AV2ENC_CDEF_SEARCH` range / `AV2ENC_DEBLOCK_DELTA`.

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
new `video_pipeline_trellis.h/.cpp`, `video_frame_sequence.cpp`,
`video_frame_internal.h`, `video_frame_decision.cpp`,
`video_frame_filters.cpp`, `video_headers.cpp`, `post/loop_filters.*`,
`docs/agent-loop-ledger-3.md`.

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

