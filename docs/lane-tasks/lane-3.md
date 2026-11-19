# Lane 3 — PIPE-TRELLIS + CDEF strength tool

Two-part program, both required:
1. **Pipeline**: a per-frame dynamic program over legal partition/mode leaves
   under `AV2ENC_PIPE_TRELLIS` (exact syntax rate + perceptual D), not greedy.
2. **Tool**: a per-frame CDEF strength search (`AV2ENC_CDEF_SEARCH`) instead
   of the fixed strength set; sweep the search range.

Each iteration advances one part. Keep the fixes.

## Read only
this file; last 15 lines of `docs/agent-loop-ledger-3.md`; `docs/video-partition-rd.md`,
`docs/video-nightly.md` CDEF section.

Do not run `git log`/`branch`/`remote`. You may add analysis tooling under `tools/lane-N/`; never edit the gate/scoring tools or pins.

## Allowed files
new `video_pipe_trellis.h/.cpp`, `video_frame_decision.cpp`,
`video_frame_internal.h`, `video_frame_filters.cpp`, `video_headers.cpp`,
`post/loop_filters.*`, `docs/agent-loop-ledger-3.md`.

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
  --base-env '{"AV2ENC_PIPE_TRELLIS":"1"}' --min-arms 3 \\
  --ceiling-segment f0-32 --ceiling-floor 90.197
```
PASS only if `PASS`: the envelope beats the **pipeline's own base arm**
(tool off) by ≥ +0.15 K, with ≥3 arms, and the f0-32 ceiling delta vs that
base is ≥ +0.15. 85.099 is the end target, not this gate.
Continue the program after a failure; append one ledger row per iteration.

