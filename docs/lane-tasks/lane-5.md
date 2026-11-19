# Lane 5 — new pipeline + motion tool

**New pipeline (from scratch)**: measured two-pass driver
`AV2ENC_PIPE_2PASS`: pass 1 decides and measures, pass 2 re-decides and
re-encodes. Its own frame loop, leaf loop and partition/RD order; no
`encode_leaf`.
**Tool it may improve**: motion (`video_frame_me.cpp`,
`video_syntax_inter.cpp`): adaptive MVD / per-block precision, plus the pass-2
lambda.

## Mandate
Build a **new encoding pipeline from scratch**: its own frame loop, leaf loop,
partition/RD order and leaf body, in the new `video_pipeline_2pass.*` files. It
may call and, where listed, edit the coding tools. It may **not** call or copy
the existing loop or decision drivers (`encode_leaf`,
`InterFrameEncoder::decide`/`deblock`, `partition::decide_frame`, `enter_sb`,
code moved out of `video_frame_decision.cpp` / `video_frame_sequence.cpp`).
Prove in the ledger that the new path never enters them.

## Rules
Do not run `git log`/`branch`/`remote`. You may add analysis tooling under
`tools/lane-5/`; never edit the gate/scoring tools or pins. Default-off env gate
only; no new syntax beyond AV2 spec; no third-party code. The existing P-frame
loop and decision helpers are read-only references.

## Allowed files
new `video_pipeline_2pass.h/.cpp` (loop, decision order, leaf body),
`video_frame_me.cpp`, `video_syntax_inter.cpp` (the tool),
one dispatch seam: a <=10-line env-checked entry call in
`video_frame_decision.cpp` or `video_frame_sequence.cpp` (no decision logic
there), `CMakeLists.txt` (add the new sources only; flag it in the ledger),
`docs/agent-loop-ledger-5.md`.

## Gate
Spec. `qp` 189 is the zero-tool 100 kbps centre; set it to your own measured
bracket centre once the pipeline moves the rate.
```json
{"ref":"testdata/nasa_480x240_30fps_60f.y4m","bin":"build-release/av2enc",
 "frames":60,
 "base_args":["--video","--frames","60","--cpu-threads","8"],
 "qp":189,"target_kbps":100.0,
 "arms":[{"name":"base","env":{}},{"name":"a1","env":{}},{"name":"a2","env":{}},{"name":"a3","env":{}}]}
```
Fill a1/a2/a3 with the knob values named above. Run:
```
python3 tools/envelope.py --spec build/agent-loop/spec.json --worktree "$PWD" \\
  --base-env '{"AV2ENC_PIPE_2PASS":"1"}' --min-arms 3 \\
  --ceiling-segment f0-32 --ceiling-floor 90.197
```
PASS only if `PASS`: envelope beats the pipeline base arm by +0.15, >=3 arms,
gated ceiling delta >= +0.15. The `--ceiling-floor` value is reported, not
gated. Append one ledger row per iteration.
