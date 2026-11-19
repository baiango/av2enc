# Lane 3 — tool: CDEF strength search

The CDEF strength set is fixed; the encoder never searches it. Add a per-frame
strength search (the tool already exists as a plan) so the tail's error is
filtered by the best set, default off, gated `AV2ENC_CDEF_SEARCH`, swept over
the search range.

## Read only
this file; last 15 lines of `docs/agent-loop-ledger-3.md`; `docs/video-nightly.md`
CDEF section.

Do not run `git log`/`branch`/`remote`. Analysis tooling under `tools/lane-3/`;
never edit the gate/scoring tools or pins.

## Allowed files
`video_frame_filters.cpp`, `video_headers.cpp`, `post/loop_filters.*`,
`docs/agent-loop-ledger-3.md`.

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

