# Lane 1 — PIPE-EXACT: a new exhaustive-RD pipeline

Program (multi-iteration): build a **new decision pipeline** from scratch under
`AV2ENC_PIPE_EXACT`, default off. Do not extend eff-perc's branch stack: write a
new leaf decision path that enumerates skip / inter-residual / intra for every
leaf, prices each with the exact syntax probe (`rate_probe_`/`pipe_probe_`),
uses the perceptual D, and picks min J. Reuse the decoder-exact primitives, the
CDF carry and the per-block interp; keep every bug fix.

Each iteration: advance one stage, build, run the gate, and record it. This is
a program, not a single patch.

## Read only
- this file; last 15 lines of `docs/agent-loop-ledger-1.md`;
- `docs/pipe-perc.md`, `docs/eff-perc.md`.

Do not run `git log`/`branch`/`remote`. Do not edit `tools/`.

## Allowed files
new `src/pipeline/video/video_pipe_exact.h/.cpp`,
`video_frame_decision.cpp`, `video_frame_internal.h`,
`docs/agent-loop-ledger-1.md`.

## Gate
At least three knob arms of the new pipeline (e.g. lambda 0.15/0.20/0.30) on
top of the eff-perc base env. Spec:
```json
{"ref":"testdata/nasa_480x240_30fps_60f.y4m","bin":"build-release/av2enc",
 "frames":60,
 "base_args":["--video","--frames","60","--cpu-threads","8","--nightly",
  "--video-drift-refresh-threshold","1.5","--video-drift-refresh-budget","3250"],
 "qp":119,"target_kbps":100.0,
 "arms":[{"name":"base","env":{}},
  {"name":"l15","env":{"AV2ENC_PIPE_EXACT":"1","AV2ENC_PIPE_EXACT_LAMBDA":"0.15"}},
  {"name":"l20","env":{"AV2ENC_PIPE_EXACT":"1","AV2ENC_PIPE_EXACT_LAMBDA":"0.20"}},
  {"name":"l30","env":{"AV2ENC_PIPE_EXACT":"1","AV2ENC_PIPE_EXACT_LAMBDA":"0.30"}}]}
```
Run:
```
python3 tools/envelope.py --spec build/agent-loop/spec.json --worktree "$PWD" \
  --base-env '{"AV2ENC_EFF_PERC":"1"}' --best-known 85.099 --min-arms 3 \
  --ceiling-segment f0-32 --ceiling-floor 90.197
```
PASS only if `PASS`: envelope ≥ 85.249, ≥3 arms, ceiling delta vs base ≥ +0.15.
On FAIL leave the pipeline code uncommitted and continue the program next
iteration; the runner reverts only if the iteration regresses the base.
