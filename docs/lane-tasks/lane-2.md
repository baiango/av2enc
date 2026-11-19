# Lane 2 — temporal-coherence weight in the eff-perc distortion

Target: raise the 100 kbps K envelope. Bar: **envelope ≥ 85.249** (85.099+0.15).

## Read only
- this file; last 15 lines of `docs/agent-loop-ledger-2.md`;
- `docs/eff-perc.md` (the D term and its knobs).

Do not run `git log`/`branch`/`remote`. Do not edit `tools/`.

## Hypothesis
eff-perc's D is a spatial CSF. Its skip decisions still treat a blurred
reference as free. Add a default-off temporal-coherence term to D: weight a
leaf's error by the primary reference's measured deficit, gated
`AV2ENC_EFF_PERC_TEMPORAL`, and sweep the weight. One mechanism, three arms.

## Allowed files
`src/pipeline/video/video_eff_perc.h`,
`src/pipeline/video/video_frame_decision.cpp`,
`src/pipeline/video/video_perceptual.h`,
`docs/agent-loop-ledger-2.md`.

## Steps
1. Default off, env-gated, no preset/pin change, no new syntax.
2. Build.
3. Spec:
```json
{"ref":"testdata/nasa_480x240_30fps_60f.y4m","bin":"build-release/av2enc",
 "frames":60,
 "base_args":["--video","--frames","60","--cpu-threads","8","--nightly",
  "--video-drift-refresh-threshold","1.5","--video-drift-refresh-budget","3250"],
 "qp":119,"target_kbps":100.0,
 "arms":[{"name":"base","env":{}},
   {"name":"tw02","env":{"AV2ENC_EFF_PERC_TEMPORAL":"0.2"}},
   {"name":"tw05","env":{"AV2ENC_EFF_PERC_TEMPORAL":"0.5"}},
   {"name":"tw10","env":{"AV2ENC_EFF_PERC_TEMPORAL":"1.0"}}]}
```
4. Run:
```
python3 tools/envelope.py --spec build/agent-loop/spec.json --worktree "$PWD" \
  --base-env '{"AV2ENC_EFF_PERC":"1"}' --best-known 85.099 --min-arms 3 \
  --ceiling-segment f0-32 --ceiling-floor 90.197
```
5. PASS only if `PASS`: envelope ≥ 85.249, ≥3 knob arms, and the
   ceiling[f0-32] delta vs the base arm is at least +0.15 K.
6. On FAIL: `git checkout -- src && git clean -fd src`. On PASS: leave `src/`.
7. One ledger row; stop.

## Closed
Allocation, key/intra, RDOQ/trellis, multi-ref, TIP, BAWP, filters, frame-global
interp, skip variants.
