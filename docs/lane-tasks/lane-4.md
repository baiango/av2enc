# Lane 4 — frequency-aware TX on the tail

Target: raise the 100 kbps K envelope. Bar: **envelope ≥ 85.249**.

## Read only
- this file; last 15 lines of `docs/agent-loop-ledger-4.md`;
- `docs/luma-audit.md` (1-4 px band, tail residual share).

Do not run `git log`/`branch`/`remote`. Do not edit `tools/`.

## Hypothesis
Same mechanism as lane 1 but restricted to the tail (frames 33-59), where most
error is coded residual: allow 4x4/8x8 TX where the 2-4 px energy is high,
gated `AV2ENC_TX_DETAIL_SEARCH_TAIL`, and sweep the threshold.

## Allowed files
`src/pipeline/video/video_tx_search.cpp`, `video_tx_search.h`,
`video_frame_decision.cpp`, `video_frame_internal.h`,
`docs/agent-loop-ledger-4.md`.

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
   {"name":"t20","env":{"AV2ENC_TX_DETAIL_SEARCH_TAIL":"1","AV2ENC_TX_DETAIL_MIN":"0.20"}},
   {"name":"t35","env":{"AV2ENC_TX_DETAIL_SEARCH_TAIL":"1","AV2ENC_TX_DETAIL_MIN":"0.35"}},
   {"name":"t50","env":{"AV2ENC_TX_DETAIL_SEARCH_TAIL":"1","AV2ENC_TX_DETAIL_MIN":"0.50"}}]}
```
4. Run:
```
python3 tools/envelope.py --spec build/agent-loop/spec.json --worktree "$PWD" \
  --base-env '{"AV2ENC_EFF_PERC":"1"}' --best-known 85.099 --min-arms 3
```
5. PASS only if `PASS`: envelope ≥ 85.249 with ≥3 arms.
6. On FAIL: `git checkout -- src && git clean -fd src`. On PASS: leave `src/`.
7. One ledger row; stop.

## Closed
Deblock/CDEF/GDF/LR, allocation, key/intra, RDOQ, entropy.
