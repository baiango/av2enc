# Lane 5 — per-leaf inter TX type search with exact plan rate

Target: raise the 100 kbps K envelope. Bar: **envelope ≥ 85.249**.

## Read only
- this file; last 15 lines of `docs/agent-loop-ledger-5.md`;
- `docs/negative-results.md`, the row on intra TX-type sets.

Do not run `git log`/`branch`/`remote`. Do not edit `tools/`.

## Hypothesis
The inter TX type is inherited from the TX search, not chosen per leaf against
the exact plan rate. Add a default-off per-leaf type search over a selectable
candidate set (`AV2ENC_INTER_TX_TYPE_SEARCH`, `AV2ENC_INTER_TX_TYPE_SET` 1/2/3),
priced by the existing exact plan rate. Non-trellis, no RDOQ. Sweep the set.

## Allowed files
`src/pipeline/video/video_tx_search.cpp`, `video_tx_search.h`,
`video_frame_decision.cpp`, `video_frame_internal.h`,
`docs/agent-loop-ledger-5.md`.

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
   {"name":"ty1","env":{"AV2ENC_INTER_TX_TYPE_SEARCH":"1","AV2ENC_INTER_TX_TYPE_SET":"1"}},
   {"name":"ty2","env":{"AV2ENC_INTER_TX_TYPE_SEARCH":"1","AV2ENC_INTER_TX_TYPE_SET":"2"}},
   {"name":"ty3","env":{"AV2ENC_INTER_TX_TYPE_SEARCH":"1","AV2ENC_INTER_TX_TYPE_SET":"3"}}]}
```
4. Run:
```
python3 tools/envelope.py --spec build/agent-loop/spec.json --worktree "$PWD" \
  --base-env '{"AV2ENC_EFF_PERC":"1"}' --best-known 85.099 --min-arms 3 \
  --ceiling-segment f0-32 --ceiling-floor 90.197
```
5. PASS only if `PASS`: envelope ≥ 85.249, ≥3 arms, and the
   ceiling[f0-32] delta vs the base arm is at least +0.15 K.
6. On FAIL: `git checkout -- src && git clean -fd src`. On PASS: leave `src/`.
7. One ledger row; stop.

## Closed
RDOQ/trellis, key/intra, dead-zone quantisers, allocation, filters, multi-ref.
