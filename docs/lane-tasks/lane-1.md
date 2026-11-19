# Lane 1 — frequency-aware inter TX size/type enumeration

Target: raise the 100 kbps K envelope. The bar is the possible K over the
tool's knobs: **envelope ≥ 85.249** (best known 85.099 + 0.15).

## Read only
- this file; the last 15 lines of `docs/agent-loop-ledger-1.md`;
- `docs/luma-audit.md` section on the 1-4 px band (grep `1-4 px`).

Do not run `git log`/`branch`/`remote`. Do not edit `tools/`.

## Hypothesis
The inter TX menu is chosen globally. Allow 4x4/8x8 TX where the leaf's 2-4 px
energy is high, gated by `AV2ENC_TX_DETAIL_SEARCH`, and **sweep the detail
threshold** (`AV2ENC_TX_DETAIL_MIN`). One mechanism, three arms.

## Allowed files
`src/pipeline/video/video_tx_search.cpp`, `video_tx_search.h`,
`video_frame_decision.cpp`, `video_frame_internal.h`,
`docs/agent-loop-ledger-1.md`.

## Steps
1. Edit only the allowed files. Default off, env-gated, no preset/pin change,
   no new syntax, no third-party code.
2. Build: `cmake --build build-release -j8`.
3. Write `build/agent-loop/spec.json` with a **knob sweep**:
```json
{"ref":"testdata/nasa_480x240_30fps_60f.y4m","bin":"build-release/av2enc",
 "frames":60,
 "base_args":["--video","--frames","60","--cpu-threads","8","--nightly",
  "--video-drift-refresh-threshold","1.5","--video-drift-refresh-budget","3250"],
 "qp":119,"target_kbps":100.0,
 "arms":[{"name":"base","env":{}},
   {"name":"t20","env":{"AV2ENC_TX_DETAIL_SEARCH":"1","AV2ENC_TX_DETAIL_MIN":"0.20"}},
   {"name":"t35","env":{"AV2ENC_TX_DETAIL_SEARCH":"1","AV2ENC_TX_DETAIL_MIN":"0.35"}},
   {"name":"t50","env":{"AV2ENC_TX_DETAIL_SEARCH":"1","AV2ENC_TX_DETAIL_MIN":"0.50"}}]}
```
4. Run from the worktree root:
```
python3 tools/envelope.py --spec build/agent-loop/spec.json --worktree "$PWD" \
  --base-env '{"AV2ENC_EFF_PERC":"1"}' --best-known 85.099 --min-arms 3 \
  --ceiling-segment f44-59 --ceiling-floor 88.995
```
5. PASS only if `PASS`: envelope ≥ 85.249, ≥3 knob arms, and the
   ceiling[f44-59] delta vs the base arm is at least +0.15 K.
6. On FAIL: `git checkout -- src && git clean -fd src`. On PASS: leave `src/`
   uncommitted. Never commit source.
7. Append one ledger row: hypothesis, envelope, best arm, PASS/FAIL.
8. Stop. No second hypothesis.

## Closed
Key/intra, HF RDOQ/trellis, dead-zone quantisers, multi-ref, TIP, BAWP,
frame-global interp, head/tail QP allocation, skip variants.
