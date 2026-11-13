# CPU feature completion program

Scope: every AV2 encoder feature that the pinned `dav2d` is **not known to hard-reject**, including the
parity bugs (decodes but reconstructs differently), the tools `dav2d` already accepts on a still/component
carrier, the decoder-unproven tools (DIP/IBP), and the composition/schedule/format limits.

Rules: CPU-only builds, `dav2d-macos` (`f9dafbc`) `--strict 1` gates, one commit per stage, no push/merge by
agents, defaults and pins unchanged when a flag is off.

Status key: **DONE** merged to `main`; **RUN** agent working; **QUEUE** not started; **BLOCKED** documented
repro and kept rejected.

## Wave 1 — parity (decodes, wrong reconstruction)

| # | Feature | Branch | Status |
|---|---|---|---|
| 1 | `--video-opfl` | `integration/cpu-opfl-exact` | DONE `7cb9082` |
| 2 | `--video-refine-mv` | `integration/cpu-opfl-exact` | DONE `7cb9082` |
| 3 | `--video-drl-reorder` (CONSTRAINT inert on low-delay P; `--video-drl-reorder-always` exercises it) | `integration/cpu-drl-exact` | DONE `3b49e28` |
| 4 | TIP hole fill (`allow_tip_hole_fill`) | `integration/cpu-tip-holefill` | DONE `b98c8d4` |
| 5 | Chroma LR 4:2:2/4:4:4 (CfL alias parity) | `integration/cpu-chroma-lr-422` | DONE `a7ee6b4` |

## Wave 2 — tools `dav2d` already decodes (wiring gaps)

| # | Feature | Carrier today | Status |
|---|---|---|---|
| 6 | 128×128 / 256×256 superblocks (`--video-sb128/256`) | `integration/cpu-sb-ext` | DONE `b1b984d` |
| 7 | SDP (`--video-sdp`) | `integration/cpu-sdp-ddt` | DONE `babf157` |
| 8 | MRLS (`--video-mrl`) | `integration/cpu-intra-tx` | DONE `7906c99` |
| 9 | Intra IST (`--video-ist`) | `integration/cpu-intra-tx` | DONE `7906c99` (inter IST stays off) |
| 10 | TCQ (`--video-tcq`) | `integration/cpu-entropy-tcq` | DONE `8fde128` |
| 11 | CCTX (`--video-cctx`) | `integration/cpu-entropy-tcq` | DONE `8fde128` |
| 12 | Palette selection | `--video-palette` (CPU keyframe) + still palette gate | DONE video (`gate_video_scc.sh`) |
| 13 | General IBC | `--video-ibc` (CPU keyframe) + `--still-ibc` | DONE video (`gate_video_scc.sh`) |
| 14 | Inter DDT (`--video-ddt`) | `integration/cpu-sdp-ddt` | DONE `babf157` |
| 15 | Ext/uneven partitions (`--video-ext-partitions`) | `integration/cpu-sb-ext` | DONE `b1b984d` |

## Wave 3 — decoder-unproven and prediction surface

| # | Feature | Status |
|---|---|---|
| 16 | DIP (needs a passing `dav2d --strict 1` gate first) | QUEUE |
| 17 | IBP (needs a passing `dav2d --strict 1` gate first) | QUEUE |
| 18 | Sub-8 leaves + shared chroma | QUEUE |
| 19 | Weighted CWP (non-equal) | QUEUE |
| 20 | Implicit mask blending (`enable_imp_msk_bld`) | QUEUE |
| 21 | DF sub-PU | QUEUE |
| 22 | Adaptive MVD / MVD sign derivation / flexible MV resolution | QUEUE |
| 23 | `num_same_ref_compound > 0`, refmvbank, short refresh, `mv_traj` | QUEUE |
| 24 | TIP-as-reference in tile frames, warp/wedge temporal fields | QUEUE |

## Wave 4 — headers, formats, rates, schedules

| # | Feature | Status |
|---|---|---|
| 25 | High tier, `seq_header_id` bank / multi-sequence, full LCR/OPS fields | QUEUE |
| 26 | Separate UV/DC delta-Q (`separate_uv_delta_q`, `equal_ac_dc_q`) | QUEUE |
| 27 | 10-bit / non-4:2:0 video in-loop filters | QUEUE |
| 28 | CRF/VBR with B frames, temporal layers, SVC | QUEUE |
| 29 | Multi-GOP B frames, multi-reference hierarchical P | QUEUE |
| 30 | `SEG_LVL_SKIP`/`SEG_LVL_GLOBALMV`, temporal segmentation, 16 segments | QUEUE |
| 31 | Film grain / lossless on the other schedules and formats | QUEUE |
| 32 | 8-bit non-4:2:0 still writer filter stage | QUEUE |
| 33 | 4:2:2 16-pixel leaves | QUEUE |
| 34 | `cfl_ds_filter_index` 1..3 on video | QUEUE |
| 35 | Screen-content force-on, `reduce_pb_aspect_ratio` | QUEUE |

## Hard dav2d rejections (out of scope; stay documented)

BRU, cross-frame CDF carry, user QM OBU, nested decoder model, 1-D negative H/V_DCT, multi-xlayer SVC inter
streams. Cross-xlayer SVC prediction is spec-forbidden (§6.8.2); 12-bit is reserved in AV2 v1.0.0.
