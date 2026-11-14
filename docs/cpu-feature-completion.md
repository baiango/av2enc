# CPU feature completion program

Scope: every AV2 encoder feature that the pinned `dav2d` is **not known to hard-reject**, including the
parity bugs (decodes but reconstructs differently), the tools `dav2d` already accepts on a still/component
carrier, the still/video-gated DIP carrier, the decoder-unproven IBP tool, and the composition/schedule/format limits.

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
| 16 | DIP (`--still-dip`, `--video-dip`) | DONE `integration/cpu-dip` (still gate + `video_dip_gate.sh`) |
| 17 | IBP (`--video-ibp`) | DONE `1491868` |
| 18 | Sub-8 leaves + shared chroma (`--video-min-block 4`) | DONE `1491868`..`851774c` |
| 19 | Weighted CWP (non-equal) | DONE (`--video-cwp-weighted`; `gate_video_pred_surface.sh`) |
| 20 | Implicit mask blending (`enable_imp_msk_bld`) | DONE (`--video-imp-mask`; `gate_video_pred_surface.sh`) |
| 21 | DF sub-PU | DONE (`--video-df-sub-pu`; sub-PU grid exact with `--video-opfl --video-deblock`) |
| 22 | Adaptive MVD / MVD sign derivation / flexible MV resolution | DONE `f838a9a` |
| 23 | `num_same_ref_compound > 0`, short refresh, `mv_traj` (refs 2); `refmvbank` rejected | DONE three tools: `--video-same-ref-compound`, `--video-short-refresh`, `--video-mv-traj` (64x64 + 640x360, dav2d exact). `--video-refmvbank` is wired but **rejected by name**: decoder diverges at the third superblock row (docs/negative-results.md). `mv_traj` refs>=3 also rejected |
| 24 | TIP-as-reference, warp/wedge §7.22 fields | DONE (`ec7fcf6`); `RefFrame[0] == TIP_FRAME` block symbols remain |

## Wave 4 — headers, formats, rates, schedules

| # | Feature | Status |
|---|---|---|
| 25 | High tier, `seq_header_id` bank / multi-sequence, full LCR/OPS fields | DONE (`--video-tier high`, `--video-multi-sequence`, `--video-lcr-full`; `gate_video_headers.sh`) |
| 26 | Separate UV/DC delta-Q (`separate_uv_delta_q`, `equal_ac_dc_q`) | DONE (`--video-uv-delta-q`; `gate_video_deltaq.sh`) |
| 27 | 10-bit / non-4:2:0 video in-loop filters | DONE `610e92b` (4:2:2 16-px and 8-bit 4:0:0 PC-Wiener rejected) |
| 28 | CRF/VBR with B frames, temporal layers, SVC | QUEUE |
| 29 | Multi-GOP B frames, multi-reference hierarchical P | QUEUE |
| 30 | `SEG_LVL_SKIP`/`SEG_LVL_GLOBALMV`, temporal segmentation, 16 segments | DONE (`--video-seg-skip/--video-seg-globalmv/--video-seg-temporal/--video-segments`; `gate_video_deltaq.sh`) |
| 31 | Film grain / lossless on the other schedules and formats | QUEUE |
| 32 | 8-bit non-4:2:0 still writer filter stage | DONE `610e92b` |
| 33 | 4:2:2 16-pixel leaves | BLOCKED: pinned dav2d frame-header parse error; rejection retained |
| 34 | `cfl_ds_filter_index` 1..3 | DONE `610e92b` (format video core; M2 8-bit 4:2:0 rejects) |
| 35 | Screen-content force-on, `reduce_pb_aspect_ratio` | QUEUE |

## Hard dav2d rejections (out of scope; stay documented)

BRU, cross-frame CDF carry, user QM OBU, nested decoder model, 1-D negative H/V_DCT, multi-xlayer SVC inter
streams. Cross-xlayer SVC prediction is spec-forbidden (§6.8.2); 12-bit is reserved in AV2 v1.0.0.
