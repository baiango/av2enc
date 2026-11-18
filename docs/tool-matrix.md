# AV2 tool composition matrix

Status: **read-only audit of `0bf1d001` on `exp/luma-matrix`.** No encoder code
was changed. Every rejection quoted below was re-run against
`build-release/av2enc` (the `release` preset, CPU-only) on
`tests/data/basic_64x64_1f.y4m` or
`testdata/nasa_480x240_30fps_60f.y4m`, with `--video --frames 2` unless noted.
The 64x64 fixture is used only for usage validation; no quality claim is made
from it.

The question is not "does tool X exist". It is "which tools can be combined,
on which carrier, and which combinations have actually been selected and
measured". A tool that is implemented and decoder-gated but rejected by every
preset is an inventory item, not a composition primitive.

## 0. Columns and presets

| Column | Meaning |
|---|---|
| `D/Q/N/E` | Enabled in `D` = bare `--video`; `Q` = `--video-rd-preset quality`; `N` = `--nightly`; `E` = eff-perc (`--nightly` + `AV2ENC_EFF_PERC=1` + `--video-drift-refresh-threshold 1.5 --video-drift-refresh-budget 3250`, the recipe in `docs/eff-perc.md` and `docs/oracle-search.md`). `off` = not enabled; a number = default value; `-` = not applicable because the owning tool is off. |
| family | prediction, transform, entropy, filter, allocation, structure |
| carrier required | the syntactic/structural surface the tool needs |
| state | `gated` = has a decoder/recon gate; `gated, no RD` = correctness only; `rejected` = named decoder rejection; `untested` = accepted but never measured on K |
| legal carriers | where the validation actually accepts it (see legend below) |
| untapped | composition that is legal but unmeasured, or measured dead |

Carrier legend: **FP** = flat key+P, one tile, 8-bit 4:2:0, `--rc cqp`;
**B** = closed IPBB GOP; **TL** = one hierarchical-P GOP (`--video-tlayers`);
**TIP** = key + future anchor P + TIP output frames; **T** = `--tiles > 1x1`;
**SVC** = `--svc-layers`; **W** = frame-type wrappers (`--video-long-term/--video-olk/--video-leading/--video-sef/--video-switch/--video-ras`);
**SC** = screen-content keyframe (palette/IBC/morph); **LL** = lossless
writer; **FMT** = 10-bit or non-4:2:0 format core; **GPU** = `--video-gpu`.

Baseline measured points (`docs/video-nightly.md`, `docs/luma-audit.md`,
`docs/oracle-search.md`):

| preset | QP | bytes | kbps | JOD | K @100 |
|---|---:|---:|---:|---:|---:|
| default `--video` | 189 | 25,009 | 100.036 | 4.8021 | — |
| `quality` | 190 | 25,141 | 100.564 | 6.3021 | — |
| `--nightly` | 106 | 25,008 | 100.032 | 9.7334 | **82.155** |
| tuned drift | 110 | 25,146 | 100.584 | 9.7605 | **83.795** |
| eff-perc | 119 | 25,230 | 100.920 | 9.7713 | **85.10** |

`--nightly` is a CPU-only aggregate. It rejects `--still`, `--still-gpu` and
`--video-gpu` by name. `--video-rd-preset quality` is a subset: inter residual,
deblock, CCSO, TX search, full intra.

## 1. Structure and reference composition

| flag | family | D/Q/N/E | carrier required | named conflicts / rejections | state | measured result | legal carriers | untapped |
|---|---|---|---|---|---|---|---|---|
| `--video` | structure | on/on/on/on | any video writer | rejects `--still`, `--still-gpu`, `--lossless` | core, gated everywhere | nightly 82.155; drift 83.795; eff 85.10 | FP, B, TL, TIP, T, SVC, W, LL, FMT | — |
| `--video-rd-preset quality` | structure | off/on/on/on | FP | GPU; still; unknown name | gated (`gate_video_rd_preset.sh`) | Q 25,141 B / 6.3021 JOD at QP190 (`video-rd-preset.md`) | FP | `--nightly` wraps it |
| `--p-frames N` | structure | all/all/all/all | FP | still modes; `--frames 1` | core | part of every P measurement | FP, B, TL, TIP | — |
| `--video-refs N` | structure | 1/1/1/1 | FP or B | `--nightly` sets `--video-ref-signal-compact` (needs refs 1); B owns its list | gated (`video_cli_gate.sh`) | refs2/3 lose 0.67/0.86 K vs nightly; −2.35/−2.54 vs blocker control (`video-multiref-p.md`, `reference-structure.md`) | FP (refs 2–7), B (refs 1) | refs 2–4 already measured dead; no untapped count |
| `--video-num-ref-frames N` | structure | auto/auto/auto/auto | FP | ref-compact sets NumRefFrames 1; B/TL/W own the field | gated | no RD claim | FP | — |
| `--video-implicit-ref-map` | structure | off/off/off/off | FP, B, TL | ref-compact; compound; tiles; W; SVC; FMT | gated (`gate_implicit_ref_map.sh`) | derived list changes decisions, stream not smaller (`negative-results.md`) | FP, B, TL | derived-map + refs>1 untested |
| `--video-ref-signal-compact` | entropy | off/off/on/on | FP, refs 1 | refs>1; short-refresh; mfh/timing/multi-seq; non-8bit420 | gated (`gate_video_ref_signalling.sh`) | −0.07 K marginal (`nightly-ablation.md`); −61 B at 60 frames | FP | cheapest nightly off-switch for header work |
| `--video-short-refresh` | structure | off/off/off/off | FP | reference-management tools reject the CDF chain; ref-compact owns refresh flags | gated (`video_short_refresh_gate.sh`) | byte-identical no-op at NumRefFrames 1 (`reference-structure.md`) | FP | dead at refs 1 |
| `--video-refmvbank` | structure | off/off/off/off | FP, refs ≥2 | reference-management CDF rejection; tiles; implicit map; mfh/timing | gated (`video_mv_traj_gate.sh`) | exact but no RD measured | FP | bank + refs2 + cdf-chain-off untested |
| `--video-ref-mvs` | structure | off/off/off/off | FP refs≥2, B | edge-inter; tiles; mv precision; 8-bit420 | gated (`video_ref_mvs_gate.sh`) | +0.09 K over its control at one ref; −2.21 K with refs2; blocker costs 4.71 K (`reference-structure.md`) | FP, B | refs2+ref-mvs measured dead; `--video-ref-mvs-stress` is a gate-only candidate-forcing mode |
| `--video-bru` | structure | off/off/off/off | none | pinned dav2d does not parse `enable_bru/use_bru` | rejected by name (`negative-results.md`, `video_bru_reject_gate.sh`) | decoder rejection | none | closed |
| `--video-mv-traj` | structure | off/off/off/off | FP refs≥2 | needs `--video-ref-mvs`; edge-inter; tiles | gated (`video_mv_traj_gate.sh`) | no RD claim | FP | with ref-mvs, untested on K |
| `--video-same-ref-compound` | prediction | off/off/off/off | B or refs2 compound | reference-management CDF rejection; compound; refs≥2 | gated (`video_same_ref_compound_gate.sh`) | −7.2 to −11.1 K (`reference-structure.md`) | B, FP refs2 | measured dead |
| `--video-drl-reorder` | structure | off/off/off/off | FP | B; SVC; gme | gated (`gate_video_drl_reorder.sh`) | decoder suppresses the sort on low-delay P; bit-exact but inert (`negative-results.md`) | FP | ALWAYS is the only exercising form |
| `--video-drl-reorder-always` | structure | off/off/off/off | FP | same as above | gated | exercises §7.12.2.19; no RD claim | FP | untested on K |
| `--b-frames N` | structure | off/off/off/off | B closed GOP | nightly edge path, BAWP, drift, partition probe, key offset, RDOQ, ref-compact, tile-group; `--video-refs` must stay 1 | gated (`gate_video_schedule_rc.sh`) | best B arm K 36.85 vs 82.15 (`negative-results.md`, `bframes.md`) | B | measured dead at 240p |
| `--video-gops N` | structure | 1/1/1/1 | B | only with `--b-frames` | gated | B family negative | B | measured dead |
| `--video-tlayers N` | structure | off/off/off/off | TL, exactly one 4/8-frame GOP | B; W; edge path; partition probe; RDOQ; ref-compact; tile-group | gated (`gate_video_schedule_rc.sh`) | 4/8-frame streams exact; not a 60-frame RD lever (`negative-results.md`) | TL | TIP-like future state untested |
| `--video-tlayer-drop L` | structure | - | TL | needs tlayers; max L by plan | gated | subset stream exact; no RD | TL | — |
| `--video-tip` | structure/prediction | off/off/off/off | TIP, 3–8 frames | edge path; partition probe; RDOQ; loop filters; TX search; ref-compact; tile-group | gated (`video_tip_gate.sh`) | no composite K; decoder gate only (`tip-decoder-gate.md`) | TIP | **the one untested prediction class; rank 1** |
| `--video-tip-no-hole-fill` | structure | off/off/off/off | TIP control | needs `--video-tip` | gated | pre-fix A/B stream | TIP | — |
| `--video-tip-block-ref` | prediction | off/off/off/off | TIP + ref-mvs + force-single | needs `--video-force-single` | gated | §5.20.7.10 block symbols exact; no RD | TIP | untested |
| `--video-long-term` | structure | off/off/off/off | W | deblock/filters/TX/RDOQ/probe/edge; 16-frame OrderHint cap | gated | header-only no-op on 15-frame prefix (`reference-structure.md`) | W | measured inert |
| `--video-olk` | structure | off/off/off/off | W, exactly 4 frames | same as long-term; pinned dav2d ignores spec OLK list | gated | no RD | W | measured no-op |
| `--video-leading N` | structure | off/off/off/off | W + open GOP | same blockers; N+2 frames | gated | no RD | W | untested |
| `--video-sef` | structure | off/off/off/off | W, 2 frames | same blockers | gated | no RD | W | duplicate-frame result negative |
| `--video-switch` | structure | off/off/off/off | W, exactly 2 frames | same blockers | gated | no RD | W | untested |
| `--video-ras` | structure | off/off/off/off | W, exactly 3 frames | same blockers | gated | normative RAS refresh re-mapped for pinned decoder | W | untested |
| `--video-mfh` | structure | off/off/off/off | plain video header | ref-compact; W | gated (`gate_mfh.sh`) | header-only | FP | untested with nightly |
| `--video-timing` | structure | off/off/off/off | plain video header | ref-compact; W | gated (`gate_timing.sh`) | header-only | FP | — |
| `--video-tier high` | structure | off/off/off/off | plain, level 4.0+ | no RD effect | gated | header bit only | FP | — |
| `--video-multi-sequence` | structure | off/off/off/off | plain video header | ref-compact | gated | header-only | FP | untested |
| `--video-lcr-full` | structure | off/off/off/off | plain video | no LCR/OPS by default | gated (`gate_video_headers.sh`) | avmdec parses; dav2d ignores | FP | untested |
| `--video-decoder-model` | structure | off/off/off/off | plain video | ref-compact; pinned dav2d rejects nested block | gated (avmdec) | avmdec exact; dav2d fails (`negative-results.md`) | FP (avmdec) | — |
| `--video-brt` | structure | off/off/off/off | plain video | no known RD effect | gated | header-only | FP | — |
| `--tiles` | structure | 1x1/1x1/1x1/1x1 | T | deblock, Wiener/chroma-LR, edge path, drift, partition probe, RDOQ, segmentation, QM; SVC | gated (17 cases, `tiles.md`) | entropy parallelism, not RD: ≈5% bytes on one 720p frame | T with CDEF/CCSO/GDF | GPU throughput only |
| `--svc-layers` | structure | off/off/off/off | SVC | key offset, drift, TX search, partition probe, RDOQ, refs/tlayers, tiles, W | gated (`gate_cli_svc.sh`, avmdec) | no RD claim; dav2d under-decodes multi-xlayer inter | SVC | embedded quality ladder untested on K |
| `--svc-lcr-id` | structure | 1/1/1/1 | SVC | same | gated | header-only | SVC | — |
| `--svc-temporal N` | structure | 1/1/1/1 | SVC | not with embedded >1 | gated | no RD | SVC | — |
| `--svc-embedded N` | structure | 1/1/1/1 | SVC | not with temporal >1; profile caps N | gated (avmdec) | no RD | SVC | untested |
| `--oppoint/--alllayers/--spatial-layers` | structure | off | SVC | one selector at a time | gated | output selection only | SVC | — |
| `--video-lossless` | structure | off/off/off/off | LL | nightly preset's inter residual/intra-full cannot be switched off; Q=0 | gated (`video_lossless_gate.sh`) | lossless rate, no K comparison | LL | not an RD lever |
| `--video-lossless-b` | structure | off/off/off/off | LL + tlayers 4/8 | lossless only | gated | exact B schedule | LL | — |
| `--video-lossless-intra-period N` | allocation | 0 | LL | lossless only | gated | exact; no K | LL | — |

## 2. Prediction

| flag | family | D/Q/N/E | carrier required | named conflicts / rejections | state | measured result | legal carriers | untapped |
|---|---|---|---|---|---|---|---|---|
| `--video-inter-residual` | prediction | off/on/on/on | FP, B, TL, T | lossless; GME; BAWP without carrier | part of `quality`; gated | quality stage table (`video-rd-preset.md`); enables RDOQ | FP, B, TL, T | — |
| `--video-edge-inter` | prediction | off/off/on/on | FP, bottom rows | B/TL/TIP/W; tiles; ref-mvs; GPU | gated (`gate_video_edge_inter.sh`) | +2.39 JOD over the old quality preset; edge pair −4.10 K marginal (`video-rd-preset.md`, `nightly-ablation.md`) | FP | in aggregate |
| `--video-edge-mask` | allocation | off/off/on/on | FP + edge-inter | needs edge-inter; tiles; GPU | gated (`gate_video_edge_mask.sh`) | +0.07 JOD; −1.00 K marginal (`video-rd-preset.md`) | FP | in aggregate |
| `--video-intra-full` | prediction | off/on/on/on | FP | SDP; SSE-only mode decision | part of `quality` | +4.9% rate / −0.016 dB at QP90 default (`negative-results.md`); no isolated nightly K | FP, B, TL | mode RD already negative; no |
| `--video-cfl` | prediction | off/off/off/off | FP | IBP; SDP; no CDF conflict | gated | **−0.497 K** with nightly, QP104/106/108 (`tool-improvement-queue.md` attempt 4) | FP, B, TL | measured negative |
| `--video-mhccp` | prediction | off/off/off/off | FP | `--avg-cdf/--adaptive-cdf`; W | gated | no K; implies CfL | FP with `--video-cdf-chain-off` | CDF-reset composition untested |
| `--video-mrl` | prediction | off/off/off/off | FP | GPU; tiles; SVC; IBP; needs intra-full | gated (`video_mrl_ist_gate.sh`) | RD selects it on 64/189 key leaves without a measured win (`negative-results.md`) | FP, B, TL | measured neutral/negative |
| `--video-ibp` | prediction | off/off/off/off | FP | CfL/MHCCP; MRL; SC; inter-intra; B/TL/TIP/W; GPU/tiles/SVC | gated (`video_ibp_gate.sh`) | **−0.820 K** with nightly (`tool-improvement-queue.md` attempt 6) | FP | measured negative |
| `--video-dip` | prediction | off/off/off/off | FP | SDP; lossless; B/TL/TIP/SVC; GPU/tiles | gated (`video_dip_gate.sh`) | **−0.391 K** with nightly (`tool-improvement-queue.md` attempt 8) | FP | measured negative |
| `--video-palette` | prediction | off/off/off/off | SC | CDF chain; TX search; tiles; B/TL/TIP/W; GPU | gated (`gate_video_scc.sh`) | no K | SC | screen-content only; nightly needs ~11.5 K of ablations |
| `--video-ibc` | prediction | off/off/off/off | SC | CDF chain; segmentation/per-seg; DRL_ALWAYS; tiles; B; GPU | gated (`gate_video_scc.sh`) | no K; inert at 64x64 (`negative-results.md`) | SC | same; needs ≥2 SB rows |
| `--video-scc-force` | prediction | off/off/off/off | SC | needs palette/IBC/morph; GPU | gated | forced §5.4.7 form | SC | untested |
| `--video-force-integer-mv` | prediction | off/off/off/off | SC | needs scc-force; precision tools; GME | gated | no K | SC | untested |
| `--video-morph` | prediction | off/off/off/off | SC | loop filters; CDF chain; tiles; B/TL/TIP/W; DRL_ALWAYS; seg/UV | gated (`gate_video_morph_interintra.sh`) | no K; keyframe-only by §5.20.5.4 | SC | screen-content only |
| `--video-compound` | prediction | off/off/off/off | B or refs2 P | BAWP; ref-compact; drift; CDF carry; W; DRL | gated (`gate_video_pred_surface.sh`) | −0.17 K vs its own control; context 3.65 K down (`reference-structure.md`) | B, FP refs2 | measured dead family |
| `--video-compound-wedge` | prediction | off/off/off/off | compound | needs compound; `implicit_ref_map` | gated | ≈−3.0 K (`reference-structure.md`) | B, FP refs2 | measured dead |
| `--video-compound-diffwtd` | prediction | off/off/off/off | compound | needs compound | gated | −0.2 K (`reference-structure.md`) | B, FP refs2 | measured dead |
| `--video-compound-near-near` | prediction | off/off/off/off | B + compound | needs B; CWP/IMP/OPFL/refine require it | gated (`gate_video_near_near.sh`) | no K | B | carrier for untested overlays |
| `--video-cwp-weighted` | prediction | off/off/off/off | B + near-near | needs carrier; OPFL/refine | gated | no K | B | untested |
| `--video-imp-mask` | prediction | off/off/off/off | B + near-near | needs carrier; OPFL/refine | gated | no K | B | untested |
| `--video-opfl` | prediction | off/off/off/off | B + near-near | needs carrier; CWP/IMP | gated (`gate_video_opfl_refine.sh`) | no K; always-refine behaviour fixed | B | untested |
| `--video-refine-mv` | prediction | off/off/off/off | B + near-near | needs carrier; CWP/IMP | gated | no K; per-sub-block search fixed | B | untested |
| `--video-compound-interintra` | prediction | off/off/off/off | single-ref FP | compound; refs≠1; BAWP; morph; B/TL/TIP/W | gated (`gate_video_morph_interintra.sh`) | no K | FP refs1 | untested |
| `--video-df-sub-pu` | prediction/filter | off/off/off/off | P/B single tile | TIP; BRU; LL; TL; W; 8bit420 | gated (`gate_video_pred_surface.sh`) | accepted with nightly; no K | FP, B | untested with nightly |
| `--video-gme` / `--video-warp` | prediction | off/off/off/off | skip-only FP | inter-residual; B/TL/W; compound; DRL; tiles | gated (`video_ref_mvs_gate.sh`) | −0.75 K at 720p after removing blockers (`gme-720p.md`) | FP skip-only | measured dead |
| `--video-bawp` | prediction | off/off/on/on | FP | B/TL/W; compound; GME; residual needs carrier; tiles | gated | −0.36 K marginal (`nightly-ablation.md`) | FP | in aggregate |
| `--video-bawp-carrier` | prediction | off/off/on/on | FP | same; CPU only | gated | +0.04 JOD over residual path (`video-nightly.md`) | FP | in aggregate |
| `--video-adaptive-mvd` | prediction | off/off/off/off | FP | BAWP; compound; GME; OPFL/refine; §5.20.7.6 order | gated (`video_adaptive_mvd_gate.sh`) | neutral on quality preset (`video-rd-preset.md`); with the BAWP-off/quarter-pel rescue it is **−2.486 K** (`tool-improvement-queue.md` attempt 9) | FP with `--video-bawp-off` | measured negative via the rescue |
| `--video-mvd-sign-derive` | prediction | off/off/off/off | FP quarter+flex | BAWP; compound; GME; needs flex at quarter | gated (`video_mvd_sign_derive_gate.sh`) | no K | FP with `--video-bawp-off` | untested trio |
| `--video-flex-mvres` | prediction | off/off/off/off | FP | BAWP; compound; GME | gated (`video_flex_mvres_gate.sh`) | no K | FP with `--video-bawp-off` | untested trio |
| `--video-mv-precision` | prediction | eighth | FP, B, TL | ref-mvs; W; force-integer; GME | gated | coarser precision exact; no RD | FP, B, TL | — |
| `--video-subpel-eighth` | prediction | off/off/on/on | FP, B, TL | GPU parity | gated | +0.02–0.03 JOD (`video-rd-preset.md`) | FP, B, TL | in aggregate |
| `--video-sdp` | structure/prediction | off/off/off/off | FP | partition probe; intra-full; tiles; B/TL/TIP/SVC; segmentation; CFL/MHCCP; CCSO decode failure | gated (`gate_video_sdp_ddt.sh`) | −1.2..−1.6 K matched QP; wash on quality (`partition-tools.md`) | FP | closed |

## 3. Transform, quantisation and entropy

| flag | family | D/Q/N/E | carrier required | named conflicts / rejections | state | measured result | legal carriers | untapped |
|---|---|---|---|---|---|---|---|---|
| `--video-tx-search` | transform | off/on/on/on | FP, B, TL, T | W; SVC; TIP (encode) | gated | −0.99 K marginal (`nightly-ablation.md`) | FP, B, TL, T | in aggregate |
| `--video-key-intra-txpart` | transform | off/off/on/on | FP | needs TX search; GPU; 8-bit420 | gated (`video_key_intra_txpart_gate.sh`) | +0.065 JOD at iso-rate (`video-key-intra-txpart.md`) | FP | in aggregate |
| `--video-tx-context-rate` | entropy | off/off/off/off | FP | needs TX search; GPU | gated | +0.044 JOD on quality at 100.04; measured negative stacked on nightly (`video-tx-context-rate.md`, `video-nightly.md`) | FP | measured dead combination |
| `--video-tx-lambda S` | transform | 256/256/—/— | FP | needs TX search | gated | 128/512/1024 all plateau (`video-rd-preset.md`) | FP, B, TL | nightly scales by QP |
| `--video-key-tx-lambda S` | transform | 0/0/—/— | FP | needs TX search; 0 inherits | gated | no standalone K | FP | — |
| `--video-idtx` | transform | off/off/off/off | TX search FP/B/TL | GPU; `--video-min-block 4`; needs TX search | gated (`video_idtx_gate.sh`) | **−0.219 K** with nightly (`tool-improvement-queue.md` attempt 7) | FP, B, TL | measured negative |
| `--video-idtx-1d` | transform | off/off/off/off | none | one-axis V/H_DCT needs `dc_sign_horz_vert`; pinned dav2d rejects negative values | rejected by name (`negative-results.md`) | decoder rejection | none | closed |
| `--video-ist` | transform | off/off/off/off | FP/B/TL | GPU; tiles; SVC; QM | gated (`video_mrl_ist_gate.sh`) | **−0.008 K** with nightly (+11 B at QP106, inert) (`tool-improvement-queue.md` attempt 5) | FP, B, TL | measured inert |
| `--video-tcq` | transform | off/off/off/off | FP/B/TL | RDOQ; QM; segmentation/UV; LL; tiles; W; TIP; SVC | gated (`video_tcq_gate.sh`) | 6.223 JOD at 98.1 kbps, 60x cost for ~+0.01 JOD (`video-rd-preset.md`) | FP, B, TL with `--video-rdoq-off` | TCQ+CCTX untested |
| `--video-cctx` | transform | off/off/off/off | FP/B/TL | same family as TCQ | gated (`video_cctx_gate.sh`) | neutral (`video-rd-preset.md`) | FP, B, TL with `--video-rdoq-off` | TCQ+CCTX untested |
| `--video-ddt` | transform | off/off/off/off | FP/B/TL/T | TIP; SVC; LL; GPU | gated (`gate_video_sdp_ddt.sh`) | −1.22 K matched rate (`partition-tools.md`) | FP, B, TL, T | measured dead alone |
| `--video-rdoq` | entropy/transform | off/off/on/on | FP | inter-residual; no TCQ/QM/seg/UV/LL/T/W/TIP/SVC | gated (`gate_video_rdoq.sh`) | +0.202 JOD / +0.87 dB; −3.68 K marginal (`video-rdoq.md`) | FP | in aggregate |
| `--video-rdoq-scale S` | transform | 0.1 | FP | needs RDOQ | gated | calibration knob | FP | — |
| `--video-qm` | transform | off/off/off/off | FP, qp 90 | TX search; RDOQ; IST; UV delta; seg; TCQ; CCTX; qp≠90 | gated | no K; gated at QP90 | FP qp90 | untested |
| `--video-user-qm` | transform | off/off/off/off | FP, qp 90 | QM set; pinned dav2d rejects OBU | gated (avmdec, `video_user_qm_gate.sh`) | identity matrix; no K | FP qp90 (avmdec) | — |
| `--adaptive-cdf` | entropy | off/off/on/on | video | needs avg-cdf; per-frame QP reset for delta/seg | gated (`gate_avg_cdf.sh`) | −13.9% rate at identical decisions (`video-nightly.md`) | FP, B, TL, T | in aggregate |
| `--avg-cdf` | entropy | off/off/on/on | video, >1 tile for real average | adaptive-cdf pairing | gated | part of CDF chain | FP, B, TL, T | — |
| `--cdf-continuity` | entropy | off/off/on/on | FP | requires avg+adaptive; `--video-cdf-per-frame-qp`; delta/seg; ref tools; compound desync | gated (`gate_video_cdf_carry.sh`) | −2.77% rate at identical decisions (`video-cdf-carry.md`) | FP | in aggregate |
| `--video-cdf-per-frame-qp` | entropy | off/off/off/off | FP | needs avg+adaptive; no continuity; single tile | gated (`gate_avg_cdf.sh`) | costs 2.9% vs carry at QP106 (`video-aq.md`) | FP | required for delta-Q/AQ/seg |
| `--video-tile-flush-min` | entropy | off/off/on/on | FP lossy single tile | tiles; LL | gated (`gate_tile_flush.sh`) | −0.11 K marginal (`nightly-ablation.md`) | FP | in aggregate |
| `--video-tile-group-compact` | entropy | off/off/on/on | FP single tile | tiles; B/TL/TIP/W | gated (`gate_video_tile_group.sh`) | −0.04 K marginal; −22 B at 60 frames | FP | in aggregate |

## 4. Filters

| flag | family | D/Q/N/E | carrier required | named conflicts / rejections | state | measured result | legal carriers | untapped |
|---|---|---|---|---|---|---|---|---|
| `--video-deblock` | filter | off/on/on/on | FP, B, TL | W; tiles | gated (`gate_video_filters.sh`) | −0.53 K marginal (`nightly-ablation.md`) | FP, B, TL | in aggregate |
| `--video-cdef` | filter | off/off/on/on | FP, B, TL, T | W | gated | +0.20 JOD (`video-nightly.md`); loop-filter stack −5.85 K | FP, B, TL, T | in aggregate |
| `--video-cdef-rd` | filter/entropy | off/off/on/on | same as CDEF | same | gated | −63 B / −0.25% at identical pixels (`video-nightly.md`) | FP, B, TL, T | in aggregate |
| `--video-cdef-rd-lambda L` | filter | 0/0/0/0 | CDEF | changes padded reconstruction when >0 | gated | no iso-rate K | FP, B, TL, T | **untested priced form** |
| `--video-cdef-strength N` | filter | 1/1/1/1 | CDEF | `step`/`fast` search need 1 | gated | 4 sets costs ~6.4 B/frame (`video-nightly.md`); **+0.065 K** matched but 80.164/82.230/81.936 per QP, sign-inconsistent (`tool-improvement-queue.md` attempt 2) | FP, B, TL, T | measured inside the ripple |
| `--video-cdef-damping N` | filter | 3/3/3/3 | CDEF | 3..6 | gated | no K | FP, B, TL, T | untested |
| `--video-cdef-search M` | filter | full | CDEF | `step`/`fast` need strength 1 | gated | speed only | FP, B, TL, T | — |
| `--video-ccso` | filter | off/on/on/on | FP, B, TL, T | W | gated | +0.15 JOD with rate-aware CCSO (`video-nightly.md`) | FP, B, TL, T | in aggregate |
| `--video-ccso-rd` | filter | off/off/on/on | 8-bit420 | same as CCSO | gated | part of +0.15 JOD | FP | in aggregate |
| `--video-ccso-bands N` | filter | 4/4/6/6 | CCSO | 0..6 | gated | 64-band is the measured form (`video-ccso-overhead.md`) | FP | in aggregate |
| `--video-ccso-lambda/refit/unit-bits` | filter | -/-/22,2,2/22,2,2 | CCSO-RD | knobs; explicit values win | gated | measured at 22/2/2 (`video-nightly.md`) | FP | in aggregate |
| `--video-ccso-header-rd` | filter/entropy | off/off/on/on | 8-bit420 | GPU | gated (`gate_video_ccso_overhead.sh`) | +0.015–0.02 JOD; −0.25 K marginal (`video-ccso-overhead.md`) | FP | in aggregate |
| `--video-wiener` | filter | off/off/off/off | FP single tile | tiles; W | gated | +0.03 K plain (tie), −1.35 K tuned (`lr-quality.md`) | FP | measured dead on drift |
| `--video-pc-wiener` | filter | off/off/off/off | FP single tile | Wiener; tiles; W | gated, recon mismatch documented | −1.44 K plain, −3.10 K tuned (`lr-quality.md`, `negative-results.md`) | FP | measured dead |
| `--video-chroma-lr` | filter | off/off/off/off | FP single tile | tiles; W | gated | rate explosion, −25.6 K (`lr-quality.md`) | FP | measured dead |
| `--video-cfl-ds-filter N` | filter | 0 | format 10-bit chroma LR | 8-bit 4:2:0 core keeps index 0 | gated (`gate_video_filters.sh`) | indices 1–3 exact on 10-bit 4:2:0; no K | FMT | — |
| `--video-gdf` | filter | off/off/off/off | FP, B, TL, T | W | gated | −0.012 JOD at 3.8x CPU (`negative-results.md`) | FP, B, TL, T | measured dead |
| `--video-grain` | filter | off/off/off/off | FP, B, TL, TIP, T, FMT | W rejected by name in older docs; wrapper carriers now landed | gated (`video_grain_gate.sh`) | output-side only; no K | FP, B, TL, TIP, T, FMT | no RD lever |

## 5. Allocation and partition

| flag | family | D/Q/N/E | carrier required | named conflicts / rejections | state | measured result | legal carriers | untapped |
|---|---|---|---|---|---|---|---|---|
| `--video-partition-rd` | allocation | off/off/on/on | FP single tile 8-bit420 | B/TL/TIP/W/SVC; tiles; sb128/256; SDP; LL | gated | +0.41 JOD (8.747→9.155); −1.51 K marginal (`video-partition-rd.md`) | FP | in aggregate |
| `--video-partition-rd-lambda S` | allocation | -/-/2.75e-6 | probe | needs probe | gated | measured 2.75e-6 (`video-partition-rd.md`) | FP | — |
| `--video-partition-rd-depth N` | allocation | -/-/1 | probe | 1..3 | gated | depth 1 measured | FP | deeper untested |
| `--video-leaf-rd` | allocation | off/off/on/on | FP | GPU | gated (`gate_video_nightly.sh`) | +0.054 JOD / +0.21 dB (`video-nightly.md`) | FP, B, TL | in aggregate |
| `--video-leaf-rd-lambda S` | allocation | 125/125/125/125 | leaf RD | explicit value wins | gated | 80 gives +0.019 JOD / −0.13 dB (`video-inter-residual-rd.md`) | FP | measured inside ripple |
| `--video-key-qp-offset N` | allocation | 0/0/-27/-27 | FP cqp | B/TL/TIP/W/SVC; `--rc crf/vbr`; CDF chain needs reset/carry | gated | +0.22 JOD; −3.26 K marginal (`video-nightly.md`) | FP | in aggregate |
| `--video-drift-refresh` | allocation | off/off/on/on | FP single tile cqp 8-bit420 | B/TL/TIP/W; compound; seg/delta unless QP 0; rc | gated (`video_drift_refresh_gate.sh`) | +0.022 JOD solo; tuned recipe 83.795; −4.46 K marginal (`video-drift-refresh.md`) | FP | in aggregate |
| `--video-drift-refresh-threshold/qp/budget` | allocation | 2.5/0/1200; E 1.5/0/3250 | drift | qp>0 needs CDF reset | gated | tuned 1.5/3250 = 83.795 (`reference-structure.md`) | FP | retune done |
| `--video-qp-lambda` | allocation | off/off/on/on | FP cqp CPU | rc crf/vbr; GPU | gated (`gate_video_qp_lambda.sh`) | +0.012 JOD/+0.69 dB at 240p; +0.046 JOD at 720p (`qp-scaled-lambda.md`) | FP, B, TL | in aggregate |
| `--video-aq` (+strength/clamp/bias) | allocation | off/off/off/off | FP single tile cqp + CDF reset | segmentation surfaces; delta-q; UV delta; CDF carry | gated (`gate_video_aq.sh`) | best +0.05 K inside ripple; −0.47 K under nightly (`video-aq.md`) | FP with `--video-cdf-per-frame-qp` | measured dead combination |
| `--video-delta-q` (+res) | allocation | off/off/off/off | FP single tile cqp + CDF reset | per-segment; CDF chain; QM; frame schedules | gated (`gate_video_deltaq.sh`) | 6.161 JOD vs preset 6.302 (`video-rd-preset.md`) | FP with CDF reset | measured negative |
| `--video-segmentation` | allocation | off/off/off/off | FP single tile cqp + CDF reset | RDOQ; AQ; per-seg; SDP; IBC; QM; frame schedules | gated | no K | FP with `--video-rdoq-off --video-cdf-per-frame-qp` | untested |
| `--video-per-segment-q` (+seg-delta) | allocation | off/off/off/off | FP + CDF reset | delta-q; RDOQ; seg-globalmv; SDP | gated | ~+0.02 JOD marginal (`video-rd-preset.md`) | FP with `--video-rdoq-off --video-cdf-per-frame-qp` | measured marginal |
| `--video-uv-delta-q D` | allocation | off/off/off/off | FP | RDOQ; TCQ; CCTX; QM; segmentation; CDF chain | gated | worse at matched rate (`video-rd-preset.md`) | FP with CDF reset + rdoq-off | measured negative |
| `--video-seg-skip` | allocation | off/off/off/off | segmentation FP | compound surfaces; frozen seq table | gated | no K | FP | untested |
| `--video-seg-globalmv` | allocation | off/off/off/off | segmentation FP | per-seg Q; compound; frozen | gated | no K | FP | untested |
| `--video-seg-temporal` | allocation | off/off/off/off | segmentation FP | B frames; frozen | gated | no K | FP | untested |
| `--video-segments N` | allocation | 2 | segmentation | AQ; 1 with seg-skip/globalmv | gated | no K | FP | untested |
| `--video-seq-seg-info` (+frozen) | entropy | off/off/off/off | segmentation FP single tile | frozen with seg-skip/globalmv | gated | no K | FP | untested |
| `--video-skip-collapse` | allocation/partition | off/off/off/off | FP/B/TL single tile 8-bit420 | segmentation/delta-Q/per-seg; GPU | gated (`video_skip_collapse_gate.sh`) | wash at same QP; −1.39% on moving clip (`video-skip-collapse.md`) | FP, B, TL | measured wash |
| `--video-ext-partitions` | allocation/partition | off/off/off/off | FP | SVC; sb256; GPU | gated (`gate_video_ext_nightly.sh`) | −0.47 K on tuned recipe (`partition-tools.md`); +0.425 dB at −0.018 JOD front-end (`video-ext-nightly.md`) | FP, B, TL | measured mixed/negative |
| `--video-sb128` | allocation/partition | off/off/off/off | stripped FP | probe; all loop filters; drift; Y | gated | −0.83 K; 1.58x (`partition-tools.md`) | FP stripped | measured dead |
| `--video-sb256` | allocation/partition | off/off/off/off | stripped FP | probe; all loop filters; drift; ext partitions | gated | −0.54 K; 1.59x (`partition-tools.md`) | FP stripped | measured dead |
| `--video-min-block N` | allocation/partition | 16 | FP 8-bit420 | 4 + idtx; 4 is 8-bit420 only | gated (`gate_video_sub8.sh`) | min-block 4: −0.74 K on tuned recipe (`partition-tools.md`) | FP, B, TL | untested pair with ext |
| `--video-max-depth/--video-me-range` | allocation | 4/16 | FP | debug/partition knobs | gated | no K | FP | — |
| `--video-reduce-pb-aspect` (+pb-aspect-2) | allocation/partition | off/off/off/off | FP single tile | B/TL/TIP/W; tiles | gated (`gate_video_partitions.sh`) | no K | FP | untested |
| `--video-force-single` | structure | off/off/off/off | debug | no RD claim | gated | reconstruction identical, smaller tree | FP | debug only |
| `--rc cqp/crf/vbr` (+crf/bitrate) | allocation | cqp | FP, B, TL, SVC | `--video-qp-lambda`; W for olk/leading/ras; CDF chain without reset | gated (`gate_video_schedule_rc.sh`) | VBR ±10%; no quality claim (`negative-results.md`) | FP, B, TL, SVC | not a 240p RD lever |

## 6. Ablation switches (only suppress a `--nightly` element)

These are not coding tools. They are the legal escape hatch that makes a
composition expressible. Off by default; they leave every pin byte-identical
(`docs/reference-structure.md`).

| switch | suppresses | measured marginal if the element is removed |
|---|---|---:|
| `--video-edge-inter-off` | bottom-edge inter | requires `--video-edge-mask-off` too |
| `--video-edge-mask-off` | bottom-edge residual mask | −1.00 K |
| `--video-bawp-off` | BAWP + carrier | −0.36 K |
| `--video-drift-refresh-off` | drift refresh | −4.46 K |
| `--video-cdf-chain-off` | avg/adaptive/cdf-continuity | −4.51 K |
| `--video-key-qp-offset-off` | key/first-P offset | −3.26 K |
| `--video-deblock-off` | deblock only | −0.53 K |
| `--video-loop-filters-off` | deblock + CDEF + CCSO | −5.85 K |
| `--video-tx-search-off` | TX search (+ key intra TX part) | −0.99 K |
| `--video-rdoq-off` | RDOQ | −3.68 K |
| `--video-partition-rd-off` | measured partition probe | −1.51 K |
| `--video-tile-flush-off` | minimum tile flush | −0.11 K |
| `--video-ref-signal-off` | compact reference signalling | −0.07 K |
| `--video-tile-group-off` | sequence tile config | −0.04 K |
| `--video-ccso-header-off` | CCSO params row pricing | −0.25 K |

## 7. Implemented but not composable with plain `--nightly`

"Not composable" here means `av2enc … --nightly <flag>` exits 2 (or, for TIP,
fails at encode) before a stream is produced. The right-hand column says
whether a named ablation buys legality and what that ablation is worth.

| tool | first `--nightly` rejection | rescue | cost of rescue |
|---|---|---|---|
| `--b-frames`, `--video-gops` | `--video-edge-inter is measured on the low-delay P chain; drop the B-frame / reorder / temporal-layer mode` | edge-inter/mask, BAWP, drift, RDOQ, probe, key offset, ref-compact, tile-group, tile-flush off; TX search may stay on | ≈13.4 K of measured marginal before the B tool is measured; B arm is 45 K down (`bframes.md`) |
| `--video-tlayers`, `--video-tlayer-drop` | same edge-inter rejection | same strip as B | same structural loss; one 4/8-frame GOP only |
| `--video-tip` | same edge-inter rejection; with the strip: `--video-tip does not compose with --video-tx-search/--video-deblock` | strip all 11 flat-P decisions + `--video-loop-filters-off --video-tx-search-off` | high; untested RD |
| `--video-long-term`, `--video-olk`, `--video-leading`, `--video-sef`, `--video-switch`, `--video-ras` | `--video-deblock does not compose with --video-long-term/--video-olk, --video-leading, --video-sef, --video-switch or --video-ras`; then TX search, RDOQ, probe | loop filters, deblock, TX search, RDOQ, probe, edge, key offset, BAWP, drift, CDF, ref-compact, tile-group off | ≈13.4 K; long-term also capped at 16 frames |
| `--video-compound`, `-wedge`, `-diffwtd`, `-near-near`, `-interintra`, `-same-ref` | `--video-bawp is gated on the plain single-prediction P chain …` | ref-compact, BAWP, drift, CDF-chain off; refs 2 / B | −3.65 K control before the tool; tool ≤0 (`reference-structure.md`) |
| `--video-cwp-weighted`, `--video-imp-mask`, `--video-opfl`, `--video-refine-mv` | `--video-compound-near-near is gated on the two-reference B-frame path …` | B carrier plus all compound blockers | measured family dead |
| `--video-gme` / `--video-warp` | `--video-gme/--video-warp are gated on the plain skip-only P chain: … --video-inter-residual …` | none: nightly forces inter residual and has no `--video-inter-residual-off` | hard-blocked; −0.75 K when measured outside nightly (`gme-720p.md`) |
| `--video-sdp` | `--video-partition-rd does not compose with --video-sdp` | probe off, but then `--video-sdp does not compose with --video-intra-full`; nightly forces intra-full and has no off switch | hard-blocked; measured −1.2 K on the probe-off control |
| `--video-tcq` | `--video-rdoq replaces the scalar quantiser, it does not compose with --video-tcq` | `--video-rdoq-off` | −3.68 K; TCQ itself ~neutral at 60x |
| `--video-qm`, `--video-user-qm` | `--video-qm is gated at --qp 90 …` | `--video-rdoq-off --video-tx-search-off --qp 90` | −3.68 K RDOQ + −0.99 K TX; no measured QM gain |
| `--video-segmentation`, `--video-per-segment-q`, `--video-seg-skip/globalmv/temporal`, `--video-segments`, `--video-seq-seg-info` | `segmentation/delta-Q composes with the CDF chain only under --video-cdf-per-frame-qp` | `--video-cdf-per-frame-qp`; segmentation also needs `--video-rdoq-off` | 2.9% CDF reset + −3.68 K RDOQ; no K measured |
| `--video-aq` | same CDF-chain rejection | `--video-cdf-per-frame-qp` | measured −0.47 K vs nightly (`video-aq.md`) |
| `--video-delta-q`, `--video-uv-delta-q` | same CDF-chain rejection | `--video-cdf-per-frame-qp`; UV also needs `--video-rdoq-off` | 2.9% reset; both measured negative |
| `--video-mhccp` | `--video-mhccp does not compose with --avg-cdf/--adaptive-cdf` | `--video-cdf-chain-off` | −4.51 K; no K measured |
| `--video-palette` | `--video-palette uses per-frame §9.3 palette/IBC CDFs; it is not wired with --avg-cdf/--cdf-continuity` | `--video-cdf-chain-off --video-loop-filters-off --video-tx-search-off --video-ref-signal-off --video-tile-group-off --video-tile-flush-off` | ≈11.5 K of measured marginal; screen content only |
| `--video-ibc` | same CDF-chain rejection | same strip; segmentation also excluded | ≈11.5 K; needs ≥2 SB rows |
| `--video-morph` | `--video-morph does not compose with the loop filters` | same strip; morph additionally rejects CDF/DRL/seg | ≈11.5 K; keyframe-only |
| `--video-short-refresh` | `reference-management tools are not wired with the average-CDF/CDF-continuity chain` | `--video-cdf-chain-off --video-ref-signal-off` | −4.51 K; tool is a no-op at refs 1 |
| `--video-refmvbank` | same reference-management rejection | `--video-cdf-chain-off` (plus explicit map) | −4.51 K; no RD measured |
| `--video-mv-traj` | `--video-edge-inter cannot carry the temporal MV field` | `--video-edge-inter-off --video-edge-mask-off`; needs `--video-ref-mvs` | −4.10/−1.00 K; no RD |
| `--video-ref-mvs` | same edge-inter rejection | edge-inter/mask off | −4.71 K; tool +0.09 K |
| `--video-refs N>1` | `--video-ref-signal-compact needs --video-refs 1` | `--video-ref-signal-off` | −0.07 K; refs 2–4 measured negative |
| `--video-num-ref-frames` | `--video-ref-signal-compact sets NumRefFrames = 1` | `--video-ref-signal-off` | −0.07 K |
| `--video-implicit-ref-map` | `--video-ref-signal-compact already selects the derived §7.7 map` | `--video-ref-signal-off` | −0.07 K; derived map not smaller |
| `--video-mfh`, `--video-timing`, `--video-decoder-model`, `--video-multi-sequence` | `--video-ref-signal-compact is not wired with --video-mfh/--video-timing/--video-multi-sequence` | `--video-ref-signal-off` | −0.07 K; header-only, no RD |
| `--tiles` | `--video-edge-inter needs the single-tile writer` | edge pair, deblock, drift, probe, RDOQ, seg off; tile-flush off | loses edge+deblock+probe ≈5.6 K; CDEF/CCSO/GDF stay legal |
| `--video-sb128`, `--video-sb256` | `--video-partition-rd does not compose with the 128/256 superblock trees` | probe, loop filters, drift off | loop filters alone −5.85 K; measured −0.83/−0.54 K on stripped chain |
| `--video-adaptive-mvd`, `--video-mvd-sign-derive`, `--video-flex-mvres` | `… does not compose with --video-compound, --video-compound-interintra, --video-bawp, --video-gme/--video-warp …` | `--video-bawp-off` | −0.36 K; **cheapest real composition candidate** |
| `--video-force-integer-mv` | `--video-scc-force needs one of --video-palette/--video-ibc/--video-morph` | screen-content carrier plus strip | ≈11.5 K; screen only |
| `--rc crf/--rc vbr` | `--video-qp-lambda needs a fixed --qp stream` | none: nightly always enables `--video-qp-lambda` and has no off switch | hard-blocked; rate control not available in `--nightly` |
| `--video-lossless`, `--video-lossless-b` | `--video-rdoq is not applied by the --video-lossless writer`; then `--video-intra-full is not applied …` | none: nightly's `--video-intra-full` and `--video-inter-residual` have no off switches | hard-blocked; separate writer |
| `--svc-layers` and `--svc-*` | `--video-key-qp-offset is wired for the plain --video chain, not the SVC layers` | key-offset, probe, RDOQ, TX search, BAWP, drift, CDF, ref-compact, edge off; refs/tlayers/tiles already excluded | high; no RD measured |
| `--video-idtx-1d` | `--video-idtx-1d is rejected: the one-axis V/H_DCT … pinned dav2d f9dafbc rejects …` | none | decoder rejection |
| `--video-bru` | `--video-bru is not decoder-gated: the pinned dav2d does not parse enable_bru/use_bru` | none | decoder rejection |
| `--svc-inter-layer` | `--svc-inter-layer is not a conforming AV2 v1.0.0 feature: §6.8.2 requires lcr_dependent_xlayers_flag = 0` | none | non-conforming |

Tools that **do** compose with plain `--nightly` (no ablation): `--video-cfl`,
`--video-mrl`, `--video-ibp`, `--video-dip`, `--video-ist`, `--video-idtx`,
`--video-cctx`, `--video-ddt`, `--video-wiener`, `--video-pc-wiener`,
`--video-chroma-lr`, `--video-gdf`, `--video-grain`, `--video-ext-partitions`,
`--video-min-block 4`, `--video-skip-collapse`, `--video-df-sub-pu`,
`--video-tx-context-rate`, `--video-cdef-strength/search/damping/rd-lambda`,
`--video-ccso-*`, `--video-mv-precision`, `--video-subpel-eighth`,
`--video-drl-reorder`, `--video-drl-reorder-always`, `--video-force-single`,
`--video-max-depth`, `--video-me-range`, `--video-reduce-pb-aspect`,
`--video-tile-flush-min`, `--video-ref-signal-compact`,
`--video-tile-group-compact`, `--video-lcr-full`, `--video-tier high`,
`--video-brt`, and the debug knobs.

## 8. Exact rejection log (build-release/av2enc, 0bf1d001)

All commands are `build-release/av2enc <input> -o /tmp/x.ivf --video --frames 2
--qp 110 --nightly <flags>` on `tests/data/basic_64x64_1f.y4m` unless marked
`nasa` (60-frame clip, `--frames 4`). The first line is verbatim.

```text
--b-frames 1
av2enc: --video-edge-inter is measured on the low-delay P chain; drop the B-frame / reorder / temporal-layer mode

--video-tlayers 4
av2enc: --video-edge-inter is measured on the low-delay P chain; drop the B-frame / reorder / temporal-layer mode

--video-tip
av2enc: --video-edge-inter is measured on the low-delay P chain; drop the B-frame / reorder / temporal-layer mode
# with the flat-P decisions + loop filters + TX search stripped:
av2enc: error: --video-tip does not compose with --video-tx-search/--video-deblock

--video-compound --video-ref-signal-off
av2enc: --video-bawp is gated on the plain single-prediction P chain: it cannot be combined with --b-frames, --video-tlayers, a frame-type wrapper, --video-compound or --video-gme/--video-warp (and --video-inter-residual needs --video-bawp-carrier)

--video-compound-interintra
av2enc: --video-compound-interintra cannot be combined with --video-bawp/--video-morph: §5.20.7.14 returns before AllowedMotionModes[INTERINTRA] when use_bawp is set

--video-compound-near-near --video-compound --video-cwp-weighted
av2enc: --video-compound-near-near is gated on the two-reference B-frame path (one past and one future reference): pass --b-frames

--video-gme
av2enc: --video-gme/--video-warp are gated on the plain skip-only P chain: they cannot be combined with --b-frames, --video-tlayers, a frame-type wrapper, --video-compound, --video-inter-residual or --video-drl-reorder

--video-sdp
av2enc: --video-partition-rd does not compose with --video-sdp (the SDP tree has a separate cost model)
# with --video-partition-rd-off:
av2enc: error: --video-sdp does not compose with --video-intra-full yet

--video-tcq
av2enc: --video-rdoq replaces the scalar quantiser, it does not compose with --video-tcq (its half-step trellis owns the levels)

--video-qm
av2enc: --video-qm is gated at --qp 90: the shared device QM stage builds q2 from the pinned 528 quantizer step
# with --video-rdoq-off --video-tx-search-off --qp 90 the error becomes:
av2enc: --video-rdoq does not compose with --video-qm: the per-coefficient QM quantiser is not the scalar step the trellis distorts against

--video-segmentation
av2enc: segmentation/delta-Q composes with the CDF chain only under --video-cdf-per-frame-qp: the per-tile delta/segment CDFs are not part of the §8.2.4 image, so the image is re-derived from the §9.3 defaults per frame rather than carried

--video-aq
av2enc: segmentation/delta-Q composes with the CDF chain only under --video-cdf-per-frame-qp: the per-tile delta/segment CDFs are not part of the §8.2.4 image, so the image is re-derived from the §9.3 defaults per frame rather than carried

--video-palette
av2enc: --video-palette uses per-frame §9.3 palette/IBC CDFs; it is not wired with --avg-cdf/--cdf-continuity

--video-ibc
av2enc: --video-ibc uses per-frame §9.3 palette/IBC CDFs; it is not wired with --avg-cdf/--cdf-continuity

--video-morph
av2enc: --video-morph does not compose with the loop filters: the IBC copy/morph template must read the unfiltered reconstruction

--video-mhccp
av2enc: --video-mhccp does not compose with --avg-cdf/--adaptive-cdf: the §8.2.4 bank does not carry the §5.20.7.32 cfl_mhccp/cfl_mh_dir CDFs

--video-short-refresh
av2enc: error: reference-management tools are not wired with the average-CDF/CDF-continuity chain

--video-refmvbank
av2enc: error: reference-management tools are not wired with the average-CDF/CDF-continuity chain

--video-ref-mvs
av2enc: --video-edge-inter cannot carry the temporal MV field: the bottom SVB rows have no coded MI grid

--video-mv-traj --video-ref-mvs
av2enc: --video-edge-inter cannot carry the temporal MV field: the bottom SVB rows have no coded MI grid

--video-refs 2
av2enc: --video-ref-signal-compact needs --video-refs 1: the §7.7 derived map must hold exactly the one live slot

--video-num-ref-frames 4
av2enc: --video-ref-signal-compact sets NumRefFrames = 1; drop --video-num-ref-frames

--video-implicit-ref-map
av2enc: --video-ref-signal-compact already selects the derived §7.7 map; drop --video-implicit-ref-map

--video-mfh
av2enc: --video-ref-signal-compact is not wired with --video-mfh/--video-timing/--video-multi-sequence

--video-timing
av2enc: --video-ref-signal-compact is not wired with --video-mfh/--video-timing/--video-multi-sequence

--video-multi-sequence
av2enc: --video-ref-signal-compact is not wired with --video-mfh/--video-timing/--video-multi-sequence

--tiles 2x1
av2enc: --video-edge-inter needs the single-tile writer; the tiled path keeps its tile-local reconstruction
# with the edge pair, drift and probe off, the next error is:
av2enc: --video-deblock does not support --tiles > 1x1: the pinned decoder filters across interior tile edges, so the CPU reference cannot match it
# with deblock off too:
av2enc: --video-wiener/--video-chroma-lr do not support --tiles > 1x1: the pinned decoder's loop-restoration unit and window handling at shared tile edges does not match a frame-level filter (docs/negative-results.md)

--video-sb128
av2enc: --video-partition-rd does not compose with the 128/256 superblock trees

--video-sb256
av2enc: --video-partition-rd does not compose with the 128/256 superblock trees

--video-adaptive-mvd
av2enc: --video-adaptive-mvd does not compose with --video-compound, --video-compound-interintra, --video-bawp, --video-gme/--video-warp or --video-opfl/--video-refine-mv yet (§5.20.7.6 symbol order)

--video-flex-mvres
av2enc: --video-flex-mvres does not compose with --video-compound, --video-compound-interintra, --video-bawp, --video-gme/--video-warp or --video-opfl/--video-refine-mv yet (§5.20.7.6 symbol order)

--video-mvd-sign-derive --video-mv-precision quarter --video-flex-mvres
av2enc: --video-mvd-sign-derive does not compose with --video-compound, --video-compound-interintra, --video-bawp, --video-gme/--video-warp or --video-opfl/--video-refine-mv yet (§5.20.7.6 symbol order)

--video-scc-force --video-force-integer-mv
av2enc: --video-scc-force needs one of --video-palette/--video-ibc/--video-morph: the forced sequence form only changes how those tools are signalled

--svc-layers 0=64x64
av2enc: --video-key-qp-offset is wired for the plain --video chain, not the SVC layers

--rc crf --crf 32
av2enc: --video-qp-lambda needs a fixed --qp stream: the rate-controlled paths own base_q_idx and would re-scale every frame

--video-lossless --qp 0
av2enc: --video-rdoq is not applied by the --video-lossless writer
# with RDOQ/TX/loop filters/probe stripped:
av2enc: --video-intra-full is not applied by the --video-lossless writer

--video-idtx-1d
av2enc: --video-idtx-1d is rejected: the one-axis V/H_DCT, V/H_ADST and V/H_FLIPADST intra types need dc_sign_horz_vert (§5.20.7.27) and the class-specific scans, and pinned dav2d f9dafbc rejects every negative H_DCT/V_DCT coefficient under --strict 1 (repro: hv_neg.ivf from src/codec/intra/supplemental/, strict 0 decodes, strict 1 fails; avmdec fails too).  Use --video-idtx for the 2-D identity type

--video-bru
av2enc: --video-bru is not decoder-gated: the pinned dav2d does not parse enable_bru/use_bru (Error parsing frame header).  The BRU component stream and the reproducer are in src/pipeline/video/video_bru_reject_gate.sh

--svc-inter-layer (exact 3-frame SVC command)
av2enc: --svc-inter-layer is not a conforming AV2 v1.0.0 feature: §6.8.2 requires lcr_dependent_xlayers_flag = 0, so no cross-xlayer reference or resampling can be signalled and there is no decoder process to gate.  Repro: av2enc in.y4m -o out.ivf --video --frames 3 --svc-layers 0=640x360,1=1280x720 --svc-inter-layer.  Use --svc-embedded N for cross-mlayer (quality) refinement
```

## 9. Top 10 untapped legal combinations

These are accepted by validation (or become accepted with the named switch) and
have **no composite-K measurement**. They are not claims.

| # | combination | carrier | why untested | expected |
|---|---|---|---|---|
| 1 | `--nightly --video-tip` plus the flat-P strip (`--video-edge-inter-off --video-edge-mask-off --video-bawp-off --video-drift-refresh-off --video-cdf-chain-off --video-key-qp-offset-off --video-rdoq-off --video-partition-rd-off --video-ref-signal-off --video-tile-group-off --video-tile-flush-off --video-loop-filters-off --video-tx-search-off`) | TIP, 3–8 frames | only decoder-gated; the oracle queue's one open prediction class | highest structural headroom; expensive and not a 60-frame sequence |
| 2 | ~~`--nightly --video-cdef-strength 4`~~ | FP | **measured: +0.065 K matched, per-QP 80.164/82.230/81.936 (sign-inconsistent)** | closed (`tool-improvement-queue.md` attempt 2) |
| 3 | ~~`--nightly --video-cfl`~~ | FP | **measured: −0.497 K** | closed (`tool-improvement-queue.md` attempt 4) |
| 4 | ~~`--nightly --video-bawp-off --video-adaptive-mvd --video-flex-mvres --video-mvd-sign-derive --video-mv-precision quarter`~~ | FP | **measured: −2.486 K** (the BAWP/quarter-pel rescue dominates) | closed (`tool-improvement-queue.md` attempt 9) |
| 5 | ~~`--nightly --video-ibp`~~ | FP | **measured: −0.820 K** | closed (`tool-improvement-queue.md` attempt 6) |
| 6 | ~~`--nightly --video-ist`~~ | FP | **measured: −0.008 K (inert)** | closed (`tool-improvement-queue.md` attempt 5) |
| 7 | ~~`--nightly --video-dip`~~ | FP | **measured: −0.391 K** | closed (`tool-improvement-queue.md` attempt 8) |
| 8 | ~~`--nightly --video-idtx`~~ | FP/B/TL | **measured: −0.219 K** | closed (`tool-improvement-queue.md` attempt 7) |
| 9 | `--nightly --video-rdoq-off --video-tcq --video-cctx` | FP/B/TL | TCQ and CCTX are individually neutral; the pair composes and is untested | low; TCQ costs ~60x |
| 10 | `--nightly --video-cdf-chain-off --video-mhccp` | FP | only blocker-free form; no K | low; pays −4.51 K for the CDF chain |

Honourable mentions, legal but with a measured blocker: screen-content
palette/IBC/morph on the stripped carrier; `--video-refs 2 --video-ref-mvs
--video-refmvbank --video-mv-traj` on the edge/CDF-stripped carrier;
`--video-ext-partitions --video-min-block 4` together; `--video-grain` on the
full nightly aggregate (output-side, no K).

## 10. Top 5 composition candidates, ranked

Ranked by expected oracle headroom, not by implementation convenience. Each
names exact flags and a legal carrier. None is a cap claim.

| rank | candidate | exact flags | carrier | status | expected headroom | cost |
|---:|---|---|---|---|---|---|
| 1 | TIP future anchor | `--nightly --video-tip --video-loop-filters-off --video-tx-search-off --video-rdoq-off --video-partition-rd-off --video-bawp-off --video-drift-refresh-off --video-cdf-chain-off --video-key-qp-offset-off --video-edge-inter-off --video-edge-mask-off --video-ref-signal-off --video-tile-group-off --video-tile-flush-off` | TIP (3–8 pictures: key + future anchor P + TIP output) | untested; decoder-gated only | the only untested prediction class in the queue; intermediate gate K > 86 before any further work (`oracle-search.md` §4) | high: 8-frame sequences, strip 13.4 K of flat-P marginal, new oracle ladder |
| 2 | CDEF per-block strength | `--nightly --video-cdef-strength 4` (keep `--video-cdef-rd`, which is already on) | FP | untested on K; CDEF itself is 60-frame gated | small, ≤0.2 K; plausible BA path because BA carries the gap | low: one encode ladder, ≈1.3x, +6.4 B/frame |
| 3 | CfL chroma on the nightly tree | `--nightly --video-cfl` | FP | accepted on a 2-frame probe; 60-frame composition untested | small; chroma is 0.61 K of the SVT gap (`eff-perc.md`) | low: one ladder, ≈1.05x |
| 4 | Adaptive MVD + flex precision + sign derivation | `--nightly --video-bawp-off --video-adaptive-mvd --video-flex-mvres --video-mvd-sign-derive --video-mv-precision quarter` | FP | each is gated alone; the trio is untested | small; MV syntax only | very low: `--video-bawp-off` costs 0.36 K, one ladder |
| 5 | IBP on every intra leaf | `--nightly --video-ibp` | FP | gated alone; 60-frame nightly composition untested | small; mostly keyframe, which is not the dominant gap | low: one ladder |

Why not the measured-dead families: B/compound, GME, SDP, refs 2–4, AQ,
delta-Q, ext partitions, min-block 4, skip-collapse, Wiener/PC-Wiener/
chroma-LR/GDF, and TX context rate all have matched-rate measurements or
measurement-backed negatives; they are not candidates. See §11.

## 11. Measured dead vs untested

**Measured dead combinations** (cite the measurement, do not re-queue as
"untested"):

- B frames / closed GOPs: K 36.85 vs 82.15 (`negative-results.md`, `bframes.md`).
- Compound family: tool ≤0 vs control, context 3.65 K down; wedge −3.0 K; same-ref −7.2..−11.1 K (`reference-structure.md`).
- Multi-reference P refs 2/3/4: −0.67/−0.86 K vs nightly; −2.35..−2.75 K vs control (`video-multiref-p.md`).
- GME/warp: −0.75 K at 720p after removing four blockers (`gme-720p.md`).
- AQ: best +0.05 K inside ripple; −0.47 K under nightly (`video-aq.md`).
- Delta-Q: 6.161 JOD vs 6.302 preset (`video-rd-preset.md`).
- Segmented/per-segment Q, UV delta-Q: marginal or worse (`video-rd-preset.md`).
- SDP: −1.2..−1.6 K matched QP; −6..−7 K on CCSO-free control (`partition-tools.md`).
- sb128/sb256: −0.83/−0.54 K (`partition-tools.md`).
- Ext partitions: −0.47 K on tuned recipe (`partition-tools.md`).
- Min-block 4: −0.74 K on tuned recipe (`partition-tools.md`).
- Skip collapse: wash at same QP (`video-skip-collapse.md`).
- Wiener: −1.35 K on drift; PC-Wiener −3.10 K; chroma-LR −25.6 K (`lr-quality.md`).
- GDF: −0.012 JOD for 3.8x CPU (`negative-results.md`).
- TX context rate: +0.044 JOD on quality, negative stacked on nightly (`video-tx-context-rate.md`, `video-nightly.md`).
- RDOQ HF/trellis variants: below the wall (`oracle-search.md` §4).
- Ref-MVs with refs 2: −2.21 K; one-ref +0.09 K cannot pay the 4.71 K blocker (`reference-structure.md`).
- Short refresh: no-op at refs 1 (`reference-structure.md`).
- Long-term: inert/header-only and capped at 16 frames (`reference-structure.md`).

**Untested** (legal but no composite K): TIP, CDEF multi-strength,
CDEF priced-drop lambda, CFL on nightly, adaptive MVD/flex/sign trio, IBP,
IST, DIP, IDTX, MHCCP with CDF reset, TCQ+CCTX together, screen-content
palette/IBC/morph on the stripped carrier, ref-MV bank + trajectory on refs 2,
and the ext-partition + min-block-4 pair.

## 12. Reproduction

```sh
cd <repo>
cmake --preset release && cmake --build --preset release -j 8
B=build-release/av2enc
I=tests/data/basic_64x64_1f.y4m

# one rejection probe
$B $I -o /tmp/x.ivf --video --frames 2 --qp 110 --nightly --video-tcq

# one accepted composition (untested RD)
$B testdata/nasa_480x240_30fps_60f.y4m -o /tmp/ibp.ivf \
    --video --frames 60 --cpu-threads 8 --qp 106 --nightly --video-ibp
```

No source, build-system, or gate file was modified by this audit. The only
repository change is this document and the composition section in
`docs/oracle-search.md`.
