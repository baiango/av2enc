# AV2 v1.0.0 §7.16–§7.20 in-loop filter coverage audit

Scope: TIP deblocking (§7.16), deblocking filter (§7.17), CDEF (§7.18),
CCSO (§7.19), and loop restoration (§7.20), spec pages 528–560 of
`spec/AV2_Spec_v1.0.0.pdf`.

Method: read the normative pseudocode in the pinned PDF (page index = spec
page − 1), then trace every filter from its syntax writer through the pixel
core to the reconstruction/reference used by the encoder. Evidence cites the
implementation and the relevant gate. No source files were changed.

## Status legend

| Status | Meaning |
|---|---|
| **WIRED** | Reachable from a real encode path (M1 CPU, M1 GPU, or a multi-frame gate), with syntax, pixel processing and output/reference use consistent. |
| **COMPONENT** | Implemented and unit-tested (often with a `dav2d` gate), but not reachable from the production `av2enc` encode path with these parameters. |
| **PARTIAL** | Implemented but limited: a subset of planes/bit depths/subsampling/modes, or a normative detail is missing. |
| **ABSENT** | Not implemented or never signalable by an encoder path. |

Production paths in this tree:

* `av2enc` M0/M1 still-picture encoder (`src/main.cpp`; `--still` selects M1).
* M1 GPU still-picture path (`--still-gpu`).
* M2 two-frame/multi-frame encoder is a gate/library path
  (`src/encoder/m2/video_frame.cpp`), not exposed by `src/main.cpp`.
* Dedicated gates under `src/encoder/m2/...`, `src/cuda/m1/...`,
  `src/encoder/filters/lr_chroma/gate/`.

## Executive summary

* **M1 CPU** is the only path that actually runs the full normative chain
  deblock → CDEF → CCSO on reconstruction. Deblock is on by default; CDEF and
  CCSO are opt-in CLI flags. All three are stream-gated with pinned `dav2d`
  decodes and recon equality (`tools/still_verify/feature_gate.sh`, 30/30 pass).
* **M1 GPU** runs deblock only. `--still-cdef` and `--still-ccso` are rejected
  (`src/cuda/m1/still_gpu.h:284-296`); the M1 restoration stage exists but is not
  called by `still_gpu.cu`.
* **M2 production** writes all filter flags off (`enable_cdef = 0`,
  `enable_restoration = 0`, `enable_ccso = 0`, `apply_deblocking_filter[] = 0`)
  and pushes the raw reconstruction as the next frame's reference. Every
  in-loop filter is therefore **not applied to the reference used by later
  frames**.
* **§7.16 TIP deblocking is entirely absent**: `enable_tip = 0` in every
  sequence header, M1 is intra-only, and there is no `tipSize` code.
* **§7.20 luma LR/PC-Wiener/GDF** are component-level only. The one exception
  is chroma LR, which has a real two-frame closed-loop `dav2d` gate where the
  filtered frame 0 is the P-frame reference; that gate passes bit-exactly.
* Main normative gaps that are not simply "not wired": delta-Q deblock
  strength is never exercised and the `DfDeltaQ[1] = DfDeltaQ[0]` fallback is
  not implemented; deblock BRU gating is absent; CCSO temporal reuse is
  rejected; LR temporal frame-filter prediction is rejected by the M1/M2
  adapters; all filters are 8-bit 4:2:0 only.

## §7.16 TIP deblocking (pages 528–529)

| Tool/rule | §/page | Status | Evidence | Gap |
|---|---|---|---|---|
| TIP deblock invocation and `tipSize = 8x8 / 16x16` selection | §7.16, p528–529 | **ABSENT** | `src/encoder/m2/video_headers.cpp:169` writes `enable_tip = 0`; `src/encoder/m1/still_headers.cpp` has no TIP fields; `src/encoder/inter/temporal/` implements TIP prediction but no §7.16 filter. | No TIP frames are ever encoded, so no TIP deblock can run. |
| TIP deblock plane/pass traversal, `baseFilterLevel` + chroma `DeltaQ`/`BaseUVAcDeltaQ`, `qThr`/`side` | §7.16, p528–529 | **ABSENT** | No `tipSize`, `is_vert_tile_edge`, or TIP-specific `filter_choice` caller exists (`grep tipSize` only in temporal tests/comments). | Whole TIP strength/edge path missing. |
| TIP deblock `filter_maximum_width` / `filter_choice` / `sample_filtering` use | §7.16, p529 | **ABSENT** | The shared §7.17 helpers exist (`deblock_core.h:324-567`) but nothing calls them with TIP block geometry. | Reusable kernels, no TIP driver. |

## §7.17 Deblocking filter (pages 530–540)

| Tool/rule | §/page | Status | Evidence | Gap |
|---|---|---|---|---|
| Plane/pass traversal and `apply_deblocking_filter[0..3]` gating | §7.17.1, p530 | **WIRED** | `deblock_core.h:782-796`; M1 calls it at `src/encoder/m1/still_encoder.cpp:1495-1516`; M1 GPU at `src/cuda/m1/still_gpu.cu:1674-1684`; flags from `still_encoder.cpp:1819-1822`. | None for 4:2:0 8-bit; M2 production has flags off. |
| Vertical boundaries strictly before horizontal | §7.17.1, p530 | **WIRED** | `deblock_core.h:784-792` (`plane` then `pass` 0/1). | None. |
| Chroma `rowStep`/`colStep = 1 << Subsampling` | §7.17.1, p530 | **WIRED** | `deblock_core.h:784-786`; 4:2:0 only (`deblock.cpp:27,58`). | 4:2:2/4:4:4 rejected. |
| `tileVertEdge`/`tileHorzEdge`, `sbEdge` = 64-grid edge or vertical tile edge, `disable_loopfilters_across_tiles` | §7.17.2, p531–532 | **COMPONENT** | `deblock_core.h:586-616`; M2 post passes tile starts (`src/encoder/m2/post/loop_filters.cpp:317`). | M1 writes `disable_loopfilters_across_tiles = 0` and is single-tile (`still_headers.cpp:145`); no production multi-tile encode exercises it. |
| BRU gating (`use_bru`, `BruModes != BRU_ACTIVE`) | §7.17.2, p531 | **ABSENT** | No `bru` field in `DeblockCoreFrame`/`DeblockOptions`; `deblock/README.md:203` lists it out of scope. | Inter-only feature, encoder writes `enable_bru = 0` (`video_headers.cpp:177`). |
| `onScreen` / frame-edge early return | §7.17.2, p532 | **WIRED** | `deblock_core.h:607-615`. | None. |
| `prevRow`/`prevCol` derivation | §7.17.2, p532 | **WIRED** | `deblock_core.h:620-625`. | None. |
| Sub-PU edges (`allow_df_sub_pu`, `SubPu*Base`), `isSubPuEdge` | §7.17.2, p532–534 | **COMPONENT** | `deblock_core.h:628-631,656-666`; M2 sequence writes `enable_df_sub_pu = 0` (`video_headers.cpp:181`); M1 intra. | Never enabled by an encoder stream. |
| Chroma bottom-right mode-info adjustment (`MiRowBase/MiColBase`, `Num_4x4_Blocks_*`) | §7.17.2, p533 | **WIRED** | `deblock_core.h:635-648`; M1 replicates chroma grids (`still_encoder.cpp:1395-1403`). | Caller must replicate chroma metadata; contract enforced by tests, not type system. |
| `skip = Skips[row][col]`, reset for `INTRA_REGION` / `(FrameIsIntra && enable_sdp)` | §7.17.2, p533 | **WIRED** | `deblock_core.h:650-655`; M1 intra metadata sets `skip = 0`, `region_intra = 1` (`still_encoder.cpp:1395-1397`). | M1 passes `enable_sdp = false` (`still_encoder.cpp:1508`). |
| `isBlockEdge`, `isTxEdge`, `filt_max_size()` | §7.17.2, p533–534 | **WIRED** | `deblock_core.h:657-666,242-289`. | None. |
| `applyFilter` condition chain (TX/sub-PU edge, Q/side, block/skip) | §7.17.2, p534 | **WIRED** | `deblock_core.h:678-690`. | None. |
| `filterSize = Min(Tx_Width/Height[prevTxSz], txSz])` | §7.17.4, p536 | **WIRED** | `deblock_core.h:692-698`. | None. |
| Screen clipping of `filterSize` (16 luma / 8 chroma) | §7.17.2, p534–535 | **WIRED** | `deblock_core.h:700-711`. | None. |
| Both-sides-lossless early return and per-side write suppression | §7.17.2, p535; §7.17.7.1, p538 | **COMPONENT** | `deblock_core.h:713-715,556-566`; tested in `deblock_test.cpp`. | Encoder cannot signal lossless (`base_q_idx = 0` rejected, `segmentation_enabled = 0`), so never exercised in a stream. |
| Filter maximum width table and `sbEdge` clamp (`1/3/6/8`, chroma `1/3/4/4`, neg clamp `6/2`) | §7.17.3, p535–536 | **WIRED** | `deblock_core.h:324-341`. | None. |
| Adaptive strength: `qindex = get_qindex(0, segment)`, `DeltaQUAc/DeltaQVAc + BaseUVAcDeltaQ`, `q_clamped`, `lvlSeg = qc + 8*DfDeltaQ[i]` | §7.17.5–§7.17.6, p536–537 | **PARTIAL** | Formula at `deblock_core.h:294-321`; options at `deblock_core.h:152-157`. | No stream uses nonzero deltas: M1 writes `delta_q_present = 0` and `DfDeltaQ = 0` (`still_headers.cpp:178,193-195`); M2 writes the same (`video_headers.cpp:285`). Segment Δ is a caller contract, not reconstructed. `DfDeltaQ[1] = DfDeltaQ[0]` when the pass-1 present bit is 0 is **not** implemented, and `write_video_deblock_params()` (`loop_filters.cpp:437-469`) writes only the present bit, not the `df_delta_q` value. |
| `qThr`/`side` merge and sub-PU `>>3` | §7.17.2, p535 | **WIRED** | `deblock_core.h:717-746`. | None. |
| Filter choice (`maxSamples*`, second derivatives, `side>>2/>>3`, `Q_First`, `endThr`) | §7.17.7.2, p538–540 | **WIRED** | `deblock_core.h:393-547`. | None. |
| Sample filtering (`qThrClamp`, `W_Mult`, `Round2(...,3+DF_SHIFT)`, clip, lossless sides) | §7.17.7.1, p538 | **WIRED** | `deblock_core.h:549-567`. | None. |
| 8-bit / 4:2:0 scope | §7.17, p530–540 | **PARTIAL** | `deblock.cpp:27,55-58` rejects other depths/subsampling; HBD still path disables deblock (`hbd_headers.h:56-61`). | 10-bit, 4:2:2, 4:4:4 absent. |

## §7.18 CDEF (pages 540–545)

| Tool/rule | §/page | Status | Evidence | Gap |
|---|---|---|---|---|
| 8×8 luma block traversal, one parameter set per 64×64 | §7.18, p540 | **WIRED** | `cdef.cpp:314-322`; M1 runs it at `still_encoder.cpp:1624-1660`; `cdef_idx` per 64×64 written by `still_syntax.cpp:1411-1426`. | None for M1; M2 has no CDEF in production. |
| Copy block to `CdefFrame`; `idx == -1` copy-only | §7.18.1, p540–541 | **WIRED** | `cdef.cpp:155-168`; `-1` handling at `cdef.cpp:81,93`. | M1 search always emits `[0, num_strengths)`; `-1` is not produced. |
| `skip` from four `Skips` cells and lossless; `cdef_on_skip_txfm_frame_enable` | §7.18.1, p541 | **WIRED** | `cdef_core.h:167-193`; M1 option at `still_encoder.h:62`, search at `cdef_search.h:114-141`. | M1 intra path passes `skip = 0`; the `on_skip` branch is wired but has no practical effect for intra. |
| `skipChroma` from `ChromaSegmentIds` | §7.18.1, p541 | **WIRED** | `cdef_core.h:180-191`; M1 grids at `still_encoder.cpp:1585-1587`. | Lossless chroma cannot be signalled by M1. |
| Direction process: 8 directions, `Div_Table`, variance `>>10` | §7.18.2, p542–543 | **WIRED** | `cdef_core.h:340-420` (64-bit partial/cost). | 8-bit formula only (`>> (BitDepth-8)`). |
| Luma strength derivation: `priStr`, `varStr`, `priStr = f(var)`, damping | §7.18.1, p542 | **WIRED** | `cdef_core.h:139-150`. | None. |
| Chroma direction `Cdef_Uv_Dir`, unscaled strengths, `damping - 1` | §7.18.1, p542 | **WIRED** | `cdef_core.h:152-161,76-79`. | None. |
| Primary/secondary taps, `constrain`, `Clip3(min,max,...)` | §7.18.3, p544–545 | **WIRED** | `cdef_core.h:243-338`. | None. |
| `is_inside_filter_region` / tile-bounded taps | §7.18.3, p545; §5.20.9.3, p243 | **COMPONENT** | `cdef_core.h:197-230,268-288`. | M1 is single-tile and writes `disable_loopfilters_across_tiles = 0`; no production multi-tile CDEF. |
| `CdefOnSkipTxfm` modes: DISABLED / ALWAYS_ON / ADAPTIVE | §5.4.10 + §5.18.7.10 | **WIRED** | Writer/reader implement all three (`filters_stream.cpp:321-348,445-466`); the still path codes the ADAPTIVE frame bit (`--still-cdef-on-skip-txfm`), and the M2 tiled gate selects DISABLED/ALWAYS_ON/ADAPTIVE with CDEF enabled (`src/pipeline/video/post/tiled_gate.cpp`, 2-frame `dav2d` gate). | The M2 product video writer still hardcodes `enable_cdef = 0`; only the tiled component gate exercises the sequence modes. Intra blocks derive `skip = 0`, so the frame bit changes no samples there. |
| Encoder strength/index search and legal signalling | §5.18.7.10 + §5.20.10.1 | **WIRED** | `cdef_search.h:114-214,372-590`; M1 `still_cdef_plan` at `still_encoder.cpp:1594`; index emit `still_syntax.cpp:1411`. Search SSE includes chroma. | No RD cost; SSE-only greedy search. |
| `CdefStrengths == 1` (no index) vs `>=2` (per-block index) | §5.20.10.1 | **WIRED** | `cdef_search.h:12-22`; `still_syntax.cpp:1414-1426`. | Per-block search is greedy, not RD-optimal. |
| 8-bit / 4:2:0 scope | §7.18, p540–545 | **PARTIAL** | `cdef.cpp:186,214-218` rejects other depths/subsampling. | 10-bit/4:2:2/4:4:4 absent. |
| M1 GPU CDEF | §7.18 | **ABSENT** | `src/cuda/m1/still_gpu.h:284-291` rejects `--still-cdef`; `still_gpu.cu` never calls the CDEF stage. | GPU pipeline has deblock only. |

## §7.19 CCSO (pages 545–548)

| Tool/rule | §/page | Status | Evidence | Gap |
|---|---|---|---|---|
| Per-plane enable, CCSO unit grid, `read_ccso()` tile symbols | §7.19, p546; §5.20.10.2 | **WIRED** | `still_encoder.cpp:1671-1800`; header `still_headers.cpp:229-260`; tile symbols `still_syntax.cpp:1319-1365`; core `ccso_core.h:231-240`. | M1 only; M2 production has CCSO off. |
| `CcsoLumaSizeLog2` derivation (6/7/8, `ccso_unit_matches_sb_size`, tile alignment) | §5.18.7.12 | **PARTIAL** | `restoration_common.h:182`; header at `filters_stream.cpp:702-716`; M1 fixes 6 with 64×64 SB (`still_encoder.cpp:1750`). | Encoder never uses 128/256 SBs or non-matching units, so sizes 7/8 are unreachable. |
| Band-offset-only mode (`ccso_bo_only`) | §7.19.1, p546–547 | **WIRED** | `ccso_core.h:207-218`; M1 sets `bo_only = true` and fits 0–6 bands per plane (`still_encoder.cpp:1710-1742`). | Only mode used by the encoder. |
| Edge-classified mode (`ccso_score`, `quantStep`, `edge_clf`, `ext_filter`, 7 `Ccso_Pos`) | §7.19.1, p547–548 | **COMPONENT** | `ccso_core.h:148-224`; header `filters_stream.cpp:755-815`; tests `ccso_test.cpp` (5,518 checks). | M1 `still_encoder.cpp:1737-1741` hard-codes `quant_idx = 0`, `ext_filter = 0`, `edge_clf = 0`, `bo_only = true`; `still_headers.cpp:237` asserts `bo_only`. |
| Luma band source from `CurrFrame` pre-CDEF, tile-clamped | §7.19.1, p547 | **WIRED** | `still_encoder.cpp:1624-1628` snapshots pre-CDEF planes; `ccso_core.h:104-112`. | M1 uses `disable_loopfilters_across_tiles = 0`; clamp path not exercised. |
| Lossless segment skip | §7.19.1, p547 | **COMPONENT** | `ccso_core.h:170-176`; `restoration.cpp` validation. | No lossless stream. |
| BRU-inactive superblock skip | §7.19.1, p546–547 | **COMPONENT** | `ccso_core.h:157-159`. | M1/M2 write `enable_bru = 0`; never exercised. |
| Tile extent clamp for the luma tap fetch | §7.19.1, p547 | **COMPONENT** | `ccso_core.h:104-112,114-160`; M2 post supplies tile grids (`loop_filters.cpp:417-435`). | M1 single tile; M2 post not wired. |
| Offset table (`CcsoFilterOffset`), clip to `2^BitDepth-1`, partial units | §7.19.1, p547–548 | **WIRED** | `ccso_core.h:199-226`; offset fill `restoration.cpp:153-180`; tests cover all 7 extension filters. | Wired only for `bo_only`; full class table is component-tested. |
| Chroma planes (U and V) | §7.19, p546 | **WIRED** | M1 fits and enables all three planes (`still_encoder.cpp:1688-1750`); core iterates `plane 0..2`. | Chroma offset fit can be all-zero but planes stay enabled, which is legal. |
| CCSO temporal reuse (`reuse_ccso`, `sb_reuse_ccso`) | §5.18.7.12 | **ABSENT** | `filters_stream.cpp:777-789` explicitly fails on reuse for non-intra, “CCSO reference reuse is not supported (intra only)”. | Inter-frame CCSO reuse cannot be signalled. |
| CCSO output used as reference for later frames | §7.19 | **ABSENT** | M1 is a single still picture; M2 production never calls `ccso_frame`; no two-frame CCSO `dav2d` gate exists. | Never used to predict a later frame. |

## §7.20 Loop restoration (pages 548–560)

| Tool/rule | §/page | Status | Evidence | Gap |
|---|---|---|---|---|
| Framework: `LrFrame := CdefFrame`, skip when `UsesLr = 0 && gdf_frame_enable = 0` | §7.20, p548 | **COMPONENT** | `restoration.cpp:574-604`; `wiener_core.h:423-437`. | No production caller; M1 sequence writes `enable_restoration = 0` (`still_headers.cpp:148`); M2 writes 0 (`video_headers.cpp:208`). |
| Unit sizing (`LoopRestorationSize`, shift flags, `count_units_in_frame`, conformance) | §5.18.7.11; §7.20.1, p549–550 | **COMPONENT** | `wiener_core.h:113-124`; `wiener_syntax.cpp:189-262,934-1108`; `filters_stream.cpp:1080-1155`; validation `restoration.cpp:543-550`. | Encoder-side search exists but is not connected to a production encoder. |
| `loop_restore_block`: unit row/col offsets, stripe row/num, tile-relative extents | §7.20.1, p549–551 | **COMPONENT** | `wiener_core.h:139-280`. | Component-tested only. |
| Source sampling: CdefFrame inside stripe, CurrFrame outside, two-line clamp, tile clamp | §7.20.2, p552 | **COMPONENT** | `wiener_core.h:285-305`. | Component-tested only. |
| Luma non-separable Wiener, 32 taps | §7.20.3, p553–554 | **COMPONENT** | `wiener_core.h:335-422`; tables `kWienerNsConfigY`. | No `dav2d` stream gate for luma LR. |
| Chroma non-separable Wiener, 12 chroma taps + 6 luma (CfL) taps | §7.20.3, p553–554 | **COMPONENT** (closed-loop gate) | `wiener_core.h:335-422`; `kWienerNsConfigUv`; `lr_chroma_gate` ran PASS with 13/18 active chroma units. | Gate-only; production M1/M2 never signal it. |
| `wiener_ns_uv_sym` mirroring for j >= 12 | §5.20.10.6; §7.20.3 | **COMPONENT** (closed-loop gate) | `wiener_syntax.cpp:1480-1600`; gate exercised 5 `uv_sym` units in frame 1. | Gate-only. |
| `cfl_ds_filter_index` and `Wiener_Filters_420` (index 3 aliases 0) | §7.20.3, p554–555 | **COMPONENT** | `wiener_core.h:307-330`; `kWienerFilters420`. | Luma LR path only component tests. |
| Per-unit coefficient syntax (`read_lr_unit`/`read_wienerns_filter`, subsets, `merged`, bank) | §5.20.10.5–.6; §7.20.3 | **COMPONENT** | `wiener_syntax.cpp:1351-1613`; `filters_stream.cpp:1886-2000`. | Not wired to production. |
| `frame_filters_on` + `FrameLrWienerNs[plane][class]` | §5.18.7.11; §7.20.3 | **COMPONENT** | `wiener_core.h:360-382`; `wiener_syntax.cpp:593-933`; `restoration.cpp:498-514` validation. | Single-class is usable; production never enables it. |
| `NumFilterClasses` / `SubclassLookup` / 256-class `FilterClass` map | §5.18.7.11; §7.20.3–§7.20.4 | **PARTIAL** | `wiener_class.cpp:109-180` implements the classifier; tables now generated in `spec_tables/`; core accepts 256 classes (`wiener_core.h:50-56`). | M1/M2 adapters expose only a 16-entry `WienerNsClassMap` and reject/avoid the 256-class map; no encoder path supplies `Pc_Wiener_Lut_To_Class` from `spec_tables`. |
| Pixel-classified Wiener (§7.20.4): features, LUT class, `Pc_Wiener_Sub_Classify`, 25 taps | §7.20.4, p555–557 | **COMPONENT** | Full reference `spec_tables/pc_wiener.cpp` + `pc_wiener_tables.h`; 1,595 checks. | LR core rejects `RESTORE_PC_WIENER` (`restoration.cpp:452-453`); `wiener_syntax.cpp:950-953` refuses to write PC-Wiener. Not integrated. |
| GDF (§7.20.5): `grad`, `gdfCls`, alpha/weight/bias/error tables, `GdfPixScale` | §7.20.5, p557–560 | **COMPONENT** | Full reference `spec_tables/gdf.cpp` + `gdf_tables.h`; 321 checks. | LR core rejects `gdf_frame_enable` (`restoration.cpp:431-432`); no GDF syntax or application path. |
| LR coefficient banks: `merged`, `use_bank`, `RefLrWienerNs` ring, `WienerNsPtr/Size` | §5.20.10.5–.6 | **COMPONENT** | `wiener_syntax.h:218-222`; `wiener_syntax.cpp:1351-1515`; `filters_stream.cpp:1886-2000`. | Not wired to production. |
| Frame-filter group search and temporal frame-filter prediction (`search_frame_filters`, `rst_ref_pic_idx`, `RefFrameLrWienerNs`) | §5.20.10.6; §5.18.7.11 | **PARTIAL** | Full writer/search in `wiener_syntax.cpp:534-933,934-1108`. | `filters_stream.cpp:1493-1496,1582-1584` hard-rejects temporal frame-filter prediction; M1/M2 adapters use only the deterministic group-0 layout and do not search bank reuse. |
| Chroma LR unit sizing / `lr_tools_uv_present` / U and V per-unit tools | §5.18.7.11; §5.20.10.4–.6 | **COMPONENT** (closed-loop gate) | `chroma_lr.cpp`; gate `lr_chroma_gate.cpp:11-13,149-150`; observed 0 diffs on both frames. | Gate-only; production headers leave `enable_restoration = 0`. |
| LR lossless, BRU-inactive, and tile-bounded blocks | §7.20.1, p549–551 | **COMPONENT** | `wiener_core.h:160-205,231-262`. | Never exercised by a production stream; lossless/BRU unreachable. |
| LR output used as reference for later frames | §7.20 | **PARTIAL** | Only `lr_chroma_gate` applies chroma LR to frame 0 and predicts frame 1 from it; `video_frame.cpp:1027,1052` uses raw recon. | Luma LR, PC-Wiener and GDF are never reference filters; production M2 uses no LR at all. |
| 8-bit / 4:2:0 scope | §7.20, p548–560 | **PARTIAL** | `restoration.cpp:24-28` rejects other depths/subsampling; plane type is `uint8_t*`. | 10-bit/4:2:2/4:4:4 absent. |

## Cross-cutting coverage

| Dimension | Status | Evidence | Gap |
|---|---|---|---|
| 8-bit filters | **WIRED/PARTIAL** | All four cores are 8-bit; M1 8-bit paths work. | 10-bit filter paths do not exist; HBD still headers disable deblock/CDEF/CCSO/LR (`hbd_headers.cpp:162-165,196-201`). |
| 4:2:0 | **WIRED** | All cores validate `subsampling_x == 1 && subsampling_y == 1`. | 4:2:2 / 4:4:4 absent (`deblock.cpp:27`, `cdef.cpp:186`, `restoration.cpp:82`). |
| Multi-tile | **PARTIAL** | Deblock/CDEF/CCSO/LR cores accept tile extents; M2 post can pass tile grids. | M1 product path is single-tile; M2 post is not wired; the tiled still gate notes a known per-tile deblock mismatch (`tiled_still_gate.cpp:603-605`). |
| Filter order | **WIRED (M1 CPU)** | `still_encoder.cpp` deblock → CDEF → CCSO; `run_video_post_filters()` enforces the same order (`loop_filters.cpp:270-435`). | No LR stage in either chain. |
| Reference update | **ABSENT (multi-frame)** | M2 pushes raw recon (`video_frame.cpp:1027,1052`); M2 headers disable all tools. | See next section. |

## Verification performed

* Static trace of every table row through syntax writer → pixel core →
  reconstruction/reference.
* `tools/still_verify/feature_gate.sh --bin build-m1/av2enc --dump build-m1/still_dump`
  with `dav2d-macos`: **30 passed, 0 failed**, including
  `still_64_ccso`, `still_64_ccso_cdef`, `still_360_ccso`, `still_360_ccso_cdef`,
  `still_360_ccso_rdoq`. The gate decodes with `dav2d --strict 1` and compares
  decoder output with encoder reconstruction.
* `src/encoder/filters/lr_chroma/gate/build_and_test.sh` with
  `DAV2D=tools/bin/dav2d-macos`: **PASS**, decoded frame 0 and frame 1 both
  `0 diffs` vs encoder §7.20 output; frame 0 U/V both changed; frame 1 used 18
  active chroma units and 5 `uv_sym` units.
* No production multi-frame filter gate exists beyond the above; M2 CDEF and
  post-chain gates were inspected, not treated as production evidence.

## Filters not applied to the reference used by later frames

The only multi-frame production-style entry is `encode_video_sequence()`
(`src/encoder/m2/video_frame.cpp`); dedicated gates build streams through their
own vendored writers. It never calls `run_video_post_filters()`; it
pushes `encoder.recon()` directly into `result.recon`, and the next P frame
uses `result.recon.back()` as its reference (`video_frame.cpp:1027`). M2 also
writes every filter flag off (`video_headers.cpp:205-213,286-288`), so the stream
is self-consistent — but no filter is applied to the reference.

| Filter | Applied to a later-frame reference? | Where/why |
|---|---|---|
| §7.16 TIP deblock | **No** | TIP never enabled (`enable_tip = 0`); no TIP frame exists. |
| §7.17 deblock | **No in M2 production** | M2 `apply_deblocking_filter[] = 0`; reference is raw recon. M1 deblocks only its single still output. |
| §7.18 CDEF | **No in M2 production** | M2 `enable_cdef = 0`; reference is raw recon. M1 CDEF affects only the still output. The M2 CDEF gate deliberately forces the P frame intra because the core always predicts from unfiltered recon (`video_cdef_gate.cpp:20-22,464`). |
| §7.19 CCSO | **No** | M1 single frame; M2 production off; no two-frame CCSO gate. |
| §7.20 LR, luma | **No** | Never signalled by M1/M2; no luma LR stream gate. |
| §7.20 LR, chroma | **Yes, gate only** | `lr_chroma_gate` applies chroma LR to frame 0 and predicts frame 1 from the filtered output; `dav2d` matched 0 diffs. Not reachable from `av2enc`. |
| §7.20 PC-Wiener | **No** | Reference implementation only in `spec_tables`; LR core rejects it. |
| §7.20 GDF | **No** | Reference implementation only in `spec_tables`; LR core rejects it. |

## Counts

| Section | WIRED | COMPONENT | PARTIAL | ABSENT | Rows |
|---|---:|---:|---:|---:|---:|
| §7.16 TIP deblock | 0 | 0 | 0 | 3 | 3 |
| §7.17 deblock | 15 | 3 | 2 | 1 | 21 |
| §7.18 CDEF | 10 | 1 | 2 | 1 | 14 |
| §7.19 CCSO | 5 | 4 | 1 | 2 | 12 |
| §7.20 LR | 0 | 15 | 4 | 0 | 19 |
| **Total** | **30** | **23** | **9** | **7** | **69** |

Interpretation:

* 30 rules are genuinely wired into an encode path, almost all via M1 CPU.
* 23 rules are implemented but only component/gate-reachable.
* 9 rules are partial, dominated by the missing delta-Q deblock path,
  multi-class LR, temporal frame-filter prediction and 8-bit-only scope.
* 7 rules are absent: all of §7.16, deblock BRU, CCSO temporal reuse, and
  CCSO reference use.

## Update 2026-11-13 — 10-bit and non-4:2:0 product filters

The CPU format still writer (`src/formats/high_bitdepth/`, `--still --bit-depth
10 --chroma 420|422|444|400`) now runs the in-loop filter chain at 10 bits:

| Tool | Implementation | Gate |
|---|---|---|
| §7.17 deblock | `deblock_frame16` (already landed) | `format_gate.sh` |
| §7.18 CDEF | `cdef_core.h` templated + `cdef_frame16`; `hbd_cdef.h` plan | `format_gate.sh` `10bit_*_cdef_*`, `hbd_test` |
| §7.19 CCSO | `ccso_core.h` templated + `ccso_frame16`; `hbd_ccso.h` band/edge fit | `format_gate.sh` `10bit_*_ccso_*`, `hbd_test` |
| §7.20 Wiener NS | `wiener_core.h` templated + `loop_restoration_frame16`; `hbd_lr.h` per-unit fit | `format_gate.sh` `10bit_*_wiener*`, `hbd_test` |
| §7.20 PC-Wiener | §7.20.4 fixed bank, per-unit on/off (`hbd_lr.h`) | `format_gate.sh` `10bit_*_pc_wiener*` |
| §7.20.5 GDF | `gdf_decision`/`tables/gdf` (already bit-depth generic) | `format_gate.sh` `10bit_*_gdf*` |

`src/formats/high_bitdepth/gate_10bit_filters.sh` runs the full matrix: every
product chroma format at 64x64 and 640x360, q90 and q303, each filter alone
and the whole chain, requiring `dav2d --strict 1` and a 0-LSB decoder/recon
match; the all-off stream is byte-stable across runs.

Remaining gaps kept **rejected by name**:

* §7.20 chroma LR at 4:2:2/4:4:4 (1-LSB stripe-boundary diffs; repro in
  `docs/negative-results.md`).  Luma Wiener, PC-Wiener and GDF are exact.
* The `--video --bit-depth 10` / `--video --chroma` writer still rejects every
  loop-filter flag: its own header/tile writer has no filter plan or syntax
  hooks.  The 8-bit 4:2:0 M2 video chain is unchanged.
* The 8-bit non-4:2:0 still writer (`src/formats/chroma/`) has no filter stage.
