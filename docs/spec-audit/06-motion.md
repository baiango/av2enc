# Spec audit 06 — motion field, TIP, MV contexts, MV prediction

Scope: AV2 v1.0.0 §7.9–§7.12 (spec pages 401–447). Normative text read from
`spec/AV2_Spec_v1.0.0.pdf` (page index = spec page − 1). Code read from
`src/encoder/inter/**`, `src/encoder/m2/**`, and the sequence/frame header
writers.

Question answered per process: **does the encoder implement the bitstream
capability**, not whether a decoder-side helper exists in the tree.

Legend:

| Status | Meaning |
|---|---|
| **WIRED** | Called from the M2 encode path; affects emitted syntax and/or reconstruction; gated against `dav2d` (M2 gate, or the component's own gate where noted). |
| **PARTIAL** | Called, but a normative branch that the current syntax can reach (or one flag away from reachable) is missing or simplified. |
| **COMPONENT** | Spec-complete (or near-complete) implementation with tests, but **no** M2 path calls it and the sequence header disables the tool. |
| **ABSENT** | No implementation of the process anywhere in `src/`. |
| **DECODER-ONLY** | Not applicable here / not observed. |

Baseline facts used throughout (they decide most statuses):

* `src/encoder/m2/m2_headers.cpp:155-182` writes `enable_ref_frame_mvs = 0`,
  `reduced_ref_frame_mvs_mode = 0`, `enable_refmvbank = 0`,
  `disable_drl_reorder = 1`, `enable_tip = 0`, `enable_mv_traj = 0`,
  `enable_bawp/cwp/imp_msk_bld/df_sub_pu = 0`, `enable_opfl_refine = REFINE_NONE`,
  `enable_refinemv = 0`, `enable_global_motion = 0`; all
  `seq_enabled_motion_modes[1..4] = 0`.
* `m2_headers.cpp:266-267` — inter frame header: `use_ref_frame_mvs` is absent
  (inferred 0 because `enable_ref_frame_mvs == 0`, §5.18.2 p102) and
  `TipFrameMode = TIP_FRAME_DISABLED`; no `tmvp_sample_step_minus_1`, no TIP
  block, no `global_motion_params()`.
* `m2_frame.cpp:313-315` resets the 4×4 `mv_grid_` **per frame**; no saved MV
  field is kept for a reference slot. `merge::build_stack_checked()` is called
  per leaf (`m2_frame.cpp:478-481`) and again in the syntax writer
  (`m2_syntax.cpp:1652-1660`).
* The M2 build (`src/encoder/m2/build_and_test.sh:45-50`) compiles only
  `inter/me/me.cpp`, `inter/mc/mc.cpp`, `m2/merge/merge.cpp`. Nothing from
  `inter/temporal/`, `inter/gme/`, `inter/drl_reorder/`, `inter/advanced/`
  is in the core binary.

Note: `src/encoder/inter/me/` and `me2/` are block motion **search**; they are
not the §7.9 motion-field process and are not counted below.

---

## §7.9 Motion field estimation (p401–413)

| Process | §/page | Encoder capability | Status | Evidence | Gap |
|---|---|---|---|---|---|
| §7.9.1 General: `sortRef`, `curIdx`, `topo_sort_refs`, `is_ref_overlay`, `record_projection_with_type`, `has_past/future_ref`, closest past/future recording | 7.9.1 / p401–405 | Full transcription incl. `reduced_ref_frame_mvs_mode` truncation and `use_bru` guard | **COMPONENT** | `inter/temporal/motion_field.cpp:450` (`motion_field_estimation`), `:118` (`topo_sort_refs`), `:144` (`record_projection_with_type`), `:513` (sortRef), `:651` (reduced mode) | No caller. Requires a `SavedMvField` per reference slot (`temporal_types.h:235`) that M2 never builds. `MotionFieldResult::closest_past/future` unused, so `OrigClosestPast/Future` and `useTemporalFirst` (§7.12.2 p423) cannot exist. |
| §7.9.2 Fill trajectory MV gap (`enable_mv_traj`) | 7.9.2 / p406–407 | Implemented inline in the estimation driver | **COMPONENT** | `motion_field.cpp:666` (`fill_traj_mv` calls), `get_tmvp_shift/unit/phase/tmvp_avail` in `temporal_math.h` | `enable_mv_traj = 0` in the sequence header; unreachable. |
| §7.9.3 Projection process (per-src loop, trajectory intersection, `get_sampled_position`, `check_block_position`, MV/offset store) | 7.9.3 / p408–410 | Implemented incl. `check_traj_intersect` | **COMPONENT** | `motion_field.cpp:303` (`projection`), `:188` (`check_traj_intersect`), `:346` | No caller; needs saved ref MVs. |
| §7.9.4 Get MV projection (`Div_Mult[32]`, `Round2Signed`, `MV_LOW/UPP` clip) | 7.9.4 / p411 | Exact, all 32 entries | **COMPONENT** (plus a WIRED duplicate) | `temporal_math.h:46` (`kDivMult`), `:56` (`get_mv_projection`); duplicate at `m2/merge/merge.cpp:61` (`project_mv`) used by §7.12.2.16 | The wired copy is only reached for multi-reference derived candidates; the baseline is single-reference, so it never executes. |
| §7.9.5 Get MV projection clamp (`±REFMVS_LIMIT`) | 7.9.5 / p412 | Exact | **COMPONENT** | `temporal_math.h:82` (`get_mv_projection_clamp`) | Only used by §7.10.2/.5. |
| §7.9.6 Get sampled position (+ `ProjStep == 2` even-position adjust) | 7.9.6 / p412 | Exact | **COMPONENT** | `temporal_math.h:151` | — |
| §7.9.7 Get block position no constraint | 7.9.7 / p412–413 | Exact | **COMPONENT** | `temporal_math.h:129` | — |
| §7.9.8 Check block position (TMVP-unit constraint, `ProjStep` half-unit) | 7.9.8 / p413 | Exact | **COMPONENT** | `temporal_math.h:170` | — |

## §7.10 Setup TIP motion field / Build TIP (p413–419)

| Process | §/page | Encoder capability | Status | Evidence | Gap |
|---|---|---|---|---|---|
| §7.10.1 General + conformance conditions (frame-header trigger, `TipFrameMode`, `allow_tip_hole_fill`, `TipGlobalMv`, `tip_global_wtd_index`, `TipInterpFilter`) | 7.10.1 / p413–414 | Frame-level capability never signalled | **ABSENT** | `m2_headers.cpp:169` (`enable_tip = 0`), `:266-267` (`TipFrameMode = TIP_FRAME_DISABLED`, no block written) | No `tip_frame_mode`, `frame_opfl_refine_type()`, `allow_tip_hole_fill`, `tip_global_wtd_index`, `tip_sharp/tip_regular`, `tip_mv_*` bits; no `OBU_REGULAR_TIP` emitter. `temporal_syntax.h/.cpp` can write the bits but M2 never calls it. |
| §7.10.2 TIP temporal scale motion field | 7.10.2 / p414 | Exact (`get_mv_projection_clamp`, `MotionFieldOffset = refOffset`) | **COMPONENT** | `motion_field.cpp:821` (scale loop) | No caller. |
| §7.10.3 TIP fill motion field holes (`Tip_Dirs`, SB/128-block containment) | 7.10.3 / p415 | Exact | **COMPONENT** | `motion_field.cpp:862` (`kTipDirs`), same function | No caller. |
| §7.10.4 TIP block average filter (`Weight_Div_Mult`, `avgValid`) | 7.10.4 / p415–416 | Exact | **COMPONENT** | `motion_field.cpp` inside `setup_tip_motion_field` | No caller. |
| §7.10.5 Fill temporal MV sample gap (`fill_tpl`, `calc_avg`) | 7.10.5 / p416–418 | Exact; also callable for non-TIP frames | **COMPONENT** | `motion_field.cpp:757` (`fill_motion_field_sample_gap`) | §5.18.2 calls this at the end of estimation; nothing does. |
| §7.10.6 Build TIP process | 7.10.6 / p418–419 | Per-block MV plan + 4×4 grid, and a `tip_predict_block()` blend helper; **no** full-frame TIP construction | **PARTIAL** | `motion_field.cpp:951` (`build_tip_plan` — MVs only), `inter/temporal/tip.cpp` (`tip_predict_block`), `store_motion_field_entry` (`motion_field.cpp:1208`) | The process body (set `RefFrames`/`YMode = NEWMV`, `CwpIdx`, `predict_inter` per 8×8/16×16 with `TipInterpFilter`, `LumaUseOptflowRefinement` → `motion_field_motion_vector_storage`) is not assembled anywhere; `build_tip_plan()` stops at `Mvs[]`. Also no OPFL/refine-MV result composition into the stored MVs (caller is expected to, but no caller exists). |

## §7.11 Motion vector context processes (p419–422)

| Process | §/page | Encoder capability | Status | Evidence | Gap |
|---|---|---|---|---|---|
| §7.11.1 General / `find_mode_ctx` entry | 7.11.1 / p419 | Called per inter block from the tile writer | **WIRED** | `m2_syntax.cpp:1461` (`new_mv_context()`), used at `:1698` for `single_mode`/`drl_mode` CDF selection; `merge::derive_new_mv_context()` (`merge.cpp:1019`) is the tested reference | Two implementations (writer-inline + `merge`); the writer one is authoritative. |
| §7.11.2 Find mode context (`leftA/B`, `aboveA/B`, `nearestMatch`, `NumMvCount`) | 7.11.2 / p419–420 | Single-prediction subset | **PARTIAL** | `m2_syntax.cpp:1461-1483` | No `scan_point_warp_ctx()` calls, no `WarpMvCount`/`WarpSampleFound` (§7.11.4), no compound arm and no TIP arm (`RefFrames[..][0] == TIP_FRAME && ClosestPast/Future`). Reachable today only as single-reference SIMPLE, so no desync yet. |
| §7.11.3 Scan point context (`found`, `NewMvCount ≤ 3`, `has_newmv_for_list`) | 7.11.3 / p420 | `has_newmv` = `YMode == NEWMV` | **WIRED** | `m2_syntax.cpp:1464-1482`; spec reference `merge.cpp` `derive_new_mv_context` | M2 does not compare the neighbour's `RefFrames[list]` with `RefFrame[0]` before counting `NewMvCount` (spec p420). Equivalent while `NumTotalRefs == 1`; a CDF-context divergence the moment multi-ref is enabled (`TileDrlModeCdf` ctx shifts). |
| §7.11.4 Scan point warp context (SB-border even-column alignment, `WarpSampleFound`, `ExtendDeltaRow/Col`) | 7.11.4 / p421 | none | **ABSENT** | `grep -rn "WarpMvCount\|WarpSampleFound\|ExtendDeltaRow" src/` matches only two README lines in `m2/advanced/`; no code | `DeriveWrl` is never 1, so the process is skipped by the decoder too; enabling any `seq_enabled_motion_modes[LOCALWARP]` needs this plus §7.12.3. |

## §7.12 Motion vector prediction (p422–446)

| Process | §/page | Encoder capability | Status | Evidence | Gap |
|---|---|---|---|---|---|
| §7.12.1 General / `find_mv_stack` entry | 7.12.1 / p422 | Called per leaf, twice (decision + syntax validation) | **WIRED** | `m2_frame.cpp:478-481`, `m2_syntax.cpp:1652-1660` | — |
| §7.12.2 ordered steps 1–5 (init, `setup_global_mv`, corners, temporal-first gate) | 7.12.2 / p423 | Init + step-16/17 subset | **PARTIAL** | `merge.cpp:130-240` | Step 1–3 always produce `GlobalMvs = {0,0}`; step 4 (`generate_points_from_corners` when `DeriveWrl`) absent; step 5–6 (`useTemporalFirst`) hard-false (`merge.h:297`, `m2_frame.cpp:473`). |
| §7.12.2.1 Setup global MV | 7.12.2.1 / p424 | Identity only | **PARTIAL** | `m2_frame.cpp:469-470` (`global_mv = {0,0}`), `m2_headers.cpp:180` (`enable_global_motion = 0`) | `gm_params[ref]` / `GmType > IDENTITY` never used; `gme::global_mv_for_block()` exists but has no caller. |
| §7.12.2.2 Get warp motion vector | 7.12.2.2 / p425 | Full (Q16 projection, `MV_LOW/UPP` clip, `clamp_mv_row/col`, `lower_mv_precision`) | **COMPONENT** | `inter/gme/gme.h:220` (`get_warp_motion_vector`), tests `gme_test.cpp`; gme has its own `dav2d` gate (6 streams, 0 mismatches) | Not called by the encoder; `DeriveWrl`/GLOBALMV-with-`GmType` unreachable. |
| §7.12.2.3 Generate points from corners (`CornersFound == 3`, `allMvsSame`, `wmmat`, `reduce_warp_model`) | 7.12.2.3 / p425–427 | none | **ABSENT** | — | Needs §7.12.2.4 + `WarpParams` MI grid + `insert_warp_candidate`. |
| §7.12.2.4 Warp corner process | 7.12.2.4 / p427–428 | none | **ABSENT** | — | `get_warp_motion_vector_xy_pos` also missing; `advanced/warp` has only forward prediction (§7.13.3.19/.20) and `estimate_local_warp()` on caller-supplied samples. |
| §7.12.2.5 Scan col process | 7.12.2.5 / p428 | Exact (`deltaCol += MiCol & 1`, `MiColBase` equality, `bh4 > 1` second point) | **WIRED** | `merge.cpp:447` (`scan_col`), `:431` (`scan_point_if_valid`) | — |
| §7.12.2.6 Scan point process (SB-border even col, `weight = 0` for `(-1,-1)`/`deltaCol < -1`, `MAX_REF_MV_STACK_SIZE` bail) | 7.12.2.6 / p428–429 | Exact incl. `RefFrames[..][0]` written check | **WIRED** | `merge.cpp:408`; `mi_at(...).ref[0] == kRefNone` guard | Add-warp half of the step is missing (see §7.12.2.9). |
| §7.12.2.7 Temporal scan process (`stepW4/H4`, `rowEnd/colEnd ≤ 16`, two `add_tpl_ref_mv`) | 7.12.2.7 / p429 | Exact | **COMPONENT** | `motion_field.cpp:1140` (`temporal_scan`) | `useTemporal = 0` because `use_ref_frame_mvs = 0`; nothing calls it. |
| §7.12.2.8 Temporal sample (`get_motion_field_mv`, `WeightStack`, `PruneCount/MAX_PR_NUM`, compound pair) | 7.12.2.8 / p430–431 | Exact | **COMPONENT** | `motion_field.cpp:1048` (`temporal_sample`), `:1023` (`get_motion_field_mv`) | No caller; depends on §7.9 output. |
| §7.12.2.9 Add warp motion vector (`WarpMvCount`, `MotionModes >= LOCALWARP`, `WarpParams`) | 7.12.2.9 / p431 | none | **ABSENT** | `merge.cpp:381` (`add_reference_mv`) never consults motion modes or warp params | Latent desync if `LOCALWARP`/`DeriveWrl` is enabled: the writer's stack would lack warp candidates the decoder inserts. |
| §7.12.2.10 Add reference motion vector (single / compound / TIP / warped dispatch, `is_derivable_ref_frame`) | 7.12.2.10 / p431–432 | Single + compound subset | **PARTIAL** | `merge.cpp:381`; TIP bail at `:399` | TIP branches (`derive_ref_mv_candidate_from_tip_mode`, `compound_add_derived` vs `compound_search_stack` vs TIP path) and the warp call are absent; compound always goes through the non-TIP dispatch. |
| §7.12.2.11 Insert warp candidate (`WarpParamStack`, `MAX_WARP_REF_CANDIDATES`) | 7.12.2.11 / p432–433 | none | **ABSENT** | no `WarpParamStack` outside `merge.h`'s comment (`merge.h:31`, "deliberately out of scope") | Warp syntax (`is_warp`, `warp_idx`) can therefore never be emitted. |
| §7.12.2.12 Search stack process (single list, `PruneCount`, row/col offsets) | 7.12.2.12 / p433 | Exact | **WIRED** | `merge.cpp:248` | — |
| §7.12.2.13 Compound search stack (`candCwp`, `large`, `GmType > IDENTITY` → `GlobalMvs`) | 7.12.2.13 / p433–434 | Exact modulo `GmType` (always IDENTITY) | **WIRED** | `merge.cpp:376`, `:329`; exercised in `m2/multiref/gate` (3-frame compound stream, `dav2d` recon exact) | Not reachable from the M2 core itself (`num_total_refs = 1`, `reference_select = 0`). |
| §7.12.2.14 Compound add derived | 7.12.2.14 / p434–436 | Non-temporal subset (`MAX_DR_STACK_SIZE`, `comp_mv_in_stack`, `SingleMv`) | **WIRED** | `merge.cpp:329` | `enable_mv_traj` block (TrajValid/TrajMv projection) omitted — no-op while the flag is 0. |
| §7.12.2.15 Derive ref MV candidate from TIP mode | 7.12.2.15 / p436–437 | none | **ABSENT** | — (`merge.cpp:399` drops any TIP-reference neighbour outright) | Requires `get_tip_cand()` (exists in `motion_field.cpp:1168`) plus compound TIP refs. |
| §7.12.2.16 Single add derived (`FrameDistance` projection, `MAX_DR_PR_NUM`) | 7.12.2.16 / p437–438 | Projection-only subset | **WIRED** | `merge.cpp:301`, `project_mv` at `:61` | Trajectory branch (`use_ref_frame_mvs && enable_mv_traj` → `TrajMv` difference) missing, and `make_stack_params()` never sets `frame_distance`/`num_refs` (`m2_frame.cpp:455-476`), so the process returns at `merge.cpp:307-309`. With one total reference every neighbour matches `RefFrame[0]` and takes the `search_stack` path anyway — so today this is a safe no-op, not a working projection. |
| §7.12.2.17 Derive single ref MV candidate from TIP mode | 7.12.2.17 / p438 | none | **ABSENT** | — | — |
| §7.12.2.18 TIP add derived (`SubMvs` linear MV, `get_tip_offsets`) | 7.12.2.18 / p438–439 | none | **ABSENT** | — | — |
| §7.12.2.19 Sorting process (one move-to-front over `[0, numNearest)`, whole-record swap) | 7.12.2.19 / p439–440 | Normative transform exists and is gated; the wired sort is the **wrong** full selection sort | **COMPONENT** | `inter/drl_reorder/drl_reorder.cpp:116` (`reorder_stack`), `:72` (`use_sort`), gate 3 streams / 0 recon diffs incl. negative control; wired path `merge.cpp:455` (`sort_nearest`) | (a) never invoked: `m2_headers.cpp:159` writes `disable_drl_reorder = 1`, `m2_frame.cpp:473` and `m2_syntax.cpp:1656` hard-code `DrlReorder::kDisabled`; (b) the wired implementation is a full selection sort, i.e. the historical desync (see below). |
| §7.12.2.20 Extra search process (clamp, global candidate, `>32` MVP combinations, warp bank, IBC candidates) | 7.12.2.20 / p440–442 | Global candidate + gated MVP combinations | **PARTIAL** | `merge.cpp:200-235` (global), `:508` (`insert_mvp_candidate`), gate in `mvp_combinations_allowed()` (`merge.cpp:760`) | Warp candidates (`WarpBankParams[ref]`, `gm_params[ref]`, two `Default_Warp_Params`) and the `use_intrabc` `add_to_ref_bv()` candidates are absent. Encoder additionally caps NEARMV indices at `min(count, 2)` for blocks with a dimension ≤ 32 (`m2_frame.cpp:613-614`) — safe but leaves candidates unused. |
| §7.12.2.21 Fill MVP from ref MV bank (`get_rmb_list_index`, key match, `check_rmb_cand`, out-of-frame rejection) | 7.12.2.21 / p442–443 | Full incl. `MAX_RMB_SB_HITS` bookkeeping | **COMPONENT** | `inter/temporal/ref_mv_bank.cpp:209` (`fill_mvp_from_ref_mv_bank`), `:56` (`get_rmb_list_index`), `:105` (`update_ref_mv_bank`), `:169` (`reset_ref_mv_bank`) | `enable_refmvbank = 0`; no caller; no per-SB bank reset/maintenance in the M2 frame loop. `MAX_REF_MV_STACK_SIZE` clamp documented as a deliberate deviation (6 vs `max_drl_bits_minus_1 + 2`). |
| §7.12.2.22 Fill MVP from derived SMVP (`stack_match`, `max_drl_bits_minus_1 + 2` cap) | 7.12.2.22 / p443–444 | Exact | **WIRED** | `merge.cpp:474` (`fill_derived`) | Effectively empty in the baseline because no derived candidates are produced (`frame_distance == nullptr`). |
| §7.12.2.23 Clamping process (`clamp_mv_row/col`, `MV_BORDER`) | 7.12.2.23 / p444 | Exact | **WIRED** | `merge.cpp:542` (`picture_clamp`), `:564` (`clamp_all`); formula matches §5.20.9.4/.5 (p243) | Applies to both lists for single blocks (harmless: list 1 mirrors list 0). |
| §7.12.3 Find warp samples + §7.12.3.2 Add sample (`above_sample_stored`, `doTopLeft/doTopRight`, `LEAST_SQUARES_SAMPLES_MAX`, `CandList`) | 7.12.3 / p444–446 | none (only an estimator that consumes pre-built samples) | **ABSENT** | `inter/advanced/warp.h:93` (`estimate_local_warp(samples, count, …)`); `inter/advanced/README.md` lists "**`find_warp_samples`** (§7.12.3) … Not implemented" | Blocks local-warp encoding: without the normative sample set the encoder cannot reproduce the decoder's `num_samples`/`CandList` and therefore its `warp_idx` stack. |

---

## Counts

| Status | Count | Rows |
|---|---:|---|
| WIRED | 11 | §7.11.1, §7.11.3, §7.12.1, .5, .6, .12, .13, .14, .16, .22, .23 |
| PARTIAL | 6 | §7.10.6, §7.11.2, §7.12.2 steps 1–5, §7.12.2.1, §7.12.2.10, §7.12.2.20 |
| COMPONENT | 17 | §7.9.1–.8 (8), §7.10.2–.5 (4), §7.12.2.2, .7, .8, .19, .21 |
| ABSENT | 10 | §7.10.1, §7.11.4, §7.12.2.3, .4, .9, .11, .15, .17, .18, §7.12.3 |
| DECODER-ONLY | 0 | — |
| **Total** | **44** | |

Reading of the counts: the temporal half of this range (§7.9, §7.10, §7.12.2.7/.8/.21)
is **implemented and tested but entirely unwired** — 17 of 44 processes are
`COMPONENT`, and every one of them sits behind a sequence flag M2 writes as 0.
The spatial MV-stack half (§7.11, §7.12.2.5/.6/.12/.13/.14/.16/.20/.22/.23) is
wired and `dav2d`-gated, and is the part that actually produces bits today.

## Where our syntax would desynchronise the decoder

Nothing below desynchronises the **current** bitstream (all gates pass, 0 recon
diffs). These are the places where flipping one flag in
`m2_headers.cpp:155-182` produces a stream the decoder parses but reconstructs
differently.

1. **DRL reorder — already proven.** `merge.cpp:455` `sort_nearest()` runs a full
   selection sort over `[0, numNearest)`; §7.12.2.19 is a *single* move-to-front
   (only `start == 0` is ever invoked). For weights `{1,2,3,4}` the encoder order
   is `{4,3,2,1}` but the decoder's is `{4,2,3,1}`, so signalled index 2 means MV
   `30` to the decoder and MV `20` to us. `src/encoder/m2/README.md`
   ("Feature 2 status") records exactly this: `DRL_REORDER_CONSTRAINT` desyncs at
   frame 2 of the 8×720p clip (first diff at (768,129)), `DRL_REORDER_ALWAYS`
   desyncs at frame 7 with inter residual. The correct transform exists
   (`drl_reorder.cpp:116`) and is `dav2d`-gated, but no M2 path calls it, and
   `merge::choose_drl_index_for_mv()` (`merge.cpp:1436`) searches the
   **pre**-reorder stack — with reordering on, the pre→post index mapping
   (`drl::signal_index_for_pre`) is mandatory.
2. **Warp stack / warp samples.** `seq_enabled_motion_modes[LOCALWARP] = 1` or any
   `DeriveWrl = 1` path makes the decoder run §7.11.4, §7.12.2.3/.4/.9/.11 and
   §7.12.3. We implement none of them, so `WarpMvCount` (the `TileIsWarpCdf`
   context) and `WarpParamStack` (the `warp_idx` candidate list) would both be
   built from a different state. This is a context *and* candidate-order
   desync that still parses.
3. **`use_ref_frame_mvs = 1` + multi-reference.** Two separate desyncs:
   (a) the frame header must then code `tmvp_sample_step_minus_1` when
   `NumTotalRefs > 1 && SbSize != BLOCK_64X64` (§5.18.2 p102) — M2 writes no
   such bit and hard-codes `ProjStep = 1`; that is a bit-level desync from that
   point on, not just a reconstruction mismatch. (b) the temporal scan candidates
   (§7.12.2.7/.8) are inserted between spatial steps 13 and 15, shifting every
   subsequent `RefStackMv` index.
4. **`enable_refmvbank = 1`.** §7.12.2 step 20 appends bank candidates before
   derived-SMVP/extra-search. Our `fill_derived()` would start from a shorter
   stack, so every DRL index past the divergence point names a different MV
   (`ref_mv_bank.cpp:209` is not called, and no per-SB `reset_ref_mv_bank()`
   exists in `m2_frame.cpp`).
5. **`skip_mode`/compound in the M2 core.** The compound stack is wired
   (`merge.cpp:376`) and gated only through the separate `m2/multiref` stream
   writer. The M2 core hard-codes `reference_select = 0` and `num_total_refs = 1`
   (`m2_headers.cpp:261-262, 291`), and `m2_syntax.cpp` has no `comp_mode`/
   `comp_ref`/`compound_mode_non_joint`/`comp_group_idx` symbols — enabling
   compound without porting the `multiref` writer would emit a tile the decoder
   parses with different RefFrame pairs.
6. **Context-only divergence (no length change):** the writer's
   `new_mv_context()` (`m2_syntax.cpp:1461`) omits the neighbour-ref comparison
   that §7.11.3 p420 requires before incrementing `NewMvCount`. With one
   reference this is unobservable; with two it silently selects the wrong
   `TileDrlModeCdf[min(idx,2)][ctx]` row — the stream still parses, the MV
   differs.
7. **Writer/decision parameter drift.** `m2_syntax.cpp:1656` hard-codes
   `params.reorder = DrlReorder::kDisabled` and takes
   `max_drl_bits_minus_1` from config, while `m2_frame.cpp:473-475` hard-codes
   `kDisabled` and `4`. Any change to only the header writer (the usual
   integration mistake) makes the validation stack disagree with the decoder with
   no compile-time signal. Similarly, `merge.h:55` caps `kMaxSupportedRefs = 4`
   against the spec's 16 exactly where the temporal/ref-bank code would need it.

## What would move the temporal half out of COMPONENT

Minimum M2-owned edits (each currently outside this audit's write scope), in
dependency order:

1. Keep a per-slot `SavedMvField` (8×8 refs + compressed MVs + order hints)
   instead of resetting `mv_grid_` per frame (`m2_frame.cpp:313`), then call
   `motion_field_estimation()` + `fill_motion_field_sample_gap()` at frame start.
2. `M2Config` flags for `enable_ref_frame_mvs` / `reduced_ref_frame_mvs_mode` /
   `enable_mv_traj` / `enable_refmvbank`, plus the frame header's
   `use_ref_frame_mvs` and `tmvp_sample_step_minus_1` bits.
3. Splice `temporal_scan()` into `StackBuilder::run()` between step 13 and 15,
   and `fill_mvp_from_ref_mv_bank()` at step 20 (needs the per-SB bank hooks).
4. Replace `sort_nearest()` with `drl::reorder_m2_stack()` and map indices with
   `drl::signal_index_for_pre()`, then flip `disable_drl_reorder`.
5. TIP last: `enable_tip`, the frame TIP block, `tip_mode`/`tip_pred_mode`, and an
   actual `build_tip_plan()`-consuming `predict_inter` loop for
   `TIP_FRAME_AS_OUTPUT`.

## Verification

```sh
# Temporal component (motion field, TIP setup, temporal scan, ref bank)
./src/encoder/inter/temporal/build_and_test.sh          # 8,980 checks

# DRL reorder, incl. its own dav2d gate and the legacy-desync negative control
DAV2D=tools/bin/dav2d-macos ./src/encoder/inter/drl_reorder/build_and_test.sh

# Global motion (§7.12.2.1/.2), 6-stream dav2d gate
./src/encoder/inter/gme/build_and_test.sh

# M2 baseline (spatial stack) sample-exact gate
./src/encoder/m2/build_and_test.sh --frames 8

# Compound stack via the multiref gate writer
./src/encoder/m2/multiref/build_and_test.sh
```
