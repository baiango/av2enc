# AV2 v1.0.0 §7.13 — Prediction coverage audit

Scope: **§7.13.1–§7.13.8**, spec pages **447–515** (intra prediction, inter
prediction, palette, CfL, MHCCP). Normative text extracted from
`spec/AV2_Spec_v1.0.0.pdf` (pypdf, PDF page index = spec page − 1).

This is a **code audit**, not a restatement of the component READMEs. Where a
README and the code disagree, the code is reported (stale README/plan claims are
listed at the end).

## Status vocabulary

| Status | Meaning |
|---|---|
| **WIRED** | An encoder path in this repository can emit the syntax and reconstruct it bit-exactly, and a `dav2d` decoder gate (plus recon comparison) exists for that path. Opt-in flags are named in Evidence. |
| **PARTIAL** | The process is implemented and reachable, but only through a constrained/gate-only writer, or only a subset of its signalled forms is exercised. |
| **COMPONENT** | Complete reference/CUDA implementation with unit or parity tests, but **no bitstream path can signal it** (header bit hard-coded 0, or caller absent). |
| **ABSENT** | No implementation. |
| **N/A** | Not an AV2 tool, or the governing sequence bit is 0 everywhere so the process cannot run. |

Important context for the "WIRED" calls: the **M2 encoder is not linked into the
CLI** (`CMakeLists.txt` `av2enc_core` contains no `src/encoder/m2/*` and
`src/main.cpp` has no M2 option). Its gate is a standalone encoder
(`src/encoder/m2/build_and_test.sh`) that produces real IVF streams and compares
them against `dav2d`. It is treated as an encoder path here because it emits
bitstreams, but it is not the shipped CLI.

## Gate inventory (short names used in Evidence)

| Name | What it proves |
|---|---|
| **G-M1-FULL** | `tools/m1_verify/feature_gate.sh:129,151` — `--m1-full-intra --m1-rdo-intra`: pinned IVF MD5 + `dav2d --strict 1`, and with `--dump` an exact per-sample recon compare (`tools/m1_verify/verify.py`). |
| **G-M1-UVFULL** | `feature_gate.sh:130-131,152-153` — `--m1-uv-full` (full chroma mode search), same MD5 + recon check. |
| **G-M1-MATRIX** | `tools/m1_verify/run_matrix.sh` — resolutions × `dc/force:N/full` recon-exact runs (documented in `tools/m1_verify/README.md`). |
| **G-MRL** | `src/encoder/m1/mrl_stream` — 8 streams with `enable_mrls=1`, `dav2d --strict 1`, luma byte-exact (45,056 samples). |
| **G-PAL** | `src/encoder/palette/build_and_test.sh` (2 streams) and `src/encoder/m1/palette_stream/build_and_test.sh` (4 streams), recon diff 0. |
| **G-IBC** | `src/encoder/m1/ibc_stream/build_and_test.sh` — 4 streams, recon exact. |
| **G-M2** | `src/encoder/m2/build_and_test.sh` — 2-frame and 8-frame 1280×720, `dav2d`, Y/U/V exact. |
| **G-M2-COMPOUND** | `src/encoder/m2/multiref/gate` — 3-frame 128×128 and 1280×704, compound P, recon exact. |
| *(none)* | DIP, IBP, MHCCP, warp, BAWP, inter-intra, TIP, OPFL, refine-MV — no decoder-gated stream exists. |

---

## 1. §7.13.2 — Intra prediction

| # | Predictor / rule | § / page | Status | Evidence | Gap |
|---|---|---|---|---|---|
| I1 | Edge construction: `AboveRow`/`LeftCol`, `w+h` extent, unavailable fillers 127 / 129 / 128, corner `-1`, clamping to `maxX`/`maxY` | §7.13.2.1 / p447-451 | **WIRED** | `src/encoder/intra_modes/intra_modes.cpp:433-542`; CUDA twin `src/cuda/intra_modes/intra_predict_device.cuh`; 48,339 CPU checks, 101.06 M GPU samples exact; exercised by every intra block in G-M1-* | Only `MrlIndex = 0`. MRL and DIP use their own edge extents/counts (I4, I7-I10). All fillers are 8-bit shaped (`mid±1`). |
| I2 | `num4AboveRight` / `num4BelowLeft` extension (`min(max, x+w+4*num4-1)`) | §7.13.2.1 / p447-450 | **WIRED** | `intra_modes.cpp:494-521`; caller derives in `m1_still.cpp`; edge/partial-block sweeps in `src/cuda/intra_modes/intra_modes_test.cu` | Caller-supplied. Passing 0 is conformant but shortens edges (documented). |
| I3 | Chroma > 32 disables above-right / below-left | §7.13.2.1 / p447 | **PARTIAL** | `intra_modes.cpp:465-472`; unit-tested in `intra_modes_test.cpp` | Not reachable from any current stream: M1 max chroma block is 32×32 (64×64 luma), and the 128/256-SB still gates use DC chroma. Code path is untested by a decoder. |
| I4 | MRL edge selection: `mrl_index 0..3`, `aboveMrlIndex=0` at SB boundary, negative indices, `AboveSecRow`/`LeftSecCol`, chroma forced 0 | §7.13.2.1 / p449-450 | **PARTIAL** | `src/encoder/mrl/mrl.cpp:263-360`; CUDA `src/cuda/mrl/mrl_device.cuh`; 16,904 emulation checks; **G-MRL** 8/8 streams | Only the `m1/mrl_stream` minimal writer signals it; production writers hard-code `enable_mrls=0` (`m1_headers.cpp:116`, `m2_headers.cpp:143`, `m0_headers.cpp:220`, `m10_headers.cpp:132`). |
| I5 | `CflRef` capture + `get_cfl_luma_sample` + `cfl_ref_luma_avail` (CFL_MULTI only) | §7.13.2.1 / p448-449 | **COMPONENT** | `src/encoder/cfl/mhccp/mhccp_common.h` (`capture_refs`, `cfl_luma_sample`); 660,691 checks + 16,803 emulation checks | `enable_mhccp=0` in every writer. |
| I6 | Basic intra / Paeth | §7.13.2.2 / p451 | **WIRED** | `intra_modes.cpp:622-651`; in the 61-candidate luma search (`m1_still.cpp:640`); **G-M1-FULL** | None for 8-bit. |
| I7 | DIP dispatch (`useDip && plane==0`) | §7.13.2.3 / p451 | **COMPONENT** | `src/encoder/dip/dip.cpp:290-300`; `src/cuda/dip/dip.cu`; 3,082,063 CPU checks, 3,612 emulation cases | `enable_dip=0` in all sequence headers (`m1_headers.cpp:114`, etc.); no `dip_mode_info` in `m1::syntax`. |
| I8 | DIP 11 features + `dip_avg` (`n>>2`, `(t+(down>>1))/down`) | §7.13.2.4 / p452 | **COMPONENT** | `dip.cpp:160-204`; features tested for 8/10/12-bit and all availability combos | Same as I7. |
| I9 | DIP transform (`Dip_Weights`, `Round2(c,10)`, `Clip1`, transpose) | §7.13.2.5 / p452 | **COMPONENT** | `dip.cpp:206-230`; `dip_weights_data.inc` shared CPU/CUDA; table checksum + 4,224 values pinned | Same as I7. |
| I10 | DIP resample 8×8 → w×h | §7.13.2.6 / p453 | **COMPONENT** | `dip.cpp:232-275`; literal-transcription test | Same as I7. Intended for M4b; no stream claim. |
| I11 | Angle mapping: `Mode_To_Angle + angle_delta*3` (+ `Mrl_Index_To_Delta`) | §7.13.2.7 / p454 | **WIRED** | `intra_modes.cpp:22,310-333`; 61 luma candidates (`m1_still.cpp:640`); signalled via `m1_syntax.cpp:105-114,1452`; **G-M1-FULL** | MRL delta only in I4. |
| I12 | Wide-angle mapping (WAIP) for rectangular blocks | §7.13.2.7 p454 (WAIP §5.20.7.29) | **WIRED** | `intra_modes.cpp:316-331`; `src/cuda/intra_modes/intra_modes.h:205-219`; wide-angle goldens in both tests; **G-M1-FULL** | Applied only when `is_inter==0`; the intra API has no `is_inter` input (see I28). |
| I13 | Edge-filter orchestration: corner, per-edge filter types, `needRight/needBottom`, `numPx`, strength selection, apply order | §7.13.2.7 / p454-456 | **WIRED** | `intra_modes.cpp:710-739`; **G-M1-FULL** with `enable_intra_edge_filter=1` (default) | The IBP variant (separate filter types, inverted angle adjustments, corner rule) lives only in the IBP component (I27). |
| I14 | Single directional, luma IDIF: 4-tap `Dr_Interp_Filter[shift]`, `Round2(...,7)`, `Clip1` | §7.13.2.8 / p457-459 | **WIRED** | `intra_modes.cpp:137-190` (`kDrInterpFilter` at :37); independent CPU reference in `src/cuda/intra_modes/intra_modes_test.cu`; **G-M1-FULL** | 8-bit only in the wired path. |
| I15 | Single directional, chroma bilinear: 2-tap, `Round2(...,5)` | §7.13.2.8 / p457-458 | **WIRED** (opt-in) | `enable_idif = plane==0` at `intra_modes.cpp:142`; **G-M1-UVFULL** | Default M1 chroma is DC/CfL; directional chroma only under `--m1-uv-full`. |
| I16 | Edge extension / `minBase` / `maxBase` clamp, 90°/180° row/column copy | §7.13.2.8 / p456-458 | **WIRED** | `intra_modes.cpp:143-160,253-262`; goldens for 45°/135°/wide-angle/above-right/below-left | `MrlIndex=0` constants; MRL has its own copy (I26). |
| I17 | `ibp_weights` (16×16) + `resolve_divisor` | §7.13.2.9 / p459 | **COMPONENT** | `src/encoder/ibp/ibp.cpp:357-374`, `ibp_tables.h`; 56,234,221 checks | `enable_ibp=0` everywhere (`m1_headers.cpp:120`). |
| I18 | DC intra: both / left-only / above-only / none, `approx_divide`, `Clip1(avg)`, `Round2(sum,log2)` one-sided | §7.13.2.10 / p459-460 | **WIRED** | `intra_modes.cpp:544-580`; DC is the M0/M1 default; 48,339 checks | `approx_divide` uses `resolve_divisor` + `Div_Lut` (I24 helper). |
| I19 | DC intra **subsampled** (`stepW/stepH = 2` when extent > 32) | §7.13.2.11 / p460-461 | **ABSENT** | No code: repo-wide search finds only the note in `src/encoder/intra_modes/README.md:42` | Would be needed for CfL on a >32-sample chroma block. Unreachable today (see I3); latent desync if such a block is ever coded with CfL. |
| I20 | IBP DC post-filter (`Ibp_Weights[5][16]`) | §7.13.2.12 / p461-462 | **COMPONENT** | `ibp.cpp:311-350,376`; DC/IBP sweep in `ibp_test.cpp` | Same as I17. |
| I21 | SMOOTH / SMOOTH_V / SMOOTH_H | §7.13.2.13 / p462-463 | **WIRED** | `intra_modes.cpp:582-620`; **G-M1-FULL** | 8-bit wired path only. |
| I22 | Filter corner (`5/6/5`, `Round2(...,4)`, `w+h>=24`) | §7.13.2.14 / p463 | **WIRED** | `intra_modes.cpp:130-136,719-723` | Corner rule differs for IBP (I27) and is absent there in production. |
| I23 | Filter type above / left (neighbour smooth flags, OR for non-IBP) | §7.13.2.15-.16 / p463-464 | **WIRED** | `intra_modes.cpp:710-718`; flags derived in `m1_still.cpp:635-638`; **G-M1-FULL** | Caller must use chroma MI coordinates under `SHARED_PART`; M1 does. |
| I24 | Edge filter strength selection (buckets / thresholds) | §7.13.2.17 / p464-465 | **WIRED** | `intra_modes.cpp:336-399`; both filter types and every bucket boundary tested | — |
| I25 | 5-tap edge filter on an **unfiltered snapshot**, `(s+8)>>4`, strengths 1..3 | §7.13.2.18 / p465 | **WIRED** | `intra_modes.cpp:401-427` (snapshot `edge[]`); fix recorded in `docs/negative-results.md:27` | Was in-place before commit `3bca233`; the CUDA intra README still calls it a blocker (stale, see below). |
| I26 | MRL secondary-line blend (`(pred+pred2+1)>>1`) | §7.13.2.7 / p455 | **PARTIAL** | `src/encoder/mrl/mrl.cpp:465-474`; **G-MRL** covers `mrl_sec_index=1` | Only via the gate-only MRL writer. |
| I27 | IBP two-angle blend + IBP edge-filter variant | §7.13.2.7 / p454-456 | **COMPONENT** | `src/encoder/ibp/ibp.cpp:610-642` (blend), `:486` (applicability) | Not signalled; `applyIbp` is decoder-forced whenever `enable_ibp=1`, so flipping that bit without switching predictors would desync every non-4×4 directional block. |
| I28 | Inter-intra intra predictor variant (`is_inter==1`: no WAIP) | §7.13.2.7 p454 / §7.13.3.29 p506 | **COMPONENT** | `src/encoder/ibp/ibp.h` carries `is_inter`; `src/encoder/inter/advanced/compound.cpp` builds II predictions | The main `intra` API has no `is_inter` field. Harmless today because II modes are DC/V/H/SMOOTH and 90°/180° are WAIP-invariant. |
| I29 | Plane-select of angle delta: `AngleDeltaY` (luma) vs `AngleDeltaUV` (chroma, derived = `AngleDeltaY` iff `UVMode==YMode`, else 0) | §7.13.2.7 p454 / §6.19.5.6 | **WIRED** | `m1_still.cpp:1286-1310` (derivation + per-candidate delta); `intra_modes.cpp:142` (plane selects IDIF); **G-M1-UVFULL** | No separate UV delta syntax exists, so the encoder must not search one; it does not. |

---

## 2. §7.13.3 — Inter prediction

| # | Predictor / rule | § / page | Status | Evidence | Gap |
|---|---|---|---|---|---|
| N1 | Block inter prediction: two-pass 8-tap, 16 phases, `Clip3(first,last)` reference bounds, `Round2` per pass, `Clip1` | §7.13.3.18 / p486-490 | **WIRED** | `src/encoder/inter/mc/mc.cpp:240-320`, `mc_core.h`; `mc_subpel_tables.h` generated from the PDF (`gen_subpel_tables.py`, `--check-tables`); **G-M2** 8-frame 720p exact | Unscaled references only; see N5. |
| N2 | Interpolation-filter signalling: 6 filter families + frame switchable | §7.13.3.18 / p487-490 | **PARTIAL** | All 6 tables in `mc_subpel_tables.h`; but `m2_headers.cpp:273-274` always writes `is_filter_switchable=0`, `interpolation_filter=EIGHTTAP`, and `m2_frame.cpp:488-509` always calls `kEighttap` | SMOOTH / SHARP / 4-tap / 4-tap-smooth are component-only. No stream can signal a non-EIGHTTAP filter. |
| N3 | Four-tap downgrade when a pass extent ≤ 4 | §7.13.3.18 / p487-488 | **WIRED** | `mc.cpp:266-267` (`mc_pass_filter`), `mc_core.h:70`; 4×4 chroma / 8×8 luma in G-M2 full-tree runs | — |
| N4 | Rounding variables `InterRound0=3`, `InterRound1=11` (single) / `7` (compound), `InterPostRound` | §7.13.3.16 / p484 | **WIRED** | `src/encoder/inter/mc/mc_types.h:32-33`; `src/encoder/inter/advanced/advanced_common.h:65-67`; compound path `src/encoder/m2/multiref/compound_predict.cpp`; **G-M2**, **G-M2-COMPOUND** | Compound rounding only proven in the gate writer. |
| N5 | MV scaling: 1/8 luma, prescaled 1/16, scaled references (`xScale/yScale`, `stepX/stepY`) | §7.13.3.17 / p485-486 | **PARTIAL** | `mc.cpp` derives 1/8 locations; 1/16 prescale only in `src/encoder/inter/temporal/*`; scaled-ref branch in `src/encoder/inter/advanced/warp_common.h:276,544-575` | Wired MC rejects `is_scaled()` refs (`src/encoder/inter/mc/README.md` scope). Superres/scaled references cannot be encoded. |
| N6 | Reference-area clipping (`RefFirst/RefLast`, `useRefArea`) | §7.13.3.4-.5 / p471-472 | **PARTIAL** | Base clip to the MI-aligned plane in `mc.cpp`; `compute_ref_area` in `src/encoder/inter/temporal/refinemv.cpp`; candidate-ref-area in `multiref` | Only the temporal/refinemv components use `useRefArea`; wired MC always clips to the full plane. |
| N7 | TIP `refFrames` selection + `Tip_Weighting_Factor` | §7.13.3.1 / p465-468 | **COMPONENT** | `src/encoder/inter/temporal/tip.cpp`; 8,980 checks; table pinned | `enable_tip=0`; M2 has no TIP syntax. |
| N8 | Predict TIP (subblock loop, `tipSize` 8×8/16×16, `RefineMvs` fill) | §7.13.3.2 / p468-469 | **COMPONENT** | `temporal/tip.cpp`, `motion_field.cpp` (`build_tip_plan`) | No stream. |
| N9 | Predict refine MV + `offset_refinemv` | §7.13.3.3 / p469-471 | **COMPONENT** | `temporal/refinemv.cpp` (`offset_refinemv`, `refine_mv_syntax`) | `enable_refinemv=0`. |
| N10 | Search refine MV (24 neighbours, SAD gating, RefArea) | §7.13.3.6 / p472-473 | **COMPONENT** | `temporal/refinemv.cpp`; search/gating tests | No stream. |
| N11 | Predict block orchestration (optflow / warp / ref-area dispatch) | §7.13.3.7 / p473-475 | **COMPONENT** | `temporal/opfl.cpp`, `refinemv.cpp` | M2 uses a direct `predict_plane`, not this dispatcher. |
| N12 | Predict optflow block (subblock size, SAD shortcut, OpflMvs) | §7.13.3.8 / p475-476 | **COMPONENT** | `temporal/opfl.cpp` | `enable_opfl_refine=REFINE_NONE`. |
| N13 | Get optflow based MV + distance reduction | §7.13.3.9 / p476-477 | **COMPONENT** | `temporal/opfl.cpp`, `temporal_math.h` (`opfl_scale_distances`) | No stream. |
| N14 | Optflow difference / gradient / `compute_opfl_mv` (LS solve, ±16 clamp) | §7.13.3.10-.12 / p477-480 | **COMPONENT** | `temporal/opfl.cpp`; zero-diff + solve tests | No stream. |
| N15 | Predict subblock (per-list loop, warp dispatch `useWarp` 1/2, interp selection) | §7.13.3.14 / p481-482 | **COMPONENT** | `temporal/*` orchestration; `multiref/compound_predict.cpp` for the compound subset | Not the wired M2 path. |
| N16 | Compound AVERAGE + CWP (`Cwp_Weighting_Factor`, `(cw*p0+(16-cw)*p1)`) | §7.13.3.14 / p483 | **PARTIAL** | `src/encoder/inter/advanced/advanced_common.h:123-125`, `compound_common.h`, `multiref/compound_predict.cpp`; **G-M2-COMPOUND** (AVERAGE + CWP forced) | Only via the 3-frame gate writer; `reference_select=0` and `enable_cwp=0` in the M2 core (`m2_headers.cpp:172`). |
| N17 | Implicit mask blend `get_mask` (ref on/off-screen weights 0/1/2) | §7.13.3.14 / p483-484 | **PARTIAL** | `multiref/compound_predict.cpp` (`implicit_mask_eligible`), `advanced_common.h`; unit-tested | The gate forces `GLOBAL_GLOBALMV`, which **disqualifies** the implicit mask, so no decoded stream exercises `get_mask`. |
| N18 | Wedge mask (`WedgeMasks`, sharp/smooth boundary, 68 codebook entries, sign flip) | §7.13.3.27 / p503-506 | **PARTIAL** | `advanced_common.h:304-330` + generated `tables/wedge_tables.h`; all 68×2 masks vs literal init; **G-M2-COMPOUND** uses index 22 | One index/sign in the gate stream; the rest are unit-tested only. |
| N19 | Difference-weight mask (`38 + diff/16`, `mask_type`) | §7.13.3.28 / p505-506 | **PARTIAL** | `compound_common.h:104` (`diffwtd_mask_value`), both `mask_type` values tested | Gate uses `mask_type=0` only. |
| N20 | Intra-mode variant mask (`Ii_Weights_1d[128]`) | §7.13.3.29 / p506 | **COMPONENT** | `tables/wedge_tables.h`, `inter/advanced/compound.cpp` | Inter-intra is not signalled (N31). |
| N21 | Mask blend (`Round2(m*p0+(64-m)*p1, 6+InterPostRound)`) | §7.13.3.30 / p506-507 | **PARTIAL** | `compound_common.h:151-162`; **G-M2-COMPOUND** (wedge + diffwtd) | Chroma subsampling of the luma mask is unit-tested, not gate-isolated. |
| N22 | Block warp, 8×8, variable phase, `Warped_Filters` | §7.13.3.19 / p490-492 | **COMPONENT** | `inter/advanced/warp.cpp`, `warp_common.h`; literal parity | `seq_enabled_motion_modes=0`; no warp stream. |
| N23 | Extended warp, 4×4 fixed phase, scaled and unscaled | §7.13.3.20 / p492-495 | **COMPONENT** | `warp_common.h` (`ext_warp_*`), `warp.cpp`; parity for both branches | No stream. |
| N24 | Setup shear (alpha/beta/gamma/delta, `warpValid`) | §7.13.3.21 / p495 | **COMPONENT** | `warp_common.h` (`setup_shear`); literal parity | No stream. |
| N25 | Warp estimation (least squares, `reduce_warp_model`, `get_warp_translation`) | §7.13.3.23 / p496-498 | **COMPONENT** | `warp.cpp` (`estimate_local_warp`), `warp_common.h`; random-model parity | Missing `find_warp_samples` / warp candidate stack (external). |
| N26 | Extend warp estimation (neighbour model extension) | §7.13.3.24 / p498-500 | **COMPONENT** | `warp.cpp`; `Default_Warp_Params` path | `ExtendDeltaRow/Col` plumbing is caller-owned and not integrated. |
| N27 | BAWP (fitted / explicit / chroma, conformance checks) | §7.13.3.25 / p500-503 | **COMPONENT** | `inter/advanced/bawp.cpp`, `bawp_common.h`; 40 random jobs + explicit rules | `enable_bawp=0`; gate impossible without M2 edits (`src/encoder/m2/advanced/README.md`). |
| N28 | Morphological prediction (`morphPred` / IBC template) | §7.13.3.26 / p503 | **COMPONENT** | `bawp.cpp:66` (`morph_pred` selects `job.cur`), `inter/advanced/bawp.h:28` | IBC explicitly excludes `morph_pred` (`src/encoder/ibc/README.md:70,198`). |
| N29 | Save sub-PU size (`SubPuSize`, `SubPuColBase/RowBase`) | §7.13.3.15 / p484 | **ABSENT** | Only the deblock core carries a `sub_pu_size` field (`src/encoder/filters/deblock/deblock_core.h:118-127`) | Only used when `enable_df_sub_pu=1`; `m2_headers.cpp` writes 0. |
| N30 | OBMC | not in AV2 | **N/A** | `spec/tables/constants.csv:202` (`MOTION_MODES,5`); `inter/advanced/obmc_common.h` documents the removal | Encoder-only experiment in `obmc.*`; never signalled. |
| N31 | Inter-intra (`COMPOUND_INTRA`, `interintra_mode`, `wedge_interintra`, blend) | §7.13.3.29-.30 / p506 | **COMPONENT** | `src/encoder/m2/advanced/advanced_decision.cpp:235`, `advanced_syntax.cpp` (1,964,773 entropy round-trip checks); `inter/advanced/compound.cpp` | M2 cannot signal `seq_enabled_motion_modes[INTERINTRA]`; no `dav2d` stream. |

---

## 3. §7.13.4 — Palette prediction

| # | Predictor / rule | § / page | Status | Evidence | Gap |
|---|---|---|---|---|---|
| P1 | Palette reconstruction `CurrFrame[0][startY+i][startX+j] = colors[ColorMapY[y*4+i][x*4+j]]` | §7.13.4 / p507 | **PARTIAL** | `src/encoder/palette/palette_core.h:193-206` (`palette_predict_tx`); **G-PAL** 2+4 streams, recon diff 0 | Luma only; the gate writers are minimal 64×64-SB writers (`NONE/HORZ/VERT`), not the M1/M2 partition tree. |
| P2 | Luma-only palette; chroma uses normal intra (DC/CfL) after palette luma | §7.13.4 / p507 + §5.20.8.1 | **PARTIAL** | `palette_core.h:19-23`; `chroma_rule()` asserted in `palette_stream_test.cpp`; **G-PAL** chroma is DC | AV2 v1.0.0 has no chroma palette syntax; the production M1 writer has no palette path. |
| P3 | Transform-skip palette block (`eob=0` on Y/U/V), edge replication for partial blocks | §7.13.4 p507 + §5.20.8.4 | **PARTIAL** | `palette.cpp` (`palette_residual`), `scc_stream.cpp`; **G-PAL** 112/134-byte tiles exact | Always `TX_MODE_LARGEST`/single transform in the gate; 8-bit 4:2:0 only. |

---

## 4. §7.13.5 — CfL (predict chroma from luma)

| # | Predictor / rule | § / page | Status | Evidence | Gap |
|---|---|---|---|---|---|
| C1 | Luma average: `stepW/H`, `lumaAbove[]`, `lumaLeft[]`, `prevSample` replication, SB-top clamp | §7.13.5 / p508-510 | **WIRED** | `src/encoder/cfl/cfl.cpp:195-279`; CPU 13,423 checks, CUDA 1,022,200 samples exact; default M1 chroma path (`m1_still.cpp:1094-1104,1330-1337`); **G-M1-*** recon exact | `coded_width/height` and chroma-MI availability are caller inputs; documented contract in `src/encoder/cfl/README.md`. |
| C2 | `L` array (3 fractional bits) with `Cfl_Filters_420` / `Cfl_Filters_422` (all four downsampling filters) | §7.13.5 / p509-511 | **WIRED** | `cfl.cpp:367-385`; filter tables entry-checked; CUDA kernel same header | All four filters implemented, but only index 0 is signalled (C3). |
| C3 | `cfl_ds_filter_index` selection (0..3) | §7.13.5 p508 / §5.4.5 | **PARTIAL** | Writers hard-code `cfl_ds_filter_index=0` (`m1_headers.cpp:118`); `m1_still.cpp:1217` sets `ds_filter=0` | Filters 1/2/3 are tested in `ds_filters/` and the CUDA Cfl test but cannot appear in a stream. |
| C4 | Explicit alpha: sign, magnitude 1..8 eighths, joint signs, `cfl_alpha_u/v` | §7.13.5 p511 / §5.20.7.32 | **COMPONENT** | `cfl.cpp:118-165,430-505` (`search_alpha`, exhaustive), `cfl_common.h`; 13,423 checks | The M1 search only calls `derive_implicit_alpha`; `m1_syntax.cpp:794-795` always emits `cfl_index = CFL_DERIVED_ALPHA`. Explicit `cfl_index=0` is never coded. |
| C5 | Derived alpha (`implicitAlpha`, `resolve_division(nor,der,8)`) | §7.13.5 / p510-511 | **WIRED** | `cfl.cpp:281-364`; used for both U and V in `m1_still.cpp:1100-1104`; **G-M1-*** | Zero-variance → 0 (DC) per spec. |
| C6 | Prediction precision: `Clip1(dc + Round2Signed(alpha*(L-lumaAvg),11))` | §7.13.5 / p511 | **WIRED** | `cfl.cpp:367-382` (`kScaleShift = 11`); CUDA `src/cuda/cfl/cfl.cu`; recon-exact gates | `alpha` is Q8 (`CflAlpha*32`), matching spec. |

---

## 5. §7.13.6–§7.13.8 — MHCCP

| # | Predictor / rule | § / page | Status | Evidence | Gap |
|---|---|---|---|---|---|
| M1 | MHCCP prediction (`vec[0..2]`, `mul_fixed32_adapt`, `Clip1`) | §7.13.6 / p511-512 | **COMPONENT** | `src/encoder/cfl/mhccp/mhccp_common.h` (`predict_sample`), `mhccp_gpu.cu`; 660,691 CPU checks + 16,803 emulation | `enable_mhccp=0` in all writers (`m1_headers.cpp:119`); `m1::syntax` has no `cfl_mhccp`/`cfl_index=CFL_MULTI` writer. |
| M2 | Derive multi params (3×3 normal equations, `CflRef` statistics, Q16 normalisation) | §7.13.7 / p512-514 | **COMPONENT** | `mhccp_common.h:496-575`; real-block parity (237 + 18,636 blocks/plane) | Same as M1. |
| M3 | Gaussian elimination + `get_division_scale_shift` + `Division_Pow2_{O,B,W}` | §7.13.8 / p514-515 | **COMPONENT** | `mhccp_common.h:165-260,576-610`; 20,000 random `mul_fixed32_adapt` cases | Same as M1. |
| M4 | MHCCP syntax: `cfl_mhccp`, `cfl_index`, `cfl_mh_dir` | §5.20.7.32 / §6.19.7.25 | **COMPONENT** | `mhccp_syntax.cpp`; every sign/direction round-trips through the normative §8.2 coder | No stream: `enable_mhccp=0`; the `is_cfl` CDF is only coded for `cflAllowed \|\| is_mhccp_allowed()` in the decoder, so enabling the bit alone would desync the UV-mode syntax. |

---

## 6. Stale documentation found (code wins)

- `src/cuda/intra_modes/README.md` blocker #1 claims the CPU edge filter is
  in-place and that CPU/CUDA reconstruction cannot match. The CPU code now
  snapshots the edge (`src/encoder/intra_modes/intra_modes.cpp:401-427`, commit
  `3bca233`) and `docs/negative-results.md:27` records the fix. The blocker is
  stale.
- `src/encoder/m2/README.md` "Not implemented yet" lists compound, warp,
  inter-intra, BAWP, and multiple references; those now exist as gated/component
  layers (`m2/multiref`, `m2/advanced`) even though the M2 core still pins the
  sequence flags off.
- `docs/validation.md` "Current scaffold status" still says M0 emits placeholder
  payloads; M0/M1/M2 streams are decoder-gated now.
- `src/encoder/cfl/README.md` "Non-CfL notes" item 2 says directional/smooth
  prediction diverges when forced; the `--m1-uv-full` and
  `--m1-full-intra --m1-rdo-intra` gates are now recon-exact, and the CUDA
  chroma header states directional UV is bit-exact.

---

## 7. Counts

| Status | Count |
|---|---:|
| WIRED | 23 |
| PARTIAL | 15 |
| COMPONENT | 32 |
| ABSENT | 2 |
| N/A | 1 |
| **Total audited rules** | **73** |

Breakdown: intra 16 W / 3 P / 9 C / 1 A = 29; inter 3 W / 8 P / 18 C / 1 A /
1 N/A = 31; palette 3 P; CfL 4 W / 1 P / 1 C = 6; MHCCP 4 C.

Interpretation: the **encoder's intra predictor is genuinely wired and
decoder-gated** for the tools it actually signals (DC, V/H/directional with
angle deltas, wide-angle, IDIF/bilinear, smooth, Paeth, corner + edge filter),
and its **single-reference inter MC is gated exact**. Everything from the M4
tool set — DIP, MRL, IBP, MHCCP, warp, BAWP, inter-intra, TIP/OPFL/refine-MV,
explicit CfL alpha — is a tested *component* with no bitstream path. Palette,
compound/CWP/wedge/diffwtd and MRL sit in between: they have working
purpose-built stream writers with `dav2d` recon-exact gates, but no production
encoder or CLI route.

## 8. Modes the encoder can signal but not reconstruct identically

**No currently emitted mode is known to desync.** The default and opt-in gated
paths (M1 DC/CfL, `--m1-full-intra`, `--m1-uv-full`, palette/MRL/IBC minimal
streams, M2 single-ref, M2 compound gate) all compare byte-exact against
`dav2d`. The following are latent hazards, ordered by risk:

1. **CfL on a chroma block larger than 32 samples** — §7.13.2.11 (DC
   subsampled) is ABSENT (I19). A future 128/256-SB intra coding unit with a
   64×64 chroma block and `UV_CFL_PRED` would make the decoder use a
   subsampled DC average while the encoder uses full DC (`intra_modes.cpp:544`).
   Today the maximum chroma block is 32×32, and the sb128/sb256 still gates use
   non-CfL DC chroma, so it cannot be signalled.
2. **`enable_ibp = 1`** — the sequence bit exists and is written as 0
   (`m1_headers.cpp:120`). Flipping only that bit would make the decoder apply
   IBP edge filtering (separate filter types, inverted angle adjustments,
   different corner rule) to every non-4×4 directional block, plus the
   two-angle blend where `useIBP` holds. The production predictor is the
   non-IBP `intra` module, so recon would diverge. The IBP predictor exists but
   is not dispatchable (I27).
3. **`enable_mrls = 1`** without the MRL block syntax/MRL-aware predictor —
   the decoder would read `mrl_index`/`mrl_sec_index` after every directional
   Y mode. The stock writer has no such fields (`m1_syntax.h` `BlockSyntax`) and
   the stock predictor fixes `MrlIndex = 0`.
4. **`enable_mhccp = 1`** or `cfl_index = CFL_MULTI` — the UV-mode bit layout
   changes (`cfl_mhccp` before `is_cfl`/`cfl_index`), and prediction must switch
   to §7.13.6. Neither is wired in `m1_syntax.cpp`/`m1_still.cpp`.
5. **`enable_dip = 1`** — the decoder then expects `use_dip`/`dip_transpose`/
   `dip_mode` after Y mode for eligible DC blocks; the stock writer emits none
   and the stock predictor is ordinary DC.
6. **`allow_screen_content_tools/allow_palette`** — a palette block is only
   reachable through the gate writer; the stock M1 writer has no
   `palette_mode_info`/`palette_tokens` call, so enabling the flag alone would
   change the intra-mode bit layout.
7. **Inter-intra / warp / BAWP / TIP / OPFL / refine-MV** — sequence bits exist
   (`seq_enabled_motion_modes`, `enable_bawp`, `enable_tip`, `enable_opfl_refine`,
   `enable_refinemv`) but are hard-coded 0; flipping any of them adds syntax the
   M2 tile writer cannot emit and prediction the M2 predictor does not run.
8. **`--m1-uv-full` + non-CfL chroma modes** is gated exact at 64×64/640×360,
   but only with `cfl_ds_filter_index=0` and 8-bit. Non-zero ds filters and
   directional/CfL chroma in 4:2:2/4:4:4 are component-tested only; the
   existing chroma-format gate streams (`src/encoder/format/chroma`) use plain
   DC chroma.

Non-prediction mismatches already known and disabled (not counted above):
DRL reordering (`disable_drl_reorder=1`, `m2_headers.cpp:159`), `skip_mode`
(needs `reference_select`), scaled references, and tiled deblocking
(`docs/negative-results.md:54`).

