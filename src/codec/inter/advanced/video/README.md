# M2 advanced inter syntax — warp, inter-intra, BAWP

Exact bit-level syntax, CDFs and encoder decisions for the three AV2 advanced
inter tools that the M2 baseline disables. **Component only**: it does not edit
`src/encoder/m2/` core, `src/encoder/inter/advanced/`, `src/main.cpp` or
`CMakeLists.txt`. Integration calls are listed below and are deliberately
explicit.

AV2 v1.0.0 (`spec/AV2_Spec_v1.0.0.pdf`, commit `38f28e7`) referenced throughout.

## Scope

| Tool | Symbols | Status |
|---|---|---|
| Warped motion / affine | `is_warp`, `warp_mv`, `warp_idx`, `warpmv_with_mvd`, `warp_delta_precision`, `warp_delta_param_{low,high,sign}` | writer + reader + decision |
| Motion modes | `use_extend_warp`, `use_local_warp` | writer + reader |
| Inter-intra | `inter_intra`, `warp_inter_intra`, `interintra_mode`, `wedge_interintra`, `wedge_{quad,angle,dist1,dist2}`, `wedge_sign` | writer + reader + search mapping |
| BAWP | `use_bawp`, `explicit_bawp`, `explicit_bawp_scale`, `use_bawp_chroma` | writer + reader + decision |
| Sequence bits | `seq_enabled_motion_modes[1..4]`, `seq_frame_motion_modes_present_flag`, `enable_six_param_warp_delta`, `enable_masked_compound`, `enable_bawp`, `enable_cwp`, `enable_imp_msk_bld` | exact `BitWriter` helpers |
| Frame bits | `frame_enabled_motion_modes[1..4]`, `allow_bawp`, `allow_warpmv_mode` | exact `BitWriter` helpers |

**OBMC is not implemented and cannot be signalled.** AV2 `MOTION_MODES` is
`SIMPLE, INTERINTRA, LOCALWARP, DELTAWARP, EXTENDWARP`; there is no OBMC tool.
Nothing here writes or reads an OBMC symbol.

The composite block writer is single-reference (the M2 frame layout,
`NumTotalRefs = 1`). Compound/wedge-compound and CWP are outside this module;
the inter-intra path and the `WARP*` modes are the single-reference cases the M2
stream can carry.

## Files

```text
src/encoder/m2/advanced/
  advanced_cdf_tables.h       generated §9.2/§9.3 tables (22 CDF arrays)
  generate_advanced_tables.py regenerates the header from spec/tables/*.json
  advanced_syntax.h/.cpp      CDF state, exact staged writers/readers,
                              full ordered block writer/reader, frame/seq bits
  advanced_decision.h/.cpp    warp/inter-intra/BAWP decisions and helpers
  advanced_test.cpp           1,964,773 checks; every symbol round-trips through
                              the normative av2::Decoder
  build_and_test.sh           standalone build/run/test
  README.md                   this file
```

Reused unchanged:

* `src/encoder/inter/advanced/warp.*`, `warp_common.h` — block/extended warp,
  `reduce_warp_model`, `setup_shear`, `get_warp_translation`, estimation.
* `src/encoder/inter/advanced/compound.*`, `compound_common.h` — inter-intra
  search, `intra_mode_mask_value`, wedge masks.
* `src/encoder/inter/advanced/bawp.*`, `bawp_common.h` — fitted/explicit BAWP,
  chroma `BawpAlpha`.
* `src/encoder/inter/advanced/tables/wedge_tables.h` — `Wedge_Bits`,
  `Wedge_Angle_Dist_2_Index`.
* `src/entropy/av2/av2_entropy.{h,cpp}` — normative §8.2 coder. The tests use
  `av2::Decoder`, not a private inverse.

## Exact symbol/CDF mapping

| Syntax (clause) | Writer/reader | CDF |
|---|---|---|
| `is_warp` (§5.20.7.6) | `write_is_warp` / `read_is_warp` | `TileIsWarpCdf[WarpMvCount]` |
| `warp_mv` | `write_warp_mv` / `read_warp_mv` | `TileWarpMvCdf` |
| `warp_idx` loop | `write_warp_idx` / `read_warp_idx` | `TileWarpIdxCdf[idx]` |
| `warpmv_with_mvd` | `write_warpmv_with_mvd` / `read_warpmv_with_mvd` | `TileWarpWithMvdCdf` |
| `warp_delta_precision` (§5.20.7.7) | `write_warp_delta` / `read_warp_delta` | `TileWarpDeltaPrecisionCdf[MiSize]` |
| `warp_delta_param_low` | same | `TileWarpDeltaParamLowCdf[idx==3 || idx==4]` |
| `warp_delta_param_high` | same | `TileWarpDeltaParamHighCdf[idx==3 || idx==4]` |
| `warp_delta_param_sign` | same | `TileWarpDeltaParamSignCdf` |
| `use_extend_warp` (§5.20.7.14) | `write_motion_mode` / `read_motion_mode` | `TileUseExtendWarpCdf[ctx]` |
| `use_local_warp` | same | `TileUseLocalWarpCdf[ctx]` |
| `inter_intra` / `warp_inter_intra` (§5.20.7.15) | `write_inter_intra` / `read_inter_intra` | `TileInterIntraCdf[Size_Group]`, `TileWarpInterIntraCdf[Size_Group]` |
| `interintra_mode` | same | `TileInterIntraModeCdf[Size_Group]` (n = 4) |
| `wedge_interintra` | same | `TileWedgeInterIntraCdf` |
| `wedge_quad` / `wedge_angle` | `write_wedge_mode` / `read_wedge_mode` | `TileWedgeQuadCdf`, `TileWedgeAngleCdf[quad]` |
| `wedge_dist1` / `wedge_dist2` | same | `TileWedgeDist1Cdf` (n = 4), `TileWedgeDist2Cdf` (n = 3) |
| `wedge_sign` `L(1)` | `write_wedge_sign` / `read_wedge_sign` | raw bit |
| `use_bawp` (§5.20.7.6) | `write_bawp_symbols` / `read_bawp_symbols` | `TileUseBawpCdf` |
| `explicit_bawp` | same | `TileExplicitBawpCdf[ctx]` |
| `explicit_bawp_scale` | same | `TileExplicitBawpScaleCdf` |
| `use_bawp_chroma` | same | `TileUseBawpChromaCdf` |

Warp delta arithmetic (`read_warp_delta_param()`):

```
v = warp_delta_param_low
if (highPrecision && v == 7) v += warp_delta_param_high
if (v != 0) v = sign ? -v : v
delta = v << (WARP_DELTA_STEP_BITS + 1 - highPrecision)   // 1024 or 2048
```

`apply_warp_delta()` performs the exact `read_warp_delta()` sequence: start from
`WarpParamStack[RefWarpIdx]`, zero `p0/p1`, add `p2..p5` (or derive
`p4=-p3, p5=p2` for the four-parameter form), `reduce_warp_model()`, then
`get_warp_translation()`. A six-parameter delta is legal only when
`enable_six_param_warp_delta && RefWarpIdx == 1`.

Wedge mode is the exact inverse of `Wedge_Angle_Dist_2_Index`:
`wedge_quad = angle/5`, `wedge_angle = angle%5`, and `wedge_dist2 = dist-1` when
`angle >= 10 || angle == 5 || angle == 0` (the `WEDGE_0`/`WEDGE_90`/second-half
rules), otherwise `wedge_dist1 = dist`.

## Neighbour rules (§7.11.2/§7.11.4)

`compute_mode_context()` returns:

* `warp_mv_count` — `WarpMvCount`, the `is_warp` context. It scans the four
  §7.11.2 points through `scan_point_warp_ctx()`, including the
  `deltaCol -= MiCol & 1` adjustment at the top superblock row, and counts
  neighbours whose `MotionModes >= LOCALWARP` and whose
  `RefFrames[..][0]/[1]` matches `RefFrame[0]`.
* `warp_sample_found[2]` — used by `AllowedMotionModes[LOCALWARP/EXTENDWARP]`.
* `extend_delta_row/col` — the first matching candidate, as the spec records it.
* `local_warp_ctx` / `extend_warp_ctx` — the `use_local_warp` and
  `use_extend_warp` CDF contexts from `NPos` (`add_neighbor()` order:
  left-bottom, above-right, left, above; only the first two in the same
  superblock row).
* `npos_count` / `npos[][]` — the decoded neighbour positions.

The grid is a caller-owned `NeighborGrid` of `NeighborMi` (written flag,
`is_inter`, `ref[2]`, `y_mode`, `motion_mode`, `interp_filter`). The M2 writer
must fill the same values it just coded, including `MotionModes` for every leaf.

## Decision API

```cpp
// --- warp ---------------------------------------------------------------
adv::WmParams wm = adv::default_wm_params();           // identity
adv::WmParams target = *adv::estimate_local_warp(...); // reused component
adv::WarpDeltaChoice q = adv::choose_warp_delta_precision(
    candidate, target, /*six_param=*/false, mid_x, mid_y, block_mv);
adv::WarpCodingDecision wc = adv::select_warp_coding({
    .candidates = warp_param_stack, .num_candidates = 2,
    .target = &target, .ref = ref_plane, .geom = geom,
    .source = source, .source_stride = stride,
    .scratch = pred_scratch, .scratch_stride = w,
    .enable_six_param = true, .mid_x = mid_x, .mid_y = mid_y,
    .block_mv = block_mv});
// wc.use_warp, wc.warpmv (WARPMV vs WARP_NEWMV), wc.warp_idx,
// wc.six_param, wc.high_precision, wc.delta, wc.model, wc.sse
// Compare wc.sse with SIMPLE-mode SSE; this module does not own lambda.

// --- inter-intra --------------------------------------------------------
adv::InterIntraCodingDecision ii = adv::decide_inter_intra({
    .search = intra_search_input, .is_warp = false,
    .inter_sse = simple_inter_sse});
adv::MotionModeSymbols motion{};
adv::fill_inter_intra_symbols(ii, &motion);

// --- BAWP ---------------------------------------------------------------
adv::BawpDecision bd = adv::decide_bawp({.job = job, .source = src,
    .source_stride = stride, .scratch = scratch,
    .scratch_stride = stride, .allow_bawp = allow_bawp,
    .has_chroma = true, .chroma_job = &chroma_job,
    .chroma_source = chroma_src, .chroma_source_stride = cstride,
    .chroma_scratch = chroma_scratch, .chroma_scratch_stride = cstride,
    .y_mode = 14, .order_hint = oh, .ref_order_hint = ref_oh});
adv::BawpSymbols bs = adv::bawp_symbols_from_decision(bd);
adv::apply_bawp_decision_to_job(bd, /*chroma=*/false, &job);
adv::apply_bawp_decision_to_job(bd, /*chroma=*/true, &chroma_job);
// bd.use_bawp / explicit_bawp / explicit_bawp_scale / use_bawp_chroma,
// bd.luma_alpha (BawpAlpha), bd.chroma_alpha
```

`decide_bawp()` evaluates `{off, fitted, explicit scale 0, explicit scale 1}` by
SSE and, when a chroma job is supplied, tests `use_bawp_chroma` with
`alpha = BawpAlpha`. Chroma rules are exactly §7.13.3.25: plane > 0 uses
`BawpAlpha`; a zero-sample template forces `alpha = 1 << 8`; `explicit_bawp` is
ignored for `morphPred`.

`explicit_bawp_scale` and the context are derived exactly:

```
listIndex = NEARMV ? 0 : (NEWMV && use_amvd ? 1 : 2)
firstRefDist = Abs(get_relative_dist(ref_order_hint, order_hint))
scale = listIndex + 1 + (firstRefDist > 4)
alpha = 256 + 16 * (explicit_bawp_scale ? scale : -scale)
explicit_ctx = (YMode == NEARMV) ? 0 : (YMode == NEWMV && use_amvd ? 1 : 2)
```

## Exact M2 integration call sites

### 1. Tile block — `src/encoder/m2/video_syntax.cpp::write_inter_mode_info()`

The current body writes `code_single_mode()` then DRL/MV. Replace that section
with this order (all advanced calls are in `av2enc::video::adv`):

```cpp
// before is_warp/single_mode: build the NeighborGrid from mv_grid_ and call
adv::ModeContext mc = adv::compute_mode_context(
    grid, node.mi_row, node.mi_col, node.bsize, /*sb_mi_size=*/16,
    /*ref0=*/inter::mc::kRefLast, /*ref1=*/-1);
// and extend find_mv_stack to fill WarpParamStack[0..3] / NumWarpFound.

if (allow_warpmv_mode && min(w, h) >= 8)
  adv::write_is_warp(enc_, cdfs_, mc.warp_mv_count, node.is_warp, &error_);

if (node.is_warp) {
  if (!force_integer_mv)
    adv::write_warp_mv(enc_, cdfs_, node.warp_mv, &error_);
  // YMode = warp_mv ? WARPMV : WARP_NEWMV
} else {
  code_single_mode(ctx, single_mode);                 // existing M2 call
  if (allow_bawp && !ref_scaled && !switch_frame && min(w, h) >= 8 &&
      YMode != GLOBALMV)
    adv::write_bawp_symbols(enc_, cdfs_, bawp_symbols, HasChroma, &error_);
}

// motion mode (skip for WARPMV: read_motion_mode returns DELTAWARP)
adv::write_motion_mode(enc_, cdfs_, node.bsize, motion_symbols, &error_);

// warp_idx / warpmv_with_mvd
if (YMode == WARPMV || motion_mode == DELTAWARP)
  adv::write_warp_idx(enc_, cdfs_, RefWarpIdx, &error_);
if (YMode == WARPMV && RefWarpIdx < 2)
  adv::write_warpmv_with_mvd(enc_, cdfs_, warpmv_with_mvd, &error_);

if (has_newmv(YMode) || has_nearmv(YMode)) code_drl_index(ctx, drl);
if (has_mvd) write_mv(mvd_y, mvd_x);
if (motion_mode == DELTAWARP && YMode == WARP_NEWMV)
  adv::write_warp_delta(enc_, cdfs_, node.bsize, warp_delta, six_param, &error_);
if (YMode == WARPMV)
  adv::write_inter_intra(enc_, cdfs_, /*is_warp=*/true, node.bsize, motion,
                         &error_);
// then update the NeighborMi grid with MotionModes/RefFrames[1]/InterpFilters.
```

`adv::write_advanced_inter_block()` packages the identical order with callbacks
for the existing M2 `code_single_mode`/`code_drl_index`/`write_mv`; use it if
replacing the whole block body is easier. `adv::read_advanced_inter_block()`
is the matching decoder-side order and is exercised by the tests.

For `WARPMV` with no delta, the decoder still runs `read_warp_delta()` and
recomputes `LocalWarpParams` through `reduce_warp_model()` /
`get_warp_translation()`. Use `adv::warpmv_local_model(candidate, mid_x, mid_y,
final_block_mv)` for the prediction model. With `warpmv_with_mvd`, the final MV
includes the MVD.

### 2. Sequence header — `src/encoder/m2/video_headers.cpp::write_sequence_header()`

Add fields to `VideoConfig` and replace:

* the `for (mode = 1; mode < 5)` motion-mode loop and the following
  `enable_masked_compound` bit with
  `adv::write_sequence_motion_mode_config(&writer, seq, &error)`;
* the `enable_bawp` / `enable_cwp` / `enable_imp_msk_bld` bits (after
  `enable_mv_traj`) with `adv::write_sequence_bawp_cwp_bits(&writer, seq, &err)`;
* the `enable_global_motion` bit with
  `adv::write_sequence_global_motion_bit(&writer, seq, &err)` (must stay 0;
  global-motion estimation is outside this module).

`enable_six_param_warp_delta` is emitted by the first helper only when
`seq_enabled_motion_modes[DELTAWARP]` is set.

### 3. Frame header — `src/encoder/m2/video_headers.cpp::write_frame_header_syntax()`

Two insertion points:

* immediately after the interpolation-filter bits and **before**
  `writer.put_bit(c.disable_cdf_update)`, call
  `adv::write_frame_motion_mode_bits(&writer, frame, &error)`; then
  `write_frame_bawp_warp_bits()` must **not** be called here;
* after the existing `reference_select` and `skip_mode_present` bits (the
  `frame_reference_mode()` / `skip_mode_params()` position), call
  `adv::write_frame_bawp_warp_bits(&writer, frame, &error)` before
  `reduced_tx_set`.

`write_frame_bawp_warp_bits` emits `allow_bawp` (iff `!FrameIsIntra &&
enable_bawp`) and then `allow_warpmv_mode` (iff `!FrameIsIntra &&
frame_enabled_motion_modes[DELTAWARP]`), in that order.

## Decoder gate

`build_and_test.sh` runs a **normative entropy-decoder gate**: every advanced
symbol sequence is encoded with `av2::Encoder` and decoded with `av2::Decoder`
(§8.2.1–§8.2.6), including CDF adaptation and `exit_symbol()` trailing-bit
validation. The full ordered block test covers every branch (NEARMV/NEWMV/
GLOBALMV, WARP_NEWMV with EXTENDWARP/LOCALWARP/DELTAWARP, WARPMV with/without
MVD, inter-intra, BAWP explicit/fitted/chroma, six-parameter delta, and
`force_integer_mv`).

A **full `dav2d` stream gate is not possible without M2 edits**: the pinned M2
sequence header hard-codes `enable_bawp = 0`, all `seq_enabled_motion_modes = 0`,
`enable_masked_compound = 0`, and the frame header omits `allow_bawp` /
`allow_warpmv_mode`; the tile writer also has no call site for any advanced
symbol. The module supplies the exact payload bits and frame bits, but the M2
core must call them at the sites above before a conforming stream can be
emitted. Once those three files are wired, the existing
`src/encoder/m2/build_and_test.sh` decode/recon comparison is the dav2d gate.

## Tests

```sh
./src/encoder/m2/advanced/build_and_test.sh
SANITIZE=1 WERROR=1 ./src/encoder/m2/advanced/build_and_test.sh
./src/encoder/m2/advanced/build_and_test.sh --check-tables
```

Current result (Apple clang 21, `-O2`):

```text
advanced_test: 1964773 checks passed
```

ASan/UBSan with `SANITIZE=1` passes. Coverage:

* all 22 CDF arrays are regenerated from `spec/tables/default_cdf.json` and
  checked by `--check-tables`; CDF state equality is verified after every
  adaptive round trip;
* all 68 wedge indices × both signs round-trip, and every mask matches
  `wedge_mask_value` for every wedge-capable block size (including `8x32`,
  `32x8`, `16x64`, `64x16`, `8x64`, `64x8`);
* warp delta symbols for both precisions, 4/6 parameters, all low/high/sign
  combinations; quantise → symbols → decode → `apply_warp_delta` is
  bit-identical;
* 200 random neighbour grids compare `compute_mode_context()` with a literal
  transcription of §7.11.2/§7.11.4, plus concrete NPos/LOCALWARP cases;
* warp decision on a synthetic reference: direct WARPMV selection, delta
  selection, high-vs-standard precision, and encoder/decoder model equality;
* inter-intra mode/wedge round trips for every MiSize and mode, and the search
  decision matches `search_inter_intra()`; weights match §7.13.3.29;
* BAWP symbol combinations; explicit-alpha rule for list 0..2, distance 0..7,
  both signs; fitted/explicit/chroma decisions on synthetic planes;
* sequence/frame bit order and bit counts, plus rejection of inconsistent
  configs (present-flag without modes, six-param without DELTAWARP,
  `allow_bawp` on an intra frame, mismatched frame mode without a present flag).

## Limitations / traps

1. **No OBMC.** It is not an AV2 tool; see `src/encoder/inter/advanced/README.md`.
2. **Single reference only.** Compound prediction, `reference_select`, CWP and
   compound wedge/diffwtd are outside this layer. Inter-intra here is the
   single-reference `COMPOUND_INTRA` path and the WARPMV `warp_inter_intra`
   path.
3. **Warp candidate generation is external.** `find_warp_samples()` /
   `insert_warp_candidate()` need the full §7.12.2 MV-stack and history banks.
   `select_warp_coding()` accepts the resulting `WarpParamStack[0..3]` and
   `RefWarpIdx`; it does not build them.
4. **`get_warp_motion_vector()` is external.** The caller must resolve the MV
   for WARPMV (and add MVD when `warpmv_with_mvd`). The warp *model* used for
   prediction is then `warpmv_local_model(candidate, mid_x, mid_y, final_mv)`.
5. **Global motion estimation is not implemented.** `enable_global_motion`
   stays 0; `gm_params` is not estimated or written.
6. **`VideoConfig`/`NeighborGrid` are not modified.** The integration notes above
   are the required M2 edits; this directory deliberately does not touch them.
7. **`wedge_sign` for inter-intra is inferred 0** by `read_interintra_mode()`;
   only compound wedge codes the sign. The writer enforces that.
8. **`Wedge_Bits` comes from the shared generated table**, not a local copy.
   `8x32`, `32x8`, `16x64`, `64x16`, `8x64`, `64x8` are wedge-capable;
   `4x32` and `32x4` are not (`Wedge_Bits[25] == Wedge_Bits[26] == 0`). Do not
   use a size heuristic.
9. **CDF state is per tile.** Call `AdvancedCdfState::reset()` once per tile
   when `disable_cross_frame_cdf_init = 1`; the M2 core's own CDF struct is
   separate.
