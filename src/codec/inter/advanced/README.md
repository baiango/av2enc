# Advanced inter — B-frames, compound, warp, BAWP, inter-intra

CPU reference + CUDA kernels for the AV2 v1.0.0 advanced inter tools
(`docs/tool-roadmap.md` M4a/M4c), workstream 13 of `docs/agent-waves.md`.

**Component only.** Nothing here writes bits, edits the M0/M1 path, or touches
`src/entropy/`, `src/bitstream/`, `src/main.cpp` or the root `CMakeLists.txt`.
There is **no integration and no BD-rate claim**: the decoder gate has not been
run for any of these tools, so per `docs/tool-roadmap.md` §0 they are `U1`/`U2`
and unproven.

## Status

| Tool | Spec | CPU | CUDA | Tests | Notes |
|---|---|---|---|---|---|
| Compound average + CWP | §7.13.3.14, §5.20.7.6 | **complete** | **complete** | literal parity | `cwp_idx` 0..4, weights from `Cwp_Weighting_Factor` |
| Compound implicit mask | §7.13.3.14 `get_mask` | **complete** | **complete** | literal parity | needs `enable_imp_msk_bld` + `RefStartX/Y` |
| Wedge compound | §7.13.3.27, `.30` | **complete** | **complete** | table + literal parity | 68 wedge types × 2 signs, on-the-fly mask |
| Diff-weighted compound | §7.13.3.28 | **complete** | **complete** | literal parity | both `mask_type` values |
| Inter-intra | §7.13.3.29, `.30` | **complete** | **complete** | literal parity | DC/V/H/smooth + wedge-inter-intra |
| Block warp (8x8) | §7.13.3.19, `.21`, `.22` | **complete** | **complete** | literal parity | shared 15×8 intermediate |
| Extended warp (4x4) | §7.13.3.20 | **complete** | **complete** | literal parity | scaled and unscaled reference branches |
| Warp estimation | §7.13.3.23 | **complete** | host only | literal parity | least squares from caller samples |
| Warp model reduce / translation | §7.13.3.23 | **complete** | **complete** | literal parity | |
| Motion-vector scaling | §7.13.3.17 | **complete** | **complete** | numeric | needed by ext warp |
| BAWP | §7.13.3.25 | **complete** | **complete** | literal parity | luma fitted/explicit, chroma from `BawpAlpha` |
| Morphological prediction | §7.13.3.26 | **complete** | **complete** | covered by BAWP | BAWP with `morph_pred = 1` |
| B-frame reference selection | §7.7 | **complete** | n/a (per frame) | ranking/dropping | must match the decoder exactly |
| Past/future/cur lists | §7.8 | **complete** | n/a | classification | |
| Order-hint lift | §5.18.2/§7.13.3.1 | **complete** | n/a | wrap round-trip | `get_disp_order_hint` |
| Hierarchical GOP scheduler | encoder-only | **complete** | n/a | simulated decode | dyadic B pyramid, temporal layers |
| OBMC | **not in AV2** | experiment | experiment | deterministic | see below |

Not implemented / TODO:

* **Global motion parameter estimation** (§7.12.2.1). The warp predictor accepts a
  model, and `setup_shear`/`block_warp` consume `gm_params`, but the
  rotation-zoom/affine least-squares that produces `gm_params` from a reference
  frame is not here. It is a frame-level search and belongs with the global-motion
  agent work, not this block-level component.
* **`find_warp_samples`** (§7.12.3) and the warp-candidate stack
  (§7.12.2.9–.11). `estimate_local_warp()` takes ready-made `WarpSample`s; the
  candidate scan over neighbouring MVs is part of the MV-prediction stack.
* **`Temporal scan` / `get_warp_motion_vector`** (§7.12.2.2/.7) — same reason.
* **TIP / OPFL / refine-MV** — explicitly out of scope (M4c, other workstreams).
  `motion_vector_scaling()` is shared, so those can reuse it.
* **Rate term / RDO.** The searches return SSE only. `src/encoder/rdo/` owns
  lambda and the entropy rate estimate.
* **Reference slot *allocation* policy** for long sequences. `build_hierarchical_gop`
  gives each frame a distinct slot; a real encoder must recycle 16 slots and
  re-derive `refresh_frame_flags`. `validate_gop()` shows how to prove the
  recycled layout still yields one past + one future reference.
* **`reference_select` / per-superblock compound flag** — the bitstream writer
  codes it; this component only produces the prediction.

## OBMC is not part of AV2

AV2 v1.0.0 has **no OBMC motion mode**. `MOTION_MODES = 5`
(`SIMPLE`, `INTERINTRA`, `LOCALWARP`, `DELTAWARP`, `EXTENDWARP`,
`spec/tables/constants.csv`), and there is no OBMC predictor anywhere in
§7.13.3. AV1's `MOTION_MODE_OBMC` was replaced by local warp (an affine field over
8×8 sections) plus inter-intra for occlusion.

`obmc.h` therefore implements a documented, isolated **encoder-only experiment**
(the usual AV1-style linear ramp: the neighbour owns the samples at the shared
edge and fades over `overlap` samples). It must never be enabled in a conforming
stream — the decoder would not reproduce it. It exists to answer "how much would
overlapped MC buy us" before investing in local-warp tooling, and it is excluded
from every other component.

## Files

| Path | Purpose |
|---|---|
| `advanced_common.h` | §4.8 math, Div_Lut/`resolve_division`, POD plane/rect descriptors, enums, wedge-mask and inter-intra-mask evaluation, compound/BAWP per-sample helpers. Shared verbatim with CUDA. |
| `tables/warp_filters.h` | §9.5 `Warped_Filters[449][8]` and `Ext_Warped_Filters[65][6]`, generated from the PDF. |
| `tables/wedge_tables.h` | §5.20.7.18 angle→index map, §7.13.3.27 LUTs + 68-entry codebook, §9.2 `Wedge_Bits` and block sizes, §7.13.3.29 `Ii_Weights_1d`. |
| `tables/extract_spec_tables.py` | Regenerates both headers from `spec/AV2_Spec_v1.0.0.pdf`; `--check` verifies they are current. |
| `compound_common.h` | `CompoundJob` + the single per-sample transcription (`compound_sample_at`). |
| `compound.h/.cpp` | CPU wrapper, validation, wedge-mask dump, compound and inter-intra SSE searches. |
| `warp_common.h` | `setup_shear`, `reduce_warp_model`, warp estimation, MV scaling, block/ext warp passes (per-index functions so the kernel can parallelise). |
| `warp.h/.cpp` | CPU block driver + literal-friendly geometry, `warp_predict_u8`, estimate/SSE helpers. |
| `bawp_common.h` | `BawpContext`, template plan, fit and apply. |
| `bawp.h/.cpp` | Job validation incl. the §7.13.3.25 conformance checks, `bawp_apply_block`. |
| `b_frame.h/.cpp` | §7.7/§7.8/§7.23 reference bookkeeping, order-hint lift, dyadic GOP builder + validator. |
| `obmc_common.h`, `obmc.h/.cpp` | Encoder-only OBMC experiment. |
| `advanced_test.cpp` | 1,033,996 checks. |
| `../../../cuda/inter/advanced/` | CUDA kernels, CUDA README, CPU-vs-GPU test. |

Build and run:

```sh
./src/encoder/inter/advanced/build_and_test.sh            # g++, ~1.03M checks
SANITIZE=1 WERROR=1 ./src/encoder/inter/advanced/build_and_test.sh
./src/encoder/inter/advanced/build_and_test.sh --check-tables
```

Expected: `advanced_test: 1033996 checks passed`.

## Inputs this component does not compute

To avoid duplicating `src/encoder/inter/mc/`:

* **Compound predictions** are `int16_t` at **1/16 sample precision**
  (`Preds[]` after `block_inter_prediction()` with `isCompound = 1`,
  i.e. InterRound1 = 7). Ordinary single-reference MC output is not accepted.
* **Inter-intra** takes an already-clipped `uint8_t` inter prediction in `dst`
  and one intra prediction plane per mode (`II_DC/V/H/SMOOTH`).
* **Warp** takes a `PlaneView` whose `max_x`/`max_y` are the spec's padded bounds
  `(RefMiCols * MI_SIZE >> subX) - 1`, not the visible crop.
* **BAWP** takes the current-frame plane (for above/left reconstruction) and the
  reference plane; the caller supplies `AvailU`/`AvailL`, `MiRow`/`MiCol`,
  `MiCols`/`MiRows` and the block MV.

## Integration sketch

```cpp
// --- one inter block, two references, wedge compound -----------------------
av2enc::inter::adv::CompoundJob job{};
job.plane = 0;                       // luma; then repeat with plane/sub for chroma
job.rect = {x, y, w, h};             // plane samples
job.bsize = MiSize;                  // selects the wedge boundary + mask size
job.p0 = preds0; job.p0_stride = stride;   // int16, 1/16 sample precision
job.p1 = preds1; job.p1_stride = stride;
job.dst = recon; job.dst_stride = frame_stride;
job.type = CompoundType::kWedge;
job.wedge_index = WedgeIndex; job.wedge_sign = wedge_sign;
job.mask_out = luma_mask; job.mask_out_stride = w;   // needed by chroma
av2enc::inter::adv::apply_compound(job);             // false = invalid job

// chroma: same job, plane 1/2, sub_x = sub_y = 1, rect halved,
// luma_mask + luma_mask_stride = w_luma.

// --- encoder decision (SSE only; RDO owns lambda) -------------------------
av2enc::inter::adv::CompoundSearchInput in{ /* predictions, source, scratch */ };
if (auto best = av2enc::inter::adv::search_compound(in)) {
  // best->type / cwp_weight / wedge_index / wedge_sign / mask_type / use_imp_mask
  // are exactly the syntax values the entropy pass must code.
}

// --- inter-intra -----------------------------------------------------------
// `dst` already holds the clipped inter prediction; `intra` is the intra
// prediction for the chosen interintra_mode (DC/V/H/smooth, generated by the
// intra module). For wedge-inter-intra pass type kWedge + wedge_index.
av2enc::inter::adv::CompoundJob ii{};
ii.rect = {x, y, w, h};
ii.bsize = MiSize;
ii.is_inter_intra = 1;
ii.interintra_mode = InterIntraMode::kSmooth;
ii.type = CompoundType::kIntra;          // or kWedge for wedge_interintra
ii.intra = intra_pred; ii.intra_stride = w;
ii.dst = recon; ii.dst_stride = frame_stride;
av2enc::inter::adv::apply_compound(ii);

// --- local warp ------------------------------------------------------------
auto params = av2enc::inter::adv::estimate_local_warp(
    samples, num_samples, mid_x, mid_y, block_mv);
if (params) {
  auto geom = av2enc::inter::adv::WarpBlockGeometry{ /* rect, sub, mi, ref sizes */ };
  geom.dst_x = 0; geom.dst_y = 0;      // block-local prediction buffer
  if (av2enc::inter::adv::warp_needs_extended(*params, geom)) {
    av2enc::inter::adv::warp_predict_ext(*params, ref, geom, /*compound=*/true,
                                         preds0, w);
  } else {
    av2enc::inter::adv::warp_predict(*params, ref, geom, /*compound=*/true,
                                     preds0, w);
  }
}

// --- BAWP ------------------------------------------------------------------
av2enc::inter::adv::BawpJob job{};
job.ctx = {.plane = 0, .sub_x = 0, .sub_y = 0, .mi_col = MiCol, .mi_row = MiRow,
           .mi_cols = MiCols, .mi_rows = MiRows, .avail_u = AvailU,
           .avail_l = AvailL, .bit_depth = 8};
job.rect = {x, y, w, h};
job.mv = {mv_row, mv_col};                 // 1/8 luma sample units
job.cur = cur_plane; job.prev = ref_plane;
job.dst = cur_plane;                       // in-place adjustment
job.explicit_bawp = use_explicit_bawp;
job.explicit_bawp_scale = explicit_bawp_scale;
job.list_index = bawp_list_index(is_near_mv, is_new_mv, use_amvd);
job.first_ref_dist = Abs(get_relative_dist(order_hint, ref_order_hint));
av2enc::inter::adv::BawpFit fit;
av2enc::inter::adv::bawp_apply_block(job, &fit);   // fit.alpha -> chroma BawpAlpha

// --- B-frame scheduling ----------------------------------------------------
av2enc::inter::adv::GopConfig cfg{ .gop_size = 8, .start_order_hint = 0 };
auto gop = av2enc::inter::adv::build_hierarchical_gop(cfg);
// per frame: order_hint, temporal_layer, coding order, explicit ref map,
// refresh_frame_flags. validate_gop(gop, cfg) simulates §7.23 + §7.8 once.
```

For frames that do **not** use `explicit_ref_frame_map`, feed the decoder's slot
state to `build_ref_frame_list()` and `get_past_future_cur_ref_lists()` and use the
result instead of `GopFrame::ref_slot`. That is the only way to be sure the
decoder selects the same references the encoder assumed.

## Spec mapping

| Spec | Implementation |
|---|---|
| §4.8 `Round2`, `Round2Signed`, `Clip1`, `Clip3`, `FloorLog2` | `advanced_common.h` |
| §7.13.3.22 `resolve_divisor`, §7.13.3.23 `resolve_division` | `advanced_common.h` |
| §7.13.3.13/.14 `get_mask`, CWP average, mask blend | `compound_common.h` |
| §7.13.3.27 wedge tables/masks | `wedge_tables.h`, `wedge_master_value`, `wedge_mask_value` |
| §7.13.3.28 difference weight mask | `diffwtd_mask_value` |
| §7.13.3.29 intra-mode variant mask | `intra_mode_mask_value` |
| §7.13.3.30 mask blend | `mask_blend_sample`, `inter_intra_blend_sample` |
| §7.13.3.16 rounding variables | `kCompoundPostRound`, `kInterRound1*` |
| §7.13.3.17 motion vector scaling | `motion_vector_scaling` |
| §7.13.3.19/.20 warp passes | `block_warp_*`, `ext_warp_*` |
| §7.13.3.21 setup shear | `setup_shear` |
| §7.13.3.23/.24 warp estimation | `warp_estimate`, `reduce_warp_model`, `get_warp_translation` |
| §7.13.3.25 BAWP | `bawp_fit`, `bawp_apply` |
| §7.13.3.26 morph pred | `BawpJob::morph_pred` |
| §5.18.2/.3 order hints, `get_relative_dist` | `display_order_hint`, `get_relative_dist` |
| §7.7 get ref frames | `build_ref_frame_list` |
| §7.8 past/future/cur lists | `get_past_future_cur_ref_lists` |
| §7.23 reference update | `refresh_reference_slots` |
| §9.2 `Wedge_Bits`, block sizes | `wedge_tables.h` |
| §9.5 warp filters | `warp_filters.h` |

## What the tests prove

* Every §9.5 filter row sums to 128; the 8-tap zero-phase row is `127/1` and the
  6-tap one is an exact delta (this is a table property, see limitations).
* Every wedge mask for 8x8/16x16/32x32/64x64/8x32/32x8, all 68 indices and both
  signs, equals a literal transcription of `initialise_wedge_mask_table()`.
* Compound average (7 CWP weights), implicit mask, wedge (5 indices × 2 signs,
  luma + chroma subsampling) and diffwtd (2 mask types) equal a literal
  per-sample transcription.
* Inter-intra (4 modes + wedge) equals the literal blend.
* Warp: block and extended predictions equal independent literal transcriptions
  of §7.13.3.19/.20 over identity, integer translation and random affine models,
  in both single-reference (>>11) and compound (>>7) precision.
* Warp estimation equals a literal transcription of §7.13.3.23 over random
  sample sets, including the `det == 0` path.
* BAWP fits and output equal a literal transcription over 40 random jobs
  (block sizes, `AvailU/L` combinations, MVs), plus the `count == 0` rule and
  the explicit-alpha derivation.
* §7.7 ranks/drops references as specified (duplicate `RefCounter` slots,
  `REFS_PER_FRAME` overflow, restricted slots) and §7.8 classifies them.
* Every GOP size 2/4/8/16 produces a complete, decodable coding order with one
  past + one future reference per B frame, temporal-layer monotonicity, distinct
  slots, and `validate_gop()` passes.
* ASan/UBSan clean (`SANITIZE=1`).

## Limitations / traps

1. **The 8-tap warp filter's zero phase is not an exact delta.** §9.5's row 192 is
   `{0,0,0,127,1,0,0,0}`, so an identity warp model is a near-copy with a measured
   worst case of 4 LSB on random 8-bit data. This is a table property, not an
   implementation bug; encoders should not pick warp for pure translation.
   `test_warp_identity_and_translation` pins the bound and
   `test_warp_literal_parity` checks the loop sample by sample.
2. **`Round2` on negative warp phases must be the floor form**, not
   `Round2Signed` (§4.8's integer definition). `setup_shear`,
   `motion_vector_scaling`, `reduce_warp_model` and `get_warp_translation` do use
   `Round2Signed` as written.
3. **Wedge masks depend on `bsize`, not just w×h.** The boundary type flips at
   `BLOCK_16X16`, so 16x16 and 32x32 use different master masks.
4. **Chroma wedge/diffwtd masks come from the luma mask.** Pass the plane-0
   `mask_out` (stride = luma block width) into the chroma jobs; the helper
   subsamples with `Round2(...,2)` for 4:2:0.
5. **`apply_compound()` rejects jobs** that violate the conformance preconditions
   it can check (null planes, non-signallable wedge index, average weight outside
   -4..20). It does not check frame bounds; the caller owns those.
6. **BAWP reads `y-1` / `x-1`.** `bawp_apply_block` returns false when
   `AvailU/AvailL` are set but the block sits on the plane edge, and when the
   reference block is not at least one sample inside the frame (§7.13.3.25
   conformance).
7. **`WarpSample` units are 1/8 luma samples** for both positions and MVs, and
   `mid_x`/`mid_y` are luma samples.
8. **`get_disp_order_hint` only recovers hints within half a span** of the largest
   hint in the reference store. `order_hint_lsbs_for()` models that; do not feed
   it arbitrary absolute hints.

## Timing

The CPU reference is a correctness oracle (`docs/architecture.md`): it is not
benchmarked or optimized. The CUDA test prints a `cudaEvent`-timed
`compound wedge 64x64` throughput on the machine that runs it; no GPU number is
claimed here because the development Mac has no CUDA device. The kernels are
designed for the `compat8g` profile: no allocation inside a kernel, one launch
per job batch, ≤ 1 KiB shared memory, no Blackwell-only instructions.

Local reference run (Apple M-series, clang 21, `-O2`): 1,033,996 checks run in
< 0.1 s; `build_and_test.sh` (six translation units, build + run) takes ~3 s,
and the ASan/UBSan build ~5 s. These are build/test times, not encoder speed.
