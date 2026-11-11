# M2 P-frame baseline

Two-frame (and short-clip) inter path on top of the M1 still encoder. This is
the CPU correctness reference for the M2 milestone; it is deliberately small
and is gated sample-by-sample against the pinned `dav2d`.

## What is implemented

- One intra keyframe followed by P frames that reference slot 0 and refresh
  slot 0 (`NumRefFrames = 2`, `NumTotalRefs = 1`, explicit ref map).
- Sequence header for `single_picture_header_flag = 0`, with order hints,
  explicit reference maps, a single tile, and all advanced tools disabled.
- Keyframe tile syntax through the M2 copy of the M1 intra writer
  (`m2_syntax.cpp`): partition tree, intra modes, transform-tree, EOB and
  coefficient coding. The partition decision uses the M1 block-variance cost
  model, so both the 64x64 grid (`--single-block`) and the full
  variance-driven tree are supported.
- P-frame tile syntax:
  - `inter_frame_mode_info()`: `is_inter`, `skip_flag`, `single_mode`,
    `read_drl_idx`, and `read_mv` (shell-coded MVD).
  - `GLOBALMV` (zero MV), `NEARMV` (the spatial MV-stack / merge path) and
    `NEWMV` (ME result quantised to 1/8 pel).
  - Per-block intra fallback (`is_inter = 0`, DC intra + DCT residual).
  - Inter leaves use `skip_flag = 1` by default. The gated feature-1 path
  (`--inter-residual`) codes a quantised inter residual with the normative
  inter `TX_TYPE` and coefficient contexts (see the tile syntax section).
- Normative subpel prediction through `src/encoder/inter/mc/` (1/8 pel MV,
  eight-tap switchable filters, four-tap downgrade for <=4 sample passes).
- Motion search through `src/encoder/inter/me/`: integer + half + quarter
  stages, SAD metric, diamond search.
- Reference reconstruction: the current frame's MI-aligned reconstruction
  (`mi_cols*4 x mi_rows*4` luma) becomes slot 0 for the next P frame.

## Files

```text
src/encoder/m2/
  m2_headers.h/.cpp       sequence + frame headers, OBU assembly
  m2_cdf_tables.h         generated inter CDF defaults (§9.3)
  generate_m2_tables.py   generator for m2_cdf_tables.h
  m2_syntax.h/.cpp        key/inter tile syntax writer
  m2_frame.h/.cpp         multi-frame loop, mode decision, recon, IVF units
  m2_gate.cpp             standalone gate (encode -> dav2d -> recon compare)
  build_and_test.sh       standalone build/run (no CMake changes)
  merge/                  spec-exact §7.12.2 MV stack + skip/DRL API
  README.md               this file
```

`m2_syntax.cpp` started from the M1 syntax writer. The M1 still integration TU
was being refactored while this milestone landed, so the keyframe uses the
M2-owned copy of the same intra syntax path instead of calling
`encode_m1_still_picture()`. The M1 §9.2/§9.3 tables, partition solver, intra
prediction, transform/quant and entropy coder are reused unchanged. When the
M1 still TU is linkable, the keyframe tile can be obtained from
`M1Result::frame` by taking the trailing `M1Result::tile_bytes` bytes; that was
the original integration hook and it is documented here so it can be restored
without changing the M2 syntax writer.

## Sequence header (exact fields)

`single_picture_header_flag = 0`; profile 0; level 5.1 (idx 7); 8-bit 4:2:0.

| Field | Value |
|---|---|
| `seq_header_id` | 0 |
| `seq_profile_idc` | 0 |
| `single_picture_header_flag` | 0 |
| `seq_level_idx` | 7 |
| `seq_tier` | 0 |
| `chroma_format_idc` | 0 (4:2:0) |
| `bit_depth_idc` | 1 (8-bit) |
| `seq_lcr_id` | 0 |
| `still_picture` | 0 |
| `max_tlayer_id` / `max_mlayer_id` | 0 / 0 |
| `monotonic_output_order_flag` | 1 |
| `max_frame_width/height` | frame dimensions |
| `seq_cropping_window_present_flag` | 0 |
| `seq_initial_display_delay_present_flag` | 0 |
| `decoder_model_info_present_flag` | 0 |
| `use_256x256_superblock` / `use_128x128_superblock` | 0 / 0 |
| `enable_sdp` / `enable_ext_partitions` | 0 / 0 |
| `reduce_pb_aspect_ratio` | 0 |
| `enable_ext_seg` / `seq_seg_info_present_flag` | 0 / 0 |
| `enable_dip` | 0 |
| `enable_intra_edge_filter` | 1 |
| `enable_mrls` | 0 |
| `enable_cfl_intra` | 1 |
| `cfl_ds_filter_index` | 0 |
| `enable_mhccp` / `enable_ibp` | 0 / 0 |
| `seq_enabled_motion_modes[INTERINTRA..EXTENDWARP]` | all 0 |
| `enable_masked_compound` / `enable_ref_frame_mvs` | 0 / 0 |
| `order_hint_bits_minus_1` | 3 (`OrderHintBits = 4`) |
| `enable_refmvbank` | 0 |
| `disable_drl_reorder` | 1 (`DRL_REORDER_DISABLED`) |
| `explicit_ref_frame_map` | 1 |
| `explicit_num_ref_frames` | 1, `num_ref_frames_minus_1 = 1` (`NumRefFrames = 2`) |
| `long_term_frame_id_bits` | 0 |
| `seq_max_drl_bits_minus_1` | 4 (`m = 5`) |
| `allow_frame_max_drl_bits` | 0 |
| `seq_max_bvp_drl_bits_minus_1` | ns(3) = 0 |
| `allow_frame_max_bvp_drl_bits` | 0 |
| `num_same_ref_compound` | 0 |
| `enable_tip` / `enable_mv_traj` / `enable_bawp` / `enable_cwp` | 0 |
| `enable_imp_msk_bld` / `enable_df_sub_pu` | 0 / 0 |
| `enable_opfl_refine` | 0 (REFINE_NONE) |
| `enable_refinemv` / `enable_bru` / `enable_adaptive_mvd` | 0 |
| `enable_mvd_sign_derive` / `enable_flex_mvres` | 0 |
| `enable_global_motion` | 0 |
| `enable_short_refresh_frame_flags` | 0 |
| `seq_choose_screen_content_tools` / `seq_force_screen_content_tools` | 0 / 0 |
| `enable_fsc` / `enable_idtx_intra` / `enable_intra_ist` / `enable_inter_ist` | 0 |
| `enable_chroma_dctonly` | 1 |
| `enable_inter_ddt` / `reduced_tx_part_set` / `enable_cctx` / `enable_tcq` | 0 |
| `enable_parity_hiding` | 0 |
| `enable_avg_cdf` | 0 (CDFs reset per frame; `disable_cdf_update = 1`) |
| `separate_uv_delta_q` | 0 |
| `equal_ac_dc_q` | 1 |
| `base_uv_ac_delta_q` / `uv_ac_delta_q_enabled` | 23 (delta 0) / 0 |
| `disable_loopfilters_across_tiles` / `enable_cdef` / `enable_gdf` | 0 |
| `enable_restoration` / `enable_ccso` | 0 |
| `cdef_on_skip_txfm_always_on` / `..._disabled` | 0 / 1 |
| `df_par_bits_minus_2` | 0 |
| `seq_tile_info_present_flag` | 0 |
| `film_grain_params_present` | 0 |

`tile_info()` writes `uniform_tile_spacing_flag = 1` and, when the frame has
more than one superblock column/row, one zero
`increment_tile_cols_log2` / `increment_tile_rows_log2` bit. This is a single
tile covering the frame.

## Keyframe frame header (OBU_CLOSED_LOOP_KEY)

| Field | Value |
|---|---|
| `cur_mfh_id` / `seq_header_id_in_frame_header` | 0 / 0 |
| `immediate_output_frame` | 1 |
| `frame_size_override_flag` | 0 |
| `order_hint` | 0 |
| `refresh_frame_flags` | all frames (inferred for CLK with `max_mlayer_id = 0`) |
| `allow_screen_content_tools` | inferred 0 (no bit) |
| `allow_intrabc` | 0 |
| `disable_cdf_update` | 1 |
| `base_q_idx` | 90 |
| `segmentation_enabled` / `using_qmatrix` / `delta_q_present` | 0 |
| `apply_deblocking_filter[0..1]` | 0 |
| `tx_mode_select` | 0 (`TX_MODE_LARGEST`) |
| `reduced_tx_set` | 0 |

The 64x64 keyframe header occupies 27 syntax bits; with
`is_first_tile_group` and byte alignment the tile-group header is 4 bytes.

## P-frame frame header (OBU_REGULAR_TILE_GROUP)

| Field | Value |
|---|---|
| `cur_mfh_id` / `seq_header_id_in_frame_header` | 0 / 0 |
| `frame_is_inter` | 1 |
| `immediate_output_frame` | 1 |
| `frame_size_override_flag` | 0 |
| `order_hint` | frame index (1, 2, ...) |
| `signal_primary_ref_frame` | 0 |
| `disable_cross_frame_cdf_init` | 1 |
| `refresh_frame_flags` | 1 (2 bits, refresh slot 0) |
| `frame_explicit_ref_frame_map` | 1 |
| `num_total_refs` | 1 (3 bits, `NumTotalRefs = 1`) |
| `ref_frame_idx[0]` | 0 (1 bit, slot 0) |
| `frame_size()` | implied from the sequence header (no bits) |
| `allow_screen_content_tools` | inferred 0 (no bit) |
| `allow_intrabc` | 0 |
| `use_qtr_precision_mv` | 0 |
| `allow_high_precision_mv` | 1 (`MV_PRECISION_EIGHTH_PEL`) |
| `is_filter_switchable` / `interpolation_filter` | 0 / 0 (EIGHTTAP) |
| `disable_cdf_update` | 1 |
| `base_q_idx` | 90 |
| `segmentation_enabled` / `using_qmatrix` / `delta_q_present` | 0 |
| `apply_deblocking_filter[0..1]` | 0 |
| `tx_mode_select` | 0 |
| `reference_select` | 0 |
| `skip_mode_present` | 0 |
| `reduced_tx_set` | 0 |

The 64x64 P-frame header occupies 45 syntax bits; with
`is_first_tile_group` and byte alignment the tile-group header is 6 bytes.

`ref_order_hint` is not a frame-header syntax element; the reference slot's
saved order hint is the keyframe's `OrderHint = 0`, so `FrameDistance[0] = 1`
and the P frame's `OrderHints[0] = 0`.

## Tile syntax for a P-frame block

For each superblock, in raster order, the partition tree is coded exactly like
the M1 intra tree (`do_split` / `rect_type`, contexts from the neighbour MI
sizes). For each leaf:

1. `read_is_inter` (`TileIsInterCdf[ctx]`), context from the two decoded
   neighbours' `IsInters`.
2. If intra: M1 `intra_frame_mode_info` path (Y/UV mode, transform,
   coefficients). No `skip_flag` is coded for intra.
3. If inter:
   - `read_skip` (`TileSkipCdf[ctx]`), context from the two neighbours'
     `Skips`.
   - `single_mode` (`TileSingleModeCdf[NewMvContext]`): 0 = NEARMV,
     1 = GLOBALMV, 2 = NEWMV (`YMode = NEARMV + single_mode`).
   - `read_drl_idx` for NEARMV and NEWMV (`TileDrlModeCdf[min(idx,2)]
     [NewMvContext]`; `m = 5`).
   - NEWMV only: shell-coded MVD (`read_mv` with `MvCtx = 0`,
     `MvPrecision = 6`), using `TileJointShellSetCdf`,
     `TileJointShell6Class{0,1}Cdf`, `TileJointShellLastTwoClassesCdf`,
     `TileShellOffsetLowClassCdf`, `TileShellOffsetClass2Cdf`,
     `TileShellOffsetOtherClassCdf`, `TileColMvGreaterCdf`,
     `TileColMvIndexCdf`, then raw sign bits.
   - Interpolation filter is frame-fixed EIGHTTAP, so no per-block filter
     symbol.
   - `skip_flag = 1` suppresses `read_block_tx_size` syntax and all residual
     symbols. A skip block resets the coefficient level/DC contexts over its
     area, mirroring `reset_block_context()`.
   - With inter residual enabled (feature 1, off by default) `skip_flag = 0`
     codes the residual transform blocks. `TX_MODE_LARGEST` makes
     `read_block_tx_size()` bitless, so luma uses
     `Max_Tx_Size_Rect[MiSize]` and chroma the subsampled residual size. Each
     luma block then codes its inter `TX_TYPE` (`DCT_DCT`):
     `TX_SET_INTER_1` (type group 0 + offset 7), `TX_SET_INTER_2` (group 0 +
     offset 3), `TX_SET_DCT_IDTX` (32x32, symbol 1), `TX_SET_{WIDE,HIGH}_32`
     (`is_long_side_dct = 1` + type 0) and `TX_SET_{WIDE,HIGH}_64` (type 0;
     the long-side flag is implied). `TX_SET_DCTONLY` sizes (64x64, 64x32,
     32x64) code no type. Inter changes three coefficient contexts:
     `TileTxbSkipCdf[is_inter]`, luma `eobCtx = is_inter`, and the inter TX
     type CDF family in §8.3.2. CCTX, parity hiding, TCQ and FSC are disabled
     in the sequence header, so the coefficient loop is otherwise identical to
     the intra path.

The entropy CDFs are §9.3 defaults (`m2_cdf_tables.h`) plus the M1 intra
tables. `disable_cdf_update = 1` and `disable_cross_frame_cdf_init = 1` keep
each frame's CDF state at the defaults, so no cross-frame CDF save/load is
needed.

## Gate

```sh
./src/encoder/m2/build_and_test.sh
```

The default run:

1. Builds a host-only CPU binary.
2. Encodes a synthetic 64x64 two-frame stream, runs `dav2d`, and compares the
   decoded Y4M against the encoder reconstruction **sample by sample** for Y,
   U and V.
3. If `testdata/shine_1280x720_60f.y4m` exists, encodes a 2-frame 720p stream
   and an 8-frame 720p clip (64x64 partition grid) and repeats the
   decode/recon comparison. The 8-frame clip exercises mixed inter/intra
   frames, GLOBALMV, NEARMV and NEWMV.

Useful options:

```sh
./src/encoder/m2/build_and_test.sh --frames 8
./src/encoder/m2/build_and_test.sh --input clip.y4m --frames 8
./src/encoder/m2/build_and_test.sh --no-intra-fallback   # force inter
./src/encoder/m2/build_and_test.sh --force-intra          # intra-only P frames
./src/encoder/m2/build_and_test.sh --no-newmv             # GLOBALMV/NEARMV only
./src/encoder/m2/build_and_test.sh --force-global         # zero-MV only
./src/encoder/m2/build_and_test.sh --single-block         # one 64x64 per SB
./src/encoder/m2/build_and_test.sh --inter-residual       # feature 1: rate-based
./src/encoder/m2/build_and_test.sh --force-inter-residual # feature 1: always on
./src/encoder/m2/build_and_test.sh --min-block 8 --max-depth 6  # full tree
./src/encoder/m2/build_and_test.sh --trace
```

`DAV2D=/path/to/dav2d` overrides decoder discovery
(`tools/bin/dav2d`, `tools/bin/dav2d-macos`, `dav2d`).

### Current gate output

```text
== two-frame P-frame: 2 frame(s) 64x64 ==
  frame 0: bytes=573 tile=544 inter=0 intra=1
  frame 1: bytes=555 tile=544 inter=0 intra=1
  PASS: 2 frame(s) decoded, recon exact

== two-frame P-frame: 2 frame(s) 1280x720 ==
  frame 0: bytes=36239 tile=36208 inter=0 intra=240
  frame 1: bytes=2230 tile=2219 inter=235 intra=5
  PASS: 2 frame(s) decoded, recon exact

== short clip: 8 frame(s) 1280x720 ==
  frame 0: bytes=36239 tile=36208 inter=0 intra=240
  frame 1: bytes=2230 tile=2219 inter=235 intra=5
  frame 2: bytes=293 tile=282 inter=239 intra=1
  frame 3: bytes=250 tile=239 inter=240 intra=0
  frame 4: bytes=257 tile=246 inter=240 intra=0
  frame 5: bytes=1669 tile=1658 inter=237 intra=3
  frame 6: bytes=258 tile=247 inter=240 intra=0
  frame 7: bytes=251 tile=240 inter=240 intra=0
  PASS: 8 frame(s) decoded, recon exact
```

### Feature 1 results: inter residual

Both 2-frame and 8-frame 720p streams decode with `dav2d` and match the
encoder reconstruction exactly (0 diffs, Y/U/V). The rate-based mode
(`--inter-residual`) keeps skip blocks when the predictor is already good and
codes residual on noisy blocks; the stress mode (`--force-inter-residual`)
codes residual on every inter leaf.

Full variance-driven partition tree, 8x 1280x720:

| mode | bytes | PSNR Y | PSNR YUV |
|---|---:|---:|---:|
| baseline (`skip_flag = 1` only) | 245,609 | 45.497 | 46.516 |
| forced inter residual | 177,229 | 45.789 | 46.776 |

Single 64x64 leaf per superblock (`--single-block`), 8x 1280x720:

| mode | bytes | PSNR Y | PSNR YUV |
|---|---:|---:|---:|
| baseline | 41,447 | 35.250 | 36.919 |
| forced inter residual | 113,751 | 35.365 | 37.040 |

The single-block forced result is expected to be much larger: every one of the
240 64x64 leaves pays a full residual even when the 1/8-pel predictor is
already near-exact. On this clip the rate-based mode declines every residual
(the 8x tree output is byte-identical to the baseline), so the forced tree row
is the meaningful quality/rate point: residual coding pays off on the blocks
where motion compensation leaves error.

### Feature 2 status: true skip/merge and DRL reordering — BLOCKED

- `skip_mode` (merge) requires `is_comp_ref_allowed()`, which is false with
  `NumTotalRefs = 1` and `reference_select = 0`. It cannot be exercised until
  feature 3 (multiple references + compound) lands, so true skip/merge is
  blocked on that work rather than faked here.
- DRL reordering is **not gated and stays DISABLED in the sequence header**
  (`disable_drl_reorder = 1`). Experimental runs with
  `DRL_REORDER_CONSTRAINT` desync at frame 2 of the 8x720p clip
  (real samples differ, first diff at (768,129)); `DRL_REORDER_ALWAYS` passes
  the plain clip but desyncs at frame 7 when combined with inter residual.
  Streams still parse, so this is an MV-stack order disagreement with the
  decoder, not a syntax error. The baseline (no reorder) is unaffected.
  No `--drl-reorder` switch is exposed until the stack is reconciled.
- The MV-stack `>32` gate is now handled by the spec-exact M2 stack builder in
  `src/encoder/m2/merge/` (`merge::build_stack_checked`), which inserts
  MVP-combination candidates only when both block dimensions exceed 32
  (§7.12.2.20). The encoder and tile writer both use it; the default bitstream
  is byte-identical to the pre-feature-2 baseline. The M2 encoder still caps
  DRL indices at 0..1 for blocks with a dimension <= 32 so the index space
  never depends on shared-component behaviour.

## Not implemented yet

- B frames, `frame_type` beyond KEY/INTER, output-order reordering.
- Multiple references per frame (`NumTotalRefs > 1`), compound prediction,
  `reference_select`, compound refs.
- True `skip_mode` (merge) with two references / compound frames. The current
  "skip" is `skip_flag = 1` plus NEARMV/GLOBALMV on a single reference.
- Inter transform types beyond `DCT_DCT`: the syntax writer emits the mandatory
  inter `TX_TYPE`, but candidate selection always chooses `DCT_DCT`.
- Subpel beyond the current ME stages: the integrated path signals 1/8-pel MVs
  and MC uses the normative filters, but there is no RDO/refinement, no
  adaptive MV precision, and `enable_flex_mvres = 0`.
- Advanced inter tools: warp/affine, OBMC, inter-intra, BAWP, TIP, OPFL,
  refine-MV, temporal MV projection, ref-MV bank, local warp, CWP.
- DRL reordering, short refresh flags, long-term references, multi-tile,
  10-bit, 4:2:2/4:4:4, film grain, loop filters.
- `src/encoder/m2/merge/` provides a spec-exact stack with the §7.12.2.20
  MVP-combination gate; the encoder still caps NEARMV DRL indices to 0..1 when
  either block dimension is <= 32 for baseline stability.
- CUDA. `src/cuda/m2/` is reserved for the device implementation; this
  milestone's gate is the CPU reference.

## Stress mode: full variance-driven partition tree

The default 720p short clip uses `--single-block` (one 64x64 leaf per
superblock, with the bottom edge forced to 64x16). The full tree is also gated:

```sh
./src/encoder/m2/build_and_test.sh --input testdata/shine_1280x720_60f.y4m \
  --frames 8 --out /tmp/m2_tree8.ivf
```

```text
== two-frame P-frame: 2 frame(s) 1280x720 ==
  frame 0: bytes=43725 tile=43694 inter=0 intra=618
  frame 1: bytes=30787 tile=30775 inter=315 intra=303
  PASS: 2 frame(s) decoded, recon exact

== short clip: 8 frame(s) 1280x720 ==
  frame 0: bytes=43725 tile=43694 inter=0 intra=618
  frame 1: bytes=30787 tile=30775 inter=315 intra=303
  frame 2: bytes=28628 tile=28616 inter=342 intra=272
  frame 3: bytes=27373 tile=27361 inter=352 intra=263
  frame 4: bytes=27258 tile=27246 inter=349 intra=265
  frame 5: bytes=29959 tile=29947 inter=338 intra=277
  frame 6: bytes=28129 tile=28117 inter=342 intra=276
  frame 7: bytes=29750 tile=29738 inter=334 intra=284
  PASS: 8 frame(s) decoded, recon exact
```
