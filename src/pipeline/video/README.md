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
  (`video_syntax.cpp`): partition tree, intra modes, transform-tree, EOB and
  coefficient coding. The partition decision uses the M1 block-variance cost
  model, so both the 64x64 grid (`--single-block`) and the full
  variance-driven tree are supported.
- P-frame tile syntax:
  - `inter_frame_mode_info()`: `is_inter`, `skip_flag`, `single_mode`,
    `read_drl_idx`, and `read_mv` (shell-coded MVD).
  - `GLOBALMV` (zero MV by default; with `--video-gme` the §7.12.2.2
    projected MV, and with a non-identity model the §7.13.3.19/.20 warped
    predictor for blocks >= 8x8), `NEARMV` (the spatial MV-stack / merge path)
    and `NEWMV` (ME result quantised to 1/8 pel).
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
- Global motion / warped prediction (`--video-gme`, `--video-warp`):
  `src/pipeline/video/video_gme.cpp` builds a deterministic 16x16 integer-ME
  grid for every explicit reference, fits the §5.18.9 model with the
  `src/codec/inter/gme/` integer least squares (AFFINE, degraded to ROTZOOM or
  IDENTITY), and codes it at the `reduced_tx_set` / `film_grain_config`
  insertion point.  `GLOBALMV` then uses the §7.12.2.2 projected centre MV and,
  for a non-identity model, the §7.13.3.19/.20 warp predictor.  The pinned
  decoder derives `useWarp` from the model itself, so a non-identity model
  **must** use the warp predictor; `--video-warp` is the explicit alias for
  `--video-gme`.  The feature is gated on the plain skip-only P chain (no B /
  temporal / frame-type wrapper / compound / inter residual / tile grid).

## Files

```text
src/encoder/m2/
  video_headers.h/.cpp       sequence + frame headers, OBU assembly
  video_cdf_tables.h         generated inter CDF defaults (§9.3)
  generate_video_tables.py   generator for video_cdf_tables.h
  video_syntax.h/.cpp        key/inter tile syntax writer
  video_frame.h/.cpp         multi-frame loop, mode decision, recon, IVF units
  video_gate.cpp             standalone gate (encode -> dav2d -> recon compare)
  build_and_test.sh       standalone build/run (no CMake changes)
  merge/                  spec-exact §7.12.2 MV stack + skip/DRL API
  README.md               this file
```

`video_syntax.cpp` started from the M1 syntax writer. The M1 still integration TU
was being refactored while this milestone landed, so the keyframe uses the
M2-owned copy of the same intra syntax path instead of calling
`encode_still_picture()`. The M1 §9.2/§9.3 tables, partition solver, intra
prediction, transform/quant and entropy coder are reused unchanged. When the
M1 still TU is linkable, the keyframe tile can be obtained from
`StillResult::frame` by taking the trailing `StillResult::tile_bytes` bytes; that was
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

The entropy CDFs are §9.3 defaults (`video_cdf_tables.h`) plus the M1 intra
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

- B frames, `frame_type` beyond KEY/INTER, output-order reordering. *(Landed in
  the wave-3/4 sections below; this list is the original M2 baseline.)*
- Multiple references per frame (`NumTotalRefs > 1`), compound prediction,
  `reference_select`, compound refs. *(Landed: `NumTotalRefs` 1–7 and
  `NumRefFrames` 4–8 in the reference-management section; compound is the
  GLOBAL_GLOBALMV/AVERAGE subset.)*
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
  --frames 8 --out /tmp/video_tree8.ivf
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

---

# Wave-2 integration: opt-in `--video` path

The M2 reference above is now exposed as a product CLI path. It stays opt-in:
every default `--still`, `--still-gpu`, basic and pinned still stream is
untouched (`tools/refactor/verify_baseline.sh --check` passes).

## CLI

```sh
av2enc clip.y4m --video --frames 8 -o clip.ivf
av2enc clip.y4m --video --p-frames 5 --video-refs 2 -o clip.ivf
av2enc clip.y4m --video --video-subpel-eighth --video-mv-precision quarter -o clip.ivf
```

| flag | default | meaning |
|---|---|---|
| `--video` | off | encode a key frame plus P frames |
| `--p-frames N` | all | number of P frames after the keyframe |
| `--video-refs N` | 1 | `NumTotalRefs` per P frame, 1..7 (`REFS_PER_FRAME`) |
| `--video-num-ref-frames N` | auto | `§5.4.6 NumRefFrames`, 4..8; must be `>= --video-refs` |
| `--video-subpel-eighth` | off | add the 1/8-pel ME refinement stage |
| `--video-mv-precision P` | eighth | `eighth`, `quarter` or `half` pel |
| `--video-compound` | off | `reference_select` compound (needs `--video-refs 2`) |
| `--video-inter-residual` | off | code a quantised inter residual |
| `--video-intra-full` | off | still product's 61-candidate luma intra search (DC, V, H, 8 directional modes with angle deltas, SMOOTH*, PAETH) instead of the DC-only fallback; chroma stays DC. Default off so the shipped streams stay byte-identical; measured +4.9% rate / −0.016 dB PSNR-Y at QP 90 (SSE-only decision) |
| `--video-tx-search` | off | §5.18.8 `TX_MODE_SELECT`: luma TX size/type/partition search (inter partition + intra/inter type); `--video-tx-lambda` (256) tunes plan selection; see `src/cuda/pipeline/video/tx/README.md` || `--video-max-depth D` / `--video-min-block N` / `--video-force-single` | 4 / 16 / off | partition controls; `N` must be 8..64 (sub-8 leaves would need shared-chroma prediction) |
| `--video-recon FILE` | off | dump the encoder reconstruction Y4M (gate use) |
| `--video-tlayers N` | off | one hierarchical-P GOP of N (4/8) frames with temporal layers |
| `--video-tlayer-drop L` | off | emit only temporal layers `<= L` |
| `--b-frames N` | 0 | N B frames per GOP (one closed GOP of N+2 frames) |
| `--video-drl-reorder` | off | rejected until the decoder disagreement below is fixed |

`--frames N` caps the total number of coded frames in the video path (the
single-picture paths still clamp it to one frame). `--qp` accepts 1..255.

The Y4M reader already yields a sequence; every frame becomes one temporal unit
in the IVF (TD + SEQ for the key frame, TD + tile group afterwards).

## Reference management

* Frame 0 is an intra keyframe and refreshes all slots.
* The new single-reference path keeps the historical two-slot layout
  (`NumRefFrames = 2`).
* `--video-refs N` (1..7) rotates one refresh slot per frame and lists the N
  most recent valid slots newest-first in the explicit §5.18.2 reference map.
  Each inter block codes `read_single_ref()` and picks the reference with the
  lowest predictor SSE (GLOBALMV / NEARMV / NEWMV); a 10-frame 360p check over
  `--video-refs 7` selects all seven slots.
* `NumRefFrames` defaults to 8 for `--video-refs > 1` and 2 for the historical
  single-reference chain. `--video-num-ref-frames N` (4..8) sets the §5.4.6
  pool explicitly: the refresh mask is `f(N)`, `ref_frame_idx` is
  `f(CeilLog2(N))`, and the rotating write slot is `i % N`. The `recent_slots`
  list is most-recent-first, excludes the slot being refreshed, and holds at
  most `REFS_PER_FRAME` (7) entries, so a 4-slot pool can still reference the
  slot it is about to overwrite (§7.23 refresh happens after decoding).
* `FrameDistance[r] = relative_dist(order_hint, slot_order_hint, OrderHintBits)`
  is built for every listed slot and passed to the MV stack, so derived
  candidates match the decoder for all N.
* The §7.8 past/future/current lists only exist for `explicit_ref_frame_map =
  0`. The M2 writer always emits an explicit map, so `--video-implicit-ref-map`
  is rejected by name (the pinned decoder is not gated on the implicit form).

## Frame MV precision

`--video-mv-precision quarter|half` writes `use_qtr_precision_mv` /
`allow_high_precision_mv` and codes the NEWMV MVD with the precision-aware
`§5.20.7.20` shell syntax (`inter/mv_precision`). ME output and the MVD are
quantised to the signalled precision. Note §5.20.7.13 only lowers the
*predictor* for precisions **coarser than 1/2 pel**; for quarter/half the
predictor is used unchanged and only the MVD is in precision units.

## Loop filters (opt-in)

The M2 core codes `apply_deblocking_filter = 0` and leaves every restoration
tool off.  The opt-in loop filters run on the **completed reconstruction** and
replace it before it becomes the output picture and the reference frame for
later pictures, which is the order the decoder applies them in:

```text
reconstruct -> deblock -> CDEF -> CCSO -> LR -> GDF -> output/reference
```

Deblocking is a separate branch (`integration/video-deblock`); this one starts
at CDEF so the ordering stays compatible with §7.20.1.

| Flag | Spec | Default | Notes |
|---|---|---|---|
| `--video-cdef` | §7.18, §5.18.7.10 | off | one strength set (no tile symbol) |
| `--video-cdef-strength N` | `CdefStrengths` 1..8 | 1 | 2..8 codes a per-64×64 `cdef_idx` |
| `--video-cdef-damping N` | `CdefDamping` 3..6 | 3 | |
| `--video-ccso` | §7.19, §5.18.7.12 | off | band-offset only |
| `--video-ccso-bands N` | `CcsoMaxBandLog2` 0..6 | 4 | |
| `--video-wiener` | §7.20.3, §5.18.7.11 | off | luma Wiener NS, per unit |
| `--video-chroma-lr` | §7.20.3 | off | U/V Wiener NS, per unit |
| `--video-gdf` | §7.20.5, §5.18.7.9 | off | luma, header-only (no `use_gdf`) |

Wiring:

* Sequence `§5.4.10`: `enable_cdef` plus the `CdefOnSkipTxfm` pair, and
  `enable_ccso` plus `ccso_unit_matches_sb_size`.  The M2 baseline codes no
  skip syntax on intra leaves, so only `CdefOnSkipTxfm = DISABLED` is wired;
  other modes are rejected by name.
* Frame rows: `§5.18.7.10 cdef_params()` then `§5.18.7.12 ccso_params()`,
  between `deblocking_filter_params()` and `tx_mode_select` (`lr_params()`
  slots between them when loop restoration lands).
* Tile `§5.20.10.1 read_cdef()`: `clear_cdef()` once per superblock, then one
  `cdef_idx` per coding block at its 64×64 origin, skipped exactly where the
  decoder skips it (a skipped inter leaf with `CdefOnSkipTxfm` off).  With
  `CdefStrengths == 1` no symbol is coded at all.
* Tile `§5.20.10.2 read_ccso()`: one `ccso_blk` symbol per enabled plane at
  every 64×64 CCSO unit origin; the plan signals an all-ones unit grid.
* Frame `§5.18.7.11 lr_params()` sits between `cdef_params()` and
  `ccso_params()`.  Only the per-unit `RESTORE_WIENER_NONSEP` form is wired
  (no frame-level filters, which §5.18.7.11 only allows on intra frames).
* Tile `§5.20.10.4 read_lr()` runs once per superblock **before** the
  partition symbol: `use_wiener_ns` plus the §5.20.10.6 per-unit coefficients
  for both chroma planes.  `LoopRestorationSize` is one superblock (64 luma /
  32 chroma), so each superblock starts exactly one unit per plane.  Chroma
  coefficients are mirrored in the plan because the shared writer codes the
  pairs once (`wiener_ns_uv_sym`).
* LR reads `CurrFrame` (pre-CDEF) for samples outside the current stripe and
  filters `CdefFrame`; the output `LrFrame` is a separate buffer.
* Frame `§5.18.7.9 gdf_params()` sits between the deblocking row and
  `cdef_params()`; GDF runs last (§7.20.1) on the `LrFrame` while sampling
  `CdefFrame`, in the header-only mode (`gdf_per_block = 0`), so
  `§5.20.10.3 read_gdf()` codes nothing.  `RESTRICTED_OH` is `-1` because the
  M2 chain codes no restricted references; passing 0 made the derivation drop
  a reference whose OrderHint is 0 and diverge by ±1 on every P frame.
* §7.18.1 `Skips[]` is derived from the coded inter leaves.  The decoder leaves
  a fully-skipped 8×8 block unfiltered, so an all-zero grid diverges on the
  first P frame.
* CDEF reads the post-deblock reconstruction and writes a separate buffer
  (`CurrFrame`/`CdefFrame`).  CCSO keeps `CurrFrame` (pre-CDEF) for its band
  classifier and filters the post-CDEF buffer in place, exactly as §7.19.1
  specifies.

Gate: `src/pipeline/video/gate_video_filters.sh` (CPU: pinned default MD5,
`dav2d --strict 1`, encoder recon == decoded output) and
`src/cuda/pipeline/video/gate_video_gpu.sh` (CPU↔GPU byte identity, both
`--video` and `--video-gpu`).  Both run in CTest.

## Gate

`src/pipeline/video/video_cli_gate.sh` drives the product CLI, decodes with the
pinned `dav2d` through `tools/validate_dav2d.sh`, and compares the decoded Y4M
against `--video-recon` sample by sample for the full frame count.

```sh
src/pipeline/video/video_cli_gate.sh build-int/av2enc \
  testdata/shine_1280x720_60f.y4m 8 --video-refs 2 --video-subpel-eighth
```

Measured on 6 frames of `shine_1280x720_60f.y4m` (`--video-max-depth 3`, the
default single-block tree is also gated by `build-int/video_gate`):

| flags | IVF MD5 | bytes | recon |
|---|---|---:|---|
| default | `c6d588647aacb2275dfaea41a00fb501` | 184,966 | exact |
| `--video-refs 2` | `961eb2010df40f638bb5afb8e45b1db2` | 172,667 | exact |
| `--video-refs 4` | `487d5ec59f7c37a9024d810fd32ef833` | 170,891 | exact |
| `--video-refs 4 --video-num-ref-frames 4` | `78546e3d010f70b1c300f4050de0aee2` | 170,887 | exact |
| `--video-refs 5 --video-num-ref-frames 5` | `850226ac8b27ec0505a77c747acb4b57` | 170,869 | exact |
| `--video-subpel-eighth` | `a1499683f8ab24177950dd61a8db4c65` | 167,863 | exact |
| `--video-mv-precision quarter` | `bdceb54a66d6051165710e37e0d4733d` | 184,866 | exact |
| `--video-refs 2 --video-subpel-eighth --video-mv-precision quarter` | `252e8a1a95a131e88556da40349de3b2` | 175,477 | exact |

`build-int/video_gate --refs N --subpel-eighth --mv-precision P` runs the same
comparison without the CLI (synthetic 64x64 and the real short clip).

Multi-tile video is opt-in via `--video --tiles CxR`. Each tile is encoded as
an independent tile-local picture per frame (fresh CDF/context state and
tile-local origins) while ME/MC read the global reference frame, then
`post::assemble_video_tile_group_from_payloads()` writes `tile_info()` and the
interleaved size fields. `--tiles 1x1` is the default and stays byte-identical.
`src/pipeline/video/gate_cli_video_tiles.sh --bin build-int/av2enc` gates 720p
6-frame 2x1/2x2/4x2/8x2 with `dav2d --strict 1` and sample-exact
`--video-recon` (9/9 local cases). `--tiles 16x2`, `--video-refs > 1`,
non-eighth-pel precision and the frame-type modes are rejected for the tiled
path.

---

# Wave 3 integration: frame types (opt-in)

Long-term references, open-loop keys, leading pictures, show-existing-frame and
SWITCH/RAS frames reuse the wave-2 picture encoder and replace only the
sequence/frame headers and temporal-unit framing. Every flag is default off;
the default `--video` stream above is byte-identical.

## Long-term references and the open-loop key

| flag | default | meaning |
|---|---|---|
| `--video-long-term` | off | `long_term_frame_id_bits = 1`; the CLK carries `LongTermId 0` |
| `--video-olk` | off | open random access: CLK, OLK, P2, P3 (exactly 4 input frames) |

`--video-olk` implies `--video-long-term`. Both need `--video-refs 1` and
eighth-pel precision because the long-term reference writer freezes those
syntax fields.

`--video-long-term` codes the frame index as `OrderHint` with
`OrderHintBits = 4`, so it supports at most 16 coded frames. A larger
`--frames` is rejected with a named error before any frame is encoded; the
plain `--video` path masks the hint with `& 15` and is not capped.

`--video-olk` decode order:

```
TU0  CLK f0  LongTermId 0, refresh all slots
TU1  OLK f1  slot 1 only, non-output;  P2 f2 predicts from the OLK
TU2  P3  f3  predicts from slot 0 (the LTR), across the open RAP
```

Output is `f0, f2, f3`; `VideoSequenceResult::frame_output_index` marks the OLK
as hidden (`-1`). The pinned `dav2d` build does not parse the §5.18.2 OLK
long-term list, so the writer uses `LongTermListSyntax::kOmitPinnedDav2d`; slot
0 still survives the first-Regular-frame invalidation because
`RefLongTermId != -1` (§5.18.2). See `docs/negative-results.md`.

Gate (CPU):

```sh
OUT=/tmp/lt src/pipeline/video/video_cli_gate.sh build-int/av2enc \
  testdata/shine_1280x720_60f.y4m 6 --video-max-depth 3 --video-long-term
OUT=/tmp/olk src/pipeline/video/video_cli_gate.sh build-int/av2enc \
  testdata/shine_1280x720_60f.y4m 4 --video-max-depth 3 --video-olk
build-int/video_gate --single-block --frames 5 --long-term
build-int/video_gate --single-block --frames 4 --olk
```

Pinned streams (720p, `--video-max-depth 3`):

| flags | IVF MD5 | bytes | output frames | recon |
|---|---|---:|---:|---|
| `--video-long-term` (6 frames) | `ac65dee95869ac196ce0537cfcabd1af` | 184,966 | 6 | exact |
| `--video-olk` (4 frames) | `095b52002519b8f1dfd62f3e632df50a` | 141,822 | 3 | exact |

## Leading pictures

`--video-leading N` (1..7, default 0) turns the stream into an open GOP:

```
CLK f0            output
OLK f(N+1)        output (implicit_output_frame = 1), RAP with a repeated SEQ
LEAD f1..fN       coded non-output, reference the OLK
REG  f(N+2)..     output, reference the OLK
```

The sequence header sets `monotonic_output_order_flag = 0`, which is what lets
the OLK be output without a second output picture in its temporal unit. Leading
pictures use the §7.3.4 coded **non-output** form, so a RAP-start decode from
the OLK produces exactly the sequential output minus the pre-RAP CLK.
`frame_output_index` marks the leading pictures `-1`; the decoded frame count is
`2 + regular` for `N + 2 + regular` coded pictures.

`N = S` and `regular = R` are limited so `S + R <= 7` (eight reference slots).
`--video-leading` needs `--video-refs 1`, eighth-pel precision, and cannot be
combined with the long-term/OLK modes.

```sh
OUT=/tmp/lead src/pipeline/video/video_cli_gate.sh build-int/av2enc \
  testdata/shine_1280x720_60f.y4m 8 --video-max-depth 3 --video-leading 3
build-int/video_gate --single-block --frames 8 --leading 3
```

Pinned streams (720p, `--video-max-depth 3`):

| flags | IVF MD5 | coded | output | recon |
|---|---|---:|---:|---|
| `--video-leading 3` (6 frames) | `f705edd017fb4eadce161aa364c5a305` | 6 | 3 | exact |
| `--video-leading 3` (8 frames) | `b73552e37d45986ebce246caab07f1b7` | 8 | 5 | exact |

## Show existing frame (SEF)

`--video-sef` (default off) codes the **last** picture hidden
(`immediate_output_frame = 0`) and appends one `TD + OBU_REGULAR_SEF` temporal
unit. The SEF references the slot that holds the hidden picture with
`derive_sef_order_hint = 1` (§6.17.2: valid, unshown, hidden), so the decoder
output count and order stay `N` for `N` input pictures — no
`--outputinvisible` is needed. This is phase C of the sef component gate applied
to the product path.

`frame_output_index` marks the hidden coded picture `-1` and the SEF unit `N-1`.
`--video-sef` composes with `--video-refs` and the MV-precision flags; it
cannot be combined with the long-term or leading modes.

```sh
OUT=/tmp/sef src/pipeline/video/video_cli_gate.sh build-int/av2enc \
  testdata/shine_1280x720_60f.y4m 6 --video-max-depth 3 --video-sef
OUT=/tmp/sef2 src/pipeline/video/video_cli_gate.sh build-int/av2enc \
  testdata/shine_1280x720_60f.y4m 6 --video-max-depth 3 --video-refs 2 --video-sef
build-int/video_gate --single-block --frames 5 --sef
```

Pinned streams (720p, `--video-max-depth 3`, 6 samples):

| flags | IVF MD5 | bytes | output | recon |
|---|---|---:|---:|---|
| `--video-sef` | `00b97e6f8390e50e189f7a581a52447a` | 184,972 | 6 | exact |
| `--video-refs 2 --video-sef` | `8e957b9dab290daba79640adfa7b3327` | 172,673 | 6 | exact |

## Random-access SWITCH and RAS frames

| flag | input frames | stream |
|---|---|---|
| `--video-switch` | exactly 2 | `OBU_CLOSED_LOOP_KEY`, `OBU_SWITCH` (`restricted_prediction_switch = 1`) |
| `--video-ras` | exactly 3 | CLK (`LongTermId 0`), SWITCH, `OBU_RAS_FRAME` naming id 0 |

Both use the random-access module's planner: the SWITCH reuses the P payload
encoded against f0 (`plan_switch_frame` → `kReusePFrame`, refresh slot 1 so the
long-term slot survives), and the RAS reuses the separate `[f0, f2]` P payload
(`plan_ras_frame` → `kReusePFrame`) because it must predict from the long-term
slot, not from the switch frame. Every header goes through `validate_header`.

Pinned-dav2d quirk: `f9dafbc` reads `NumRefFrames` explicit refresh bits for
`OBU_RAS_FRAME` instead of deriving them. The gate sets
`SequenceConfig::ras_codes_refresh_frame_flags = true` and writes exactly the
§5.18.2 derived mask, so the stream is normatively consistent apart from the
redundant bits. See `docs/negative-results.md`.

Both need `--video-refs 1` and eighth-pel precision, and are mutually exclusive
with the other frame-type modes.

```sh
OUT=/tmp/sw src/pipeline/video/video_cli_gate.sh build-int/av2enc \
  testdata/shine_1280x720_60f.y4m 2 --video-max-depth 3 --video-switch
OUT=/tmp/ras src/pipeline/video/video_cli_gate.sh build-int/av2enc \
  testdata/shine_1280x720_60f.y4m 3 --video-max-depth 3 --video-ras
build-int/video_gate --single-block --frames 2 --switch
build-int/video_gate --single-block --frames 3 --ras
```

Pinned streams (720p, `--video-max-depth 3`):

| flags | IVF MD5 | bytes | output | recon |
|---|---|---:|---:|---|
| `--video-switch` (2 frames) | `2daa1cfae28664063e3f807e70b8e901` | 72,984 | 2 | exact |
| `--video-ras` (3 frames) | `9e7201eb0ca704ddae167a0e0da3d37f` | 99,733 | 3 | exact |

## Build guard for the frame-type modes

The wave-3 modes live in `src/pipeline/video/video_frames.cpp` and
`video_frame.cpp` only calls them when `AV2ENC_VIDEO_FRAMES` is defined (the
CMake build sets it on `av2enc_core`). Without the define, the plain path
compiles and links on its own, so the standalone per-component gate scripts
that compile `video_frame.cpp` still build. Those scripts also gained the
wave-2 `drl_reorder` and `mv_precision` sources they were missing; all four
wave-3 component gates now build and pass:

```sh
src/codec/inter/longterm/build_and_test.sh        # 345 checks + 3 decode phases
src/codec/inter/leading/build_and_test.sh         # 116 checks + full/RAP decode
src/codec/inter/sef/build_and_test.sh             # 3810 checks + SEF phases
src/codec/inter/random_access/build_and_test.sh   # 142 checks + 3-frame gate
```

The SEF gate's slot model now matches the wave-2 rotating refresh
(`slot = i % NumRefFrames` instead of always slot 0), which is what the
two-frame phase needed.

---

# Completion wave: temporal layers (opt-in)

`--video-tlayers N` (N = 4 or 8) replaces the flat key + P chain with one
closed **hierarchical-P GOP** planned by `src/codec/inter/tlayers`. The plan
fixes the coding order, each frame's temporal layer, its single reference and
the slot it refreshes; `validate_plan()` replays §7.23 / §5.18 before any bit
is written.

| field | value |
|---|---|
| `max_tlayer_id` | `plan.max_tlayer_id` (1 for N=4, 2 for N=8; `f(2)`) |
| `monotonic_output_order_flag` | **0** (reordered stream) |
| `tlayer_dependency_present_flag` | 0 (the default `ref <= curr` map; the pinned `dav2d` parses a later explicit-map syntax) |
| `num_ref_frames` | `plan.num_slots` = N |
| `obu_tlayer_id` | per frame OBU; TD/SEQ/CLK are 0 (§5.2.2) |
| `immediate_output_frame` / `implicit_output_frame` | first coded frame 1 / 0; every later frame 0 / 1 |

Units are written in **coding order**; the decoder emits **display order**.
`VideoSequenceResult::recon` is filled in display order and
`frame_output_index[coding_index]` holds that display position.

N=8 coding order is `0 4 2 6 1 3 5 7` with layers `0 1 1 2 2 2 2`; the
per-OBU ids are audited by the gate.

## Dropped-layer sub-stream

`--video-tlayer-drop L` encodes only the units with `obu_tlayer_id <= L`
(`plan.coding_order_up_to_layer`), after `validate_layer_drop()` proves that
no retained frame references a dropped one. The sequence header keeps the full
`max_tlayer_id`.

```sh
OUT=/tmp/tl src/pipeline/video/video_cli_gate.sh build-int/av2enc \
  testdata/shine_1280x720_60f.y4m 8 --video-max-depth 3 --video-tlayers 8
OUT=/tmp/tl0 src/pipeline/video/video_cli_gate.sh build-int/av2enc \
  testdata/shine_1280x720_60f.y4m 8 --video-max-depth 3 --video-tlayers 8 \
  --video-tlayer-drop 0
```

Pinned streams (720p, `--video-max-depth 3`):

| flags | IVF MD5 | units | decoded frames | recon |
|---|---|---:|---:|---|
| `--video-tlayers 8` | `d1b887d47ed595ad3ebc1cdd01b46cc0` | 8 | 8 | exact |
| `--video-tlayers 4` | `44307f54d20e1d6c7b3c1c21e71d5378` | 4 | 4 | exact |
| `--video-tlayers 8 --video-tlayer-drop 0` | `959896691a4bac1c669f3d0ff21d3231` | 2 | 2 | exact |
| `--video-tlayers 8 --video-tlayer-drop 1` | `7828abe69cb64858d6e927e0080fb635` | 4 | 4 | exact |

`--video-tlayers` needs one whole GOP (`frames == N`), `--video-refs 1`,
eighth-pel MV precision, `--rc cqp`, and cannot be combined with the wave-3
frame-type modes. It composes with the partition/ME flags and the inter
residual. Only one closed GOP is supported (the tlayers planner's scope); 16
frames would need `OrderHintBits = 5`.

---

# Completion wave: compound prediction (opt-in)

`--video-compound` (needs `--video-refs 2`) turns on §5.18.2
`reference_select = 1` and lets the mode decision pick a compound leaf. The
gated subset is deliberately narrow and fully normative:

| element | value |
|---|---|
| `reference_select` | 1 |
| `comp_mode` | 1 for a compound leaf, 0 for every single-prediction leaf |
| `comp_ref` | no bits: `NumTotalRefs == 2`, `num_same_ref_compound == 0` implies `(0, 1)` |
| `is_joint` | 0 |
| `compound_mode_non_joint` | 3 (`GLOBAL_GLOBALMV`) |
| `comp_group_idx` / `compound_type` | absent (`enable_masked_compound = 0` → `COMPOUND_AVERAGE`) |
| `cwp_idx` | absent (`enable_cwp = 0` → `CWP_EQUAL`) |
| residual | compound leaves are `skip_flag = 1`; no residual is wired yet |

Prediction uses the shared 1/16-sample compound path
(`multiref::build_compound_prediction` → `apply_compound_prediction`,
§7.13.3.18 `InterRound1 = 7`), so the encoder and decoder agree on the AVERAGE
blend. `enable_masked_compound`, `enable_cwp` and `enable_imp_msk_bld` stay 0 in
the sequence header: CWP, wedge/diffwtd masks and the implicit mask are not
exercised yet.

The compound candidate competes against GLOBALMV / NEARMV / NEWMV on luma SSE;
`--video-refs 2 --video-compound` on `shine` 720p picks compound on ~22 of 512
leaves in three frames.

```sh
OUT=/tmp/cmp src/pipeline/video/video_cli_gate.sh build-int/av2enc \
  testdata/shine_1280x720_60f.y4m 6 --video-max-depth 3 --video-refs 2 \
  --video-compound
OUT=/tmp/cmp2 src/pipeline/video/video_cli_gate.sh build-int/av2enc \
  testdata/shine_1280x720_60f.y4m 6 --video-max-depth 4 --video-refs 2 \
  --video-compound --video-inter-residual --qp 60
```

Pinned streams (720p, `--video-max-depth 3`, 8 frames):

| flags | IVF MD5 | bytes | recon |
|---|---|---:|---|
| `--video-refs 2 --video-compound` (8 frames) | `7f91d2498266696cbda2dfb8c60b2467` | 220,810 | exact |

`--video-compound` cannot be combined with the frame-type modes, DRL reorder or
`--video-tlayers`.

---

# Completion wave: B frames (opt-in)

`--b-frames N` (N = 1..6) codes **one closed GOP** with N consecutive B frames:
`gop_size = N + 2` display frames, coding order `I P B B ...`, display order
`I B B ... P`. The schedule comes from `src/codec/inter/bframes`
(`GopPattern::kIpbb`); `build_sequence()` + `validate_plan()` assign the coding
order, per-frame reference lists, slots, order-hint LSBs and output flags
before any bit is written.

| field | value |
|---|---|
| `monotonic_output_order_flag` | 0 |
| `immediate_output_frame` / `implicit_output_frame` | plan (`FrameHeaderFields`) |
| `order_hint` | `frame.order_hint_lsbs` (the decoder replays the absolute hint) |
| `num_total_refs` | 1 for the P anchor, 2 for every B frame |
| `reference_select` | 1 on the B frames only when `--video-compound` is also on |
| `refresh_frame_flags` | plan (one bit per non-key frame; allFrames for a key) |

Units are written in coding order and the decoder emits display order;
`VideoSequenceResult::recon` is display order and `frame_output_index` maps each
coded unit to its display position.

The B frame's two references are the past anchor and the future P anchor. Each
block selects one with `single_ref`; with `--video-compound` the two-reference
B frame also codes `comp_mode` and can pick the AVERAGE GLOBAL_GLOBALMV
compound block (see the compound section).

```sh
OUT=/tmp/bf src/pipeline/video/video_cli_gate.sh build-int/av2enc \
  testdata/shine_1280x720_60f.y4m 4 --video-max-depth 3 --b-frames 2
OUT=/tmp/bfc src/pipeline/video/video_cli_gate.sh build-int/av2enc \
  testdata/shine_1280x720_60f.y4m 5 --video-max-depth 4 --b-frames 3 \
  --video-compound
```

Pinned streams (720p):

| flags | IVF MD5 | bytes | coded | output | recon |
|---|---|---:|---:|---:|---|
| `--b-frames 1` (3 frames, depth 3) | `c7aec62bad44b2717c42f0abd9fd3bbe` | 76,374 | 3 | 3 | exact |
| `--b-frames 2` (4 frames, depth 3) | `1f5b8d13d31a869b3d3f3d923996806a` | 117,792 | 4 | 4 | exact |
| `--b-frames 2 --video-compound` (4 frames, depth 3) | `e4b5b56c01b3b00f69d00bbc4dc8f4c9` | 94,990 | 4 | 4 | exact |
| `--b-frames 3` (5 frames, depth 3) | `1e75b100872a0258ba58ff931163c9fe` | 143,530 | 5 | 5 | exact |
| `--b-frames 3 --video-compound` (5 frames, depth 4) | `cd5f20c6cf850867e83e0c739ce38ae6` | 94,671 | 5 | 5 | exact |

Limits: exactly one GOP per invocation (`frames == N + 2`), `--video-refs 1`
(the plan owns the reference lists), `--rc cqp`, and no temporal-layer or
frame-type modes. Two or more GOPs need an end-of-stream flush rule for the
pending implicit-output pictures; the pinned decoder emits only the frames whose
display gap is closed, so multi-GOP streams are rejected rather than emitted
short.
