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
- TIP (`--video-tip`, §7.10/§7.13.3.2): key + future anchor P + one
  `OBU_REGULAR_TIP` `TIP_FRAME_AS_OUTPUT` per remaining picture (3..8
  pictures).  Every reference's §7.22 field is stored from the real coded MV
  grid; each TIP frame stores its own field for the chained next TIP, and the
  sequence enables `enable_ref_frame_mvs`/`enable_tip`.  Hole fill is on (see
  `docs/tip-decoder-gate.md`).  With `--video-ref-mvs` (4..8 pictures) the last
  picture is instead a tile-coded P that references the final TIP display
  frame, so §7.9 runs over a TIP slot's field.  Gated by `video_gate --tip`
  (and `--tip --ref-mvs`) at 4 and 8 pictures with `dav2d --strict 1` and exact
  recon; the GPU path is not wired.
- Temporal reference MVs (`--video-ref-mvs`, §5.4.6/§5.18.2/§7.9/§7.12.2.7-.8):
  every reference slot stores the §7.22 8x8 motion field, the next inter frame
  runs §7.9 estimation and the merge stack inserts the temporal candidates at
  the normative step-5/step-12 points.  Working for the multi-reference P chain,
  the B-frame GOPs, `--video-gme`/`--video-warp` (per-8x8 warped MVs),
  `--video-compound-wedge` (the §7.22 wedge allow-list) and the
  `--video-tip` tail P that references the final TIP display frame;
  `--video-tlayers` is single-reference, so §7.9 produces an empty field there.
  Off by default (default streams are byte-identical).
  Gate: `src/pipeline/video/video_ref_mvs_gate.sh`; details and MD5s in
  `docs/ref-frame-mvs.md`.
- Reference reconstruction: the current frame's MI-aligned reconstruction
  (`mi_cols*4 x mi_rows*4` luma) becomes slot 0 for the next P frame.
- Arbitrary frame dimensions: the sequence/frame header signals the real
  `max_frame_width/height`, the coded picture is the §5.18.4.4 MI-aligned
  extent (`MiCols = 2*((W+7)>>3)`, `MiRows = 2*((H+7)>>3)`), and every tile
  encoder's padded reconstruction is pasted into the frame reference before the
  loop-filter pass.  The reconstructed output and `--video-recon` are cropped
  to W x H.  Each format keeps only its true constraint (4:2:0 even W/H, 4:2:2
  even W, 4:4:4/4:0:0 any W/H); the CUDA `--video-gpu` path keeps a named
  CPU-only rejection for sizes that are not multiples of 8.  Gate:
  `gate_video_any_dims.sh` (100x70/130x74/424x240/640x362 at 4:2:0 and 4:4:4,
  plus odd 101x71 at 4:4:4, 4-frame key+P with every loop-filter composition,
  `dav2d --strict 1` and byte-exact recon).
- Global motion / warped prediction (`--video-gme`, `--video-warp`):
  `src/pipeline/video/video_gme.cpp` builds a deterministic 16x16 integer-ME
  grid for every explicit reference, fits the §5.18.9 model with the
  `src/codec/inter/gme/` integer least squares (AFFINE, degraded to ROTZOOM or
  IDENTITY), and codes it at the `reduced_tx_set` / `film_grain_config`
  insertion point.  `GLOBALMV` then uses the §7.12.2.2 projected centre MV and,
  for a non-identity model, the §7.13.3.19/.20 warp predictor.  The pinned
  decoder derives `useWarp` from the model itself, so a non-identity model
  **must** use the warp predictor; `--video-warp` is the explicit alias for
  `--video-gme`.  Every explicit reference carries its own §5.18.9 model
  (§5.18.9.1), so `--video-refs 2..7` works on the plain skip-only P chain (no
  B / temporal / frame-type wrapper / compound / inter residual / tile grid).
  Candidate ties are resolved in reference-list order, exactly like the CPU
  reference.
- Segmentation / delta-Q (`--video-segmentation`, `--video-delta-q`,
  `--video-per-segment-q`, `--video-uv-delta-q`, `--video-seg-skip`,
  `--video-seg-globalmv`, `--video-seg-temporal`, `--video-segments N`,
  CPU only): the frame header codes the real
  §5.18.7.1 segmentation table and §5.18.7.8 `delta_q_params()`; every leaf
  codes `intra_segment_id()` / `inter_segment_id()` in the exact
  `inter_frame_mode_info()` / `intra_frame_mode_info()` order, and the first
  leaf of each superblock codes `read_delta_qindex()`.  The encoder plans one
  `rc::DeltaQPlanStep` per superblock (`rc::plan_delta_q`) and uses the
  resulting §7.14.2 effective qindex for the block quantizer in inter and
  intra leaves and for the §7.17 deblock metadata.  `--video-per-segment-q`
  assigns `SEG_LVL_ALT_Q` per superblock from the same luma-variance split the
  still `--still-per-segment-q` uses.  A nonzero delta step cannot be coded
  when a superblock is a single skipped 64x64 leaf, and a skipped inter leaf
  infers its segment id from the §5.20.7.4 neighbours; both are replayed
  exactly (see `src/codec/entropy/deltaq/video/README.md`).
  `--video-segments N` (1..16) splits the superblocks into N variance-ranked
  buckets; N > 8 signals §5.4.4 `enable_ext_seg = 1` (MaxSegments = 16) and
  exercises `seg_id_ext_flag`/`seg_id_ext`.  `--video-seg-skip` and
  `--video-seg-globalmv` place `SEG_LVL_SKIP` / `SEG_LVL_GLOBALMV` on the
  lowest-variance bucket; those leaves are forced to GLOBALMV/skip (an intra
  key frame suppresses SKIP, see §6.19.4.1) and an edge leaf that cannot be
  forced falls back to segment 0 with the same qindex.  `--video-seg-temporal`
  codes `segmentation_temporal_update` on P frames whose primary reference is
  an earlier P frame and replays `PrevSegmentIds` from the reference frame.
  `--video-uv-delta-q D` turns on §5.4.8 `separate_uv_delta_q=1` /
  `equal_ac_dc_q=0` and codes U=+D, V=-D for both DC and AC (`diff_uv_delta=1`);
  §7.14.2 then gives each chroma plane/component its own quantizer and §7.17.6
  receives the same UV AC deltas.  Gated by
  `src/pipeline/video/gate_video_deltaq.sh` with pinned IVF MD5s at 64x64 and
  640x360, `dav2d --strict 1`, picture counts and exact recon; the default-off
  path is byte-identical.
- SDP (`--video-sdp`, §5.4.3): the intra keyframe codes a luma partition tree
  and a chroma tree for every 64x64 superblock.  The luma tree uses
  `--video-max-depth`/`--video-min-block`; the chroma tree is pinned to one
  `PARTITION_NONE` block (see `src/codec/partition/sdp/README.md` for the
  decoder-gated policy).  CPU-only 8-bit 4:2:0, single tile, no frame-type
  wrappers/morph/CfL/segmentation.  Gate:
  `src/pipeline/video/gate_video_sdp_ddt.sh`.
- Inter DDT (`--video-ddt`, §5.4.8/§7.15.4.1): the sequence bit plus the
  per-block inter TX-type selection applies the DDTX/FDDT kernels when an
  ADST/FDST component reaches adjusted length 8 or 16.  Composes with
  `--video-inter-residual` and `--video-tx-search`.  Gate as above; it
  requires at least one coded DDT leaf and sample-exact recon.
- Reference-management tools on the flat key+P chain (all default off, so the
  default streams stay byte-identical):
  - `--video-short-refresh` (§5.4.6 `enable_short_refresh_frame_flags`,
    §5.18.2 `has_refresh_frame_flags` / `frame_to_refresh`): every P frame
    codes the compact single-slot refresh form instead of the full mask.  Gate:
    `video_short_refresh_gate.sh` (64x64 refs 1/2 and 640x360 refs 2).
  - `--video-same-ref-compound` (§5.4.6 `num_same_ref_compound = 1`,
    §5.20.7.11 `comp_ref`, §5.20.7.6 `compound_mode_same_refs`): with
    `--video-compound --video-refs 2` a compound leaf may signal the `(0, 0)`
    pair; the decision prefers it on an SSE tie with the (0,1) carrier.  Gate:
    `video_same_ref_compound_gate.sh` (64x64/640x360, counts the same-ref
    leaves and checks recon).
  - `--video-mv-traj` (§5.4.6 `enable_mv_traj`, §7.9.2 trajectory fill,
    §7.12.2.8 temporal-scan substitution and §7.12.2.14/.16 `TrajMv` derived
    candidates): requires `--video-ref-mvs`; any `NumTotalRefs >= 2` is gated.
    Gate: `video_mv_traj_gate.sh` (64x64/640x360 plus the three-reference case,
    counts trajectory-derived candidates).
  - `--video-refmvbank` (§7.12.2.21): the §5.20.2.2 per-superblock reset and
    row-above seeding, plus the §6.19.2.1 per-superblock-row size/start clear,
    replay the pinned decoder's bank.  Gate: `video_mv_traj_gate.sh`
    (64x64/640x360, with and without `--video-ref-mvs`).

### Sub-8 luma leaves (`--video-min-block 4`)

`--video-min-block 4` lowers the intra keyframe partition floor to four
samples.  A leaf whose chroma residual would be smaller than 4x4 (4x4, 4x8,
8x4, 4x16, 16x4, ...) does not own its chroma: §5.20.3.1 gives
`HasChroma = hasChroma && !chromaOffset` to the first child of the split and
the next leaf codes the whole inherited ChromaMi block.  The rect, extended and
SDP callers carry `has_chroma` / `chroma_offset` / `chroma_bsize` /
`chroma_mi_row` / `chroma_mi_col` in `BlockSyntax` (behind
`kFlagHasChromaInfo`), the leaf coder predicts/transforms chroma once at the
ChromaMi origin, and the §7.17 deblock grid replicates the chroma metadata over
the ChromaMi luma extent.  The §5.20.5.5 directional-mode list is gated on
`MiSize >= BLOCK_8X8`, so rectangular 32x4/4x32 leaves keep the dynamic list
even though one dimension is four samples.

P frames raise their search floor to 8x8 (`decide()`); a sub-8 or
shared-ChromaMi leaf stays intra because MC derives chroma from the luma leaf.
The sub-8 syntax path composes with `--video-intra-full`, `--video-mrl`,
`--video-tx-search`, `--video-deblock`, `--video-ext-partitions`,
`--video-sdp`, `--video-sb128` and `--tiles`.  `--video-sdp` with
`--video-intra-full` is rejected by name (the SDP luma tree's dynamic mode list
is not decoder-exact at any min-block).  Gate:
`src/pipeline/video/gate_video_sub8.sh` (picture counts, actual 4xN/Nx4 leaf
selection, `dav2d --strict 1` and sample-exact `--video-recon` at 64x64 and
640x360, depths 5 and 6; `--video-min-block 8` controls select no sub-8 leaf).

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
  gate_implicit_ref_map.sh §7.7 implicit-map decoder gate (flat/B/temporal)
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
| `seq_header_id` | 0; `--video-multi-sequence` carries a second header with id 1 |
| `seq_profile_idc` | 0 |
| `single_picture_header_flag` | 0 |
| `seq_level_idx` | 7 |
| `seq_tier` | 0 (main tier); 1 with `--video-tier high` (level 4.0+) |
| `chroma_format_idc` | 0 (4:2:0) |
| `bit_depth_idc` | 1 (8-bit) |
| `seq_lcr_id` | 0; 1 with `--video-lcr-full` (activates the global LCR) |
| `still_picture` | 0 |
| `max_tlayer_id` / `max_mlayer_id` | 0 / 0 |
| `monotonic_output_order_flag` | 1 |
| `max_frame_width/height` | frame dimensions |
| `seq_cropping_window_present_flag` | 0 |
| `seq_initial_display_delay_present_flag` | 0 |
| `decoder_model_info_present_flag` | 0 |
| `use_256x256_superblock` / `use_128x128_superblock` | 0 / 0 |
| `enable_sdp` / `enable_ext_partitions` | 0 / 0 |
| `reduce_pb_aspect_ratio` | 0; 1 with `--video-reduce-pb-aspect`, then `max_pb_aspect_ratio_log2_minus_1` selects 4:1 (1, default) or 2:1 (0, `--video-pb-aspect-2`) |
| `enable_ext_seg` / `seq_seg_info_present_flag` | 0 / 0; `enable_ext_seg` 1 with `--video-segments 9..16`; `seq_seg_info_present_flag` 1 with `--video-seq-seg-info` (then `seq_allow_seg_info_change` + the §5.4.9 table; `--video-seq-seg-info-frozen` codes `allow_change = 0`) |
| `enable_dip` | 0 |
| `enable_intra_edge_filter` | 1 |
| `enable_mrls` | 0 |
| `enable_cfl_intra` | 1 |
| `cfl_ds_filter_index` | 0; `--video-cfl-ds-filter N` (format video core only) |
| `enable_mhccp` / `enable_ibp` | 0 / 0; `enable_ibp` 1 only with `--video-ibp` |
| `seq_enabled_motion_modes[INTERINTRA..EXTENDWARP]` | all 0 |
| `enable_masked_compound` / `enable_ref_frame_mvs` | 0 / 0 |
| `order_hint_bits_minus_1` | 3 (`OrderHintBits = 4`) |
| `enable_refmvbank` | 0; 1 only with `--video-refmvbank` |
| `disable_drl_reorder` | 1 (`DRL_REORDER_DISABLED`); 0 with `--video-drl-reorder[-always]` |
| `explicit_ref_frame_map` | 1 |
| `explicit_num_ref_frames` | 1, `num_ref_frames_minus_1 = 1` (`NumRefFrames = 2`) |
| `long_term_frame_id_bits` | 0 |
| `seq_max_drl_bits_minus_1` | 4 (`m = 5`) |
| `allow_frame_max_drl_bits` | 0 |
| `seq_max_bvp_drl_bits_minus_1` | ns(3) = 0 |
| `allow_frame_max_bvp_drl_bits` | 0 |
| `num_same_ref_compound` | 0; 1 only with `--video-same-ref-compound` |
| `enable_tip` / `enable_mv_traj` / `enable_cwp` | 0; `enable_mv_traj` 1 only with `--video-mv-traj`, `enable_cwp` 1 only with `--video-cwp-weighted` |
| `enable_bawp` | 0; 1 only with `--video-bawp` |
| `enable_imp_msk_bld` / `enable_df_sub_pu` | 0; enabled by `--video-imp-mask` / `--video-df-sub-pu` |
| `enable_opfl_refine` | 0 (REFINE_NONE) |
| `enable_refinemv` / `enable_bru` / `enable_adaptive_mvd` | 0 |
| `enable_mvd_sign_derive` / `enable_flex_mvres` | 0 |
| `enable_global_motion` | 0 |
| `enable_short_refresh_frame_flags` | 0; 1 only with `--video-short-refresh` |
| `seq_choose_screen_content_tools` / `seq_force_screen_content_tools` | 0 / 0; the palette/IBC/morph carriers select the per-frame form (`seq_choose_screen_content_tools = 1`), `--video-scc-force` writes the forced form (`0 / 1`); `seq_choose_integer_mv` is SELECT unless `--video-force-integer-mv` writes `0 / 1` (`seq_force_integer_mv = 1`) |
| `enable_fsc` / `enable_idtx_intra` | 0; `enable_idtx_intra` 1 only with `--video-idtx` |
| `enable_mrls` | 0; 1 only with `--video-mrl` |
| `enable_dip` | 0; 1 only with `--video-dip` |
| `enable_intra_ist` / `enable_inter_ist` | 0; `enable_intra_ist` 1 only with `--video-ist` (inter IST is never signalled) |
| `enable_chroma_dctonly` | 1 |
| `enable_inter_ddt` / `reduced_tx_part_set` / `enable_cctx` / `enable_tcq` | 0 |
| `enable_parity_hiding` | 0 |
| `enable_avg_cdf` | 0 (CDFs reset per frame; `disable_cdf_update = 1`) |
| `separate_uv_delta_q` / `equal_ac_dc_q` | 0 / 1; `1 / 0` with `--video-uv-delta-q` |
| `base_uv_ac_delta_q` / `base_uv_dc_delta_q` | 23 (delta 0) / inferred 0; `--video-uv-delta-q` codes both |
| `uv_ac_delta_q_enabled` / `uv_dc_delta_q_enabled` | 0 / 0; both 1 with `--video-uv-delta-q` |
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
| `cur_mfh_id` / `seq_header_id_in_frame_header` | 0 / 0; 0 / 1 with `--video-multi-sequence` |
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
| `cur_mfh_id` / `seq_header_id_in_frame_header` | 0 / 0; 0 / 1 with `--video-multi-sequence` |
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

## Header-surface flags (all default off)

| Flag | Syntax | Decoder carrier |
|---|---|---|
| `--video-tier high` | `seq_tier = 1` (§5.4.1, only for `seq_level_idx > 3`) | `dav2d --strict 1` + sample-exact `--video-recon`; the gate parses the bit |
| `--video-multi-sequence` | two sequence headers (ids 0/1) and `seq_header_id_in_frame_header = 1` (§5.18.2) for every frame | `dav2d --strict 1` + recon; header 1 also changes `OrderHintBits` 4 → 5, so a decoder that reused header 0 desynchronises |
| `--video-lcr-full` | `seq_lcr_id = 1`, global/local LCR (§5.8) and global/local OPS (§5.10/§5.11) before the sequence header | `avmdec` parses and conformance-checks OBU types 16/18; `dav2d` ignores them (prints "Unknown OBU type 16/18") and must still decode the same pictures |

The default stream keeps `seq_tier = 0`, one `seq_header_id = 0` header, no
LCR/OPS and `seq_lcr_id = 0`; the pinned reference MD5s are unchanged.  The
gate is `src/pipeline/video/gate_video_headers.sh` (structural header parse +
`dav2d`/`avmdec` decode + pinned IVF MD5s for the default and all three
carriers).

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

### Feature 2 status: true skip/merge and DRL reordering

- `skip_mode` (merge) requires `is_comp_ref_allowed()`, which is false with
  `NumTotalRefs = 1` and `reference_select = 0`. It cannot be exercised until
  feature 3 (multiple references + compound) lands, so true skip/merge is
  blocked on that work rather than faked here.
- DRL reordering is **decoder-gated** for the bootstrapped
  `DRL_REORDER_CONSTRAINT` / `DRL_REORDER_ALWAYS` pair:
  `--video-drl-reorder` writes `disable_drl_reorder = 0,
  constrain_drl_reorder = 1` and `--video-drl-reorder-always` writes the ALWAYS
  form. Both are sample-exact against the pinned dav2d
  (`gate_video_drl_reorder.sh`, 360p depth 3/4 and 720p depth 3/4).
  `DRL_REORDER_CONSTRAINT` is *inert* on this low-delay P surface: dav2d
  f9dafbc's §7.12.2 step-16 high-priority-TMVP term is not gated by
  `use_ref_frame_mvs`, so a single reference within two POCs never sorts. The
  encoder mirrors that (`drl_constraint_sort_suppressed()`), so
  `--video-drl-reorder` is bit-exact but moves no candidates; use
  `--video-drl-reorder-always` to exercise the §7.12.2.19 transform (18 swaps
  on the 6-frame 640x360 gate fixture at depth 3, 74 at depth 4). If the pinned
  decoder changes, re-check the gate.
- The §7.12.2.19 sort now runs **inside** the stack build (step 17), so the
  §7.12.2.20 MVP-combination candidates at step 21 are derived from the sorted
  `[0, numNearest)` prefix exactly as the decoder derives them. Applying the
  transform after `merge::build_mv_stack` (the previous attempt) left those
  tail candidates in pre-sort order and desynchronised every signalled index
  past `numNearest`.
- The MV-stack `>32` gate is handled by the spec-exact M2 stack builder in
  `src/codec/inter/merge/` (`merge::build_stack_checked`), which inserts
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
- Advanced inter tools: warp/affine, OBMC, inter-intra, TIP, OPFL, refine-MV,
  temporal MV projection, ref-MV bank, local warp, CWP.  (BAWP is wired for
  the fitted luma form behind `--video-bawp`; explicit BAWP and chroma BAWP
  are not signalled.)
- DRL reordering *(landed: `--video-drl-reorder` / `-always`, see Feature 2)*,
  short refresh flags, long-term references, multi-tile, 10-bit, 4:2:2/4:4:4,
  film grain, loop filters.
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
| `--video-refs N` | 1 | `NumTotalRefs` per P frame, 1..7 (`REFS_PER_FRAME`); ignored by `--video-implicit-ref-map` |
| `--video-num-ref-frames N` | auto | `§5.4.6 NumRefFrames`, 4..8; must be `>= --video-refs`. Auto is 8 for `--video-implicit-ref-map` |
| `--video-implicit-ref-map` | off | sequence `explicit_ref_frame_map = 0`: §7.7/§7.8 derive `NumTotalRefs`, `ref_frame_idx` and the past/future/current lists; no map bits are coded. Flat P, B-frame and `--video-tlayers` carriers; see the reference-management section |
| `--video-subpel-eighth` | off | add the 1/8-pel ME refinement stage |
| `--video-mv-precision P` | eighth | `eighth`, `quarter` or `half` pel |
| `--video-adaptive-mvd` | off | §5.4.6 `enable_adaptive_mvd`: NEWMV blocks may code §5.20.7.20/.21 `mv_joint`/`amvd_index` (`use_amvd`) when the MVD is exactly representable; CPU key+P, single tile, `--video-refs N` |
| `--video-mvd-sign-derive` | off | §5.4.6 `enable_mvd_sign_derive`: the last nonzero `mv_sign` is derived from the §5.20.7.13 parity; needs `--video-mv-precision half` (or `quarter` + `--video-flex-mvres`) |
| `--video-flex-mvres` | off | §5.4.6 `enable_flex_mvres`: `UsePerBlockMvPrecision = 1`, each NEWMV block may code `use_most_probable_precision`/`pb_mv_precision` to select a coarser exact precision (1/8→1/2 or 1/4/1/2→1) |
| `--video-compound` | off | `reference_select` compound (needs `--video-refs 2`) |
| `--video-bawp` | off | §5.20.7.6/§7.13.3.25 fitted BAWP on eligible single-prediction luma blocks (skip-only plain P chain, single tile); CPU and `--video-gpu` byte-identical |
| `--video-morph` | off | §5.20.5.4/§7.13.3.26 morphological prediction: the intra keyframe codes the screen-content + global-IBC mode-info path, `morph_pred` and the fitted §7.13.3.25 adaptation of the IBC copy. CPU, single tile, no loop filters |
| `--video-palette` | off | §5.20.8 luma palette selection on the intra keyframe: real `palette_mode_info()`/`palette_tokens()` for every leaf the colour-map analysis selects (single tile, key+P chain) |
| `--video-ibc` | off | §5.20.5.3/.4 general intra block copy on the intra keyframe: `use_intrabc`, BV stack/DRL search and the exact-copy skip block (single tile, key+P chain) |
| `--video-opfl` | off | §7.13.3.8-.12 optical-flow refinement on a NEAR_NEARMV compound block (codes `use_optflow`; needs `--video-compound-near-near` + `--b-frames`); decoder-gated on the pan fixture by `gate_video_opfl_refine.sh` |
| `--video-refine-mv` | off | §7.13.3.3-.6 refine-MV on a NEAR_NEARMV compound block (derived, no bit: §5.20.7.17 makes it non-switchable); search/prediction run per 16x16 luma sub-block with the colocated MV reused for chroma; see `docs/video-opfl-refine.md` |
| `--video-compound-wedge` | off | §7.13.3.27 wedge-mask compound candidate search (needs `--video-compound`) |
| `--video-compound-diffwtd` | off | §7.13.3.28 difference-weight compound candidate search (needs `--video-compound`) |
| `--video-cwp-weighted` | off | §5.4.6 `enable_cwp` + §5.20.7.6 `cwp_idx`: searches the non-equal `Cwp_Weighting_Factor` rows on a NEAR_NEARMV `COMPOUND_AVERAGE` block (needs `--video-compound-near-near` + `--b-frames`); decoder-gated by `gate_video_pred_surface.sh` |
| `--video-imp-mask` | off | §5.4.6 `enable_imp_msk_bld` + §7.13.3.14 `get_mask()` on the eligible CWP_EQUAL NEAR_NEARMV average; the mask is decoder-derived, so the encoder never compares the plain equal-weight blend |
| `--video-df-sub-pu` | off | §5.4.6 `enable_df_sub_pu` + §5.18.7.9 `allow_df_sub_pu` and the §7.13.3.15 sub-PU grid in §7.17 deblocking; OPFL subblocks are projected per plane, so `--video-opfl --video-deblock` exercises real sub-PU edges |
| `--video-compound-interintra` | off | single-reference §7.13.3.29/.30 inter-intra (`--video-refs 1`): codes `seq_enabled_motion_modes[INTERINTRA]`, §5.20.7.14 motion mode/interintra_mode/wedge_interintra and blends the inter prediction with an intra prediction (CPU, single tile) |
| `--video-inter-residual` | off | code a quantised inter residual |
| `--video-intra-full` | off | still product's 61-candidate luma intra search (DC, V, H, 8 directional modes with angle deltas, SMOOTH*, PAETH) instead of the DC-only fallback; chroma stays DC. Default off so the shipped streams stay byte-identical; measured +4.9% rate / −0.016 dB PSNR-Y at QP 90 (SSE-only decision) |
| `--video-tx-search` | off | §5.18.8 `TX_MODE_SELECT`: luma TX size/type/partition search (inter partition + intra/inter type); `--video-tx-lambda` (256) tunes plan selection; see `src/cuda/pipeline/video/tx/README.md` |
| `--video-idtx` | off | §5.4.8 `enable_idtx_intra = 1` + the §5.20.5.5 `fsc_mode` symbol, letting `--video-tx-search` select the 2-D identity type (9); requires `--video-tx-search`; decoder-gated (`video_idtx_gate.sh`) |
| `--video-mrl` | off | §5.4.5 `enable_mrls = 1`: directional luma intra searches `mrl_index` 1..3 and `mrl_sec_index` with the §7.13.2.1 extra reference lines; the §5.20.5.5 symbols are coded and the §7.13.2.7 edge filter is forced off for `mrl_index > 0`. `--video-intra-full` is needed for a non-zero index; decoder-gated (`video_mrl_ist_gate.sh`) |
| `--video-dip` | off | §5.4.5 `enable_dip = 1`: §5.20.7.9 `use_dip`/`dip_transpose`/`dip_mode` on luma DC_PRED keyframe blocks with `W > 4`, `H > 4`, `W*H >= 128`, reconstructed with §7.13.2.3-.6. Runs on the DC-only path and composes with `--video-intra-full`, `--video-mrl`, `--video-ist` and `--video-tx-search`; decoder-gated (`video_dip_gate.sh`) |
| `--video-ist` | off | §5.4.8 `enable_intra_ist = 1`: luma intra residuals code §5.20.8.2 `sec_tx_type` + `most_probable_stx_set` and the reconstruction applies the §7.15.3 inverse; the encoder projection is the still product's analysis tool. Inter IST stays unsignalled; decoder-gated (`video_mrl_ist_gate.sh`) |
| `--video-ibp` | off | §5.4.5 `enable_ibp = 1`: the intra keyframe and every intra leaf use §7.13.2 intra bi-prediction (IBP). No per-block bit: the decoder derives `applyIbp`/`useIBP` from `YMode` + `angle_delta`, so the luma search evaluates every candidate with `ibp::predict` and the DC/chroma path uses `ibp::predict_block` (the §7.13.2.12 `Ibp_Weights` post-filter). `--video-intra-full` exercises the two-angle §7.13.2.9 blend; DC-only exercises the DC filter. CPU, 8-bit 4:2:0, single tile, plain key+P chain; decoder-gated (`video_ibp_gate.sh`) |
| `--video-idtx-1d` | rejected | the one-axis V/H_DCT/A/FLIPADST types need the class scans + `dc_sign_horz_vert`, and pinned dav2d rejects negative H_DCT/V_DCT under `--strict 1`; see `docs/video-1d-intra.md` |
| `--video-cfl` | off | §7.13.5 CfL chroma decision `{DC, CfL}` with derived alpha, shared with `--still-gpu` (`still::chroma_decide`) |
| `--video-mhccp` | off | §7.13.6-.8 MHCCP: signals `enable_mhccp` and searches the three `CFL_MULTI` directions; implies `--video-cfl` |
| `--video-pc-wiener` | off | §7.20.4 luma `RESTORE_PC_WIENER` (own `lr_params()`/`read_lr()` dispatch); composes with `--video-chroma-lr` |
| `--video-qm` | off | §5.18.6.2 `using_qmatrix = 1` with the §9.4 default matrices (level 0, no user QM OBU); gated at `--qp 90` |
| `--video-user-qm` | off | §5.13 `OBU_QUANTIZATION_MATRIX` (type 22) before the key frame with an identity all-32 user matrix for level 0, plus `using_qmatrix = 1`; implies `--video-qm`; pinned dav2d rejects every QM OBU, so the gate is avmdec (`video_user_qm_gate.sh`) |
| `--video-tcq` | off | §5.4.8/§5.20.7.27 trellis-coded quantisation: `enable_tcq = 1`, `choose_tcq_per_frame = 1`, a per-frame `allow_tcq` bit and luma TX_CLASS_2D trellis levels with the §7.14.4 extra `dqDenom` shift; composes with `--video-tx-search` and `--video-inter-residual`; `video_tcq_gate.sh` decodes at QP 60/90/120 with `dav2d --strict 1` and compares the recon exactly |
| `--video-cctx` | off | §5.4.8/§7.14.6 cross-component transform: `enable_cctx = 1`, per-block `cctx_type` on plane 1 (after the U EOB) and the joint U/V dequant rotation; composes with `--video-tcq` and `--video-tx-search`; `video_cctx_gate.sh` decodes at QP 60/90/120 with `dav2d --strict 1` and compares the recon exactly |
| `--video-grain` | off | §5.14/§5.18.10 film grain: sequence `film_grain_params_present = 1`, at most one model OBU (type 23) per temporal unit immediately before the coded-frame OBU, per-frame `film_grain_config()`; CPU-only; composes with the flat P chain, `--tiles`, `--b-frames`, `--video-tlayers`, `--video-tip`, the format cores (10-bit / non-4:2:0) and the loop filters / `--avg-cdf --adaptive-cdf` chain; the SEF/SWITCH/RAS/OLK/leading/long-term wrappers stay rejected by name; `video_grain_gate.sh` decodes every composition at 640x360 and 64x64 with dav2d `--filmgrain 0` (recon exact) and `--filmgrain 1` (synthesis), 118 checks |
| `--video-lossless` | off | Q=0 lossless sequence via `src/codec/entropy/lossless/inter/`: its own headers (`CodedLossless = 1`, `base_q_idx = 0`, `ONLY_4X4`), DPCM/reversible WHT/IDTX and FSC; `--video-lossless-intra-period N` forces one intra block per N; `--lossless-fsc` forces the FSC coefficient path; supports 8/10-bit 4:0:0/4:2:0/4:2:2/4:4:4, the hierarchical-P schedule (`--video-tlayers 4/8/16`) and the two-reference hierarchical-B schedule (`--video-lossless-b --video-tlayers 4/8`, COMPOUND_AVERAGE); `video_lossless_gate.sh` pins the plain 4/8-frame streams plus the format, temporal-layer and B cases and compares dav2d recon exactly |
| `--video-max-depth D` / `--video-min-block N` / `--video-force-single` | 4 / 16 / off | partition controls; `N` is 4..64.  `N = 4` enables §5.20.3.1 sub-8 intra keyframe leaves with shared-chroma ownership (see "Sub-8 luma leaves"); P-frame leaves stay >= 8x8 |
| `--video-recon FILE` | off | dump the encoder reconstruction Y4M (gate use) |
| `--video-tlayers N` | off | one hierarchical-P GOP of N (4/8) frames with temporal layers |
| `--video-tlayer-drop L` | off | emit only temporal layers `<= L` |
| `--b-frames N` | 0 | N B frames per GOP (one closed GOP of N+2 frames) |
| `--video-drl-reorder` | off | §5.4.6 `DRL_REORDER_CONSTRAINT`; decoder-gated but inert on the low-delay P surface (see Feature 2) |
| `--video-drl-reorder-always` | off | §5.4.6 `DRL_REORDER_ALWAYS`; exercises the §7.12.2.19 move-to-front (`gate_video_drl_reorder.sh`) |

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
* `--video-implicit-ref-map` switches the sequence header to
  `explicit_ref_frame_map = 0`, so the frame header codes no
  `frame_explicit_ref_frame_map`, `num_total_refs` or `ref_frame_idx` bits. The
  encoder mirrors the §7.23 slot state (`mv_stack_full::RefSlotBank`, including
  the KEY/SWITCH `RefValid = first` rule) and calls
  `mv_stack_full::derive_implicit_ref_map()` before every inter frame; the
  decoder's `get_ref_frames(0)/(1)` and `get_past_future_cur_ref_lists()`
  produce the same map. The mapping is a *derived* list, not `--video-refs`:
  the flat chain grows `NumTotalRefs` as slots become valid, a 4-slot pool caps
  it at four, and the B/temporal schedules can select mixed past/future refs
  (up to `REFS_PER_FRAME` = 7). `--video-trace` prints the derived list.
* `--video-implicit-ref-map` is wired for the 8-bit 4:2:0 flat P chain, the
  B-frame schedule and `--video-tlayers`. TIP, compound, tiles, the
  random-access/LTR/OLK/leading/SEF wrappers and the 10-bit/chroma format cores
  own an explicit map and reject the flag by name. `--video-gpu` rejects it
  too.
* Decoder gate: `src/pipeline/video/gate_implicit_ref_map.sh` encodes six
  carriers with the flag, asserts the derived `NumTotalRefs` varies, and
  requires `dav2d --strict 1` to decode the IVF picture count with a
  sample-exact reconstruction (pinned `f9dafbc`; the 2026-11-13 run is exact
  for flat/B/temporal at 1..8 pictures).

## Frame MV precision

`--video-mv-precision quarter|half` writes `use_qtr_precision_mv` /
`allow_high_precision_mv` and codes the NEWMV MVD with the precision-aware
`§5.20.7.20` shell syntax (`inter/mv_precision`). ME output and the MVD are
quantised to the signalled precision. Note §5.20.7.13 only lowers the
*predictor* for precisions **coarser than 1/2 pel**; for quarter/half the
predictor is used unchanged and only the MVD is in precision units.

## Adaptive MVD / MVD sign derivation / flexible MV resolution

Three opt-in sequence bits (§5.4.6) extend the NEWMV path (§5.20.7.6). All are
off by default, so the reference bitstreams are unchanged. They share the
`inter/mv_precision` module, the per-tile §9.3 CDF state and the §8.3
`NPos`/`NPosBuf` neighbour grid.

* `--video-adaptive-mvd` codes `use_amvd` after `single_mode`. When the chosen
  MVD components are in `Amvd_Index_To_Mvd = {2,4,6,8,16,32,64,128}` the block
  signals `mv_joint`/`amvd_index` instead of the shell code; `IsAdaptiveMvd`
  pins `MvPrecision` to the frame value and suppresses the per-block precision
  symbols and sign derivation.
* `--video-flex-mvres` makes the decoder derive `UsePerBlockMvPrecision = 1`.
  Every eligible NEWMV block codes `use_most_probable_precision`; a block that
  selects a coarser value also codes `pb_mv_precision`. The encoder picks the
  coarsest precision that represents the ME MVD exactly (1/8→1/2 pel, or
  1/4/1/2→1 pel), so the reconstruction is identical and only the MVD rate
  changes. Contexts: `UseMostProbablePrecisions` is 0 for intra leaves, 1 for
  inter blocks that code no symbol, and the coded value for NEWMV; `MvPrecision`
  drives the `pb_mv_precision` context bit.
* `--video-mvd-sign-derive` omits the last nonzero `mv_sign`. §5.20.7.13 derives
  it as `lastSign & 1` over the coded magnitudes, so the encoder forces that
  sign before costing the block. It is only active when `FrameMvPrecision <=
  QUARTER` and `MvPrecision < QUARTER` (a half-pel frame, or quarter-pel with
  flex selecting one-pel blocks).

Gates: `video_adaptive_mvd_gate.sh`, `video_flex_mvres_gate.sh` and
`video_mvd_sign_derive_gate.sh` encode 64x64 and 640x360 four-picture streams
with `--video-refs 2 --video-inter-residual --video-tx-search`, require a
non-zero decision count, and decode with `dav2d f9dafbc --strict 1` with an
exact `--video-recon` match and picture count. Unsupported compositions
(B/TIP/temporal/SVC, compound, BAWP, GME/warp, OPFL/refine-MV, multi-tile,
non-8-bit 4:2:0, CUDA) are rejected by name.

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

Composition:

* **B frames / temporal layers** carry each picture's own filter plan in its
  frame header (`apply_video_loop_filter_frame_header()`); the B and
  hierarchical-P loops use the same sequence flags as the flat P chain.
* **Tiles** (`--tiles`): the frame is decided and reconstructed tile by tile,
  then one picture-level filter pass runs over the assembled reconstruction
  (CDEF strengths, LR units and GDF parameters are frame decisions).  The tile
  payloads are written afterwards with the frame plan, offset by the tile
  origin.  CDEF/CCSO/GDF compose at 4 and 8 frames.
* **Loop restoration + tiles stays a named rejection**: pinned `dav2d` leaves
  the samples at a shared tile edge unfiltered, while a frame-level LR pass
  filters them (and the spec's tile-bounded LR window does not reproduce the
  decoder either).  See `docs/negative-results.md`.
* **Frame-type wrappers** (long-term/OLK/leading/SEF/SWITCH/RAS) stay rejected
  with loop filters: their component header writers hard-code the §5.4.10
  filter flags off and do not signal the frame filter rows.

| Flag | Spec | Default | Notes |
|---|---|---|---|
| `--video-cdef` | §7.18, §5.18.7.10 | off | one strength set (no tile symbol) |
| `--video-cdef-strength N` | `CdefStrengths` 1..8 | 1 | 2..8 codes a per-64×64 `cdef_idx` |
| `--video-cdef-damping N` | `CdefDamping` 3..6 | 3 | |
| `--video-cdef-search m` | encoder search | `full` | `full`/`ladder`/`step`/`fast`; `step`/`fast` (2x2 MI decimation) need `CdefStrengths 1` |
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
the sequence header unless the corresponding opt-in flags are on:
`--video-compound-wedge`/`--video-compound-diffwtd` (masked types),
`--video-cwp-weighted` (non-equal CWP) and `--video-imp-mask` (implicit mask).
CWP and the implicit mask ride the NEAR_NEARMV carrier below; the historical
GLOBAL_GLOBALMV subset keeps CWP_EQUAL.

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

## Masked compound (§7.13.3.27–.30), opt-in

`--video-compound-wedge` and `--video-compound-diffwtd` extend the compound
candidate from CWP_EQUAL AVERAGE to the §5.20.7.16 masked forms. Both require
`--video-compound` (the gated subset is still non-joint GLOBAL_GLOBALMV over
the `(0, 1)` reference pair, zero MV, `skip_flag = 1`).

| element | value |
|---|---|
| `enable_masked_compound` | 1 in the sequence header when either flag is on |
| `comp_group_idx` | coded for **every** compound leaf: 0 = AVERAGE, 1 = masked |
| `compound_type` | 0 = WEDGE, 1 = DIFFWTD (only when `comp_group_idx = 1`) |
| wedge symbols | `wedge_quad`/`wedge_angle`, `wedge_dist1` (or `wedge_dist2`) and the `wedge_sign` bit |
| diffwtd | one `mask_type` bit |
| `cwp_idx` | still absent (`enable_cwp = 0`); masked types do not code it |
| residual | compound leaves stay `skip_flag = 1` |

The decision is SSE-only, like the AVERAGE candidate: the encoder evaluates
AVERAGE, then wedge sign 0/1 over every signalable index, then DIFFWTD
`mask_type` 0/1 in that order, and keeps the first strict minimum. The CPU
reference uses `inter::adv::search_compound()`; the CUDA backend enumerates the
same jobs and blends them with the device mask helpers, so both paths pick the
same block (`compound-wedge`/`compound-diffwtd` in
`src/cuda/pipeline/video/gate_video_gpu.sh` assert this from the trace).

`dav2d f9dafbc --strict 1` decodes both modes; the 360p 4-frame and 8-frame
CPU/GPU streams and reconstruction dumps are byte-identical, and the decoded
Y4M equals the encoder reconstruction sample for sample.

Pinned 640x360 gate streams (`--video-refs 2 --video-force-single`, generated
A/B/wedge or A/B/diffwtd clip from `gate_video_gpu.sh`):

| mode | frames | IVF MD5 | decoder verdict |
|---|---:|---|---|
| `--video-compound-wedge` | 4 | `ec8c21e51e3bb612fef95d297374456e` | 4 pictures, recon exact |
| `--video-compound-wedge` | 8 | `90af227bb58cdd20564c2478b12b5d57` | 8 pictures, recon exact |
| `--video-compound-diffwtd` | 4 | `59222d14be4fd056981f16b4561a9ff0` | 4 pictures, recon exact |
| `--video-compound-diffwtd` | 8 | `173ead7e968b8100a47f3130d36a5c13` | 8 pictures, recon exact |

```sh
# wedge / diffwtd parity gate (add --frames 8 for the long case)
src/cuda/pipeline/video/gate_video_gpu.sh --bin build/av2enc --frames 4
```

`--video-compound-interintra` is the single-reference §7.13.3.29/.30 path,
despite its historical name.  §5.20.7.14 `motion_mode_allowed()` sets
`AllowedMotionModes[INTERINTRA]` only when `!isCompound` and the block is a
single-reference 8x8..64x64 leaf, so a two-reference compound block can never
carry `interintra_mode` or `wedge_interintra`.  The flag therefore:

* writes `seq_enabled_motion_modes[INTERINTRA] = 1` with
  `seq_frame_motion_modes_present_flag = 0` (§5.4.6), so every inter frame
  inherits it without extra bits;
* codes `inter_intra` (and, when set, `interintra_mode`, `wedge_interintra` and
  `read_wedge_mode()`) after `single_mode`/`use_bawp` and before `drl_idx`
  (§5.20.7.6/.14/.15);
* searches the four `II_DC/V/H/SMOOTH` intra modes plus the 68 wedge masks
  against the inter prediction (`inter::adv::search_inter_intra`) on luma and
  chroma, and picks the blend (`COMPOUND_INTRA` mask or wedge mask) when its
  luma+chroma SSE beats the plain inter prediction.  A wedge is kept only when
  it strictly beats the plain mask, because it codes `read_wedge_mode()`.

The surface is `--video-refs 1`, CPU-only, single tile, plain key+P chain and
no `--video-compound`/`--video-bawp`/`--video-morph`/average-CDF chain.  Pinned
stream (`dav2d f9dafbc --strict 1 --threads 1`, recon byte-identical):

| fixture | pictures | IVF MD5 | feature exercised |
|---|---:|---|---|
| translating gradient/texture 320x192 | 4 | `7d52d066c7437bc3bc9c66f2fc1e52ca` | 68 inter-intra leaves, 65 `wedge_interintra = 1`, 3 `COMPOUND_INTRA` |

```sh
src/pipeline/video/gate_video_morph_interintra.sh --bin build/av2enc
```

---

# Morphological prediction on the keyframe (opt-in `--video-morph`)

`morph_pred` is read only inside §5.20.5.4 `read_intrabc_info()`, i.e. an intra
frame with `allow_screen_content_tools && enable_bawp`. The product carrier is
therefore the intra keyframe:

* §5.4.7 `seq_choose_screen_content_tools = seq_choose_integer_mv = 1`
  (the alternative forced form is `--video-scc-force`; see the forced sequence
  surface section);
* the keyframe frame header sets `allow_screen_content_tools = 1`,
  `force_integer_mv = 1`, `allow_intrabc = 1`, `allow_global_intrabc = 1`,
  `allow_local_intrabc = 0`;
* every keyframe leaf codes `use_intrabc` (§5.20.5.3); an IBC leaf additionally
  codes `skip_flag = 1`, `intrabc_mode`/DRL/`assign_mv` and `morph_pred`;
* non-IBC leaves code the mandatory `palette_mode_info()` (`has_palette_y = 0`,
  §5.20.5.5) that `allow_screen_content_tools` makes present;
* the encoder picks a global BV from the §7.12.2 stack / periodic candidates,
  copies the reconstructed block and evaluates the §7.13.3.25 template fit.
  `morph_pred = 1` is signalled only when the adapted copy lowers luma SSE
  (and §6.19.5.4 `is_offset_mv_valid(-1,-1) == 1`, enforced by
  `is_bv_encodable()`); a plain copy tie also picks IBC because it codes no
  residual.

The carrier is gated on the plain key+P chain: single tile, no
deblock/CDEF/CCSO/Wiener/chroma-LR/GDF (the IBC copy and morph template must
read the unfiltered reconstruction) and no average/CDF-continuity chain. P
frames inherit the sequence `enable_bawp`, so they code `allow_bawp = 1` and
`use_bawp = 0`; the rest of the P-frame path is unchanged.

Pinned streams (`dav2d f9dafbc --strict 1 --threads 1`, recon byte-identical
to `--video-recon`):

| fixture | pictures | IVF MD5 | feature exercised |
|---|---:|---|---|
| vertical gradient 256x192 | 1 | `1100a99b1c8f2e20d17ec2c55fe09ebe` | 1 IBC leaf, `morph_pred = 1`, SSE 0 |
| vertical gradient 256x192 | 4 | `596e2af5db198939b5e8fa20df1a21d5` | keyframe IBC + P frames with `use_bawp = 0` |
| repeated 64-row texture 256x192 | 1 | `0ddbee1a49758c3c64e15a1389687184` | 18 IBC leaves, `morph_pred = 0` |

```sh
src/pipeline/video/gate_video_morph_interintra.sh --bin build/av2enc
```

The gate generates the fixtures, pins the MD5s, runs `dav2d --strict 1`,
compares the decoded Y4M with `--video-recon` and requires `morph=1` and
`morph=0` leaves in the trace. The default `--video` stream is pinned in the
same script (`ef85e55bd85357f965169051dd691612` for the 640x360 1-frame clip).

---

# Screen-content keyframe: palette and general IBC (opt-in)

`--video-palette` and `--video-ibc` wire the still-picture screen-content tools
into the CPU `--video` keyframe.  Both reuse the gated modules
`src/codec/intra/palette/` and `src/codec/intra/ibc/`; they are CPU-only and
default off, so every historical stream is byte-identical.

## Signalling

Both carriers select the §5.4.7 per-frame screen-content form
(`seq_choose_screen_content_tools = seq_choose_integer_mv = 1`) and code the
§5.18.3.3/.4 keyframe bits:

| bit | palette only | IBC (or both) | both + `--video-bawp`/`--video-morph` |
|---|---|---|---|
| `allow_screen_content_tools` | 1 | 1 | 1 |
| `force_integer_mv` | 1 | 1 | 1 |
| `allow_intrabc` | 0 | 1 | 1 |
| `allow_global_intrabc` | absent | 1 | 1 |
| `allow_local_intrabc` | absent | 0 | 0 |
| `morph_pred` per IBC leaf | absent | absent | present (§5.20.5.4 reads it whenever `enable_bawp = 1`) |

P frames inherit the sequence `enable_bawp` bit like the morph carrier, but code
`allow_screen_content_tools = 0` / `allow_intrabc = 0`; the tools never appear
outside the intra keyframe.

## Decisions

* **Palette** (§5.20.8, §7.13.4).  For every keyframe leaf the encoder queries
  `get_palette_cache()` from the same per-MI `PaletteState` the writer
  maintains, runs `analyze_palette()` (exact palette for <= 8 distinct values,
  median-cut + Lloyd above that), rejects non-delta-representable colour sets,
  and predicts the block with `predict_palette_block()`.  The palette
  prediction is compared with the **intra prediction** SSE (the palette leaf
  codes `eob = 0` on all three planes, so the residual-coded intra
  reconstruction is not the right baseline).  A winning leaf forces
  `YMode = DC_PRED` (the §5.20.5.5 palette condition), codes the palette
  symbols after the UV mode, and recomputes chroma with DC from the
  palette-predicted luma.  CfL is cleared on palette leaves because its
  derived alpha would read the palette luma rather than the intra luma.
* **IBC** (§5.20.5.3/.4, §7.12.2, §7.13.3.18).  The keyframe evaluates the
  §7.12.2 BV stack (`build_bv_stack()` over the decoder's MI neighbour map and
  ring buffer) followed by periodic offsets `-8, -16, -32, -64, -128` in both
  directions.  A candidate must pass `is_bv_encodable()` (range, geometry,
  global wavefront/delay window, ±1 reference border) and reproduce the luma
  **and** chroma source exactly; the skip block codes no residual, so a
  non-exact copy could only lose quality.  A tie with the intra/palette
  distortion prefers IBC (fewer bits).  `intrabc_mode`/`intrabc_drl_mode` are
  NEARMV when the vector is a stack entry and NEWMV otherwise.

Both decisions run in that order on the same leaf, so a block can be palette,
IBC or ordinary intra.  On the 640x360 gate fixture the keyframe selects 117
palette leaves, 798 IBC leaves and 5 ordinary intra leaves.

## Gating

The carrier is the plain key+P chain (not B frames/temporal layers/TIP/SEF or
the frame-type wrappers), one tile, 8-bit 4:2:0, no average-CDF/CDF-continuity
chain.  `--video-palette` is also rejected with `--video-tx-search` (a palette
leaf has no luma TX partition decision; TX_MODE_SELECT needs one per leaf).
`--video-ibc` is rejected with `--video-drl-reorder-always` (§7.12.2.19 ALWAYS
also reorders the decoder's BV stack; CONSTRAINT is exact) and with
`--video-segmentation`/`--video-per-segment-q` (the per-SB delta-Q state and
the IBC skip block disagree).  The CLI and `encode_video_sequence()` reject
every other combination by name.  `--video-ibc` composes with
`--video-palette`, `--video-morph`, `--video-bawp`, `--video-intra-full`, the
reference/compound P-frame tools and every opt-in loop filter.

Two filter-path rules matter on this carrier and are now wired:

* §7.18.1 `Skips[]`: an IBC skip leaf codes `skip_flag = 1`; with
  `CdefOnSkipTxfm DISABLED` the decoder leaves it out of CDEF and §7.20.1
  LrTxSkip.  The keyframe loop-filter plan therefore imports the leaf list and
  marks IBC leaves skipped (ordinary and palette intra leaves stay unskipped).
  Without this, an IBC + CDEF keyframe diverged over most of the frame.
* §5.20.10.2 `read_ccso()` has no CDEF-style skip early return.  The CCSO unit
  symbol is coded independently of the §5.20.10.1 `cdef_idx` skip; suppressing
  it in the same branch desynchronised IBC + CDEF + CCSO streams.

## Gate

`src/pipeline/video/gate_video_scc.sh` generates deterministic 64x64 and
640x360 fixtures, pins the IVF MD5s, runs `dav2d f9dafbc --strict 1 --threads 1`
and requires the decoded picture count plus a sample-exact `--video-recon`
match.  Trace counts assert palette/IBC leaves actually appear.

| case | fixture | flags | IVF MD5 |
|---|---|---|---|
| default | 64x64 | – | `0bd528e1e7896cce1e1e1802f1155a15` |
| default | 640x360 | – | `93655a2d5bbf07ed13ee6fe3e3a25c20` |
| palette | 64x64 | `--video-palette` | `f172516a575a626a7c36d8a1f76c468f` |
| IBC flag | 64x64 | `--video-ibc` | `23b8f93c1e061e6d9289074c30664ae8` |
| both | 64x64 | `--video-palette --video-ibc` | `305312a8dbb3487b336677a1d18a0ffb` |
| both + full intra | 64x64 | `+ --video-intra-full` | `305312a8dbb3487b336677a1d18a0ffb` |
| palette | 640x360 | `--video-palette` | `5a79c9d8076f04c35c4e90b8a5f1b6db` |
| IBC | 640x360 | `--video-ibc` | `bbee14f71d1e453ef8d5247962674c88` |
| both | 640x360 | `--video-palette --video-ibc` | `cce44dd8f011babbd5d7f777a5ec81b5` |
| both + full intra | 640x360 | `+ --video-intra-full` | `cce44dd8f011babbd5d7f777a5ec81b5` |
| both, 4 frames | 640x360 | key+P | `a465b846e201521368e9c0e602910f12` |
| palette + morph | 640x360 | `--video-palette --video-morph` | `0c3de5aa02e135028ef9e2cdd33e0ddc` |
| both + morph | 640x360 | `--video-palette --video-ibc --video-morph` | `0c3de5aa02e135028ef9e2cdd33e0ddc` |
| both + all loop filters | 640x360, 4f | deblock/CDEF/CCSO/Wiener/chroma-LR/GDF | `6546d6802b8beb2652cfd0ca4d21cb3e` |

A 64x64 frame is a single superblock, so the §6.19.5.4 global-IBC delay window
admits no BV: the `--video-ibc` 64x64 stream is decoder-exact with zero IBC
leaves.  The 640x360 cases are the ones that exercise the IBC syntax (>= 600
leaves).  `--video-morph` already enables IBC (with `morph_pred`), so adding
`--video-ibc` to it does not change the stream; the palette+morph row proves
palette and the morph/IBC carrier compose.

```sh
src/pipeline/video/gate_video_scc.sh --bin build-int/av2enc
```

---

# Forced sequence surface: `--video-scc-force`, `--video-force-integer-mv`,
# `--video-reduce-pb-aspect`, `--video-seq-seg-info`

These four flags close the remaining §5.4 sequence-header combinations.  All are
CPU-only, default off, and every default stream stays byte-identical.

* `--video-scc-force` writes the §5.4.7 forced form
  (`seq_choose_screen_content_tools = 0`, `seq_force_screen_content_tools = 1`)
  instead of the per-frame SELECT form.  Every frame then infers
  `allow_screen_content_tools = 1` and codes no allow bit; the keyframe still
  codes `force_integer_mv = 1` while P frames code `force_integer_mv = 0`
  (unless pinned below).  P frames therefore code the mandatory
  `palette_mode_info()` (`has_palette_y = 0`) on eligible intra leaves, exactly
  as the decoder derives it.  Requires one of `--video-palette`,
  `--video-ibc`, `--video-morph`.
* `--video-force-integer-mv` (requires `--video-scc-force`) writes
  `seq_choose_integer_mv = 0` / `seq_force_integer_mv = 1`.  Every frame derives
  `force_integer_mv = 1` and FrameMvPrecision = `MV_PRECISION_ONE_PEL`: the
  inter header codes no `use_qtr_precision_mv`/`allow_high_precision_mv` pair,
  §5.20.7.13 lowers the predictor, the motion search quantises MVs to the
  integer grid and the MVD shell codes one-pel units.  Rejected by name with
  `--video-mv-precision`, `--video-flex-mvres`, `--video-adaptive-mvd`,
  `--video-mvd-sign-derive` and `--video-gme`/`--video-warp`.
* `--video-reduce-pb-aspect` writes `reduce_pb_aspect_ratio = 1` plus
  `max_pb_aspect_ratio_log2_minus_1 = 1` (4:1; `--video-pb-aspect-2` selects
  2:1, i.e. the extra bit = 0).  The partition search and the tile writer run
  with the derived `MaxPbAspectRatio` (2 or 4) so every coded leaf satisfies
  §6.19.4.1 and `is_partition_allowed()` recomputes the same allowed set.
  Gated on the plain key+P chain (the B/TIP/frame-type wrappers build their
  own sequence header and partition plan) and rejected with `--tiles` (the
  tiled header copy does not carry it).
* `--video-seq-seg-info` writes §5.4.4 `seq_seg_info_present_flag = 1` with
  `seq_allow_seg_info_change = 1` and the full §5.4.9 `seg_info(MaxSegments)`
  table (the key-frame form).  Each frame codes `reuse_seg_info`; the P frames
  that add a SKIP/GLOBALMV feature code their own table.  The frozen form
  `--video-seq-seg-info-frozen` writes `allow_change = 0`, so every frame reuses
  the table with no `reuse_seg_info` bit.  Composes with `enable_ext_seg`
  (`--video-segments 9..16`).  Requires `--video-segmentation` or
  `--video-per-segment-q`; rejected with `--tiles`.

`src/pipeline/video/gate_video_seq_surface.sh` pins the IVF MD5s at 64x64 and
640x360, runs `dav2d f9dafbc --strict 1 --threads 1`, checks the picture count
and requires a sample-exact `--video-recon` match.  The force-integer case
asserts at least one NEWMV block is chosen and every traced MVD component is a
multiple of 8; the reduce-aspect cases assert the 4:1, 2:1 and default trees
produce three distinct streams.

```sh
src/pipeline/video/gate_video_seq_surface.sh --bin build-int/av2enc
```

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

## NEAR_NEARMV compound carrier (opt-in, CPU)

`--video-compound-near-near` extends the compound candidate set from the
historical `GLOBAL_GLOBALMV` subset to §5.20.7.6 `NEAR_NEARMV` on a
two-reference B frame.  Both lists are merged independently (`has_second_drl`),
so the block codes `drl_idx0` + `drl_idx1` and stores the per-list MVs.  The B
frame's references are one past and one future picture, so
`opfl_allowed_for_refs()` is true — this is the carrier the temporal refinement
tools need (`--video-opfl`, `--video-refine-mv`; both are decoder-exact and
gated separately, see `src/pipeline/video/gate_video_opfl_refine.sh` and
`docs/video-opfl-refine.md`).

```sh
src/pipeline/video/gate_video_near_near.sh --bin build/av2enc \
  --dav2d tools/bin/dav2d
src/pipeline/video/gate_video_opfl_refine.sh --bin build/av2enc \
  --dav2d tools/bin/dav2d
```

Pinned (640x360, smooth (3,1)-per-frame pan fixture generated by the gate):

| flags | IVF MD5 | coded/output | NEAR_NEARMV blocks |
|---|---|---:|---:|
| `--b-frames 1 --video-max-depth 3` (3 frames) | `7c9b6213ee3eb5c16a25fcbe9e799466` | 3/3 | 140 |
| `--b-frames 3 --video-max-depth 3` (5 frames) | `8baef2a99f13a16b3df21440c678ff41` | 5/5 | 542 |
| `--b-frames 1 --video-max-depth 4` (3 frames) | `e961f4e3af862c7e8cde83e6c83b2f67` | 3/3 | 269 |

Every case is `dav2d --strict 1` exact (decoded picture count and per-sample
recon).  The flag defaults off, so `--video-compound` and every default
CPU/GPU stream keep their pinned bytes.  `--video-gpu` rejects the flag by
name (the CUDA compound backend implements only the GLOBAL subset).

## Weighted CWP and the implicit mask (opt-in, CPU)

`--video-cwp-weighted` and `--video-imp-mask` extend the NEAR_NEARMV
`COMPOUND_AVERAGE` decision; both require `--video-compound
--video-compound-near-near --b-frames` and are CPU-only.

| element | value |
|---|---|
| `enable_cwp` | 1 with `--video-cwp-weighted` |
| `cwp_idx` | coded for every `COMPOUND_AVERAGE`, `SIMPLE`-motion NEAR_NEARMV block that is not `use_optflow`/`use_refinemv`; 0 = `CWP_EQUAL` |
| weights | `Cwp_Weighting_Factor[is_same_side()]`; a B frame with one past + one future reference uses the opposite-side row `{8, 12, 4, 10, 6}` |
| `enable_imp_msk_bld` | 1 with `--video-imp-mask` |
| `get_mask()` | §7.13.3.14, applied by the decoder whenever `COMPOUND_AVERAGE && CWP_EQUAL && YMode != GLOBAL_GLOBALMV` and both references are unscaled |

The search order follows `multiref::decide_compound()`: the derived
implicit-mask average first, then the five CWP weights, all SSE-only.  A
critical detail is that when the implicit mask is eligible and
`--video-cwp-weighted` is off, the plain equal-weight blend is **not** a legal
decoder output; `search_compound()` is therefore bypassed for that one family
and only the mask blend is evaluated.  Comparing the two would let the encoder
pick a prediction the decoder cannot derive.

Gate: `src/pipeline/video/gate_video_pred_surface.sh` (64x64 3-frame and
640x360 5-frame pan fixtures).  Every stream decodes `dav2d --strict 1` with
the expected picture count and byte-exact recon; the implicit-mask stream's
reconstruction is asserted different from the equal-weight carrier, so the
mask path is exercised rather than merely signalled.

## DF sub-PU (opt-in, CPU)

`--video-df-sub-pu` signals sequence `enable_df_sub_pu = 1` and frame
`allow_df_sub_pu = 1` for every inter frame (§5.18.7.9).  The deblock grid
already stores `SubPuSize = Max_Tx_Size_Rect[MiSize]`,
`Max_Tx_Size_Rect[ChromaMiSize]` and the per-4x4 `SubPuColBase/RowBase`.

For an ordinary whole-PU leaf the §7.13.3.15 process early-returns, so
`allow_df_sub_pu` changes no filtered edge.  The flag becomes observable on the
OPFL/refine-MV carriers, whose prediction splits the block into §7.13.3.8 or
§7.13.3.1 subblocks: the encoder collects those rects per plane and feeds them
to `video_save_subpu()`.  The gate runs `--video-opfl --video-deblock
--video-df-sub-pu`, verifies byte-exact recon, and asserts the reconstruction
differs from the same stream with `allow_df_sub_pu = 0`.

Tiles and the frame-type wrappers (TIP/BRU/SWITCH/RAS/LTR/OLK/leading/SEF) are
rejected by name: their frame headers code `allow_df_sub_pu` in a different
branch.  Lossless is rejected because `deblocking_filter_params()` returns
before the bit.

---

# Codec-tools wave: CfL / MHCCP / PC-Wiener / QM (opt-in)

Four intra/restoration/quantizer tools with a device path and the usual
CPU↔GPU byte-parity gate (`src/cuda/pipeline/video/gate_video_gpu.sh`).  Every
flag defaults off, so the shipped `--video` and `--video-gpu` streams keep
their pinned bytes.

| Flag | Spec | CPU ref | CUDA |
|---|---|---|---|
| `--video-cfl` | §7.13.5 | `src/pipeline/video/video_chroma_search.cpp` | `src/cuda/pipeline/still/chroma/` (`still::chroma_decide`) |
| `--video-mhccp` | §7.13.6–.8 | same + `cfl::mhccp::search_mhccp` | same, `enable_mhccp = 1` |
| `--video-pc-wiener` | §7.20.4 | `src/codec/filters/pc_wiener/pc_wiener_stream.cpp` | `src/cuda/codec/filters/pc_wiener/` + `still_tx` |
| `--video-qm` | §5.18.6.2 | `qm::build_q2_matrix` in `compute_tx()` | `still_tx` q2 matrices |

Wiring notes:

* CfL reads the reconstructed luma of the current block, so both paths publish
  the luma reconstruction (host write + `write_recon_luma` device upload)
  before the chroma decision.  The CPU reference and the CPU-emulation backend
  share `video_chroma_search.cpp`; the CUDA backend passes the §6.19.2.3
  above-right/below-left counts to `still::chroma_decide()`.
* MHCCP signals §5.4.5 `enable_mhccp` and reuses the gated
  `cfl::mhccp::write_cfl_alphas()` adapter for `cfl_mhccp`/`cfl_index`/
  `cfl_mh_dir`.  Tie order matches the still product: DC is the incumbent,
  then derived alpha, then `CFL_MULTI`.
* PC-Wiener has its own `§5.18.7.11 lr_params()` writer (`pc_wiener_stream`)
  and a `§5.20.10.4 use_pc_wiener` per-unit dispatch in `video_syntax.cpp`; the
  chroma Wiener NS plan keeps the shared `filters_stream` writer.  The CUDA
  backend keeps a persistent `pc_wiener` workspace, the §7.20.1 `LrTxSkip`
  grid (skip || eob == 0, including the keyframe) and u16 CurrFrame/CdefFrame
  planes.
* QM signals `using_qmatrix = 1` with default level 0 only; no QM OBU is
  emitted.  The transform chain uses the per-coefficient `q2 = Round2(q*m, 5)`
  (CPU `compute_tx` and the device `still_tx` stage).  The shared device QM
  stage builds q2 from the pinned 528 step, so the gate is QP 90.
* `--video-user-qm` is the CPU-only §5.13 carrier: one type-22 OBU after the
  sequence header in the key temporal unit (level 0, identity all-32 matrix).
  `--video-qm`'s frame header is unchanged; §6.12 keeps `QmDataPresent[0]`
  across later CLK-free P frames.  The encoder state and `compute_tx()` use the
  same `UserQm[0]`, so the all-32 identity dequantizes exactly.  Pinned dav2d
  f9dafbc **rejects every QM OBU** (`QM OBUs with user-defined QMs are
  currently unsupported`), so this carrier is decoder-gated with `avmdec`:
  `video_user_qm_gate.sh` pins base/QM streams and proves the user stream's
  four decoded pictures equal `--video-recon` (0 differing samples, 720p).
* `--video-grain` plans one `GrainStreamPlanner` over the coding-order
  pictures.  A changed model emits one type-23 OBU immediately before the
  coded-frame OBU; §5.18.10.1 codes `apply_grain`/`fgm_id`/`grain_seed` in the
  frame header.  Film grain is output-side synthesis, so `dav2d --filmgrain 0`
  must equal `--video-recon` (0 samples at 720p); `--filmgrain 1` decodes the
  same picture count with different samples.  The plain chain, `--tiles`,
  `--b-frames`, `--video-tlayers`, `--video-tip`, the `--bit-depth/--chroma`
  format cores, the §7.17-§7.20.5 loop filters and the `--avg-cdf
  --adaptive-cdf` chain all carry the config and the model OBU (a TIP OBU
  codes its config right after the TIP block; every other schedule uses the
  normal `frame_header_info()` tail).  The filters and the CDF chain are
  orthogonal to grain: the encoder reconstruction is never grain-filtered and
  the model OBU is an independent unit OBU, so each filter plan and the §8.2.4
  CDF image keep their bytes.  The B/temporal schedules plan on the source
  picture in coding order, so the model is available to every unit.  The
  SEF/SWITCH/RAS/OLK/leading/long-term components own their §5.18.2 writers and
  code no `film_grain_params_present`/`film_grain_config()`, so those wrappers
  stay rejected by name.
* `--video-idtx` signals §5.4.8 `enable_idtx_intra` and codes the §5.20.5.5
  `fsc_mode = 0` symbol on every <= 32x32 luma coding block; the TX search then
  exposes the 2-D `IDTX` type (9).  Without the flag the search filters IDTX so
  `--video-tx-search` cannot select an identity type while the sequence header
  says `enable_idtx_intra = 0`.  One-axis types (10..15) are rejected by the
  writer and by the CLI (`--video-idtx-1d`): pinned dav2d rejects negative
  H_DCT/V_DCT coefficients under `--strict 1` (`docs/video-1d-intra.md`).
* `--video-mrl` reuses the still path's §5.20.5.5 syntax adapter
  (`src/codec/intra/mrl/stream/`) and §7.13.2.1 prediction core.  The writer
  keeps a per-tile `UsesMrls` grid (`MrlContext`) and its `mrl_index` /
  `mrl_sec_index` CDFs ride the §8.2 packed CDF image, so `--avg-cdf`,
  `--adaptive-cdf` and cross-frame continuity stay exact.  The encoder runs the
  still MRL candidate loop only for directional modes (mrl 0 is the ordinary
  search winner), and the §5.20.7.29 TX-type mapping adds
  `Mrl_Index_To_Delta` before wide-angle mapping.  MRL is 8-bit 4:2:0,
  single-tile, CPU-only.
* `--video-ist` reuses `src/codec/transform/secondary/` for the §5.20.8.2
  legality, kernel selection, §9.7 tables, §7.15.3 inverse and §9.3 symbol
  costs.  `txsec::project_ist_levels()` is the still encoder's
  `project_ist_levels()` moved into the component, so the video decision uses
  the exact same analysis projection.  The symbol follows the intra TX_TYPE and
  precedes the coefficients; a TX-search plan with more than one transform
  leaf keeps `sec_tx_type = 0`.  Inter IST is not implemented, so
  `enable_inter_ist` stays 0.  IST does not compose with `--video-qm` (the §7.15.3
  inverse would need the per-coefficient q2 dequantizer).

Gate evidence (360p, `--video` vs `--video-gpu`, `dav2d --strict 1`, all
sample-exact; `gate_video_gpu.sh --cfl-only|--mhccp-only|--pcw-only|--qm-only`):

| Case | IVF MD5 |
|---|---|
| `--video-cfl`, 4 frames | `e7b8e15eec0ffa098e0c30f66d78794e` |
| `--video-cfl`, 8 frames | `7ba251e6588b0767b65e7ee77a0fd2e0` |
| `--video-mhccp`, 4 frames (textured) | `4d7569186a50c9800759319a3ea4518e` |
| `--video-mhccp`, 8 frames | `c601a2813697241cae94e9cbb17898c4` |
| `--video-mhccp --tiles 2x1` | `ac7c1874b1e045904f9b2734e310dd40` |
| `--video-pc-wiener`, 4 frames | `cfbaa720530af114cf9129892e3173fd` |
| `--video-pc-wiener`, 8 frames | `c0b113358467dcda75bc98447f5a3439` |
| `--video-pc-wiener --video-chroma-lr` | `5a8d7a0bed6801672aec478f4213d89e` |
| `--video-qm`, 4 frames | `16f4a20f5bc5e3c3ae9d49ba2a00d64a` |
| `--video-qm`, 8 frames | `60006eafb2883ea103e8adeabba8b2a8` |
| `--video-user-qm`, 4 frames, 720p | `2bba524225209c18b7af16ddf298f493` (avmdec) |
| `--video-grain`, 4 frames, 720p | `e60ded0a500a856984cd1b575c89103b` |
| `--video-grain`, 8 frames, 720p | `e4c9dbd7e105160aea90c66947b69bbb` |
| `--video-lossless`, 4 frames, 720p | `3569437af68c2ff640e6deda5be2b5a5` |
| `--video-lossless --video-lossless-intra-period 4`, 4 frames | `59cf90c17e97639bfd9ed8d06670a964` |
| `--video-lossless ... --lossless-fsc`, 4 frames | `31e50ec0d113cc02d1383664ff371124` |
| `--video-lossless`, 8 frames, 720p | `19c4b81811692127c108fad1c6ec7f47` |
| `--video-lossless --video-tlayers 4`, 4 frames, 720p | `f0a2ce304b609a2312822838ab4f7125` |
| `--video-lossless --video-tlayers 8`, 8 frames, 720p | `1b02c94e0c0b044157e86e973261ac36` |
| `--video-lossless`, 10-bit 4:2:0, 4 frames, 360p | `3b8293a818b771cdab190a53f5635813` |
| `--video-lossless --video-tlayers 4`, 10-bit 4:2:0, 4 frames | `2cf0113f2a27957a86e836b0344e9335` |
| `--video-lossless`, 10-bit 4:2:2/4:4:4, 4 frames, 360p | `b5835150471b2d3f163df693cc6e4061` / `37921674d1ef264eec9194af26cf38ce` |
| `--video-lossless`, 8-bit 4:2:2/4:4:4, 4 frames, 360p | `34304fc9f5dd82347bf60dd548fd670e` / `ea09a31d9544d4eaf476a6c1461b0df1` |
| `--video-tx-search`, 4 frames, 720p | `be155c435a6c4d591bb5b7ab7d084c7e` |
| `--video-tx-search --video-idtx`, 4 frames | `13c96481b49d6150d7cb9fa4240f5c69` |
| `--video-tx-search --video-idtx`, 8 frames | `c4e7ceab9bf98e672c992b30b93aa871` |

The gate also runs **CPU-reference-only** cases (`PASS video-cpu ...`) for every
new tool, including `--video-force-single --video-cfl|--video-mhccp`.  The
bottom superblock is a 64x64 `PARTITION_NONE` leaf whose chroma block extends
past the coded MI-aligned plane (legal while `has_rows` is true); the CPU
encoder has no inter candidate there, so it codes the leaf intra.  CfL/MHCCP
used to write the DC/CfL prediction past the chroma plane in that case
(glibc `double free or corruption`, exit 134).  `chroma_search::choose()` now
builds a padded scratch copy of the chroma plane when the block does not fit,
and the CUDA backend clips `write_recon_420`/`write_recon_luma` to the plane
bounds like `write_plane_block()` does.

## Remaining gap in this wave

The segmentation/delta-Q surface is wired and gated.  It still does not compose
with `--video-qm`, `--video-tcq`, `--video-cctx`, `--adaptive-cdf`,
`--cdf-continuity`, multiple tiles, B/temporal-layer/random-access/TIP
schedules, or the 10-bit/non-4:2:0 video core.  `--video-seg-globalmv` is not
combined with `--video-per-segment-q` (an edge intra leaf would have to change
segment and therefore the quantizer); the skip/globalmv feature segments keep
`ALT_Q = 0` on inter frames for the same fallback reason.  A key frame cannot
activate `SEG_LVL_SKIP` at all (§6.19.4.1 requires it to be 0 for every intra
block), so the key frame table keeps ALT_Q and the P-frame table enables the
feature; the table is per frame.  The pinned `dav2d` f9dafbc also reads
`is_inter` on a `SEG_LVL_GLOBALMV` block instead of inferring it, so the
carrier codes the bit (see `docs/negative-results.md`).
# Format video core (10-bit / 4:2:2 / 4:4:4 / 4:0:0)

`--video --bit-depth 10` and `--chroma 422|444|400` are served by
`src/pipeline/video/hbd/`, a `uint16_t` sibling of this core. It shares the
syntax writer and header writer (now `fmt::Format`-aware) and adds depth-aware
prediction/transform/reconstruction. The default 8-bit 4:2:0 path and every
pinned stream above are unchanged.

* CPU and CUDA share one orchestration; the CUDA backend is
  `src/cuda/pipeline/video/hbd/` (uses `hbd_tx_batch`).
* Flat key + P chain, 1..7 references, partition/ME/MV-precision flags, DC
  intra fallback and `--video-inter-residual` are wired.
* Deblock, loop filters, tiles, B frames, temporal layers, frame-type
  wrappers, TX search, full intra, GME/warp, compound, DRL, average CDF,
  MFH/timing and CRF/VBR are rejected by name on these formats.
* 4:2:2 defaults to `--video-max-depth 2`; 16-pixel 4:2:2 leaves
  (`--video-max-depth 3/4`) are decoder-exact now that `partition` models
  4:2:2 chroma, and are gated by `gate_video_422_depth.sh`.
* Gate: `src/cuda/pipeline/video/gate_video_formats.sh` (CTest
  `gate_video_formats`) at 640x360 / 4 frames, all seven formats.
## Data-driven intra prediction (opt-in, CPU)

`--video-dip` writes §5.4.5 `enable_dip = 1` in the non-tiled sequence header
and codes §5.20.7.9 `dip_mode_info()` after `uv_mode`/palette and before the
residual for every luma `DC_PRED` block that satisfies

```
Block_Width > 4 && Block_Height > 4 && Block_Width * Block_Height >= 128
```

`use_dip` is coded for both key and P-frame intra leaves (the symbol is present
for every eligible block), but the mode decision only runs on the intra
keyframe: `dip::decide()` scores the twelve normative `(dip_mode, transpose)`
candidates by SSE against the incumbent prediction and replaces it only on a
strict win.  When it wins the leaf keeps `YMode = DC_PRED` and codes
`use_dip = 1`; MRL and palette state are cleared because both are incompatible
with a DIP block.  Prediction uses `dip::prepare_edges()` (the `w + w/4` /
`h + h/4` sample counts), never the ordinary `intra::prepare_edges()`.

`video_syntax.cpp` keeps a per-MI `UseDip` grid and derives the §8.3.1
`TileUseDipCdf` context from the same §5.20.4.1 `collect_npos()` set the
decoder uses; the two DIP CDF arrays ride the tile CDF bank, so
`--avg-cdf/--adaptive-cdf` streams stay exact.

Gate: `src/pipeline/video/video_dip_gate.sh` (registered as `gate_video_dip`).
It pins 64x64 and 640x360 streams with and without `--video-intra-full` and
`--video-tx-search`, checks that each case codes at least one DIP block, and
compares the encode-side `--video-recon` to `dav2d f9dafbc --strict 1` output
sample for sample.  Default `--video` cases keep their IVF MD5 and code no DIP
block.

Rejected by name: `--tiles` (the tiled sequence-header writer is a separate
copy), `--video-sdp` (the SDP tree path is not DIP-gated), `--video-lossless`
(its own writer) and the B/temporal-layer/random-access/TIP/SVC frame
schedules.  Non-8-bit/4:2:0 formats and `--video-gpu` are rejected too.
