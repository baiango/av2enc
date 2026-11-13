# delta-Q / segmentation glue for M2

Owner: delta-Q/segmentation syntax agent.  Path: `src/encoder/m2/deltaq/`.

The normative syntax, CDFs, readers and writers live in
`src/encoder/m1/deltaq/` (namespace `av2enc::deltaq`).  This directory contains
only the M2 frame/block glue:

* `FrameQpPlan` (RC decisions + `primary_ref_is_none`);
* `frame_qp_header()` and the frame-header splice points;
* `FrameQpContext` (derived segmentation, tile QP state, segment grid, CDFs);
* the exact `inter_frame_mode_info()` / `intra_frame_mode_info()` call order;
* a decoder gate that splices the header into a real M2 stream.

## `primary_ref_is_none` — the subtle part

`FrameQpPlan::primary_ref_is_none` is **not** `VideoFrameHeader.keyframe`.

`choose_primary_secondary_ref_frame()` only considers references whose frame
type is `INTER_FRAME`.  M2's first P frame references the key frame
(`KEY_FRAME`), so `DerivedPrimaryRefFrame` is `PRIMARY_REF_NONE` even though the
frame itself is an inter frame.  Consequence (§5.18.7.1):

* `segmentation_update_map` is **not coded** (inferred 1);
* `segmentation_temporal_update` is **not coded** (inferred 0);
* the module must not write those two bits.

A P frame that references an earlier `INTER_FRAME` must set
`primary_ref_is_none = false`; the two bits are then coded.  `VideoFrameHeader`
does not carry the reference types, so the integration must compute this in
`src/encoder/m2/video_frame.cpp` (where `RefFrameType` / `first_slot_with_ref` are
available) and pass it through `FrameQpPlan`.

`disable_cross_frame_cdf_init = 1` does **not** force
`DerivedPrimaryRefFrame` to NONE; it only selects CDF initialization.

## API

```cpp
av2enc::video::deltaq::FrameQpPlan plan;
plan.base_q_idx = 90;
plan.delta_q = rc_delta_q;
plan.segmentation = rc_segmentation;
plan.enable_ext_seg = false;          // M2 sequence writes enable_ext_seg = 0
plan.primary_ref_is_none = true;      // P frame whose ref is a key frame

// Frame header: write after base_q_idx, before deblocking_filter_params().
av2enc::video::deltaq::write_frame_qp_syntax(writer, plan, frame_header, &error);
```

`FrameQpContext` + block hooks:

```cpp
av2enc::video::deltaq::begin_frame(context, plan, frame_header, mi_cols, mi_rows,
                        prev_segment_ids, &error);
av2enc::video::deltaq::begin_superblock(context, node.mi_col);   // at each SB root

// §5.20.7.1 inter_frame_mode_info()
if (context.segmentation.seg_id_pre_skip)
  write_inter_segment_id_pre(context, enc, node, segment_id, skip_flag, &error);
// ... code_is_inter() ...
// ... if (is_inter) code_skip_flag() else skip_flag = 0 ...
if (!context.segmentation.seg_id_pre_skip)
  write_inter_segment_id_post(context, enc, node, segment_id, skip_flag, &error);
write_delta_qindex(context, enc, node, skip_flag, reduced_delta, &error);
// ... write_inter_mode_info / write_intra_leaf ...
commit_segment_block(context, node, segment_id, &error);
```

For a key frame (`intra_frame_mode_info()`):

```cpp
write_intra_segment_id(context, enc, node, segment_id, /*skip=*/false, &error);
write_delta_qindex(context, enc, node, /*skip=*/false, reduced_delta, &error);
commit_segment_block(context, node, segment_id, &error);
```

`write_delta_qindex()` no-ops automatically when the block does not code a
delta (full-SB skip or `ReadDeltas == 0`).  `begin_superblock()` sets
`ReadDeltas = delta_q_present` and clears `LeftSegPredContext` when
`mi_col == 0`.

`write_superblock_delta_q()` takes an `rc::DeltaQPlanStep` and verifies both
`qindex_before` (tile state) and `qindex_after` (apply rule) before writing.

## Exact M2 call sites (for reference; the product wiring is below)

### `src/encoder/m2/video_headers.cpp`, `write_frame_header_syntax()`

Replace:

```cpp
writer.put_bits(c.base_q_idx, 8U);
writer.put_bit(false);              // segmentation_enabled
writer.put_bit(false);              // using_qmatrix
writer.put_bit(false);              // delta_q_present (base_q_idx > 0)
```

with:

```cpp
writer.put_bits(c.base_q_idx, 8U);
av2enc::video::deltaq::write_frame_qp_syntax(writer, config.qp_plan, header, &error);
```

`VideoConfig` (or a new per-frame config argument) must carry the
`av2enc::video::deltaq::FrameQpPlan`.  The sequence header already writes
`enable_ext_seg = 0` / `seq_seg_info_present_flag = 0`
(`video_headers.cpp` lines 136-138).

### `src/encoder/m2/video_syntax.cpp`, `Writer::write_block()`

Current order is:

```cpp
if (plan_.inter_frame) {
  code_is_inter(...);
  if (node.is_inter) {
    code_skip_flag(...);
    write_inter_mode_info(...);
    ...
  } else {
    write_intra_leaf(...);
  }
} else {
  write_intra_leaf(...);
}
```

Required order (insertions marked `+`):

```cpp
if (plan_.inter_frame) {
+ if (seg.seg_id_pre_skip) write_inter_segment_id_pre(...);  // §5.20.7.2
  code_is_inter(...);
  if (node.is_inter) {
    code_skip_flag(...);
+   if (!seg.seg_id_pre_skip) write_inter_segment_id_post(...);
+   write_delta_qindex(...);                                  // §5.20.5.11
    write_inter_mode_info(...);
    ...
  } else {
+   if (!seg.seg_id_pre_skip) write_inter_segment_id_post(...);
+   write_delta_qindex(...);
    write_intra_leaf(...);
  }
} else {
+ if (seg.seg_id_pre_skip) write_intra_segment_id(...);      // before cdef
+ // cdef/ccso are disabled in M2, so delta follows directly:
+ write_delta_qindex(...);
  write_intra_leaf(...);
}
+ commit_segment_block(...);
```

`Writer::run()` must call `begin_superblock(context, node.mi_col)` at each
superblock root and `clear_left_context()` when `node.mi_col == 0`.

`Writer::init_cdfs()` must select the coefficient slice from raw
`base_q_idx` (`deltaq::coeff_cdf_slice(config.base_q_idx)`) instead of the
hard-coded slice-0 tables.  This does not depend on delta-Q/segmentation.

## RC integration

The pipeline/RC output maps directly:

| RC | M2 |
|---|---|
| `rc::RateController::begin_frame().base_q_idx` | `FrameQpPlan::base_q_idx` |
| `rc::DeltaQParams` | `FrameQpPlan::delta_q` |
| `rc::SegmentationParams` | `FrameQpPlan::segmentation` |
| `rc::DeltaQPlan` (per tile, one step per SB) | `write_superblock_delta_q()` |
| `rc::SegmentationMap` (previous frame) | `begin_frame(..., prev_segment_ids, ...)` |

The product build adds `src/codec/entropy/deltaq/video/video_deltaq.cpp` to
`av2enc_core` and defines `AV2ENC_VIDEO_DELTAQ`; standalone gates compile the
neutral path without it.  See "Product wiring (CPU `--video`)" below for the
call sites that were actually integrated.

## Tests and decoder gate

```sh
./src/encoder/m2/deltaq/build_and_test.sh          # 121 checks
./src/encoder/m2/deltaq/build_gate.sh              # needs dav2d
DAV2D=/path/to/dav2d ./src/encoder/m2/deltaq/build_gate.sh
./src/encoder/m2/deltaq/build_gate.sh --no-decode  # bytes only
```

## Product wiring (CPU `--video`)

`src/pipeline/video/video_headers.cpp`, `video_syntax.cpp` and `video_frame.cpp`
now call this adapter behind `AV2ENC_VIDEO_DELTAQ` (defined by the root
`CMakeLists.txt` for `av2enc_core`; standalone gates compile the historical
neutral path without it).  The gated CLI surface is:

| Flag | Header | Tile side | Encoder |
|---|---|---|---|
| `--video-segmentation` | `segmentation_enabled = 1`, `SEG_LVL_ALT_Q = 0` for segment 1 | `intra_segment_id` / `inter_segment_id` from the per-SB variance split | segment id only; qindex unchanged |
| `--video-delta-q` | `delta_q_present = 1`, `delta_q_res` | `begin_superblock` + one delta step at the first leaf of every SB | per-SB `CurrentQIndex` staircase (`rc::plan_delta_q`) drives the block quantizer |
| `--video-per-segment-q` | `segmentation_enabled = 1`, `SEG_LVL_ALT_Q = delta` for segment 1 | segment ids as above | §7.14.2 effective qindex drives quantizer and deblock metadata |

Call order as coded (the adapter refuses a mismatch):

```cpp
// inter frame, write_block()
if (SegIdPreSkip) write_inter_segment_id_pre(...);   // §5.20.7.2
code_is_inter(); code_skip_flag();
if (!SegIdPreSkip) write_inter_segment_id_post(...);
write_cdef_index(); write_ccso_index();
write_superblock_delta_q(...);                       // first leaf of the SB only
write_inter_mode_info(); ...residual...
commit_segment_block(...);                           // §5.20.5.7

// intra keyframe / intra leaf in a P frame
write_intra_segment_id(...);                         // §5.20.5.3
write_cdef_index(); write_ccso_index();
write_superblock_delta_q(...);
write_intra_leaf(...);
commit_segment_block(...);
```

### Two hazards the wiring must replay

1. **Skip-first-leaf delta-Q.** `read_delta_qindex()` codes nothing when
   `MiSize == SbSize && skip_flag`, and clears `ReadDeltas` regardless.  A
   superblock whose only leaf is a skipped 64x64 inter block therefore cannot
   move `CurrentQIndex`; `write_superblock_delta_q()` verifies the achieved
   qindex and fails loudly if the plan asked for a nonzero step.  The planner
   walks its steps, rewrites the unreachable targets to the carried qindex and
   re-runs `rc::plan_delta_q()` until stable.  If the quantizer-dependent skip
   decision oscillates, the encoder falls back to an all-zero ramp.
2. **Skipped-leaf segment inference.** With `update_map = 1`,
   `SegIdPreSkip = 0` and no lossless segment, §5.20.7.2 codes no segment id
   for a skipped inter leaf: the decoder infers
   `segment_prediction(above/left/above-left)`.  The encoder replays
   `deltaq::segment_prediction()` over its own committed map, stores that id in
   `BlockSyntax::segment_id` (the module errors if it differs) and uses that
   id's §7.14.2 qindex in the §7.17 deblock grid.

### Limits of the product wiring

- CPU 8-bit 4:2:0 `--video` only (`--video-gpu` and the 10-bit / 4:2:2 / 4:4:4
  core are rejected by name).
- One tile (`--tiles > 1x1` rejected: the tiled header writer does not own the
  §5.18.7 rows), fixed `--rc cqp` base_q_idx, no `--adaptive-cdf` /
  `--cdf-continuity` (the delta/segment CDFs are not in the §8.2.4 image), no
  B/temporal-layer/random-access/TIP schedules, no `--video-qm`.
- `--video-per-segment-q` and `--video-delta-q` are mutually exclusive (the
  encoder supports `ALT_Q + delta-Q` arithmetically but the planner would need
  a joint target; not wired).
- `SEG_LVL_SKIP` / `SEG_LVL_GLOBALMV`, temporal segmentation prediction, 16
  segments and UV/DC delta-Q remain unwired, as in the still path.

The gate:

1. encodes a 64x64 2-frame M2 stream (intra key + one 64x64 GLOBALMV skip
   block) with `force_single_block`, `force_global_mv`,
   `enable_inter_residual = false`;
2. extracts the P tile and splices real `segmentation_params()` /
   `delta_q_params()` bits into the P frame header, with
   `primary_ref_is_none = true`;
3. verifies the disabled splice is byte-identical to the M2 P OBU;
4. decodes with the pinned dav2d and compares every sample to the M2 key-frame
   reconstruction.

Passing variants: `deltaq` (`delta_q_present=1`, res=3), `segmentation`
(`segmentation_enabled=1`, no delta), and `both`.

The gate covers the header syntax, the full-SB-skip delta/segment gating, and
the key-reference `update_map` inference.  The block-level symbol coders are
covered by `src/encoder/m1/deltaq/deltaq_test.cpp` (all values, escapes,
contexts, temporal update, ext segments).  A dav2d gate for those symbols
requires the M2 tile writer to call the hooks above.

## Limits

Same as `src/encoder/m1/deltaq/README.md`: no SDP/CHROMA_PART, no QM, no UV
delta-Q, 8-bit qindex domain, reduced delta magnitude <= 516.
