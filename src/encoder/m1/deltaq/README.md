# delta-Q / segmentation bitstream syntax (M1)

Owner: delta-Q/segmentation syntax agent.  Path: `src/encoder/m1/deltaq/`.

This component implements the **bitstream side** of AV2 delta-Q and
segmentation.  It does not decide quantizers, partitions, or modes; the rate
controller in `src/encoder/rc/` produces the plans and this module turns them
into syntax and the exact decoder state transitions.

Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0.

| Clause | What is implemented |
|---|---|
| §5.4.4 | `enable_ext_seg` -> `MaxSegments` (8 or 16) |
| §5.4.9 | `seg_info(numSegments)` (ALT_Q `su(10)` clipped to ±351; SKIP/GLOBALMV zero-bit features) |
| §5.18.2 | call order: `segmentation_params()`, `setup_qm_params()`, `delta_q_params()` |
| §5.18.7.1 | `segmentation_params()`; key-frame `update_map`/`temporal_update` inference; `LastActiveSegId`, `SegIdPreSkip` |
| §5.18.7.8 | `delta_q_params()` (`delta_q_present`, `delta_q_res`; **AV2 has no `delta_q_update_map` field**) |
| §5.20.1 | `CurrentQIndex = base_q_idx` at the start of every tile |
| §5.20.2.1 | `ReadDeltas = delta_q_present` at every superblock |
| §5.20.5.7/.8 | `intra_segment_id()`, `read_segment_id()`, `neg_deinterleave()`, §8.3.2 context |
| §5.20.5.11 | `read_delta_qindex()` including the `DELTA_Q_SMALL` escape |
| §5.20.5.12 | `seg_feature_active()` |
| §6.17.2 | coefficient CDF slice from **raw** `base_q_idx` |
| §7.14.2 | `get_qindex(ignoreDeltaQ, segmentId)` |

## Files

| File | Purpose |
|---|---|
| `deltaq.h` | public API |
| `deltaq.cpp` | implementation |
| `bitreader.h` | MSB-first raw-bit reader for frame-header syntax |
| `deltaq_cdf_tables.h` | generated §9.3 CDFs (segment/delta-Q + all four coefficient slices) |
| `generate_deltaq_tables.py` | regenerates the header from `spec/tables/default_cdf.json`; `--check` verifies it |
| `deltaq_test.cpp` | 16k checks: round-trips, bit goldens, state, RC integration |
| `build_and_test.sh` | builds and runs the tests |

CMake integration is intentionally not part of this change (the root
`CMakeLists.txt` is outside this component's ownership).  Add
`src/encoder/m1/deltaq/deltaq.cpp` to `av2enc_core` when wiring the module into
the product build; the standalone scripts compile it directly.

## API summary

### Frame header

```cpp
deltaq::FrameQpHeader qh;
qh.primary_ref_is_none = true;   // DerivedPrimaryRefFrame == PRIMARY_REF_NONE
qh.enable_ext_seg = false;       // sequence enable_ext_seg
qh.frame_type_switch = false;    // M1/M2 never emit SWITCH_FRAME

deltaq::write_segmentation_and_delta_q(writer, seg_params, delta_q_params,
                                       base_q_idx, qh, &error);
```

This writes exactly the three consecutive pieces the M1/M2 frame headers
currently hard-code as three zero bits (`segmentation_enabled`,
`using_qmatrix`, `delta_q_present`).  A matching reader is
`read_segmentation_and_delta_q()`.

Important AV2 details:

* A key frame (`DerivedPrimaryRefFrame == PRIMARY_REF_NONE`) **does not code**
  `segmentation_update_map` / `segmentation_temporal_update`; they are inferred
  to 1 / 0.
* `delta_q_present` is only present when `base_q_idx > 0`; `delta_q_res` is
  only present when `delta_q_present == 1`.
* `using_qmatrix = 1` is rejected: the QM payload is not part of this module.
* `enable_ext_seg = 0` means **8** segments, not 16.

### Derived state

`derive_frame_segmentation()` computes `update_map`, `temporal_update`,
`LastActiveSegId`, `SegIdPreSkip`, `HasLosslessSegment`, and the effective
feature table exactly as §5.18.7.1 / §5.18.2 do.

`effective_qindex()` is `get_qindex()` (§7.14.2) for the no-UV-delta case used
by M1/M2.

### Tile / block state

```cpp
deltaq::TileQp qp;
deltaq::begin_tile(qp, base_q_idx, delta_q, &error); // per tile; resets CurrentQIndex
deltaq::begin_superblock(qp);                        // per SB; ReadDeltas = delta_q_present
deltaq::write_delta_qindex(enc, cdfs.delta_q, qp, mi_size_is_sb, skip_flag,
                           reduced_delta, &error);
```

`write_delta_qindex()` / `read_delta_qindex()`:

* code nothing when `MiSize == SbSize && skip_flag`, or when `ReadDeltas == 0`;
* clear `ReadDeltas` afterwards, exactly like the spec;
* clip `CurrentQIndex` to [1, 255] when `delta_q_present`;
* implement the `delta_q_abs == DELTA_Q_SMALL` escape.

Segment ids:

```cpp
deltaq::write_intra_segment_id(enc, cdfs, seg, grid, block, id, skip, &error);
deltaq::write_inter_segment_id(enc, cdfs, seg, grid, block, pre_skip, id,
                               skip, coded, &error);
deltaq::commit_segment_block(grid, block, id, &error);
```

`write_inter_segment_id(pre_skip = true)` is the no-op call when
`SegIdPreSkip == 0`; the post-skip call codes the real id.  The skip branch
(`skip_flag && !HasLosslessSegment`) codes nothing and reuses the predicted id.
Temporal update codes `seg_id_predicted` with context
`LeftSegPredContext[MiRow] + AboveSegPredContext[MiCol]`.

### Coefficient CDF slice

```cpp
const deltaq::CoeffCdfSlice slice = deltaq::coeff_cdf_slice(base_q_idx);
```

`coeff_cdf_slice()` selects one of four `COEFF_CDF_Q_CTXS` slices from the
**raw** `base_q_idx` (§6.17.2):
`<=90 -> 0`, `<=140 -> 1`, `<=190 -> 2`, else `3`.
Delta-Q / segmentation must not change this selection.  Slice 0 is checked
byte-for-byte against `av2enc::m0::k*Cdf` in the tests.

### RC planner integration

```cpp
deltaq::DeltaQPlanView view;
deltaq::make_delta_q_plan_view(rc_plan, sb_cols, sb_rows, view, &error);
const rc::DeltaQPlanStep* step = deltaq::delta_q_plan_step(view, sb_x, sb_y);
deltaq::check_plan_step(qp, *step, &error);   // qindex_before/after chain
```

`rc::plan_delta_q()` already produces one reduced delta per superblock in
raster order.  The decoder resets `CurrentQIndex` at every tile, so a plan is
per tile; `make_delta_q_plan_view()` enforces that shape and the qindex chain.
`rc::SegmentationMap` (4x4 MI grid) is loaded as `PrevSegmentIds` with
`load_segmentation_map()`.

## M1 integration call sites (core edits, not made here)

`src/encoder/m1/m1_headers.cpp`, `write_frame_header_syntax()`:

```cpp
writer.put_bits(c.base_q_idx, 8U);
// replace the three fixed zero bits with:
deltaq::FrameQpHeader qh;
qh.primary_ref_is_none = true;   // OBU_CLOSED_LOOP_KEY
deltaq::write_segmentation_and_delta_q(writer, c.segmentation, c.delta_q,
                                       c.base_q_idx, qh, &error);
```

`M1Config` needs `rc::SegmentationParams segmentation; rc::DeltaQParams
delta_q;` (or the equivalent POD fields).  The sequence header already writes
`enable_ext_seg = 0`, `seq_seg_info_present_flag = 0` (lines 110-111), so
`MaxSegments = 8`.

`src/encoder/m1/m1_syntax.cpp`, `Writer::write_block()`, after
`write_ccso()` and before `code_y_mode()`:

```cpp
// §5.20.5.2 intra_frame_mode_info(): skip_flag is 0 for intra blocks.
deltaq::write_intra_segment_id(enc_, cdfs_.deltaq, seg_, grid_,
                               to_segment_block(node), segment_id,
                               /*skip_flag=*/false, &error);
// §5.20.5.11 read_delta_qindex() after read_cdef/read_ccso.
deltaq::write_delta_qindex(enc_, cdfs_.deltaq.delta_q, tile_qp_,
                           mi_size_is_sb, /*skip_flag=*/false,
                           planned_reduced_delta, &error);
// then code_y_mode(), ..., and finally:
deltaq::commit_segment_block(grid_, to_segment_block(node), segment_id, &error);
```

`Writer::run()` must call `begin_superblock(tile_qp_)` at each superblock root
and `clear_left_context(grid_)` when `node.mi_col == 0` (next to the existing
left-partition-context clear).  The constructor must call `begin_tile()`,
`init_segment_grid()`, `clear_above_context()`, and `cdfs_.deltaq.init()`.

`Writer::init_cdfs()` must replace the hard-coded slice-0 coefficient
`copy_array(...)` calls with copies from `coeff_cdf_slice(config.base_q_idx)`.
This is the §6.17.2 requirement: the slice is chosen from raw `base_q_idx`.

## Tests and gate

```sh
./src/encoder/m1/deltaq/build_and_test.sh
WERROR=1 SANITIZE=1 ./src/encoder/m1/deltaq/build_and_test.sh
```

The tests cover:

* every §9.3 CDF used here, and slice 0 vs the M0 tables;
* frame-header write/read for key/inter, disabled/ext-seg, update/temporal;
* pinned bit goldens (39-bit key header, 61-bit ext-seg header, 3-bit disabled
  header);
* all `delta_q_abs` values in [-516, 516], escape boundaries, CDF adaptation;
* per-tile `CurrentQIndex` reset, `ReadDeltas` gating, full-SB skip;
* `neg_deinterleave` inverse for every `(ref, max, target)` up to 16;
* intra/inter segment ids, skip branch, temporal update, ext segments;
* `rc::DeltaQPlan` and `rc::SegmentationMap` integration;
* one pinned range-coded block-stream golden (`23d2500000`).

Decoder gate: see `src/encoder/m2/deltaq/build_gate.sh`.  It proves the header
syntax end-to-end with dav2d for the M2 all-skip P-frame case.  The block-level
symbols themselves are covered by the unit tests above; a dav2d gate that
exercises them needs the M1/M2 tile writers to call the hooks.

## Limits

* No SDP / `CHROMA_PART` (M1/M2 do not emit it).
* No quantization matrices (`using_qmatrix = 1` is rejected).
* No chroma (U/V) delta-Q; M1/M2 sequence headers force `equal_ac_dc_q` and
  zero UV deltas.
* 8-bit qindex domain [0, 255].
* `read_delta_qindex` codes at most ±516 for a reduced delta.
