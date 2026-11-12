# M2 multi-reference / compound prediction

Integration layer for `NumTotalRefs > 1` and compound prediction on top of the
M2 P-frame baseline.  This directory is self-contained: it does **not** modify
`src/encoder/m2/` core, `src/encoder/inter/{advanced,mc}/`, `src/main.cpp` or
`CMakeLists.txt` (the M2 core is owned by other workstreams and is being
changed in parallel).

Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
(2026-05-28, commit 38f28e7), §5.18.2, §5.18.8.2, §5.20.7.6-.20, §7.11.2,
§7.12.2, §7.13.3.14, §7.13.3.27-.30.

## What is here

```text
src/encoder/m2/multiref/
  ref_map.h/.cpp          reference slots, NumTotalRefs, ref_frame_idx,
                          reference_select, RefFrame[]/OrderHints[]/FrameDistance,
                          read_compound_ref() pair enumeration
  compound_predict.h/.cpp 1/16-sample compound MC (MC filters), CWP/wedge
                          conversions, §7.13.3.14 implicit-mask eligibility,
                          apply AVERAGE/WEDGE/DIFFWTD through
                          inter::adv::CompoundJob, luma+chroma masks
  mvstack_multi.h/.cpp    per-list and two-list MV stacks via av2enc::video::merge,
                          compound merge candidate list, deterministic
                          selection
  decide.h/.cpp           candidate predictions + source -> mode/refs/mask/wedge
                          (SSE + rate proxy), deterministic tie-break
  gpu_batch.h/.cpp        flat, pointer-free, trivially-copyable batch description
  compound_batch_core.h   device-callable job loader / per-sample helper
  multiref_test.cpp       unit + integration tests
  build_and_test.sh       standalone build/run (no CMake changes)
  gate/                   minimal compound stream writer + dav2d gate
  README.md               this file
```

## Reference map API

`ref_map.h` models the frame-level fields the M2 core does not have yet.

```cpp
RefSlotBank slots;                        // 2 decoded slots (key, P1)
slots[0] = RefSlot{true, true, 0};        // valid, key, OrderHint 0
slots[1] = RefSlot{true, false, 1};       // valid, inter, OrderHint 1
RefFrameMap map;
const uint8_t idx[2] = {0, 1};            // ref_frame_idx[]
build_ref_frame_map(slots, idx, /*order_hint=*/2,
                    /*reference_select=*/true,
                    /*num_same_ref_compound=*/0, map, error);
```

`RefFrameMap` carries `num_total_refs`, `ref_frame_idx[]`, `order_hints[]`,
`frame_distance[]` (`get_relative_dist(OrderHint, OrderHints[i])`),
`reference_select`, `num_same_ref_compound` and provides `is_same_side()` and
`comp_allowed_at(w,h)` (`is_comp_ref_allowed()` = `Min(w,h) >= 8`).

`enumerate_compound_pairs(map)` reproduces every pair `read_compound_ref()`
can signal.  Important cases:

| map | reachable pairs |
|---|---|
| `NumTotalRefs = 2`, `NumSameRefCompound = 0` | `(0,1)` only, **no comp_ref bits** |
| `NumTotalRefs = 3`, `NumSameRefCompound = 0` | `(0,1)`, `(0,2)`, `(1,2)` |
| `NumTotalRefs = 3`, `NumSameRefCompound = 1` | adds `(0,0)` (repeat of ref 0) |

The M2 core currently uses `NumTotalRefs = 1` and `reference_select = 0`
(`video_headers.h:31`, `video_headers.cpp:261`, `video_headers.cpp:291`), so it cannot
reach any of this yet.

## Compound prediction

`build_compound_prediction()` runs the MC 8-tap filters for both references and
writes **1/16-sample** predictions.  `inter::mc::predict_plane()` rounds at
`InterRound1 = 11` (single prediction); compound prediction needs
`InterRound1 = 7` (§7.13.3.18), so this module uses the same inline arithmetic
from `inter::mc::mc_core.h` with the compound round.  References must be
unscaled and share the current frame dimensions (M2 baseline).

`apply_compound_prediction()` fills an `inter::adv::CompoundJob` and calls
`inter::adv::apply_compound()`, so the blend is one transcription shared with
the CUDA path:

| mode | §7.13.3.14 / .30 branch |
|---|---|
| `kAverage` | CWP `(cw * p0 + (16-cw) * p1 + 128) >> 8` |
| `kAverage` + `use_imp_mask` | `get_mask()` with `RefStartX/Y` from the MC sample location |
| `kWedge` | `wedge_mask_value()` luma mask, subsampled for chroma |
| `kDiffWtd` | `diffwtd_mask_value(p0, p1, mask_type, 4, 8)` |

`implicit_mask_eligible()` implements the §7.13.3.14 gate (`enable_imp_msk_bld &&
CWP_EQUAL && YMode != GLOBAL_GLOBALMV && unscaled && !compoundWarp`).

`CompoundModeParams` maps 1:1 to the §5.20.7.16 symbols.  `cwp_weight_from_index`
/ `cwp_index_from_weight` cover both `Cwp_Weighting_Factor[is_same_side]` rows;
`wedge_index_to_syntax` / `wedge_syntax_to_index` invert `read_wedge_mode()`
(quad, angle, dist1/dist2).

## MV stack / merge

`build_pair_stacks()` drives `av2enc::video::merge::build_mv_stack` (the spec-exact stack
in `src/encoder/m2/merge/`, which applies the §7.12.2.20 MVP-combination gate)
three times:

* list 0 single-reference stack (`RefFrame[0] = ref0`);
* list 1 single-reference stack (`RefFrame[0] = ref1`) — the `has_second_drl()`
  path;
* two-list stack (`RefFrame[0..1] = {ref0, ref1}`) — the skip_mode path.

`build_compound_merge_candidates()` produces

1. single-index candidates `(i, i)` from the two-list stack, then
2. two-index candidates `(i, j)`, `i != j`, in lexicographic order,

dropping duplicate MV pairs while keeping the first.  The order is fixed, so
the result is deterministic.  `select_compound_merge_candidate()` predicts the
luma block per candidate (average blend) and picks the lowest SSE, keeping the
earlier candidate on ties.  `select_best_ref_merge()` does the same for the
single-reference NEARMV/merge form with `mc::predict_plane` + `sad_block`.

The older `inter::mc::build_mv_stack` is not used: it inserts MVP-combination
candidates without the block-size gate, which would produce DRL indices the
decoder does not have.

## Decision API

```cpp
DecisionInput input;
input.refs[0] = &ref_frame0;            // or precomputed prediction
input.refs[1] = &ref_frame1;
input.bsize = 12;                       // BLOCK_64X64
input.luma_rect = {x, y, 64, 64};
input.mi_cols = ...; input.mi_rows = ...;
input.source = source_block; input.source_stride = stride;
input.map = &ref_map;
CompoundCandidateInput cand;           // ref pair + MVs (or has_prediction)
cand.ref0 = 0; cand.ref1 = 1;
cand.mv[0] = cand.mv[1] = {0, 0};
cand.global_global_mv = true;          // implicit mask not eligible
input.compound_candidates.push_back(cand);
SingleCandidateInput single;           // optional
single.ref = 1; single.mode = 0; single.sse = ...;
input.single_candidates.push_back(single);
input.options.lambda = 0.0;
DecisionResult result;
decide_compound(input, result, error);
```

`decide_compound()` calls `inter::adv::search_compound()` per candidate (CWP
row chosen from `is_same_side`, implicit mask when eligible) and compares
`cost = sse + lambda * rate_proxy` against the optional single-reference
candidates.  `DecisionResult` returns `kind` (compound/single), `ref0`/`ref1`,
`mv[2]`, the `CompoundModeParams` (type, CWP index+weight, wedge index+sign,
mask_type, implicit-mask flag), `sse`, `rate_bits` and `cost`.

Determinism: candidates are visited in input order, modes in the fixed
`search_compound()` order (implicit-mask average, average/CWP, wedge sign 0..1
× index 0..67, diffwtd mask 0..1), and only strict `<` replaces the current
best.  Compound candidates are evaluated before single candidates, so an exact
tie favours compound.

`evaluate_compound_modes()` exposes the full mode×SSE table for tests/tuning.

### Rate proxy

`compound_mode_bits()` counts the §5.20.7.6/.16 symbols only: `comp_mode`,
`is_joint` (when refs differ), `compound_mode_non_joint` (4 bits), `drl_idx`,
`comp_group_idx`, `compound_type` (2), wedge bits (`2+3+2` or `2+3+2` + sign)
or `mask_type`, `cwp_idx` (index bits), plus a caller-supplied MV cost.  It is
an encoder model, **not** a range-coder measurement; real cost belongs to the
M2 RDO/entropy layer.

## GPU-ready batch

`gpu_batch.h` defines `CompoundBatchItem` (trivially copyable, ints/offsets
only, 8-byte aligned) and `CompoundBatch` (flat `pred0`, `pred1`, `dst`,
`src`, `masks` pools).  `run_compound_batch_cpu()` loads one
`inter::adv::CompoundJob` per item through `compound_batch_core.h` and calls
`adv::run_compound_job()`, so the batch and per-job paths are identical by
construction.  `validate_compound_batch()` checks every offset before a run.

A CUDA kernel should:

1. upload the pool buffers and `items[]` with one `cudaMemcpy` each;
2. launch one thread (or warp) per item, or one thread per sample with the
   item index in a small lookup;
3. call `load_compound_job()` + `adv::run_compound_job()` from
   `src/cuda/inter/advanced/advanced.h` (both are `__host__ __device__`);
4. keep items ordered so each block's plane-0 item (mask producer) runs before
   its chroma items.

Pools must come from `src/cuda/arena.*`; no `cudaMalloc` in hot paths.

## Tests

```sh
./src/encoder/m2/multiref/build_and_test.sh          # unit + synthetic gate
./src/encoder/m2/multiref/build_and_test.sh --720p   # + real 720p gate
SANITIZE=1 ./src/encoder/m2/multiref/build_and_test.sh
```

The unit test covers, with ~44.6k checks:

* reference map, order hints, `is_same_side`, compound pair enumeration
  (including the `NumTotalRefs = 3` coded-`comp_ref` cases and same-ref pairs);
* CWP/wedge round-trips and mode-bit proxy ordering;
* 1/16 prediction exactness (`sample << 4` at integer MVs) and AVERAGE/CWP/
  implicit-mask/WEDGE/DIFFWTD sample-exact blending against the shared
  transcription, including chroma mask subsampling;
* decision determinism, CWP=12 selection for a 3:1 source, wedge selection for
  a wedge-generated source (SSE 0), diffwtd selection for a diffwtd source
  (SSE 0), and single-ref preference when it is better;
* MV stack determinism, compound merge candidate enumeration/selection, and the
  §7.12.2.20 gate;
* GPU batch equals the per-job path and rejects out-of-range offsets;
* decision with `enable_cwp = false` and implicit-mask blending enabled returns
  the `use_imp_mask` average (the only average the decoder can derive);
* real 720p (when `testdata/shine_1280x720_60f.y4m` exists): two frames encoded
  by the M2 core, then 42 interior 64×64 blocks decided with the compound API.
  Output: `720p blocks=42 compound_beats_single=11 best_sse=3107272
  single_sse=3131052 (0.03 dB proxy gain)`.

ASAN/UBSAN runs clean.

## Decoder gate

The gate in `gate/` emits a three-frame stream and checks it with the pinned
`dav2d`:

```text
frame 0  OBU_CLOSED_LOOP_KEY   M2 core intra, order_hint 0, refresh all slots
frame 1  OBU_REGULAR_TILE_GROUP M2 core single-ref P, order_hint 1, slot 0 -> 1
frame 2  OBU_REGULAR_TILE_GROUP gate writer compound P, order_hint 2,
         NumTotalRefs = 2, ref_frame_idx = {0,1}, reference_select = 1
```

The M2 core emits the keyframe and P1 tile payloads; the gate re-wraps them
with a compound-enabled sequence header and its own frame headers.  Frame 2's
tile is written by `gate/compound_tile.cpp`.

Current results (macOS host, pinned `tools/bin/dav2d-macos`):

```text
== compound gate: 128x128 ==
  key: bytes=609 tile=579
  p1:  bytes=19 tile=9
  decision histogram: average=3 wedge=0 diffwtd=1 (first four overridden)
  p2:  tile=8 blocks=4
  psnr frame 0: Y=34.326 U=99 V=99 dB
  psnr frame 1: Y=28.6583 U=24.0484 V=24.0484 dB
  psnr frame 2: Y=26.8423 U=99 V=99 dB
  frame2 Y: single-ref(slot1)=26.7073 dB, compound=26.8423 dB, delta=+0.135 dB
  PASS: 3 frame(s) decoded, recon exact

== compound gate: 1280x704 (real 720p cropped to a 64 multiple) ==
  key: bytes=36225 tile=36194
  p1:  bytes=240 tile=229
  decision histogram: average=185 wedge=27 diffwtd=8 (first four overridden)
  p2:  tile=287 blocks=220
  psnr frame 2: Y=34.8975 U=48.048 V=49.775 dB
  frame2 Y: single-ref(slot1)=34.8999 dB, compound=34.8975 dB, delta=-0.0024 dB
  PASS: 3 frame(s) decoded, recon exact
```

Both streams decode sample-exactly.  The first four superblocks are forced to
AVERAGE, CWP, WEDGE and DIFFWTD so every mode family is exercised even when the
content-based decision would pick something else.

### Why the M2 core cannot emit this stream itself

Without editing the M2 core, a compound stream is impossible because the core
pins all four gates:

* `VideoConfig::num_total_refs = 1` (`video_headers.h:31`) and
  `write_frame_header_syntax()` writes a single `ref_slot`
  (`video_headers.cpp:261-263`);
* `reference_select` is hard-coded to 0 (`video_headers.cpp:291`);
* `enable_masked_compound`, `enable_cwp`, `enable_imp_msk_bld` are hard-coded
  to 0 in the sequence header (`video_headers.cpp:154,172,173`);
* `BlockSyntax` has no compound fields and
  `syntax::write_inter_mode_info()` (`video_syntax.cpp:1641`) codes
  `single_mode`/DRL/MVD only; there is no `comp_mode`, `comp_ref`,
  `compound_mode_non_joint`, `comp_group_idx`, `compound_type`, wedge or
  `cwp_idx` symbol, and `CdfState` has no compound CDFs.

The gate therefore owns a second, minimal writer.  It is not a second
implementation of intra/partition/residual: frames 0–1 still come from the M2
core, and `gate/compound_tile.cpp` only codes the compound inter layer.

### Sequence header differences vs the M2 core

Everything is identical to `video_headers.cpp` except:

| field | M2 core | gate |
|---|---|---|
| `enable_masked_compound` | 0 | **1** |
| `enable_cwp` | 0 | **1** |
| `enable_imp_msk_bld` | 0 | **1** |

(`enable_cwp`/`enable_imp_msk_bld` are independent of `enable_masked_compound`
in the syntax, so all three must be set.)

### Frame header for the compound P frame (exact fields)

`cur_mfh_id` uvlc 0, `seq_header_id_in_frame_header` uvlc 0,
`frame_is_inter` 1, `immediate_output_frame` 1, `frame_size_override_flag` 0,
`order_hint` 4 bits (2), `signal_primary_ref_frame` 0,
`disable_cross_frame_cdf_init` 1, `refresh_frame_flags` 2 bits (1),
`frame_explicit_ref_frame_map` 1, `num_total_refs` 3 bits (2),
`ref_frame_idx[0]` 1 bit (0), `ref_frame_idx[1]` 1 bit (1),
`allow_intrabc` 0, `use_qtr_precision_mv` 0, `allow_high_precision_mv` 1,
`is_filter_switchable` 0, `interpolation_filter` 2 bits (0 = EIGHTAP),
`disable_cdf_update` 1, `tile_info()` (uniform + one zero increment bit per
SB dimension > 1), `base_q_idx` 8 bits (90), `segmentation_enabled` 0,
`using_qmatrix` 0, `delta_q_present` 0, `apply_deblocking_filter[0..1]` 0/0,
`tx_mode_select` 0, **`reference_select` 1**, `skip_mode_present` 0,
`reduced_tx_set` 0.

Frame 1 uses `num_total_refs = 1`, `ref_frame_idx[0] = 0`,
`reference_select = 0`, `refresh_frame_flags = 2`.  Frame 0 is the unchanged M2
keyframe header (`refresh_frame_flags` inferred all slots).

### Per-block symbol order for a compound block

The tile is one `PARTITION_NONE` 64×64 leaf per superblock.  After the
partition `do_split = 0` symbol (context via `partition::partition_contexts`),
an inter compound block codes:

| # | symbol | CDF / context | notes |
|---|---|---|---|
| 1 | `is_inter = 1` | `TileIsInterCdf[ctx]` | neighbour IsInters |
| 2 | `skip_flag = 1` | `TileSkipCdf[ctx]` | no residual; resets coeff contexts |
| 3 | `comp_mode = 1` | `TileCompModeCdf[ctx]` | only when `reference_select && is_comp_ref_allowed()`; ctx from neighbour refs |
| 4 | `comp_ref` bits | `TileCompRef0/1Cdf` | **none** when `NumTotalRefs == 2`; `(0,1)` is implied |
| 5 | `is_joint = 0` | `TileIsJointCdf[ctx]` | ctx from `FieldDistance` symmetry |
| 6 | `compound_mode_non_joint` | `TileCompoundModeNonJointCdf[NewMvContext]`, 5 symbols | `NEAR_NEARMV = 0`, `GLOBAL_GLOBALMV = 3` |
| 7 | `drl_idx0`, `drl_idx1` | `TileDrlModeCdf[min(idx,2)][ctx]` | only for `has_second_drl(YMode)`; `read_drl_idx(0,m)`, `read_drl_idx(start,m)` |
| 8 | — | — | `assign_mv()` codes nothing for NEAR_NEARMV / GLOBAL_GLOBALMV |
| 9 | `comp_group_idx` | `TileCompGroupIdxCdf[ctx]`, 2 symbols | `0 = AVERAGE`, `1 = masked` |
| 10 | `compound_type` | `TileCompoundTypeCdf`, 2 symbols | only when `comp_group_idx = 1`: WEDGE=0, DIFFWTD=1 |
| 11 | `wedge_quad`, `wedge_angle`, `wedge_dist1`/`dist2` | `TileWedgeQuadCdf` (4), `TileWedgeAngleCdf[quad]` (5), `TileWedgeDist1Cdf` (4) or `TileWedgeDist2Cdf` (3, value `dist-1`) | dist2 when `angle >= 10 || angle == 5 || angle == 0`; then raw `wedge_sign` |
| 12 | `mask_type` raw bit | — | DIFFWTD only |
| 13 | `cwp_idx` bits | `TileCwpIdxCdf[idx]`, 2 symbols each | AVERAGE + `NEAR_NEARMV`; `CwpIdx = Cwp_Weighting_Factor[is_same_side()][idx]` |
| 14 | — | — | no `interp_filter` (frame EIGHTAP), no motion-mode/warp/refinemv/optflow/amvd bits |

No `read_block_tx_size()` symbols appear because `skip_flag = 1` and
`tx_mode_select = 0` (`TX_MODE_LARGEST`).

## M2 integration call sites (for the core owner)

These are the exact places to wire this module in later; none is edited here.

1. **Options/config** — `src/encoder/m2/video_frame.h` `VideoOptions`: add
   `num_total_refs`, `reference_select`, `num_same_ref_compound`.  The two
   `VideoConfig` sites are `video_frame.cpp:365-373` and `video_frame.cpp:950-958`
   (`config.num_total_refs = 1`).
2. **Sequence header** — `video_headers.cpp:154,172,173` set the three tool flags
   from config instead of `false`.
3. **Frame header** — `VideoFrameHeader` (`video_headers.h:42`) gains
   `ref_frame_idx[7]`/`num_total_refs`/`reference_select`;
   `write_frame_header_syntax()` loop at `video_headers.cpp:261-263` writes the
   full list; line 291 writes `header.reference_select`.
4. **Reference slots** — the frame loop in `encode_video_sequence()`
   (`video_frame.cpp:975-1035`) needs a `RefSlotBank` updated by
   `refresh_frame_flags`, plus `build_ref_frame_map()` for each P frame.
5. **Mode decision** — `InterFrameEncoder::encode_leaf()`
   (`video_frame.cpp:536-675`) builds `predict_luma`/`predict_all` candidates.
   For `NumTotalRefs > 1`, build one `CompoundPrediction` per candidate pair
   (`build_compound_prediction`) and call `decide_compound()`; the single-ref
   candidates remain as they are.
6. **Tile syntax** — `syntax::BlockSyntax` (`video_syntax.h:48`) needs
   `compound`, `ref0/ref1`, `y_mode`, `drl_index[2]`, and a
   `CompoundModeParams`.  `syntax::write_inter_mode_info()`
   (`video_syntax.cpp:1641`) inserts symbols 3–13 above after `skip_flag`;
   `CdfState` (`video_syntax.cpp:150-230`) gains the compound CDFs from
   `gate/compound_cdf_tables.h` (or regenerate into `video_cdf_tables.h`).
7. **Contexts** — reuse `av2enc::video::merge::derive_new_mv_context()` for
   `NewMvContext`; the gate's `compound_tile.cpp` contains the other context
   transcriptions (`comp_mode`, `is_joint`, `comp_group_idx`) that would move
   into the core.

## Quality / rate proxy summary

* Synthetic 128×128: compound Y PSNR 26.84 dB vs 26.71 dB for the best
  zero-MV single reference (+0.135 dB).
* Real 720p (cropped 1280×704, 220 superblocks): decision picks average 185,
  wedge 27, diffwtd 8; compound Y PSNR 34.8975 dB vs 34.8999 dB single-ref
  (−0.0024 dB).  At zero MV and with two very close references the average is
  essentially the single reference; the gate's point is syntax/decoder
  correctness, not rate.  The rate proxy (`rate_bits`) reports the mode
  symbols only (see above), so `lambda=0` selection is pure SSE.
* The 42-block real-content probe in the unit test finds 11 blocks where a
  compound mode beats the zero-MV single reference, with an aggregate
  best-SSE gain of 0.03 dB.

## Limitations

* No B frames, output reordering, TIP, temporal MV projection or ref-MV bank.
* No inter-intra, warp/OBU/local-warp, BAWP, OPFL, refine-MV, adaptive MVD,
  flex MV resolution; all are disabled in the sequence header used here.
* Compound prediction supports unscaled references only (same as M2).
* The gate writer codes only 64×64 `PARTITION_NONE` leaves (frames must be
  multiples of 64); the library itself supports any even block up to 64×64.
* The gate forces `disable_cdf_update = 1`, so CDF adaptation is not exercised.
* No CUDA kernel is added here; `gpu_batch.h` is the upload/execution contract
  and `run_compound_batch_cpu()` is the oracle it must match.
