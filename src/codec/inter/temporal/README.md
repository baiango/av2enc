# Temporal inter — MV projection, ref-MV bank, TIP, OPFL, refine-MV

CPU correctness reference for the AV2 v1.0.0 temporal inter tools. This
directory is standalone: it does **not** write the M0/M1/M2 bitstream, does not
edit `src/encoder/m2/`, `src/encoder/inter/me/`, `src/encoder/inter/mc/`,
`src/main.cpp`, or `CMakeLists.txt`, and does not touch CUDA build files.

**No integration and no BD-rate claim.** The decoder gate for *our* syntax is
blocked on M2 header ownership (see “Decoder gate”); the only empirical decoder evidence
here is a black-box AVM/dav2d probe (§8).

Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
(2026-05-28, commit `38f28e7`), §4.8, §5.4.6, §5.18.2, §5.18.3.2, §5.20.2.2,
§5.20.7.6/.10/.17, §6.19.2.2, §7.9, §7.10, §7.12.2.7/.8/.21, §7.13.3.2–.12,
§7.22.

## Status

| Tool | Spec | Implementation | Tests |
|---|---|---|---|
| Temporal MV projection / motion field | §7.9 | complete (incl. `enable_mv_traj`) | synthetic + determinism |
| TIP motion-field setup | §7.10.2–.5 | complete | hole fill, average, gap fill |
| Build TIP structures | §7.10.6 | complete (8x8 / 16x16 blocks, per-4x4 grid) | block/MV checks |
| Motion field storage | §7.22 | complete ordering/allow-list/compression | ordering + wedge |
| Temporal scan | §7.12.2.7/.8 | complete | candidate/weight checks |
| Reference-MV bank | §5.20.2.2, §7.12.2.21 | complete ring + hit budget + MVP fill | ring/budget/prune |
| TIP weights / blend | §7.13.3.1 | complete `Tip_Weighting_Factor` | hand-checked blends |
| OPFL | §7.13.3.8–.12 | complete | zero-diff + solve checks |
| Refine-MV | §7.13.3.3–.6, §5.20.7.17 | complete | search/gating/area |
| Temporal syntax bits | §5.4.6 etc. | complete temporal slice | round trips |

## Files

```text
src/encoder/inter/temporal/
  temporal_types.h      POD types, constants, enums
  temporal_math.h       §7.9 projection, compression, sampled position, TIP offsets
  temporal_predict.h    two-pass subpel prediction (1/16 and integer), compound blend
  motion_field.h/.cpp   §7.9, §7.10, §7.12.2.7/.8, §7.22
  ref_mv_bank.h/.cpp    §5.20.2.2, §7.12.2.21
  tip.h/.cpp            Tip_Weighting_Factor prediction/blend
  opfl.h/.cpp           §7.13.3.8–.12
  refinemv.h/.cpp       §5.20.7.17, §7.13.3.3–.6
  temporal_syntax.h/.cpp temporal bit-level read/write + BitReader
  temporal_test.cpp     8,980 checks
  make_probe_clip.py    synthetic 8-frame pan clip for the decoder probe
  probe_dav2d_tip.py    OBU/IVF probe used for the decoder evidence
  build_and_test.sh     standalone build/run (--asan supported)
  README.md             this file
```

## Conventions

- Motion vectors are 1/8 luma samples, `mv[0] = row`, `mv[1] = column`.
- Motion-field locations are 8x8 luma samples (`x8`, `y8`).
- Mode-info locations are 4x4 luma samples (`mi_row`, `mi_col`).
- `Preds[]` fed to OPFL is integer 8-bit (`make_inter_predictions()` uses
  single-reference rounding). TIP/compound blends use 1/16-precision `Preds[]`
  (4 fractional bits) and round with `Round2(..., 4 + InterPostRound)`.
- Reference frame values: `0..6` inter slots, `7 = TIP_FRAME`, `8 =
  INTRA_FRAME`, `-1 = NONE`.
- Every core function takes POD views/pointers and returns
  `TemporalStatus`; no exceptions, no per-block allocation in the header
  arithmetic. The CPU driver allocates its own storage (`std::vector` in
  `opfl.cpp` for the gradient scratch).

## API

### Projection / math (`temporal_math.h`)

```cpp
Mv get_mv_projection(Mv mv, int numerator, int denominator);        // §7.9.4
Mv get_mv_projection_clamp(Mv mv, int numerator, int denominator);  // §7.9.5
std::int16_t compression_mv(std::int32_t v);                        // §7.22
std::int32_t uncompression_mv(std::int16_t v);                      // §7.22
SampledPosition get_sampled_position(int x8, int y8, int dst_sign, Mv proj_mv,
                                     int mi_cols, int mi_rows, int proj_step);
bool check_block_position(bool valid, int base_x8, int base_y8, int pos_x8,
                          int pos_y8, bool sb_64x64, int proj_step);
TipOffsets get_tip_offsets(const TipConfig&);                       // §7.22 helper
bool opfl_scale_distances(int d0, int d1, int out[2]);              // §7.13.3.9
void divide_and_round_array(const int sol[2], int den, int shift, int out[2]);
```

All 32 `Div_Mult` entries, the `REFMVS_LIMIT` clamp, the 17-bit MV clamp and the
`ProjStep` even-position adjustment are tested.

### Motion field (`motion_field.h`)

```cpp
TemporalStatus motion_field_estimation(const MotionFieldConfig&, MotionField*,
                                       TrajectoryField*, MotionFieldResult*);
TemporalStatus setup_tip_motion_field(const MotionFieldConfig&, const TipConfig&,
                                      MotionField*, const TrajectoryField*,
                                      MotionFieldScratch*, MotionFieldResult*);
TemporalStatus fill_motion_field_sample_gap(const MotionFieldConfig&,
                                            MotionField*);
TemporalStatus build_tip_plan(const MotionFieldConfig&, const TipConfig&,
                              const MotionField&, TipPlan*);
TemporalStatus temporal_scan(const TemporalScanInput&, StackMv*, int* num,
                             int* prune);
MvPair get_tip_cand(const TipCandInput&, const TipConfig&);
TemporalStatus store_motion_field_entry(const MotionFieldStoreEntry&,
                                        RefMvField*, int y8, int x8);
```

`MotionField` is `[y8][x8]` with `mv[2]`, `valid`, and `offset` arrays.
`TrajectoryField` is optional and adds `[rf][y8][x8]` MVs/validity plus the
three-phase `TrajPos` history used by `check_traj_intersect()`.
`build_tip_plan()` fills both `mv_grid` (list 0, `Mvs[..][0]`) and optional
`mv_grid1` (list 1, `Mvs[..][1]`), and the `TipBlock` array.

The §7.9 projection is a direct transcription of the spec's reference ordering,
topological sort, `record_projection_with_type()`, `projection()`, trajectory
intersection, `reduced_ref_frame_mvs_mode` truncation and `fill_traj_mv` gap
fill. `setup_tip_motion_field()` implements the §7.10.2 temporal scale,
§7.10.3 hole fill, §7.10.4 block-average filter (with `Weight_Div_Mult`) and
§7.10.5 sample-gap fill.

### Reference-MV bank (`ref_mv_bank.h`)

```cpp
int get_rmb_list_index(const int ref_frames[2]);                   // 0..8
void update_ref_mv_count(RefMvBank*, const RefMvBankBlockContext&);
bool update_ref_mv_bank(RefMvBank*, const RefMvBankBlockContext&,
                        const int ref_frames[2], const Mv mvs[2],
                        int cwp_idx, bool from_within_sb);
void reset_ref_mv_bank(RefMvBank*, int mi_row, int mi_col, int sb_size4,
                       bool frame_is_intra, bool top_row,
                       const RefMvBankRow&);
void fill_mvp_from_ref_mv_bank(const RefMvBank&, const int ref_frames[2],
                               bool is_compound, bool use_intrabc,
                               int max_drl_bits_minus_1, int mi_row, int mi_col,
                               int mi_cols, int mi_rows, int bw, int bh,
                               int max_out, StackMv*, int* num_found,
                               int* prune_count);
```

The bank is 9 rings of 4 six-word entries (`cwp, key, mv0row, mv0col, mv1row,
mv1col`). `rmb_params_equal()` compares words 1..5, so the CWP index is not part
of the key — a refresh moves the *stored* entry (with its old CWP) to the tail,
exactly as the spec does. `MAX_RMB_SB_HITS`, `RefMvRemainHits`, `RefMvUnitHits`
and the `update_ref_mv_count()` alignment rules are implemented. Candidates are
pruned against the stack (`MAX_PR_NUM`) and rejected when the integer MV points
outside the frame.

### TIP (`tip.h`, `temporal_types.h`)

`kTipWeightingFactor[8] = {8, 12, 16, 18, 20, 4, 6, -4}`. `tip_predict_block()`
handles the single-reference weight-16 case (`InterRound1 = 11`) and the
two-reference case (`InterRound1 = 7`, `Round2(cwp*P0 + (16-cwp)*P1, 8)`).
`tip_store_refined_mvs()` and `tip_luma_use_opfl()` reproduce the §5.20.7.6
conditions. 4:2:0 chroma derives the luma reference dimensions from the plane
extent and subsampling before calling §7.13.3.17, so it is not accidentally
half-scaled.

### OPFL (`opfl.h`)

```cpp
bool opfl_allowed_for_refs(int d0, int d1, bool switch_frame,
                           bool ref0_scaled, bool ref1_scaled);
TemporalStatus opfl_compute(const OpflJob&, const OpflOutput&);
```

`opfl_compute()` takes integer 8-bit predictions, the two starting MVs and the
order hints. It computes the reduced distances, `tmp`/`pDiff`, the 16x16-unit
gradients, the least-squares statistics, the `MAX_LS_BITS` reduction, the
determinant, `divide_and_round_array()`, the ±16 `OPFL_MV_DELTA_LIMIT` clamp,
`MvDeltas[]` and the prescaled 1/16 `OpflMvs[]`. A `det <= 0` block leaves its
4x4 entries untouched, as in the spec.

### Refine-MV (`refinemv.h`)

```cpp
TemporalStatus refine_mv_syntax(const RefineMvConfig&, RefineMvSyntax*);
bool refine_mv_allowed_mode(const RefineMvConfig&);
bool refine_mv_switchable(const RefineMvConfig&);
bool refine_mv_allowed_reference(const RefineMvConfig&);
MvPair offset_refinemv(const MvPair&, int dx, int dy);
TemporalStatus compute_ref_area(const RefAreaInput&, RefArea out[2]);
TemporalStatus search_refinemv(const RefineSearchInput&, RefineSearchResult*);
TemporalStatus refine_predict_block(const RefinePredictInput&);
```

`search_refinemv()` implements the asymmetric `t - 4*8` / `t + 2*8` overflow
guard, the 2-sample region expansion, the `(w*h) << 1` threshold, the
`bestSad -= bestSad >> 3` center test, the 24-entry `Refinemv_Neighbors` table
and the even-row `get_sad(w,h,1)` convention. The final prediction uses
EIGHTTAP_SHARP and the compound average.

## Availability, range and determinism

- MVs are clamped to `MV_LOW+1 .. MV_UPP-1`; projected MVs are additionally
  clamped to `±REFMVS_LIMIT` for storage and TIP scaling.
- `get_relative_dist()` returns `0`/`127`/`-127` for restricted order hints and
  otherwise `Clip3(-127, 127, a-b)`, matching §5.18.3.1.
- `ProjStep` is 1 for 64x64 superblocks; `get_tmvp_shift()` is 3 in that case
  and 4 for 128/256 superblocks with `ProjStep=2`.
- `check_block_position()` restricts a projected position to the current TMVP
  unit (plus the right half-unit for `ProjStep=1`).
- TIP weights require `0 <= tip_global_wtd_index <= 7`; an out-of-range index
  returns `kInvalidArgument`.
- All search tie-breaks are strict `<` and iteration order is fixed, so results
  are deterministic; the tests run the motion-field process twice and compare
  every MV/valid/offset.
- Edge cases covered: right/bottom partial blocks, out-of-picture projected
  positions, `det <= 0`, zero OPFL distances, `den == 0` in projection, zero
  reference area, truncated syntax, and `REFMVS_LIMIT`-sized vectors.

Known deliberate deviation: `fill_mvp_from_ref_mv_bank()` clamps
`max_ref_mv_count` to `MAX_REF_MV_STACK_SIZE` (6). The spec's expression
`max_drl_bits_minus_1 + 2` can be 7, but the stack array is six entries and the
spec's own `MAX_REF_MV_STACK_SIZE` bound is 6; the clamp prevents an
out-of-bounds write without changing any representable candidate.

## Tests

```sh
./build_and_test.sh          # build + run
./build_and_test.sh --no-run # build only
./build_and_test.sh --asan   # ASan/UBSan
CXX=clang++ ./build_and_test.sh
```

Current result: **8,980 checks, 0 failures**, ASan/UBSan clean.

Coverage:

1. `Div_Mult[32]`, projection identity/scaling/rounding/clamping, compression
   round trips and the lossy >31 bound.
2. `project_no_constraint()`, `get_sampled_position()` (`ProjStep=1/2`) and
   `check_block_position()` unit boundaries.
3. Two-reference motion-field estimation: topological order, closest
   past/future, MV/offset propagation, TIP temporal scaling, `get_tip_cand`,
   8x8/16x16 build-TIP plans, per-4x4 grid.
4. Sparse-field TIP hole fill and `Weight_Div_Mult` average.
5. §7.22 storage: list mirroring, past/future reordering, compression and
   wedge allow-list.
6. Temporal scan candidate, weight and invalid-field behavior.
7. Ref-MV bank: list index, ring insert/refresh, CWP-not-in-key behavior, hit
   budget, reset seeding, MVP append/prune and out-of-frame rejection.
8. TIP weights, hand-checked blends, weight-16 single prediction and 4:2:0
   chroma shift.
9. OPFL distance reduction, `divide_and_round_array()`, `opfl_allowed_for_refs`,
   zero-difference solve and prescaled outputs.
10. Refine-MV gating, `offset_refinemv()`, `get_ref_area()` boundaries, a
    known +2-sample reference shift search, center early exit and overflow guard.
11. Sequence/frame/block/refine-MV syntax round trips, including every
    conditional path and truncated input.

## Decoder gate — why there is no stream from this component

The text below is the honest status: **no `dav2d` gate exists for our temporal
syntax**, because a valid minimal TIP stream cannot be produced without editing
M2-owned files.

Required M2 changes (all outside this directory):

1. `src/encoder/m2/video_headers.h/.cpp`
   - `VideoConfig` needs `enable_ref_frame_mvs`, `reduced_ref_frame_mvs_mode`,
     `enable_refmvbank`, `enable_tip`, `enable_tip_hole_fill`, `enable_mv_traj`,
     `enable_opfl_refine`, `enable_refinemv`, `enable_tip_refinemv`,
     `disable_tip_output`, `enable_df_sub_pu`, `enable_bawp`, `enable_cwp`,
     `enable_imp_msk_bld`.
   - `make_sequence_header_obu()` must write the temporal slice described in the “Exact M2 integration bits” section below.
   - `VideoFrameHeader` needs `use_ref_frame_mvs`, `tmvp_sample_step_minus_1`,
     `TipFrameMode`, `TipGlobalMv`, `tip_global_wtd_index`, `TipInterpFilter`,
     `allow_tip_hole_fill`, `opfl_refine_type`.
   - `write_frame_header()` must write the TIP block described in “Exact M2 integration bits”.
2. `src/encoder/m2/video_frame.cpp`
   - At frame start: build `SavedMvField` inputs from the reference frames and
     call `motion_field_estimation()`, then `setup_tip_motion_field()` (or
     `fill_motion_field_sample_gap()` when TIP is disabled).
   - After each inter block/8x8: `store_motion_field_entry()`.
   - Mode decision: `temporal_scan()`, `fill_mvp_from_ref_mv_bank()`,
     `get_tip_cand()`, `build_tip_plan()`, `tip_predict_block()`.
   - Refinement: `opfl_compute()`, `refine_mv_syntax()`, `compute_ref_area()`,
     `search_refinemv()`, `refine_predict_block()`.
3. `src/encoder/m2/video_syntax.cpp`
   - `write_tip_mode_syntax()` in `read_ref_frames()`.
   - `write_tip_pred_mode_syntax()` in `inter_block_mode_info()`.
   - `write_refinemv_block_syntax()` after `read_compound_type()` and before
     the interpolation-filter decision.
   - `write_frame_temporal_syntax()` in the first tile-group header, at the
     positions below.

A TIP-as-output frame has no tile payload (`TipFrameMode ==
TIP_FRAME_AS_OUTPUT`), but its reference frames must carry motion fields, so
the P-frame encoder must also signal `use_ref_frame_mvs` and store MVs. M2's
current P-frame path has no such fields. Fabricating an OBU stream outside M2
would duplicate M2's tile/header ownership and is out of scope for this
component.

## Decoder evidence: what `dav2d` f9dafbc accepts

Because the AVM encoder is available as a black-box tool, the pinned decoder
was probed with an AVM-generated stream that carries the temporal tools. This
does **not** validate our syntax; it only answers whether the decoder supports
the tools.

Probe:

```sh
# 128x128, 8-frame synthetic pan (Y4M, 4:2:0)
python3 src/encoder/inter/temporal/make_probe_clip.py /tmp/pan128.y4m 8
avmenc --ivf --limit=8 --passes=1 --good --cpu-used=6 \
  -o /tmp/base.ivf /tmp/pan128.y4m
python3 src/encoder/inter/temporal/probe_dav2d_tip.py /tmp/base.ivf \
  --decode tools/bin/dav2d-macos
```

Observed with `dav2d` `f9dafbc`:

```text
unit    0 bytes=     665 obu_types=[2, 1, 4, 7]
unit    1 bytes=     119 obu_types=[2, 7, 2, 7, 2, 7]
unit    2 bytes=      14 obu_types=[2, 7]
unit    3 bytes=      34 obu_types=[2, 7, 2, 14]  <-- TIP OBU
unit    4 bytes=      13 obu_types=[2, 7]

IVF payloads: 5, TIP OBU present: True
decode exit: 0, decoded display frames: 8
```

`obu_type 14` is `OBU_REGULAR_TIP`, i.e. a TIP-as-output frame. AVM `--verbose`
for the same settings reports:

```text
TIP (1)
TIP RefineMV (1)
OpflMvSearch (2)
RefineMV mode: (1)
OptflowRefinement (1)
Refmv Bank (1)
MV traj (1)
```

So `dav2d` f9dafbc parses and decodes a stream that enables TIP output, TIP
refine-MV, OPFL, refine-MV, the reference-MV bank and MV trajectory tracking.
It is still **not** a support claim for our bitstream: the encoder that
produced it is AVM, and the test does not isolate which blocks actually used
each tool. Treat our temporal tools as `U2`/unverified until an M2-produced
stream passes `tools/validate_dav2d.sh`.

## Exact M2 integration bits

### Sequence slice (starting at `enable_ref_frame_mvs`)

```text
enable_ref_frame_mvs                         f(1)
if (enable_ref_frame_mvs)
  reduced_ref_frame_mvs_mode                 f(1)
enable_refmvbank                             f(1)
... order_hint_bits, explicit ref map, max_drl_bits, num_same_ref_compound ...
enable_tip                                   f(1)
if (enable_tip) {
  disable_tip_output                         f(1)
  enable_tip_hole_fill                       f(1)
}
enable_mv_traj                               f(1)
enable_bawp                                  f(1)
enable_cwp                                   f(1)
enable_imp_msk_bld                           f(1)
enable_df_sub_pu                             f(1)
if (EnableTipOutput && enable_df_sub_pu)
  enable_tip_explicit_qp                     f(1)
enable_opfl_refine                           f(2)
enable_refinemv                              f(1)
if (enable_tip && (enable_opfl_refine != 0 || enable_refinemv))
  enable_tip_refinemv                        f(1)
```

`write_sequence_temporal_syntax()` / `read_sequence_temporal_syntax()` cover
exactly this slice except the bracketed M2-owned fields; splice it at the right
point or extend the helper with those fields.

### Frame TIP block

```text
if (use_ref_frame_mvs && NumTotalRefs > 1 && SbSize != BLOCK_64X64)
  tmvp_sample_step_minus_1                   f(1)   // ProjStep = value + 1
...
if (enable_tip && use_ref_frame_mvs && NumTotalRefs >= 2 && !bru_inactive) {
  if (EnableTipOutput && is_tip_frame())
    TipFrameMode = TIP_FRAME_AS_OUTPUT
  else {
    tip_frame_mode                           f(1)   // 1 = AS_REF, 0 = DISABLED
    TipFrameMode = tip_frame_mode
  }
  frame_opfl_refine_type()
  if (TipFrameMode != TIP_FRAME_DISABLED && enable_tip_hole_fill)
    allow_tip_hole_fill                      f(1)
  usesEqualWeight = enable_tip_refinemv && NumFutureRefs > 0 &&
                    NumPastRefs > 0 &&
                    (opfl_refine_type != REFINE_NONE || enable_refinemv)
  if (TipFrameMode == TIP_FRAME_DISABLED || usesEqualWeight)
    tip_global_wtd_index = 0
  else
    tip_global_wtd_index                     f(3)
  if (TipFrameMode == TIP_FRAME_AS_OUTPUT) {
    tip_mv_zero                              f(1)
    if (!tip_mv_zero) {
      tip_mv_row                             f(4)
      tip_mv_col                             f(4)
      if (tip_mv_row != 0) tip_mv_row_sign   f(1)
      if (tip_mv_col != 0) tip_mv_col_sign   f(1)
    }
  }
  tip_sharp                                  f(1)
  if (!tip_sharp) tip_regular                f(1)
} else {
  TipFrameMode = TIP_FRAME_DISABLED
  if (!bru_inactive && !IsBridge) frame_opfl_refine_type()
}
```

`frame_opfl_refine_type()` (§5.18.3.2): for `TIP_FRAME_AS_OUTPUT` it derives
`opfl_refine_type = (!enable_tip_refinemv || enable_opfl_refine == REFINE_NONE)
? REFINE_NONE : REFINE_ALL`; otherwise if `enable_opfl_refine ==
REFINE_AUTO` it writes `opfl_refine_type f(1)` (the switchable flag) and, when
that is not `REFINE_SWITCHABLE`, `opfl_refine_all f(1)`.

### Block bits

```text
// read_ref_frames()
if (TipFrameMode != TIP_FRAME_DISABLED && !skip_mode &&
    Min(bw4, bh4) >= 2 && MiSize == ChromaMiSize)
  tip_mode                                   S()
  if (tip_mode) { RefFrame[0] = TIP_FRAME; RefFrame[1] = NONE; return; }

// inter_block_mode_info(), only when RefFrame[0] == TIP_FRAME
tip_pred_mode                                S()   // 0 = NEARMV, 1 = NEWMV
YMode = Tip_Pred_Index_To_Mode[tip_pred_mode]

// read_refinemv(isCompound), after the compound type
if (enable_refinemv && isCompound &&
    (Block_Width >= 16 || Block_Height >= 16) &&
    Block_Width >= 8 && Block_Height >= 8 &&
    is_refinemv_allowed_mode() && is_refinemv_allowed_reference(RefFrame)) {
  if (is_switchable_refinemv())
    use_refinemv                             S()
  else
    use_refinemv = 1                         // inferred
}
```

`TipFrameMode`, `RefFrame[0]`, `YMode`, `use_refinemv` in turn select
`tip_predict_block()`, `build_tip_plan()`, `get_tip_cand()`,
`opfl_compute()` and `search_refinemv()`.

## CUDA readiness

- `temporal_math.h`, `temporal_predict.h`, `temporal_types.h` and the
  `motion_field.h` POD views are standard-library-free and header-only; they
  can be included by a CUDA translation unit with the same arithmetic.
- `motion_field.cpp` uses fixed-size `std::array` state for the topological
  sort and takes caller-owned buffers for the motion/trajectory/TIP-average
  arrays. There is no device-side allocation.
- `opfl.cpp` allocates its `tmp`/`pDiff`/gradient planes with `std::vector` in
  the CPU driver. A CUDA port should pass those as scratch pointers (or use
  shared memory for an 8x8 block) and call the same arithmetic.
- `refinemv.cpp` and `tip.cpp` take caller scratch buffers and never allocate
  in the prediction loop.

## Limitations

- Unscaled references are the tested path. The prediction helpers pass the
  luma reference/frame dimensions to §7.13.3.17, so scaled references should
  follow the formula, but no scaled-reference test exists yet.
- 8-bit 4:2:0 only. 4:2:2/4:4:4/10-bit are not exercised.
- `store_motion_field_entry()` takes the final per-8x8 MVs. The caller composes
  `mvMethod` 1/2 (OPFL deltas or `RefineMvs`) and `get_tip_cand()` before
  calling it; the entry helper owns the §7.22 allow-list, ordering and
  compression.
- No RDO/rate term, no mode decision, no bitstream integration.
- The CUDA M2 tree (`src/cuda/m2/`) is reserved for a later port.
