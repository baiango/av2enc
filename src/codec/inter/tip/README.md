# TIP stream — frame-level signalling, build TIP, TIP MV candidates

CPU correctness component for AV2 v1.0.0 **TIP** (temporally interpolated
prediction) at the frame level.  It owns the bits that turn TIP on, the
`OBU_LEADING_TIP` / `OBU_REGULAR_TIP` framing, the complete §7.10.6 build, the
§7.12.2.15/.17/.18 stack candidates, and the output rules for TIP frames.  It
ships a `dav2d --strict 1 --threads 1` gate that decodes real TIP-as-output
frames and requires bit-exact reconstruction at 64x64 and 640x360.

Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
(2026-05-28, commit `38f28e7`).  Sections used: §5.2, §5.4.6, §5.18.2,
§5.18.3.2, §6.2.2, §6.20, §7.8, §7.9, §7.10, §7.12.2.15/.17/.18, §7.13.3,
§7.22.  The pre-existing `inter/temporal/` component supplies the §7.9/§7.10.2–
.5 arithmetic; this directory supplies the assembly and the signalling.

This component does **not** edit M2, `src/main.cpp`, `CMakeLists.txt` or any
other directory.  Everything below the "M2 integration sites" heading is a
checklist for the agent that owns those files.

## Status

| Piece | Spec | Status |
|---|---|---|
| Sequence slice: `enable_ref_frame_mvs` … `enable_tip_refinemv` | §5.4.6 | writer + reader, round-trip + byte goldens |
| Frame slice: `use_ref_frame_mvs`, `tmvp_sample_step_minus_1`, TIP block, `frame_opfl_refine_type()` | §5.18.2/.3.2 | writer + reader, byte goldens |
| `EnableTipOutput`, `disable_tip_output`, `enable_tip_hole_fill`, `enable_tip_explicit_qp`, `enable_tip_refinemv` | §5.4.6, §6.4 | derived and validated on both paths |
| `TipFrameMode`, `TipGlobalMv`, `tip_global_wtd_index`, `TipInterpFilter` | §5.18.2, §6.20 | writer + reader |
| `OBU_LEADING_TIP` (13) / `OBU_REGULAR_TIP` (14) | §6.2.2 Table 6.1 | typed helpers + Annex B framing |
| §7.12.2.15 derive ref MV candidate from TIP mode | §7.12.2.15 | implemented/tested |
| §7.12.2.17 derive single ref MV candidate from TIP mode | §7.12.2.17 | implemented/tested |
| §7.12.2.18 TIP add derived | §7.12.2.18 | implemented/tested |
| Build TIP: MV field → hole fill → block average → sample prediction | §7.10.2–.6 | implemented/tested/**decoder-gated** |
| §7.22 motion-field store of a TIP-as-output frame | §7.22 | implemented, used by the chained gate frame |
| Output rules for TIP frames | §5.4.6, §7.21.1 | implemented/tested + gate call site |
| §7.10.1 setup-TIP conformance conditions | §7.10.1 p413–414 | implemented/tested + gate call site |

## Files

```text
src/encoder/inter/tip_stream/
  tip_signalling.h/.cpp   §5.4.6 sequence slice, §5.18.2 frame slice,
                          TipObuType / OBU framing, readers
  tip_candidates.h/.cpp   §7.12.2.15/.17/.18
  tip_build.h/.cpp        complete §7.10 build + §7.22 TIP field store
  tip_output.h/.cpp       EnableTipOutput / TIP-frame output decisions
  tip_stream_test.cpp     157 checks incl. every pinned golden
  goldens/tip_goldens.txt pinned bit strings, OBU bytes, stack/MV values,
                          reconstruction hashes
  gate/tip_gate.cpp       4-frame IPTT stream + reconstruction compare + control
  build_and_test.sh       unit tests, goldens, dav2d gate, ASan mode
```

## API

### Signalling (`tip_signalling.h`)

```cpp
enum class TipObuType : std::uint8_t { kLeadingTip = 13, kRegularTip = 14 };
constexpr bool is_tip_obu_type(std::uint8_t);      // Table 6.1
constexpr bool is_tip_frame_obu(std::uint8_t);     // §5.1.2 is_tip_frame()
constexpr av2enc::basic::ObuType tip_obu_type(TipObuType);    // §5.2.2 header byte

struct TipSequenceSyntax { /* every field §5.4.6 between
                              enable_ref_frame_mvs and enable_tip_refinemv */ };
TemporalStatus write_tip_sequence_syntax(BitWriter&, const TipSequenceSyntax&);
TemporalStatus read_tip_sequence_syntax(BitReader&, TipSequenceSyntax*);
// TipSequenceSyntax::enable_tip_output() == enable_tip && !disable_tip_output

struct TipFrameSyntax { /* inputs: use_ref_frame_mvs, num_total_refs,
                           enable_tip(_output/_hole_fill/_refinemv),
                           enable_opfl_refine, num_past/future_refs,
                           is_tip_frame; writer: requested_mode,
                           tip_global_wtd_index, tip_global_mv,
                           tip_sharp/tip_regular; outputs: mode,
                           opfl_refine_type, allow_tip_hole_fill,
                           TipGlobalMv, interp_filter */ };
TemporalStatus write_tip_frame_syntax(BitWriter&, const TipFrameSyntax&);
TemporalStatus read_tip_frame_syntax(BitReader&, TipFrameSyntax*);
TemporalStatus resolve_tip_frame_syntax(TipFrameSyntax*);
TemporalStatus write_tip_opfl_refine_type(BitWriter&, const TipFrameSyntax&);

void finish_tip_obu_payload(BitWriter*);           // §5.2.3 trailing bits
std::vector<std::uint8_t> make_tip_frame_obu(TipObuType, BitWriter*);
std::vector<std::uint8_t> make_tip_slice_bytes(const TipFrameSyntax&,
                                               TemporalStatus*);
```

The sequence writer emits a **spliceable** slice: every field between
`enable_ref_frame_mvs` and `enable_tip_refinemv` is present in bitstream order,
including the non-TIP fields that sit between them.  That avoids a second,
drifted set of bit positions.

`make_tip_slice_bytes()` is the §5.18.2 TIP slice alone (for goldens/tests).
A complete TIP OBU is: surrounding frame-header fields → TIP slice →
`finish_tip_obu_payload()` → `av2enc::basic::make_obu(header{type 13/14}, payload)`.
TIP OBUs are **not** extensible (§5.2.1): the payload has no
`obu_extension_flag` bit, only `trailing_one_bit` + zeros.

### Setup-TIP conformance (`tip_signalling.h`)

```cpp
struct TipSetupConditions { switch_frame, closest_past_present,
                            closest_future_present, mode, use_ref_frame_mvs,
                            has_both_refs, num_past_refs,
                            past_type_eligible, future_type_eligible,
                            past_size_eligible, future_size_eligible };
TipSetupConditions make_tip_setup_conditions(...);
TemporalStatus check_tip_setup_conditions(const TipSetupConditions&,
                                          const char** reason = nullptr);
```

Encodes §7.10.1 p413–414 exactly: not a switch frame, both closest refs
present, `TipFrameMode != DISABLED`, `use_ref_frame_mvs == 1`,
`HasBothRefs || NumPastRefs >= 2`, at least one closest ref eligible by frame
type (`!= INTRA_ONLY && != KEY`), and **both** closest refs eligible by size
(same frame dimensions).  M2 should call this before
`build_tip_frame()`/`setup_tip_motion_field()`.

### Build TIP (`tip_build.h`)

```cpp
struct TipBuildInput {
  MotionFieldConfig config;   // mi_cols/rows, proj_step, refs[], saved[],
                              // ref_order_hint[], num_past/future_refs
  TipConfig tip;              // mode must be kAsOutput
  MotionField* field;         // caller-owned w8 x h8 scratch
  MotionFieldScratch* scratch;// required when allow_tip_hole_fill
  TipPlan* plan;              // caller-owned blocks + mv_grid + mv_grid1
  PlaneView ref[3][2];        // decoded reference planes [plane][list]
  std::uint8_t* dst[3]; int dst_stride[3];
  int dst_width[3]; int dst_height[3];
  int sub_x[3]; int sub_y[3];
  std::int16_t* pred_scratch; // 2 * 256 samples
  std::uint8_t* block_scratch;// 256 samples
};
TemporalStatus build_tip_frame(const TipBuildInput&, TipBuildOutput*);
TemporalStatus predict_tip_plan(...);                 // plan -> samples
void fill_tip_closest_refs(const MotionFieldConfig&, const MotionFieldResult&,
                           TipConfig*);

struct TipStoreInput { int mi_cols, mi_rows, order_hint;
                       int closest_past, closest_future;
                       int closest_past_order_hint, closest_future_order_hint;
                       bool weight16; };
TemporalStatus store_tip_frame_motion_field(const TipPlan&, const TipStoreInput&,
                                            RefMvField*);
SavedMvField saved_mv_view(const RefMvField&);
```

`build_tip_frame()` is the whole §7.10 chain in order:

1. `motion_field_estimation()` (§7.9) — fills `MotionField` and returns
   `ClosestPast` / `ClosestFuture` as **reference-list indices**.
2. `fill_tip_closest_refs()` copies the closest order hints into `TipConfig`
   (required by `get_tip_offsets()`).
3. `setup_tip_motion_field()` (§7.10.2 scale, §7.10.3 hole fill, §7.10.4
   block average, §7.10.5 sample gap).
4. `build_tip_plan()` (§7.10.6 per-16x16/8x8 MV plan + per-4x4 grid).
5. `predict_tip_plan()`: per plane, per block, `COMPOUND_AVERAGE` with
   `Tip_Weighting_Factor` and `TipInterpFilter`.  The prediction always runs on
   the full TIP block into `block_scratch` and only the visible part is copied;
   clipping the block rectangle would change `mc_pass_filter(w,h)` at the frame
   edge and was measured to produce ±1 chroma errors at 640x360.

`store_tip_frame_motion_field()` implements the TIP-specific part of §7.22:
`tipPred` sets `refIdx0/1 = ClosestPast/Future`, a weight of 16 clears list 1,
`bw4/bh4` are clamped to `MiCols - c` / `MiRows - r`, and each 8x8 entry goes
through `store_motion_field_entry()` (allow-lists, past/future ordering,
`compression_mv`).  A chained TIP frame needs this output as its
`SavedMvField`.

### Candidates (`tip_candidates.h`)

```cpp
TemporalStatus derive_ref_mv_candidate_from_tip_mode(
    const TipCandInput&, const TipConfig&, int weight, StackMv*, int* num, int* prune);
TemporalStatus derive_single_ref_mv_candidate_from_tip_mode(
    const TipCandInput&, const TipConfig&, int cand_list, int weight,
    StackMv*, int* num, int* prune);
struct DerivedMvStack { Mv mv[kMaxDrStackSize]; int count; int prune; };
TemporalStatus tip_add_derived(const Mv sub_mvs[2], const TipConfig&,
                               DerivedMvStack*);
```

`derive_ref_mv_candidate_from_tip_mode()` compares the full
`(list0, list1)` pair; `derive_single_ref_mv_candidate_from_tip_mode()`
compares only `RefStackMv[][0]` of the selected list, exactly like the spec.
`tip_add_derived()` is the `SubMvs[0] - SubMvs[1]` linear MV projected by
`pastOffset/refOffset`, clipped to `MV_LOW+1 .. MV_UPP-1`.

### Output rules (`tip_output.h`)

```cpp
struct TipOutputConfig { enable_tip, disable_tip_output, is_tip_frame, mode,
                         immediate_output, implicit_output, refreshes_slot,
                         has_closest_past, has_closest_future };
struct TipOutputPlan { enable_tip_output, tip_obu_allowed, output_now,
                       non_output_tip, reason };
TemporalStatus plan_tip_output(const TipOutputConfig&, TipOutputPlan*);
bool tip_implicit_output_inferred(bool monotonic_output_order_flag);
```

Rules enforced (each has a test):

1. `EnableTipOutput = enable_tip && !disable_tip_output`.  With
   `disable_tip_output = 1` a TIP-frame OBU is rejected: §6.20 requires
   `TipFrameMode == TIP_FRAME_AS_OUTPUT` for an `is_tip_frame()` OBU, and that
   value is only inferred when `EnableTipOutput` is 1.
2. A TIP OBU needs both `ClosestPast` and `ClosestFuture` (§7.10.1).
3. `output_now = immediate_output_frame || implicit_output_frame`.  A TIP frame
   with neither is a **coded non-output frame unit** (`non_output_tip`): legal
   as a reference, never displayed at that temporal unit, and it must refresh
   at least one reference slot.
4. `TIP_FRAME_AS_OUTPUT` is impossible for a non-TIP OBU.
5. With `monotonic_output_order_flag = 1`, `implicit_output_frame` is inferred
   0, so a TIP frame is either immediate-output or non-output.

## M2 integration sites and ordering constraints

The M2 path is where this becomes a *bitstream*.  Exact sites, in dependency
order.  Line numbers are for the tree at the time of writing; re-grep before
editing.

### 1. Sequence header — `src/encoder/m2/video_headers.cpp:155-190`

`enable_ref_frame_mvs`, `reduced_ref_frame_mvs_mode`, `order_hint_bits`,
`enable_refmvbank`, `disable_drl_reorder`, `explicit_ref_frame_map`,
`explicit_num_ref_frames`, `long_term_frame_id_bits`, `seq_max_drl_bits`,
`allow_frame_max_drl_bits`, bvp bits, `num_same_ref_compound`, **`enable_tip`,
`disable_tip_output`, `enable_tip_hole_fill`**, `enable_mv_traj`,
`enable_bawp`, `enable_cwp`, `enable_imp_msk_bld`, `enable_df_sub_pu`,
**`enable_tip_explicit_qp`** (only when `EnableTipOutput && enable_df_sub_pu`),
`enable_opfl_refine`, `enable_refinemv`, **`enable_tip_refinemv`** (only when
`enable_tip && (enable_opfl_refine != 0 || enable_refinemv)`).

`TipSequenceSyntax` + `write_tip_sequence_syntax()` replace that whole region
verbatim.  **Ordering constraint:** the slice starts at
`enable_ref_frame_mvs`; do not insert it after `num_same_ref_compound` or the
`num_same_ref_compound` / `enable_tip` positions swap.

For TIP output without residual/refinement the minimum is
`enable_ref_frame_mvs=1, enable_tip=1, disable_tip_output=0,
enable_tip_hole_fill=1`, everything else 0 (that is the gate's stream).

### 2. Frame header — `src/encoder/m2/video_headers.cpp:224-300`

M2 currently writes `use_ref_frame_mvs = 0, TipFrameMode = DISABLED` implicitly
(`video_headers.cpp:266-267`).  The insertion points are:

```text
refresh_frame_flags
frame_explicit_ref_frame_map
num_total_refs / ref_frame_idx[]
use_ref_frame_mvs                                f(1)   <-- insert
if (use_ref_frame_mvs && NumTotalRefs > 1 &&
    SbSize != BLOCK_64X64)
  tmvp_sample_step_minus_1                      f(1)
... FrameDistance / OrderHints (no bits) ...
TIP block (write_tip_frame_syntax)                       <-- insert
```

**Ordering constraints inside the TIP block** (all handled by
`write_tip_frame_syntax()`):

* `is_tip_frame() && EnableTipOutput` ⇒ `TipFrameMode = AS_OUTPUT`, **no
  `tip_frame_mode` bit**; otherwise one bit (1 = AS_REF, 0 = DISABLED).
* `frame_opfl_refine_type()` comes next.  For `AS_OUTPUT` it is derived
  (`enable_tip_refinemv && enable_opfl_refine != NONE ? ALL : NONE`) and writes
  no bits.  Otherwise `REFINE_AUTO` writes the switchable/all bits.
* `allow_tip_hole_fill` only when the mode is not disabled **and** the sequence
  set `enable_tip_hole_fill`.
* `tip_global_wtd_index` is absent (forced 0) when the mode is disabled or
  `usesEqualWeight` (`enable_tip_refinemv && NumFutureRefs>0 &&
  NumPastRefs>0 && (opfl != NONE || enable_refinemv)`).
* `TipGlobalMv` is only coded for `AS_OUTPUT`: `tip_mv_zero`, then 4-bit
  magnitudes and per-component sign bits.
* `tip_sharp`, then `tip_regular` only when `tip_sharp == 0`.

After the TIP block, `TipFrameMode == TIP_FRAME_AS_OUTPUT` **skips**
`screen_content_params`, `intrabc_params`, MV precision, interpolation filter,
motion modes, `quantization_params` (when `enable_tip_explicit_qp == 0`),
`allow_df_sub_pu`, `tile_info` (since `apply_deblocking_filter_tip == 0`),
`disable_cdf_update`, and `motion_field_estimation` is followed by
`setup_tip_motion_field()` and `return`.  An M2 writer that keeps writing the
regular inter tail produces a stream the decoder rejects.

The frame header also loses `disable_cross_frame_cdf_init` for a TIP frame
(§5.18.2 p99: `if (!is_tip_frame()) disable_cross_frame_cdf_init f(1)`); the
gate's `write_frame_header_bits()` shows the exact conditional.

### 3. OBU selection

`src/bitstream/basic_headers.h` only defines the M0 OBU types.  A TIP frame is
`av2enc::basic::make_obu(av2enc::basic::ObuHeader{tip_obu_type(type)}, payload)` with
`type = OBU_REGULAR_TIP (14)` for a regular frame and `OBU_LEADING_TIP (13)`
for a leading one.  The payload is `frame_header(1)` bits **plus**
`finish_tip_obu_payload()`; there is no tile payload and no
`is_first_tile_group` bit.  The TIP OBU belongs to a coded output (or
non-output) frame unit like any tile group: `TD` + TIP OBU, one temporal unit
per frame.

### 4. Frame loop — `src/encoder/m2/video_frame.cpp`

1. **Keep a saved motion field per reference slot.** M2 resets `mv_grid_`
   per frame (`video_frame.cpp:313`) and has no `SavedMvs`.  For TIP you need a
   `RefMvField` / `SavedMvField` per slot and the reference's
   `RefNumTotalRefs` + `SavedOrderHints` (§7.23).
2. Before building a TIP frame, call `build_tip_frame()` with the decoded
   reference planes and those saved fields.
3. After producing a frame, store its field (§7.22).  For a TIP-as-output
   frame call `store_tip_frame_motion_field()`; it needs the `TipPlan`, the
   frame's order hint, `ClosestPast/Future` and their order hints, and whether
   the weight is 16.
4. **Ordering:** a chained TIP frame reads the *stored* field of the earlier
   TIP frame.  Store it before the next frame's `motion_field_estimation()`;
   §7.22 storage happens in `decode_frame_wrapup`, i.e. after reconstruction.
5. For non-TIP frames with `use_ref_frame_mvs`, §5.18.2 calls
   `fill_motion_field_sample_gap()` when `TipFrameMode == DISABLED`; with
   `ProjStep == 1` it is a no-op, so the M2 baseline is unaffected.

### 5. Mode decision / tile writer — `src/encoder/m2/video_syntax.cpp`

For blocks with `RefFrame[0] == TIP_FRAME` (§7.10.6 `TIP_FRAME_AS_OUTPUT`
needs no tile, but a `TIP_FRAME_AS_REF` frame does):

* `read_ref_frames()`: `tip_mode` when the frame's `TipFrameMode != DISABLED`,
  `!skip_mode`, `Min(bw4,bh4) >= 2` and `MiSize == ChromaMiSize`; on 1 set
  `RefFrame[0] = TIP_FRAME`, `RefFrame[1] = NONE`.
* `inter_block_mode_info()`: `tip_pred_mode` (0 = NEARMV, 1 = NEWMV) when
  `RefFrame[0] == TIP_FRAME`.
* `find_mv_stack()`: TIP neighbours must use
  `derive_ref_mv_candidate_from_tip_mode()` (compound) or
  `derive_single_ref_mv_candidate_from_tip_mode()` (single), and the derived
  pass must call `tip_add_derived()` with the neighbour's `SubMvs`.  M2's
  `merge.cpp` currently drops TIP neighbours (`merge.cpp:399`), which is a
  stack-order desync the moment a TIP block appears in a tile.

## The decoder gate

```sh
./src/encoder/inter/tip_stream/build_and_test.sh            # tests + gate
DAV2D=tools/bin/dav2d-macos ./build_and_test.sh            # explicit decoder
./build_and_test.sh --no-gate                              # tests only
SANITIZE=1 ./build_and_test.sh                             # ASan/UBSan
```

Gate structure (all 8-bit 4:2:0, one tile, 64x64 superblocks, filters off):

| unit | OBU | hint | refs | TIP |
|---|---|---|---|---|
| 0 | `OBU_CLOSED_LOOP_KEY` | 0 | — | — |
| 1 | `OBU_REGULAR_TILE_GROUP` | 12 | slot 0 | — |
| 2 | `OBU_REGULAR_TIP` | 1 | slots 0, 1 | AS_OUTPUT, weight 16, `TipGlobalMv=(15,15)` |
| 3 | `OBU_REGULAR_TIP` | 9 | slots 2, 1 | AS_OUTPUT, weight 18, `TipGlobalMv=0` |

The key and P tiles come from the unmodified M2 writer (P is an all-intra
`INTER_FRAME`, so its §7.22 field is empty and the gate can synthesise it
exactly).  The two TIP OBUs have no tile payload.  Frame 2 stores its §7.22
field with `store_tip_frame_motion_field()`; frame 3 projects that field, which
is where §7.10.3 (holes) and §7.10.4 (block average) actually run in the
stream.

Decoded with the pinned decoder:

```text
dav2d -q -i <ivf> -o <y4m> --strict 1 --threads 1
```

Observed (macOS arm64, dav2d `f9dafbc`):

```text
tip_gate 64x64: units=4 bytes=1540 control_bytes=1519 control_identical=1
  tip1: blocks=16 field_valid=0/64 holes_filled=0 stored=8x8
  tip2: blocks=16 field_valid(est)=49 field_valid(setup)=64 holes=15 ...
  decode: key_diff=0(max 0) p_diff=0(max 0) tip1_diff=0(max 0) tip2_diff=0(max 0)
tip_gate: PASS (recon exact, control byte-identical)

tip_gate 640x360: units=4 bytes=31689 control_bytes=31667 control_identical=1
  tip1: blocks=920 field_valid=0/3600 holes_filled=0 stored=80x45
  tip2: blocks=920 field_valid(est)=3081 field_valid(setup)=3600 holes=519 ...
  decode: key_diff=0(max 0) p_diff=0(max 0) tip1_diff=0(max 0) tip2_diff=0(max 0)
tip_gate: PASS (recon exact, control byte-identical)
```

**`enable_tip = 0` control.** The gate re-emits frames 0 and 1 with
`enable_tip = 0`, `enable_ref_frame_mvs = 0`, `NumRefFrames = 2` and the M2
baseline's order hints, then requires the two temporal units to be
**byte-identical** to what `av2enc::video::encode_video_sequence()` produced.  This proves
the TIP writer adds no bits, changes no field order and does not perturb the
M2 baseline when the tool is off.  (`--control <path>` also writes that stream
to disk.)

Every emitted TIP slice is also parsed back with `read_tip_frame_syntax()` from
the OBU payload (skipping the pre-TIP header prefix written by the same
function) and must reproduce `TipFrameMode`, the weight index, `TipGlobalMv`,
`allow_tip_hole_fill` and `TipInterpFilter`.

The gate fails on any recon difference, any missing hole fill in frame 3, a
TIP1 frame equal to the key (i.e. build TIP silently skipped), a control
mismatch, a writer/reader disagreement, a §7.10.1 or output-plan rejection.

## Tests

`./build_and_test.sh --no-gate` → **157 checks, 0 failures**, ASan/UBSan clean.

Coverage:

1. Sequence-slice round trip; `EnableTipOutput` derivation; rejection of an
   unreachable `enable_tip_refinemv` and of `enable_tip_hole_fill` without
   `enable_tip`.
2. Frame-slice round trips for AS_OUTPUT (weight 20 / weight 16 + global MV /
   regular filter), AS_REF, and the no-TIP-block path (0 bits); rejection of a
   TIP OBU with `EnableTipOutput = 0` and of AS_OUTPUT on a non-TIP OBU.
3. OBU framing: Table 6.1 type bytes (0x38 regular = 14, 0x34 leading = 13)
   and Annex B length.
4. §7.10.1 setup-TIP conformance: switch frame, missing closest refs, type
   eligibility, size eligibility, and the failure reason.
5. `get_tip_cand` + §7.12.2.15/.17/.18 stack behaviour, weight accumulation,
   prune counters, derived-MV clip.
6. Build TIP at 64x64 and 640x360 (uniform field), weight 16 equals a plain
   single-reference `predict_plane_u8`, chained TIP1→TIP2 giving the analytically
   expected `(-3,-3)` / `(5,5)` plan MVs, and a 128x128 hole-fill case.
7. §7.22 store: weight-16 list mirroring, compression, 640x360 last-row clamp.
8. Output rules: immediate / implicit / non-output TIP, refresh requirement,
   `disable_tip_output`, missing closest refs, AS_OUTPUT on a normal frame.

### Pinned goldens

`goldens/tip_goldens.txt` is loaded by `tip_stream_test --goldens …`; the test
also fails on an unused entry, so removing a case is visible.  Regenerate with:

```sh
./build_and_test.sh --no-gate
./build-cpu/tip_stream_test --print-goldens
```

Values are bit counts + hex for syntax, full hex for the TIP OBU, stack
counts/MVs, and FNV-1a hashes of the reconstructed planes.  No clocks, no
floating point, fixed iteration order.

## Limitations

* No OPFL (`enable_opfl_refine = REFINE_NONE`) and no refine-MV
  (`enable_refinemv = 0`), so build TIP always stores `mvMethod = 0` and always
  uses `BLOCK_16X16`.  `TipConfig`/`predict_tip_plan` do not consume OPFL
  deltas; §7.13.3.8–.12 exists in `inter/temporal/opfl.*` but is not part of
  this build path.
* `TIP_FRAME_AS_REF` **blocks** are only covered by the syntax and candidate
  helpers; the gate emits AS_OUTPUT frames.  A tile-coded TIP block needs the
  M2 tile writer (integration item 5).
* 8-bit 4:2:0 only, single tile, one embedded layer.  No scaled references
  (the gate's references are unscaled; `predict_plane_*` passes the luma
  dimensions to §7.13.3.17, but no scaled-ref test exists).
* The component is CPU-only; the arithmetic is in header-only POD helpers and
  the core functions take caller-owned buffers, so a CUDA translation unit can
  call the same code later.
* No RDO: the gate chooses the TIP weight/global MV by hand and does not
  compare against non-TIP coding.

## Negative results to avoid repeating

* **Do not clip the TIP block rectangle to the visible frame before
  prediction.** It changes `mc_pass_filter(w, h)` for the bottom row
  (4:2:0 chroma h = 4) and produced ±1 chroma samples at 640x360 while the
  64x64 case passed.  Predict the full block into a scratch buffer, then copy
  the visible part (see `predict_tip_plan`).
* **Do not use `force_global_mv` for the gate's future reference at 640x360.**
  A 64x64 leaf in the 40-line bottom superblock row extends outside the frame,
  GLOBALMV prediction fails, and M2 aborts with "intra fallback unavailable".
  An all-intra `INTER_FRAME` future reference gives an empty §7.22 field that
  the gate can synthesise and decodes at every size.
* **Do not use a 4-bit order hint** when the future reference is 12 frames
  away: `get_disp_order_hint()` reconstructs a TIP hint of 1 as 17 once
  `maxDisp` is 12.  The gate uses 5 bits; the control keeps the M2 baseline's
  4 bits.
