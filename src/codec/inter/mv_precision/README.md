# MV precision / AMVD signalling (`inter/mv_precision`)

Spec-only implementation of the AV2 v1.0.0 motion-vector signalling tools that
the M2 baseline disables.  No core, CMake, or `src/main.cpp` edits; the module
is a standalone component with its own tests and decoder gate.

Normative sources: AV2 v1.0.0 §5.4.6, §5.18.2, §5.18.5.1, §5.20.7.6,
§5.20.7.13, §5.20.7.20, §5.20.7.21, §5.20.7.24, §7.9.4, §8.3, §9.3.

## What is implemented

| Syntax | § | Writer / reader |
|---|---|---|
| `seq_enabled_motion_modes[INTERINTRA..EXTENDWARP]` | 5.4.6 | `write_sequence_motion_modes` / `read_sequence_motion_modes` |
| `seq_frame_motion_modes_present_flag` | 5.4.6 | same |
| `enable_six_param_warp_delta` | 5.4.6 | same (only when `DELTAWARP`) |
| `enable_adaptive_mvd`, `enable_mvd_sign_derive`, `enable_flex_mvres` | 5.4.6 | `write_sequence_mv_res` / `read_sequence_mv_res` |
| `frame_enabled_motion_modes[]` | 5.18.2 | `write_frame_motion_modes` / `read_frame_motion_modes` |
| `use_qtr_precision_mv`, `allow_high_precision_mv` | 5.18.2 | `write_frame_mv_precision` / `read_frame_mv_precision` |
| `read_interpolation_filter()` | 5.18.5.1 | `write_frame_interp_filter` / `read_frame_interp_filter` |
| `use_amvd` | 5.20.7.6 | `write_use_amvd` / `read_use_amvd` |
| `use_most_probable_precision`, `pb_mv_precision` | 5.20.7.6 | `write_block_precision` / `read_block_precision` |
| `interp_filter S()` | 5.20.7.6 | `write_block_interp_filter` / `read_block_interp_filter` |
| `mv_joint S()` (AMVD form) | 5.20.7.20 | `write_mv_joint` / `read_mv_joint` |
| `amvd_index S()` | 5.20.7.21 | `write_amvd_index` / `read_amvd_index` |
| AMVD `read_mv()` | 5.20.7.20/.21 | `write_amvd_mv` / `read_amvd_mv` |
| `jmvd_scale_mode S()` | 5.20.7.6 | `write_jmvd_scale_mode` / `read_jmvd_scale_mode` |
| `projMv` / `assign_mv()` joint path | 5.20.7.13 | `project_joint_mvd`, `scale_joint_projection`, `assign_joint_mvd` |
| `mv_sign L(1)` with derivation | 5.20.7.13 | `write_mvd_signs` / `read_mvd_signs`, `mvd_sign_derive_allowed` |
| precision-aware shell MVD | 5.20.7.20 | `write_shell_mv` / `read_shell_mv` |

Every writer has an independent reader.  The entropy-coded symbols go through
the normative `av2::Encoder`/`av2::Decoder` (§8.2); raw `f(n)`/`L(n)` fields go
through `BitWriter`/`BitReader`.

### Precision backend

The precision model is consumed read-only from
`src/encoder/inter/me2/me2_precision.h`:

* `me2_frame_precision_from_bits` / `me2_frame_precision_to_bits`
* `me2_pb_precision_derive`
* `me2_lower_mv_precision`
* `me2_ctx_use_most_probable` / `me2_ctx_pb_precision`
* `me2_pb_precision_cdf_row`

`plan_block_precision()` is the §5.20.7.6 decision table on top of that model.
It deliberately caps `pb_mv_precision` at the three-symbol CDF alphabet `0..2`
(the me2 inverse helper searches a wider range, which cannot be signalled).

## CDFs and contexts

`mv_precision_cdf_tables.h` is generated from the pinned
`spec/tables/default_cdf.json` by `generate_mv_precision_tables.py`.  It
contains:

* `Default_Use_Amvd_Cdf[9][3]`, `Default_Mv_Joint_Adaptive_Cdf[4]`,
  `Default_Amvd_Indices_Cdf[2][8]`;
* `Default_Jmvd_Scale_Mode_Cdf[5]`,
  `Default_Jmvd_Adaptive_Scale_Mode_Cdf[3]`;
* `Default_Use_Most_Probable_Precision_Cdf[3][2]`,
  `Default_Pb_Mv_Precision_Cdf[2][3][3]`;
* `Default_Interp_Filter_Cdf[16][3]`;
* every `Default_Joint_Shell{P}Class{Q}_Cdf` for the reachable precisions
  `P ∈ {0,1,3,4,5,6}` (`P == 2` has no table and is rejected), plus the shell
  offset and column CDFs.

`MvPrecisionCdfState` is one independent copy per tile.  §8.3 context
derivation:

| context | rule |
|---|---|
| `use_amvd_cdf_index` | `NEWMV=4`, `JOINT_NEWMV=5/6`, `NEAR_NEWMV=0/2`, `NEW_NEARMV=1/3`, `NEW_NEWMV=7/8` (`use_optflow` selects the odd index) |
| `use_amvd_ctx` | count `NPosBuf` neighbours with `RefFrame[0]` matching and `UsesAmvds != 0` |
| `use_most_probable_precision_ctx` | count `NPos` neighbours with `UseMostProbablePrecisions != 0` |
| `pb_mv_precision_ctx` | 1 if any `NPos` neighbour has `MvPrecision < FrameMvPrecision` |
| `interp_filter_ctx` | `4 * is_inter_ref_frame(RefFrame[1])` then the left/above-type rule over `NPos` |

`collect_neighbors()` keeps `NPos` (same superblock row) and `NPosBuf`
(cross-row) separate, because the spec uses different lists for these
contexts.  This distinction is pinned by a golden.

## Legal value sets

* Sequence motion modes: `seq_enabled_motion_modes[INTERINTRA..EXTENDWARP]`;
  `seq_frame_motion_modes_present_flag` is derived as `any(enabled)`.
* Frame motion modes: `frame_enabled_motion_modes[i]` must be 0 when the
  sequence disabled `i`; when `seq_frame_motion_modes_present_flag == 0` the
  frame values are inferred and no bits are coded.
* `use_amvd` is present only when `enable_adaptive_mvd && allow_amvd_mode(YMode)`.
* `use_most_probable_precision` / `pb_mv_precision` are present only when
  `enable_flex_mvres && UsePerBlockMvPrecision && has_newmv(YMode)` and the
  block is not `IsAdaptiveMvd`.  `IsAdaptiveMvd` pins `MvPrecision` to
  `FrameMvPrecision` and codes no precision symbol.
* `jmvd_scale_mode` has 5 symbols when `use_amvd == 0`, 3 when `use_amvd == 1`.
* Block `interp_filter` alphabet is 3 symbols: `EIGHTTAP`, `EIGHTTAP_SMOOTH`,
  `EIGHTTAP_SHARP`.  `BILINEAR` is a frame-header value only; `SWITCHABLE`
  selects the per-block symbol.
* `enable_mvd_sign_derive` derives the last nonzero sign only when the
  §5.20.7.13 predicate holds (SIMPLE motion mode, not AMVD, not skip_mode, no
  screen-content tools, `FrameMvPrecision <= QUARTER`, `MvPrecision < QUARTER`,
  no NEARMV; compound only for `RefMvIdx == 0`).

## Tests

```
./build_and_test.sh                 # unit tests + dav2d gate
./build_and_test.sh --no-gate       # unit tests only
./build_and_test.sh --gen-goldens   # rewrite goldens/mv_precision_goldens.txt
SANITIZE=1 ./build_and_test.sh      # ASan + UBSan
```

`mv_precision_test` covers:

* every §9.3 table (dimensions, monotonicity, param/count slot, shell alphabet
  sizes) and the generated-header check;
* sequence/frame motion-mode bitstrings, including the `DELTAWARP` six-param
  flag and the "frame cannot enable a sequence-disabled mode" rule;
* frame precision bits for `HALF`/`QUARTER`/`EIGHTH`, `force_integer_mv`, and
  `read_interpolation_filter()` (fixed 0..3 and switchable);
* entropy writer/reader lockstep for `use_amvd`, AMVD `mv_joint`/`amvd_index`,
  `jmvd_scale_mode`, precision symbols, block `interp_filter`, and shell MVDs
  at every reachable precision, with and without CDF adaptation;
* `pb_mv_precision` reachability and the AMVD short-circuit;
* §8.3 context values over a grid that exercises the `NPos`/`NPosBuf` split;
* `projMv` values, `jmvd_scale_mode` scaling, `mv_clamp_to_integer`, and the
  `enable_mvd_sign_derive` truth table plus sign round-trips;
* pinned goldens.

## Decoder gate

`gate/mv_precision_gate.cpp` encodes key + P1 with the M2 core, then writes P2
with `gate/mv_precision_tile.cpp`.  P2 is a 16x16-leaf NEWMV frame whose leaves
alternate between

* AMVD (`use_amvd = 1`, `mv_joint` + `amvd_index`, `IsAdaptiveMvd`),
* flex precision (`use_most_probable_precision = 0`, `pb_mv_precision = 0`,
  resolved `MvPrecision = ONE_PEL`), and
* frame precision (`use_most_probable_precision = 1`,
  `FrameMvPrecision = QUARTER`),

with `enable_mvd_sign_derive = 1` and a switchable per-block interpolation
filter.  The sequence header enables all three MV-res tools and codes the
motion-mode slice; every inter frame codes `frame_enabled_motion_modes`.

The P2 tile uses the real M2 merge stack (`m2/merge`), the M2 `MiInfo` grid,
M2 default CDFs, and the M2 reconstruction kernel (`inter/mc`).  Prediction
references are the M2 core's key reconstruction.

The stream is decoded with the pinned `dav2d --strict 1 --threads 1`; every
sample is compared with the encoder reconstruction.

Current result (synthetic 128x128 and `testdata/shine_1280x720_60f.y4m`
cropped to 1280x704):

```
P2: tile=175 leaves=64 amvd=21 flex=21 frame_precision=22 derived_sign=21
decoded=3 frames, recon diffs=0

P2: tile=9341 leaves=3520 amvd=1173 flex=1173 frame_precision=1174 derived_sign=1173
decoded=3 frames, recon diffs=0
```

## Call sites (for a future core integration)

The module is intentionally not wired into `src/main.cpp` or the M2 core.  The
minimal integration points are:

1. **Sequence header** (`src/encoder/m2/video_headers.cpp:150-153, 178-180`) —
   replace the four zero `seq_enabled_motion_modes` bits with
   `write_sequence_motion_modes`, and the three MV-res zero bits with
   `write_sequence_mv_res`.
2. **Frame header** (`src/encoder/m2/video_headers.cpp:271-275`) — replace the
   fixed `use_qtr_precision_mv`/`allow_high_precision_mv` pair with
   `write_frame_mv_precision`, the fixed filter pair with
   `write_frame_interp_filter`, and add `write_frame_motion_modes` after it.
3. **Tile mode info** (`src/encoder/m2/video_syntax.cpp:1705-1722`) — after
   `single_mode` add `write_use_amvd`; around `code_drl_index` /
   `write_mv` add `write_block_precision`, `write_amvd_mv` or
   `write_shell_mv`, then `write_mvd_signs`; after `read_refinemv` /
   `read_compound_type` add `write_block_interp_filter`.
4. **Decision layer** — use `plan_block_precision` to choose the per-block
   precision and `assign_joint_mvd` for `JOINT_NEWMV`, exactly as the gate
   does.

Keep the header, decision, and writer in lockstep: the bitstream is only
correct when the writer's `MvPrecision` equals the decision layer's and the
contexts are derived from the same `MiInfo` grid.
