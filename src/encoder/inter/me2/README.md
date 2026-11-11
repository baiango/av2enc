# me2 — extended subpel / adaptive-precision motion estimation

Additive AV2 motion-estimation layer over `src/encoder/inter/me/` (unchanged).
It keeps the existing integer / half / quarter path and adds:

* **eighth- and sixteenth-pel coverage** using the normative
  `Subpel_Filters` two-pass interpolation already shared by ME and MC;
* a **deterministic exhaustive refinement** pass around the best
  integer/subpel candidate (the ME core's 8-neighbour hill climb can stop one
  step early);
* **`enable_flex_mvres` adaptive per-block MV precision**: the 1/16 search
  result is lowered to the frame precision or a coarser reachable precision,
  re-costed at the signalled MV, and scored with a deterministic
  distortion/rate proxy;
* an exact description of the **M2 signalling** for
  `use_qtr_precision_mv`, `allow_high_precision_mv`, `enable_flex_mvres`,
  `use_most_probable_precision` and `pb_mv_precision`;
* a **CPU reference** and a **CUDA batch kernel** that instantiate the same
  inline core, plus a host-emulation parity test that runs without `nvcc`.

No bitstream syntax is written here.  ME2 returns MVs, precisions, distortion,
a rate proxy and evaluation counts; M2 / MC / RDO own the syntax and
reconstruction.

Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
(2026-05-28, commit `38f28e7`), §5.18.2 (frame header precision),
§5.20.7.11–§5.20.7.13 (per-block precision and `read_mv`), §7.13.3.14
(`Subpel_Filters`), §7.13.3.16 (`InterRound0/1`).

## Files

```text
src/encoder/inter/me2/
  me2_precision.h          MV precision enum, lower_mv_precision, frame/block
                           signalling maps, shell-code rate proxy
  me2_core.h               host/device search + refinement + precision selection
  me2.h / me2.cpp          CPU reference API and validation
  me2_gpu_cost.cuh         warp-cooperative cost functor (shared with the kernel)
  me2_gpu.h / me2_gpu.cu   CUDA batch search/cost entry points
  me2_test.cpp             CPU tests + 720p quality/rate report
  me2_gpu_emulation_test.cpp  host parity for the warp cost and search core
  me2_gpu_test.cu          CPU-vs-CUDA bit-exact test (needs nvcc + GPU)
  build_and_test.sh        CPU + emulation build/run (--asan, --quick)
  build_gpu_and_test.sh    CUDA build/run (falls back to emulation without nvcc)
  README.md                this file
```

Nothing here edits `src/encoder/inter/me/`, `src/encoder/m2/`, `src/main.cpp`
or the root `CMakeLists.txt`.

## Units and data model

* 8-bit planar 4:2:0.  Frames/refs/jobs use the ME data model unchanged
  (`MeCoreFrame`, `MeCoreRef`, `MeCoreJob`, `MeCoreMvField`).
* ME2 works in **1/16 luma samples** internally (`me::kSubpelOne == 16`).
* AV2 signals at most **1/8 luma sample** (`MV_PRECISION_EIGHTH_PEL`).  The
  1/16 stage is an encoder-side search granularity.  `Me2Result::mv_x/mv_y` are
  1/16 units but are always a multiple of the selected precision's quantum:

  | precision | enum | quantum (1/16) | signalled bits |
  |---|---:|---:|---|
  | 8-pel | 0 | 128 | frame header only |
  | 4-pel | 1 | 64 | frame header only |
  | 2-pel | 2 | 32 | **not reachable** by the header |
  | 1-pel | 3 | 16 | `force_integer_mv` only |
  | 1/2-pel | 4 | 8 | `use_qtr=0, allow_high=0` |
  | 1/4-pel | 5 | 4 | `use_qtr=1` |
  | 1/8-pel | 6 | 2 | `use_qtr=0, allow_high=1` |

* `search_mv_x/search_mv_y` is the refined 1/16 result before lowering;
  `mv_x/mv_y` is the **signalled** MV and `cost` is its distortion.
* `me2_1_16_to_eighth()` matches `inter::mc::mv_from_me_1_16()` exactly
  (`Round2Signed(...,1)`, ties away from zero).  The tests lock this contract.
* Coarser lowering uses the normative `lower_mv_precision()` rule
  (`Round2(a-1, bits)`, ties down in magnitude, clipped to
  `MV_LOW/MV_UPP ± radix`).

## API

```cpp
namespace av2enc::me2 {

Me2Status me2_search_batch(const MeCoreFrame& cur, const MeCoreRef* refs,
                           int num_refs, const MeCoreJob* jobs, int num_jobs,
                           const Me2Options& options,
                           const MeCoreMvField* field, Me2Result* results);

Me2Status me2_cost_block(const MeCoreFrame& cur, const MeCoreRef& ref,
                         const Me2CostJob& job, std::int32_t* out_cost);

Me2Status me2_interp_block(const MeCoreRef& ref, const me::MeInterpJob& job,
                           std::uint8_t* out);
}
```

`Me2Options`:

| field | meaning |
|---|---|
| `search` | existing `MeCoreOptions` (method, metric, range, stages, filter) |
| `frame_precision` | `FrameMvPrecision` (`kMvPrecision*`) |
| `enable_flex_mvres` | allow NEWMV blocks to signal a coarser precision |
| `precision_mask` | optional per-precision bitmask; 0 = all reachable |
| `refine_radius` / `refine_step` | exhaustive post-search pass in 1/16 units |
| `mv_rate_lambda` / `mv_rate_shift` | selection cost = distortion + `(lambda*rate)>>shift` |
| `count_precision_bins` | include the per-block precision side info in `rate` |

`Me2Result` carries: signalled MV, search MV, `ref_index`, distortion at both
MVs, `evals`, `mv_rate_bins`, `precision_bins`, `total_rate_bins`, selected
`precision`, `use_most_probable_precision`, `pb_mv_precision`,
`pb_precision_coarser`, last improving `stage`, and the visible block size.

The batch entry point validates everything before touching `results`; it
allocates nothing (fixed-size stack state only).

## Search pipeline

Per reference, in job order:

1. **ME core** (`me::me_search_core`, unchanged): predictor set (explicit MVPs,
   spatial merge, zero), integer full/diamond/TZ search, then half/quarter/
   eighth subpel rings.  The 1/16 ring is masked off (see below).
2. **Exhaustive refinement** around the ME result: fixed `dy`-then-`dx` order,
   offsets in `[-radius, radius]`, step `refine_step` (1 = 1/16).  Strict `<`
   improvement only.  If the caller enables the ME sixteenth stage but leaves
   `refine_radius == 0`, a 3x3 one-step 1/16 pass is used, so the stage is
   never silently ignored.
3. **Precision selection.**  For every reachable precision, both the
   pre-refinement and refined MVs are quantized, clamped to the reference,
   and re-costed.  Selection cost is
   `distortion + (mv_rate_lambda * rate_bins) >> mv_rate_shift`.
   Deterministic order: selection cost, then fewer rate bins, then finer
   precision, then smaller `|MV|`, then `mv_y`, then `mv_x`.
4. **Reference selection**: same order; a strict tie keeps the earlier
   reference.  So the first ref wins a full tie.

Why the ME 1/16 ring is masked off: a 1/16 local minimum can quantize to a
*worse* 1/8 position than the eighth-grid result.  Keeping the pre-refinement
candidate on the signalable grid and scoring the refined candidate after
quantization makes the signalled result monotonic with respect to the
eighth-only path.  The 1/16 pass still reduces the internal search cost; on
the 720p report below it improves `searchSAD` by ~5% but cannot change the
signalled MV, because AV2 cannot signal 1/16.

### Stage mask

`Me2Options::search.subpel_stages` accepts the existing ME bits
(`kMeSubpelHalf|Quarter|Eighth|Sixteenth`).  `kMeSubpelSixteenth` maps to the
exhaustive 1/16 pass.  `kMeSubpelEighth` is the normal ME ring.  Fractional
costs use `search.subpel_interp` (normative rows 0..5).

## Adaptive precision and exact M2 signalling

### Sequence header

`enable_flex_mvres` is one bit in §5.4.6, written after
`enable_mvd_sign_derive` and before `enable_global_motion`:

```text
f(1) enable_flex_mvres;
```

The M2 baseline currently writes `0` (`m2_headers.cpp`); enabling per-block
precision requires writing `1` there (an M2-owned change).

### Frame header

§5.18.2, inside the non-key/non-intrabc branch:

```text
if (force_integer_mv) {
    FrameMvPrecision = MV_PRECISION_ONE_PEL
    UsePerBlockMvPrecision = 0
} else {
    f(1) use_qtr_precision_mv;
    if (use_qtr_precision_mv) {
        FrameMvPrecision = MV_PRECISION_QUARTER_PEL
    } else {
        f(1) allow_high_precision_mv;
        FrameMvPrecision = allow_high_precision_mv ?
            MV_PRECISION_EIGHTH_PEL : MV_PRECISION_HALF_PEL
    }
    UsePerBlockMvPrecision = enable_flex_mvres
}
MvPrecision = FrameMvPrecision
```

Exact bit cost of these two fields:

| target frame precision | `use_qtr_precision_mv` | `allow_high_precision_mv` | coded bits | flex usable |
|---|---:|---:|---:|---:|
| 1/8 | 0 | 1 | 2 | yes |
| 1/4 | 1 | not coded | 1 | yes |
| 1/2 | 0 | 0 | 2 | yes |
| 1 | not coded (`force_integer_mv=1`) | not coded | 0 | **no** |

`force_integer_mv` comes from the screen-content path; with the M2 baseline
(`seq_force_screen_content_tools = 0`) it is inferred 0, so **one-pel frame
precision is not reachable** in the current M2 configuration.  Two-pel and
coarser are not reachable from the frame header at all.

`me2_frame_precision_to_bits()` / `me2_frame_precision_from_bits()` implement
this table and round-trip it in the tests.

### Per-block precision

Only for NEWMV when `enable_flex_mvres && UsePerBlockMvPrecision`
(§5.20.7.11):

```text
f(1) use_most_probable_precision;
if (use_most_probable_precision) {
    MvPrecision = FrameMvPrecision
} else {
    S() pb_mv_precision;
    adjustedPrecision = Max(MV_PRECISION_ONE_PEL, FrameMvPrecision - 2)
                        - pb_mv_precision;
    MvPrecision = adjustedPrecision <= MV_PRECISION_TWO_PEL ?
                  adjustedPrecision - 1 : adjustedPrecision;
}
```

`me2_block_precision_to_syntax(frame, target)` is the exact inverse.  The
reachable sets are:

| frame precision | reachable per-block precisions (`use_most=1` first) |
|---|---|
| 1/8 | 1/8, 1/2 (`pb=0`), 1 (`pb=1`), 4 (`pb=2`), 8 (`pb=3`) |
| 1/4 | 1/4, 1 (`pb=0`), 4 (`pb=1`), 8 (`pb=2`) |
| 1/2 | 1/2, 1 (`pb=0`), 4 (`pb=1`), 8 (`pb=2`) |
| 1 | 1 only (`UsePerBlockMvPrecision = 0`) |

Important consequences:

* per-block precision can only get **coarser** than the frame precision;
* **1/4 is not reachable when the frame precision is 1/8** with flex on —
  the inverse skips it; 2-pel is never reachable (the `adjusted - 1` step);
* one-pel frame precision implies `force_integer_mv` and disables flex.

Entropy values and contexts (`Me2Result` + helpers):

* `use_most_probable_precision`: `TileUseMostProbablePrecisionCdf[ctx]`,
  `ctx =` number of neighbours with `UseMostProbablePrecisions != 0`
  (`me2_ctx_use_most_probable()`).
* `pb_mv_precision`: `TilePbMvPrecisionCdf[ctx][row]`, row =
  `FrameMvPrecision - MV_PRECISION_HALF_PEL` (0/1/2), `ctx = 1` if any
  neighbour has `MvPrecisions < FrameMvPrecision`, else 0
  (`me2_ctx_pb_precision()`).
  `Me2Result::pb_precision_coarser` is only this block's local part
  (`MvPrecision < FrameMvPrecision`); M2 must OR it with the neighbour state.
* NEARMV/GLOBALMV do not code either symbol and use `MvPrecision =
  FrameMvPrecision`.

The two precision CDF defaults (`TileUseMostProbablePrecisionCdf`,
`TilePbMvPrecisionCdf`) are **not present in this tree** (`m2_cdf_tables.h`
predates them).  ME2 reports the exact `use_most_probable_precision` value,
`pb_mv_precision` symbol, local context bit and CDF row; M2 must add the §9.3
default rows before it can write the symbols.

### MV coding implication

`read_mv()` (§5.20.7.20) shell-codes the MVD in units of `MvPrecision`, then
shifts it back to the 1/8 domain:

```text
shift = MV_PRECISION_EIGHTH_PEL - MvPrecision;
diffMv = shellIndex << shift;
```

When `MvPrecision < MV_PRECISION_HALF_PEL` and the block is NEWMV, the
predictor is first lowered with `lower_mv_precision(MvPrecision, PredMv)`
(§5.20.7.13).  ME2 mirrors that in its rate proxy.  For HALF/QUARTER/EIGHTH the
predictor is used at 1/8 precision.

## Rate proxy

`me2_mvd_shell_bins()` counts the `S()`/`L(1)` decisions of `read_mv()` for a
MVD at a given precision: shell set/class, class offset, column coding and the
two sign bits.  `me2_precision_side_bins()` adds the per-block precision
symbols (1 bin for `use_most=1`, 2 for the `pb` path).  This is a
**deterministic bin-count proxy**, not entropy: it ignores CDF skew and does
not model the precision CDFs (absent in-tree).  CPU and GPU use the identical
integer function.

`Me2Result::total_rate_bins = mv_rate_bins + precision_bins`.  With
`mv_rate_lambda == 0` the selected result still uses the rate tie-break for
equal distortion, so a coarser precision with the same distortion and fewer
bins wins; that is deterministic and tested.

## CPU / GPU parity

The CUDA kernel (`me2_gpu.cu`) runs one warp per job and calls the same
`me2_search_core()` with `Me2WarpCost`:

* SAD rows / 8x8 SATD tiles are split across 32 lanes;
* integer `__shfl_down_sync` reduction = scalar sum exactly;
* every search decision is uniform across the warp; lane 0 writes the result.

Parity is checked two ways:

1. `me2_gpu_emulation_test.cpp` (always built, no GPU): sums the 32 lane
   partials on the host, compares candidate costs with `me::me_cost_plane`,
   and compares every `Me2Result` field between the scalar and simulated-warp
   instantiations for full/diamond/TZ, all precisions, flex on/off and
   different lambdas.  This is the gate used on the macOS development host.
2. `me2_gpu_test.cu` (when `nvcc` + an sm_86 GPU are present): uploads frames,
   runs `cuda::me2_search_batch` / `me2_cost_batch`, and compares MVs, costs,
   precision, rate bins, evals and stage against the CPU reference.  It also
   verifies that the kernels do not write plane memory (guard canaries).

Run:

```sh
./build_and_test.sh            # CPU + host-emulation parity + 720p report
./build_and_test.sh --asan     # ASan/UBSan
./build_and_test.sh --quick    # skip the 720p report
./build_gpu_and_test.sh        # nvcc path, or emulation fallback
```

## Test coverage

`me2_test.cpp` currently runs 506 checks (plus 1,993 emulation checks):

* frame precision header bits and round-trips, including `force_integer_mv`;
* every reachable/unreachable `pb_mv_precision` case and candidate order;
* `lower_mv_precision` ties, signs, clipping and quantization idempotence;
* MC/me2 1/16->1/8 contract;
* MVD shell bins and precision side bins (including the non-representable
  case);
* deterministic refinement: exact minima, first-minimum tie-break, exact
  eval counts, hill-climb escape;
* precision rate trade-off (`HALF_PEL` chosen over `EIGHTH_PEL` when the MVD
  shell is one bin cheaper and distortion ties);
* planted half/quarter/eighth/sixteenth-pel shifts with the normative filter;
* partial blocks, guard canaries, multi-reference strict tie, batch
  determinism, spatial merge field passthrough, API validation.

### 720p quality/rate proxy

`test_quality_rate_720p()` loads up to 24 frames from
`testdata/shine_1280x720_60f.y4m` (override with `ME2_Y4M=/path/clip.y4m`),
uses frame pairs (0/1, 8/9, 16/17), and measures every second 64x64 block.
A 280-frame 720p file is not in the repository; point `ME2_Y4M` at one to run
the same harness on it (the first 24 frames are used).  PSNR is computed from
the ME prediction against the source (no residual coding), so it is a relative
quality proxy, not a reconstructed-frame metric.

Measured on the CPU reference (8-bit 4:2:0, SAD metric, diamond range 16):

```text
config              signalled SAD   PSNR     MV bins  side   search SAD  evals
quarter_search         212182       53.790     1308     0     212182     11020
eighth_search          167515       55.662     1367     0     167515     14588
sixteenth              167515       55.662     1367     0     158989     23289
flex@quarter           219413       53.423     1113   378     158989     24552
flex@eighth            167812       55.656     1254   381     158989     24973
flex@eighth lam64      168095       55.649     1236   385     158989     24973
```

Interpretation (proxy only — **no BD-rate claim**):

* Adding the 1/8 stage improves signalled SAD by 21.1% over the current
  1/4-stage path on this sample, for +4.5% MV shell bins.
* The 1/16 pass reduces the internal search SAD by 5.1% but the signalled SAD
  is unchanged: AV2 cannot signal 1/16.
* Flex trades a small SAD increase for fewer MVD bins (1254 vs 1367 at
  lambda 16; 1236 at lambda 64) but pays one `use_most_probable_precision`
  bin per NEWMV block plus one extra bin for coarser blocks.  The net rate
  proxy is higher with this bin-count model; a real entropy coder with
  `TileUseMostProbablePrecisionCdf` / `TilePbMvPrecisionCdf` is needed before
  drawing rate conclusions.  Precision histograms show most blocks stay at
  1/8, with a tail at 1/2, 1-pel, 4-pel and 8-pel.
* `flex@quarter` starts from a 1/4 frame precision, so it can never recover
  the 1/8 MVs and is strictly worse than `flex@eighth` / `eighth_search`.

## M2 integration notes

ME2 is a drop-in *producer* of decisions; it does not write syntax.

1. **Sequence header**: write `enable_flex_mvres = 1` only when using
   `Me2Options::enable_flex_mvres`.  The current M2 writer hard-codes 0.
2. **Frame header**: choose `FrameMvPrecision` with
   `me2_frame_precision_to_bits(target, flex, force_integer_mv, &s)` and write
   `s.use_qtr_precision_mv`; write `s.allow_high_precision_mv` only when
   `s.coded_bits == 2 && !s.use_qtr_precision_mv`.  The current M2 baseline
   writes `use_qtr=0, allow_high=1` (1/8 frame precision), which is exactly
   what the `eighth_search`/`sixteenth` configurations use.
3. **Per NEWMV block**: if flex is on, use
   `Me2Result::use_most_probable_precision` and `pb_mv_precision` directly;
   they already satisfy the decoder's inverse mapping.  Only NEWMV codes them.
   `Me2Result::pb_precision_coarser` and `me2_pb_precision_cdf_row()` give
   the local context bit and CDF row; add the neighbour state for the final
   `ctx`.  NEARMV/GLOBALMV omit both symbols.
4. **MVD**: the signalled MV is `Me2Result::mv_x/mv_y` (1/16 units).  Convert
   to the 1/8 domain with `me2_1_16_to_eighth()` (identical to
   `mc::mv_from_me_1_16`) before computing MVD/predictor stacks.  ME2's rate
   proxy assumes the first explicit predictor (`job.pred_mv[0]`) as the MVD
   reference; set it to the intended DRL-stack predictor for a tighter proxy,
   or use zero when no predictor is known.
5. **MC**: `mv_x/mv_y` is quantized to the selected precision, so MC can
   predict with the same `interp_filter` as before.  `me2_interp_block()`
   delegates to the ME oracle for tests.
6. **RDO**: `cost` is distortion at the signalled MV; `total_rate_bins` is a
   proxy.  Use `mv_rate_lambda` to make ME2 itself choose the precision, or
   call `me2_cost_block()` for explicit candidates at a fixed precision.
7. **CDF tables**: add the §9.3 defaults for
   `TileUseMostProbablePrecisionCdf[ctx]` and
   `TilePbMvPrecisionCdf[ctx][row]` to `m2_cdf_tables.h` (M2-owned) before
   enabling `use_most_probable_precision` / `pb_mv_precision` output.  Until
   then, keep `enable_flex_mvres = 0` and use ME2 only for 1/8 search.

## Limitations

* 1/16 MVs are never signalled; the 1/16 stage only improves the internal
  search/selection.
* `enable_flex_mvres` cannot select a finer precision than the frame, cannot
  reach 1/4 from a 1/8 frame, and cannot reach 2-pel.
* One-pel frame precision requires `force_integer_mv` (screen-content tools),
  which the M2 baseline disables.
* Luma search only; chroma ME and chroma MV derivation remain MC-owned.
* No scaled references, no compound/B-frame ME, no temporal/merge-bank
  candidates beyond the existing spatial merge field.
* The rate proxy is a bin count, not an entropy rate; precision CDFs are not
  in-tree.
* The CUDA kernel was compiled/tested only where `nvcc` and an sm_86 device are
  available; on macOS the host-emulation parity test is the gate.
* CPU timing is not a product metric (see `docs/architecture.md`).
