# Inter motion estimation — CPU reference (AV2 §7.13.3)

Wave-2 component 11 (see `docs/agent-waves.md`).  Standalone ME for 8-bit
4:2:0 luma: SAD/SATD, full / diamond / TZ-style search, integer through
1/16-sample refinement, spatial MVP/merge candidates, multiple references, and
a batch API with no per-block allocation.

**No bitstream integration.  No BD-rate claim.**  This component returns motion
vectors and costs; MC, mode decision, RDO and entropy are other components.

Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
(2026-05-28, commit `38f28e7`), §7.13.3.14 (`Subpel_Filters`, two-pass
interpolation) and §7.13.3.16 (`InterRound0`/`InterRound1`).  The filter table
is transcribed from the spec; all 16 phases sum to 128.

## Files

```text
src/encoder/inter/me/
  me_core.h         shared host/device arithmetic, cost, interpolation,
                    MV candidates and the search state machine
  me.h              CPU API
  me.cpp            validation + batch driver
  me_test_common.h  guarded synthetic frames, Y4M loader, job grid
  me_test.cpp       CPU tests (3,746 checks)
  build_and_test.sh standalone build/run (--asan supported)
  README.md         this file
```

The CUDA kernel is in `src/cuda/inter/me/` and instantiates the same
`me_core.h`, so CPU and GPU costs/MVs are bit-exact.

## Data model

- 8-bit planar 4:2:0.  `MeCorePlane` carries `data`, `stride` (samples per row),
  `width`, `height`; `MeCoreFrame` holds Y/U/V plus the luma dimensions.
- Motion vectors are signed `int32_t` in **1/16 luma samples** (`SUBPEL_BITS=4`
  per AV2).  One luma sample = 16 MV units.  Integer positions are multiples of
  16.
- Blocks are 1..128 luma samples per dimension, square or rectangular.
  Right/bottom **partial blocks** are clipped to the current frame: the result
  reports `visible_w`/`visible_h`, and all costs use only the visible area.
- Search runs on **luma**.  Chroma planes are part of the frame/ref views and
  `me_cost_block` / `me_interp_block` accept plane 1/2 with plane-relative
  coordinates and chroma-precision MVs.
- References are data inputs: `MeCoreRef` views, up to 8 active refs per job.
  References must be at least as large as the current frame in each plane.
  **Scaled references are not implemented** (no `xStep`/`yStep` resampling);
  all refs are used at native resolution.

## Cost model

- **SAD**: exact sum of absolute differences over the visible block.
- **SATD**: 8x8 2-D integer Hadamard (`int` accumulation), tiles zero-padded at
  the right/bottom edge of partial blocks, normalised by `(sum_abs + 8) >> 4`.
  For a flat residual `d` this yields `4*d` per 8x8 tile, so SATD and SAD are in
  the same numeric range.  SATD is integer-only and bit-exact CPU/GPU.
- Integer MVs copy samples directly.  Fractional MVs use the normative
  two-pass 8-tap process (`Round2(..., 3)` then `Round2(..., 11)`, clip to
  0..255) with per-tap clipping to the visible reference plane.  Filter rows:
  regular / smooth / sharp / bilinear / regular-4 / smooth-4.
- ME costs are encoder-side decisions only.  The reconstructed prediction must
  still be produced by the MC component with the chosen `interp_filter`.

## Search

All methods evaluate a predictor set first: explicit `job.pred_mv[]` entries
(NewMV/NearestMV candidates), then spatial merge MVs from an optional
`MeCoreMvField`, then `(0,0)`.  Duplicate MVs are removed.  The first-listed
predictor wins ties, and refs are scanned in job order, so the first ref wins a
cost tie.  Candidate order is fixed and comparison is strict `<`, making the
result deterministic.

| Method | Behaviour |
|---|---|
| `kMeFull` | Exhaustive integer grid in `±range` around the best predictor, `integer_step` samples.  `max_candidates` coarsens the step deterministically when set. |
| `kMeDiamond` | Coarse-to-fine 4-point diamond from `diamond_step` (auto `max(1, range/4)`), halving the step when a ring makes no progress. |
| `kMeTestZone` | TZ-style 8-neighbour coarse-to-fine grid from `tz_start_step` (auto `range`), halving each level; up to `max_iterations` moves per level. |

Integer moves are bounded by both `±range` around the predictor-stage best and
the reference plane, so iterative searches cannot drift out of the requested
window or read outside a plane.

Subpel refinement runs after the integer stage.  Stages are selected by
`subpel_stages`:

| Stage | MV step |
|---|---:|
| half | 8/16 |
| quarter | 4/16 |
| eighth | 2/16 |
| sixteenth | 1/16 |

Each stage evaluates the 8 neighbours at that step, repeating up to
`subpel_iterations` while it improves.  Fractional costs use the selected
`subpel_interp` filter.  The default is half+quarter with the regular 8-tap
filter.

## MVP / merge candidates

`me_collect_mv_candidates()` reads a 4x4 MI field (`mv_x`, `mv_y`, optional
`ref_index`, optional `valid`) in AV1/AV2 spatial order: A1, B1, B0, A0, B2.
Duplicates are removed by `(mv, ref)`; a zero-MV candidate with the supplied
default ref is appended when absent.  The search uses the first
`max_merge_candidates` entries as predictor MVs.  On CUDA the same derivation
runs on the device from a device MV field.

## API

```cpp
namespace av2enc::me {

MeStatus me_search_batch(const MeCoreFrame& cur, const MeCoreRef* refs,
                         int num_refs, const MeCoreJob* jobs, int num_jobs,
                         const MeCoreOptions& options,
                         const MeCoreMvField* field, MeCoreResult* results);

MeStatus me_cost_block(const MeCoreFrame& cur, const MeCoreRef& ref,
                       const MeCostJob& job, std::int32_t* out_cost);

MeStatus me_interp_block(const MeCoreRef& ref, const MeInterpJob& job,
                         std::uint8_t* out);

int me_collect_mv_candidates(const MeCoreMvField& field, int x, int y, int w,
                             int h, int default_ref, MeMvCandidate* out,
                             int max_out);
}
```

`MeCoreResult` contains the MV, selected `ref_index`, final `cost`,
`evals` (cost evaluations), the stage that last improved, and the visible
block size.  `evals` is part of the CPU/GPU parity contract.

The batch entry point validates planes, strides, ref indices, block geometry
and options before touching results.  It allocates nothing: predictors and merge
lists are fixed-size stack arrays.

## Tests

```sh
./build_and_test.sh            # build + run
./build_and_test.sh --asan     # ASan/UBSan
```

Coverage (`me_test.cpp`, 3,746 checks):

- hand-checked SAD and SATD (flat, random, partial 8x8 tiles);
- normative interpolation: integer copy, half/quarter phases on a linear ramp;
- full/diamond/TZ search on a known synthetic pan (SAD and SATD);
- exact half-pel, quarter-pel and diagonal fractional matches;
- partial 8x8 / 4x4 visible blocks at the right/bottom edge;
- multiple references, first-ref tie order;
- explicit fractional predictors and spatial merge candidates;
- batch determinism over a 20-block grid;
- guard canaries around every plane (no out-of-bounds reads/writes);
- real `testdata/shine_256x256_1f.y4m` with a synthetic pan (45/49 blocks reach
  zero cost) and the real 720p frame pair (search never loses to zero MV);
- API validation failures.

Measured on the CPU reference: 0 failures, ASan/UBSan clean.  CPU timing is not
a product metric (see `docs/architecture.md`).

## Integration notes

- MC (#12) consumes `mv_x`/`mv_y` (1/16 luma), `ref_index`, and the visible block
  size.  It must apply the selected `interp_filter`; `me_interp_block` is the
  shared oracle for that arithmetic.
- RDO can call `me_cost_block` (or the CUDA `me_cost_batch`) to score explicit
  candidate MVs without re-running a search.
- Rate/mode decision gets `cost` and `evals`; no rate term is included here.
- One call per frame/batch, not per block, keeps the CPU launch/control path
  thin.  The CUDA companion documents device pointers and scheduling.
