# M1 GPU CfL stage — AV2 §7.13.5

GPU chroma-from-luma prediction for M1 blocks. This is **workstream 6** of
[`docs/m1-gpu.md`](../../../../docs/m1-gpu.md); it owns `src/cuda/m1/cfl/` only.

The stage is a thin adapter over the existing CfL component
([`src/cuda/cfl/`](../../../cfl/README.md)). It does **not** re-implement any
§7.13.5 arithmetic: `lumaAvg`, the array `L`, the derived-alpha correlation,
`resolve_division` and `Clip1(dc + Round2Signed(alpha * (L - lumaAvg), 11))`
all come from `cfl_common.h` / `cfl.cu`, the same code the CPU reference
([`src/encoder/cfl/`](../../../../encoder/cfl/README.md)) uses. The M1 layer adds
pair descriptors, decoder-bounds construction, validation and the batch entry
points the M1 orchestrator can call.

## Files

| Path | Purpose |
|---|---|
| `still_cfl.h` | Public API: `StillCflPair`, `StillCflPairExplicit`, `StillCflFrame`, `StillCflRect`, builder, validation, derived/explicit batch entry points. |
| `still_cfl.cu` | Adapter: builds `cfl::Block`s, validates pairs, forwards batches zero-copy to `cfl_predict_derived_batch` / `cfl_predict_batch`. |
| `still_cfl_test.cu` | CPU-vs-GPU bit-exact test (synthetic, derived-alpha 0..511, real 64x64/640x360 frames, timing). |
| `build_still_cfl_test.sh` | Standalone `nvcc` build into `<repo>/build-m1-cfl`; generates the test frames with `ffmpeg`. |

Build/run on the sm_86 box:

```sh
src/cuda/m1/cfl/build_still_cfl_test.sh
# or build only
src/cuda/m1/cfl/build_still_cfl_test.sh --no-run
```

## API

```cpp
namespace av2enc::cuda::still {

struct StillCflFrame {                 // per frame (or tile)
  const uint8_t* luma; int luma_stride;
  int luma_max_x, luma_max_y;       // inclusive, coded plane
  int bit_depth; int ds_filter;     // 8; 0..3
  int sub_x, sub_y;                 // 1,1 for M1
};

struct StillCflRect {                  // per chroma block
  uint8_t* chroma; int chroma_stride;
  int x, y, w, h;                   // chroma samples
  int coded_width, coded_height;    // MiCols*4>>sub_x, MiRows*4>>sub_y
  int have_above, have_left;        // <0 = derive via cfl_avail_*
};

struct StillCflPair { cfl::Block u, v; };          // derived alpha (default)
struct StillCflPairExplicit { CflJob u, v; };      // explicit Q8 alpha

cfl::Block make_still_cfl_block(const StillCflFrame&, const StillCflRect&);
StillCflPair make_still_cfl_pair(const StillCflFrame&, const StillCflRect& u,
                           const StillCflRect& v);
bool still_cfl_valid_pair(const StillCflPair&, std::string* error = nullptr);

cudaError_t still_cfl_predict_pairs(const StillCflPair* d_pairs, int pair_count,
                                 int* d_alpha_q8 = nullptr,
                                 cudaStream_t stream = nullptr);
cudaError_t still_cfl_predict_pairs_explicit(const StillCflPairExplicit* d_pairs,
                                          int pair_count,
                                          cudaStream_t stream = nullptr);
}
```

`StillCflPair` is exactly two `cfl::Block`, and `StillCflPairExplicit` is exactly two
`CflJob`; `static_assert`s in the header pin the layout, so both entry points
forward the caller's device array to the component batch kernels without a
copy. `still_cfl_predict_pairs` runs all `2 * pair_count` plane jobs (U and V of
every pair) in **one launch**; `d_alpha_q8`, when non-null, receives the derived
Q8 alpha per plane (`[2i]` = U, `[2i+1]` = V).

Neither entry point allocates, copies or synchronizes. `d_pairs` and
`d_alpha_q8` are device memory; the orchestrator owns the arena and stream.

## Contract (matches the CPU M1 encoder)

* `block.chroma` holds the block's **DC prediction** on input; the stage
  overwrites the `w x h` coded samples with the CfL prediction. The chroma
  above row / left column are the already-reconstructed neighbours.
* `block.luma` is the reconstructed luma plane with the decoder's coded bounds,
  `MiCols*4-1` / `MiRows*4-1` (exactly `make_cfl_block()` in
  `src/encoder/m1/still_encoder.cpp`). The allocation is padded to a 64-sample
  multiple; those samples are read (clamped) by §7.13.5 and must hold the
  decoder's values (0 for a fresh `CurrFrame`), not an encoder-only fill.
  A padded `max_x`/`max_y` is also accepted — the component clamps reads.
* `have_above`/`have_left` are `AvailUChroma`/`AvailLChroma` at chroma-mi
  granularity (`cfl_avail_above` / `cfl_avail_left`), not `y > 0`. The builder
  derives them from `x`/`y` unless the caller overrides.
* `sb_top_y` is `((y << sub_y) / 64) * 64` through the shared
  `cfl::cfl_sb_top_y`, so the §7.13.5 above-line clamp matches the CPU.
* Only 8-bit 4:2:0 is an M1 format; other depths/subsampling are handled by the
  component but are not exercised by this stage's tests.

## Integration notes

The orchestrator's `stage::cfl(const GpuFrame&, std::string&)` hook is still
unimplemented (the weak symbol stays unset). The hook has no per-block job
array, and the M1 data flow is:

1. `stage::chroma` already produces the winning DC/CfL prediction per block
   (`src/cuda/m1/chroma/`) and writes `is_cfl` + alphas into the decision array.
2. A wavefront/scheduler pass that owns the recon dependency order can call
   `still_cfl_predict_pairs` with a device array of `StillCflPair`s built from
   `FramePlan::blocks` (`chroma_w/h`, `plane_offset_u/v`, `avail`) and the
   arena `dev_recon` pointers. For explicit-alpha decisions, fill
   `StillCflPairExplicit` instead.

The blocker is orchestrator-side, not kernel-side: there is no stage adapter
that converts `BlockPlan` + `BlockDecision` into pair descriptors yet, and the
current `stage::chroma` module already folds CfL into its prediction. Wiring
the hook should be done by the orchestrator/scheduler owner once the recon
dependency order is available, so it does not duplicate the chroma decision.

## What the test proves

Run on the 3060 (`sm_86`, CUDA 12.8):

```
still_cfl_test: 8491 cases, 7123840 samples, 7978 derived alphas, all exact
```

Coverage:

| Test | Coverage |
|---|---|
| Synthetic geometry | chroma 4x4..64x64, offsets 0/1/2/3/4/8/16/32/64, all four `cfl_ds_filter_index` values, x/y inside the first chroma mi row/column (`have_* = 0` even though samples exist), coded bounds narrower than the block (right/bottom `prevSample` replication), padded vs coded luma bounds, explicit U/V alphas with opposite signs. |
| Derived alpha 0..511 | Constructed 4x4 pairs whose `implicitAlpha` walks every value 0..511; each GPU alpha and all 16 predicted samples match the CPU. Pairs share luma and often carry different U/V alphas. |
| Explicit alpha | Every Q8 value −512..512 on an 8x8 block (U = +q, V = −q), plus signalable multiples of 32. |
| Real frames | 64x64 testsrc2 (1 block); 640x360 testsrc2 (950 M1 blocks; 640 4x4 chroma blocks, 20 bottom-partial, 155 SB-top-clamp, 45 no-above, 19 no-left); 632x356 crop (942 blocks, 11 right-partial). Chroma DC is produced by `intra::predict_block`; each block is replayed from the exact frame state M1 had (reconstructed neighbours, DC in the current block), then U+V are batched. |
| API | no-op/invalid pointers, optional alpha readback, U/V geometry validation. |

Timing (one whole-frame 640x360 batch, 50 iterations after 5 warmups, CUDA
events, kernel time only):

```
950 pairs (1900 plane jobs), 0.352 ms/frame, 185 ns/plane-job
```

## Limitations / follow-ups

* The adapter reuses the component's **one-CUDA-block-per-plane** kernel, so a
  U/V pair computes the shared `L`/`lumaAvg` twice. A fused pair kernel would
  remove that and shrink the per-job launch overhead, but the component's
  internals are private to `src/cuda/cfl/cfl.cu`; exposing a pair body there
  (owner: CfL component) is the clean next step. At 640x360 the duplication is
  0.35 ms/frame; it is throughput on 4x4/8x8 blocks, not launch overhead.
* `stage::cfl` is not wired; see the integration notes above.
* `still_cfl_valid_pair` is a host-side plan check (it calls the CPU reference's
  `cfl::valid`); it is never called from device code.
