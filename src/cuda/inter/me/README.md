# Inter motion estimation — CUDA

Device companion of `src/encoder/inter/me/` (wave-2 component 11).  One warp
per ME job; SAD rows or 8x8 SATD tiles are spread over the 32 lanes and reduced
with warp shuffles.  Search decisions are uniform across the warp, so the same
`me_search_core()` from `me_core.h` produces the CPU-identical MV.

**Component only.  Nothing here writes a bitstream, and no BD-rate is claimed.**

## Files

```text
src/cuda/inter/me/
  me.h              device API
  me.cu             search / cost / interpolation kernels
  me_test.cu        CPU-vs-GPU parity, guard checks, timing (510 checks)
  build_and_test.sh standalone nvcc build/run
  README.md         this file
```

## API

All pointers in frame/ref/job/field/output structures are **device** pointers.
The caller owns allocations and the stream; these entry points allocate,
free and synchronize nothing.

```cpp
namespace av2enc::cuda {

cudaError_t me_search_batch(const me::MeCoreFrame& cur,
                            const me::MeCoreRef* d_refs, int num_refs,
                            const me::MeCoreJob* d_jobs, int num_jobs,
                            const me::MeCoreOptions& options,
                            const me::MeCoreMvField* d_field,
                            me::MeCoreResult* d_results,
                            cudaStream_t stream = nullptr);

cudaError_t me_cost_batch(const me::MeCoreFrame& cur,
                          const me::MeCoreRef* d_refs, int num_refs,
                          const me::MeCostJob* d_jobs, int num_jobs,
                          std::int32_t* d_out, cudaStream_t stream = nullptr);

cudaError_t me_interp_batch(const me::MeCoreRef* d_refs, int num_refs,
                            const me::MeInterpJob* d_jobs, int num_jobs,
                            std::uint8_t* d_out,
                            cudaStream_t stream = nullptr);
}
```

- `d_field` may be null.  When present, it is a device `MeCoreMvField` on the
  4x4 MI grid; spatial merge candidates are derived on the device.
- `me_cost_batch` scores explicit candidates (RDO path); `me_interp_batch`
  produces the normative prediction used by MC and by the parity test.
- The API validates the host-visible frame/ref geometry and options.  Contents
  of device arrays (jobs, refs, MV field) are the caller's responsibility.
- Launch shape: one warp per job, one job per block (`kWarpsPerBlock = 1`).
  No shared memory, no barriers, no per-block allocation.

## Bit-exactness

The cost functor is the only difference from the CPU path:

- SAD: lane `i` accumulates rows `i, i+32, ...`; integer addition is
  associative, so the warp sum equals the scalar sum.
- SATD: lane `i` evaluates 8x8 tiles `i, i+32, ...`; tile sums are non-negative
  integers.
- Subpel, MV clamping, predictor/merge ordering and tie-breaking are the same
  inline code as the CPU reference.

Tests compare `mv_x`, `mv_y`, `ref_index`, `cost`, `evals`, `stage` and visible
block size, not just the MV.

## Build / test

```sh
./build_and_test.sh                 # sm_86, build + run
./build_and_test.sh --no-run
ARCH=sm_120 ./build_and_test.sh     # secondary build target
```

The script compiles the CPU reference with the host compiler, then `me.cu`,
`me_test.cu` and links them.  It does not touch the root CMake build.

Coverage (510 checks, all passing on the RTX 3060):

- 28 SAD/SATD cost jobs: both metrics, all six `Subpel_Filters` rows, luma and
  chroma, integer/fractional MVs;
- 26 interpolation blocks: all filters, negative edge phases, chroma;
- search parity for full / diamond / TZ, SAD and SATD, subpel none through
  1/16, candidate caps, partial blocks, two references;
- device merge candidates and explicit fractional predictors;
- 64 real `shine_256x256` blocks with a synthetic pan;
- plane guard canaries: the kernels are read-only on frame/ref memory
  (byte-compare after every case).

## Measured timing

RTX 3060 12 GB dev proxy (cc 8.6, 28 SMs), CUDA 12.8, `sm_86`, synthetic smooth
pan content, 16x16 blocks, TZ range 16, SAD half+quarter refinement unless
noted.  These are component timings, not an encode throughput claim; run-to-run
variation is about ±5%.  The 8 GB `compat8g` target has the same SM count but
less memory bandwidth.

| Case | Time | Blocks/s | Mpix/s |
|---|---:|---:|---:|
| 720p 16x16 TZ SAD subpel | 9.85 ms | 365,491 | 93.6 |
| 720p 16x16 TZ SATD subpel | 22.06 ms | 163,170 | 41.8 |
| 720p 16x16 full r8 SAD integer | 4.58 ms | 786,212 | 201.3 |
| 1080p 16x16 TZ SAD subpel | 21.10 ms | 386,810 | 98.3 |
| 1080p 8x8 TZ SAD subpel | 29.95 ms | 1,081,922 | 69.2 |

Kernel resources: search kernel 164 registers, 608 B local stack
(predictor/merge lists); cost kernel 96 registers, 368 B.  The search cost
functor is `__noinline__` so the large cost body is compiled once rather than
per call site.  At 1 warp/block, occupancy is register-limited to 12 warps/SM;
the measured timings already fit comfortably inside the 1 s/frame product
budget, but a future pass can lower the stack/register footprint if ME becomes
the frame-time bottleneck.

## Integration notes

- Upload `MeCoreRef` structs and jobs once per batch.  Do not copy per block.
- Keep `MeCoreFrame` plane pointers device-resident; the frame is passed by
  value as a small POD.
- Results are POD `MeCoreResult`; copy them back once per batch if the mode
  decision is on the host, or consume them in a following device kernel.
- MC should use `me_interp_batch` (or the same `me_sample_at`/`me_interp_sample`
  inline code) to keep prediction bit-exact with the search cost.
