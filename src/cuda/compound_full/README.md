# Compound prediction — full CUDA path

GPU counterpart of the M2 compound prediction / OPFL / refine-MV stack:

- `src/encoder/inter/advanced/compound_common.h`, `compound.cpp` (§7.13.3.14,
  .27–.30),
- `src/encoder/m2/multiref/compound_predict.*` (1/16 predictions, implicit mask,
  per-plane apply),
- `src/encoder/m2/multiref/gate/compound_stream.cpp` (the reconstruction the
  gate stream performs),
- `src/encoder/inter/temporal/opfl.*` (§7.13.3.8–.12),
- `src/encoder/inter/temporal/refinemv.*` (§7.13.3.3–.6).

Component only: no bitstream, no root `CMakeLists.txt`, `main.cpp`, or pipeline
edits.

## Status

| Feature | GPU | Oracle |
|---|---|---|
| AVERAGE + all CWP weights | complete | `adv::run_compound_job` (8-bit) |
| Implicit mask (`get_mask`) | complete | `adv::run_compound_job` |
| WEDGE, 68 indices × 2 signs | complete | `adv::run_compound_job` |
| DIFFWTD, both `mask_type` | complete | `adv::run_compound_job` |
| Inter-intra (DC/V/H/smooth + wedge) | complete | `adv::run_compound_job` |
| Chroma mask subsample (4:2:0/4:2:2/4:4:4) | complete | `adv::run_compound_job` |
| 10-bit (8 → 1023 Clip1) | complete | independent literal reference |
| Skip-mode derived prediction (same MV, no residual) | complete | `build/apply_compound_prediction` (8-bit) |
| OPFL `OpflMvs` / `MvDeltas` | complete | `temporal::opfl_compute` (8-bit) |
| Refine-MV 24-neighbour pixel search | complete | `temporal::search_refinemv` |
| Refine-MV final EIGHTTAP_SHARP + average | complete | `temporal::refine_predict_block` |
| §7.13.3.4/.5 `get_ref_area` | complete | `temporal::compute_ref_area` |
| Batch descriptors in the arena | complete | batch and per-job runs agree |

All block sizes from 4×4 to 64×64 are covered, including edge rectangles,
4-wide/4-tall filter downgrade, and chroma half geometry.

## Files

```text
src/cuda/compound_full/
  compound_full.h                  public API, job PODs, Arena workspace
  compound_full_device.cuh         shared device body (kernel + emulation)
  compound_full.cu                 kernels, launchers, validation
  compound_full_test.cu            CPU-vs-GPU parity + arena + batch test
  compound_full_emulation_test.cpp host-emulation parity test
  compound_full_reference.h        independent 10-bit blend/skip reference (test)
  compound_full_cpu_oracle.h/.cpp  host-only bridge to adv::run_compound_job
  build_and_test.sh                nvcc -> GPU test, else emulation
  build_gpu_test.sh                nvcc build/run
  build_emulation_test.sh          host-only build/run
  README.md
```

## API

```cpp
#include "cuda/compound_full/compound_full.h"
using namespace av2enc::cuda::compound_full;

// One-time persistent allocation from the runtime arena.
CompoundFullConfig config;   // max_jobs = 64, wedge_cache = 1
CompoundFullWorkspace ws;
std::uint64_t required = 0;
Status s = workspace_create(arena, config, &ws, &required, &err);
// required == required_device_bytes(config) == 2,281,472 bytes for 64 jobs
wedge_cache_build(ws);       // one async kernel, optional but recommended

// Blend: one block per job, one thread per sample.
BlendJob job;                // all pointers are device pointers
job.type = kCompoundWedge;   // or kAverage / kDiffWtd / kIntra
job.p0 = preds0_1_16;        // int16_t, 1/16 sample precision
job.p1 = preds1_1_16;
job.dst = recon;             // uint8_t* or uint16_t* (dst_16bit)
job.mask_out = ws.masks + slot * config.max_mask_samples;  // plane 0
blend_batch_ws(ws, &job, 1, stream);

// Skip mode: both lists use one MV, average, no residual; Y/U/V in one job.
skip_batch(device_skip_jobs, count, stream);

// OPFL refinement.
opfl_batch(device_opfl_jobs, count, stream);   // OpflMvs + MvDeltas

// Refine-MV.
refine_search_batch(device_search_jobs, device_results, count, stream);
refine_predict_batch(device_predict_jobs, count, stream);

// §7.13.3.4/.5, host or device callable.
RefAreaInput in; int area[2][4]; ref_area_compute(in, area);
```

Launchers are asynchronous, allocate nothing and never synchronize. The `_ws`
forms copy a small host descriptor array into `workspace.jobs` (and friends)
and then launch. `job_count == 0` is a no-op; a null descriptor array with
`job_count > 0` returns `cudaErrorInvalidValue`.

Chroma DIFFWTD can either consume the plane-0 mask (`luma_mask`) or derive it
from the luma predictions (`luma_p0`/`luma_p1`). The second form removes any
job-order dependency from a batch; both are bit-identical.

## Arena contract

- `required_device_bytes(config)` returns the exact persistent size, including
  the arena's 256-byte rounding of every allocation:
  - descriptors: `BlendJob`, `SkipJob`, `OpflJob`, `RefineSearchJob`,
    `RefinePredictJob`, `RefineResult` × `max_jobs`;
  - luma mask pool: `max_mask_samples` × `max_jobs` (4096 = 64×64 default);
  - wedge-mask cache: 1,958,400 bytes (`wedge_cache = 1`).
- `workspace_create` allocates once with `Arena::allocate_checked`. A short
  arena returns `Status::kArenaExhausted`, sets `*out_required_bytes`, and
  rewinds the arena (no leak). `kInvalidArgument` / `kNotInitialized` are
  distinct.
- `workspace.persistent_mark` is the arena watermark after the persistent set.
  Launchers never move it. Callers that want temporary scratch can
  `arena.mark()` / `arena.rewind()` around their own work without touching the
  workspace.
- There is **no** `cudaMalloc`, `cudaFree`, `cudaMallocHost` or per-launch
  allocation in this module. The tests also allocate every fixture from the
  same arena.

Sizes measured by the tests:

| `max_jobs` | `required_device_bytes` |
|---|---|
| 8 | 1,999,360 B |
| 64 | 2,281,472 B |

## Bit-exactness

`compound_full_device.cuh` is the single transcription. It is compiled by nvcc
for the kernels and as host code by the emulation test, so both run the same
integer code. Where the CPU reference exposes a device-safe helper (wedge
masks, §4.8 rounding, `get_relative_dist`, MV scaling) the header calls it
instead of restating it.

- **8-bit**: `clamp` is `Clip1(..., 8)`, so the blend is byte-identical to
  `adv::run_compound_job()`. OPFL and refine-MV are compared field-by-field
  against `inter::temporal`.
- **10-bit**: the CPU advanced module hard-clips compound output to 8 bits, so
  it is not a 10-bit oracle. The blend and skip paths are compared against
  `compound_full_reference.h`, an independent literal transcription with
  `Clip1(..., BitDepth)` and uint16 samples. OPFL/refine-MV for 10-bit apply the
  spec's `Round2Signed(..., BitDepth - 8)` / SAD shift and are checked GPU vs the
  same shared header executed on the host.
- No floating point, no tolerances.

## Device-code audit (host emulation cannot catch these)

The shared header was audited for the three failure classes seen in other
modules on real `nvcc` builds:

1. **Helpers called from kernels** — every free function in
   `compound_full_device.cuh` is `AV2ENC_CF_HD` (`__host__ __device__` under
   nvcc): `wedge_cache_*`, `cf_wedge_*`, `load/store_sample`, `ref_read`,
   `location_*`, `filter_sample`, `predict_unscaled_u8`, `implicit_mask_sample`,
   `luma_mask_value`, `mask_sample`, `blend_sample_at`, `mask_out_at`,
   `run_*_device`, `ref_area_compute`, `get_msb`, `opfl_*`,
   `divide_and_round_array`, `refine_candidate_sad`.
2. **Namespace-scope tables** — the wedge/intra tables are copies of
   `encoder/inter/advanced/tables/wedge_tables.h` with
   `AV2ENC_CF_DEV_TABLE`, which is `__device__ constexpr` in the device pass
   and `inline constexpr` in the host pass:
   `kCfWedgeCosLut`, `kCfWedgeSinLut`, `kCfPos/NegDist2BldWeight`,
   `kCfWedgeCodebook`, `kCfWedgeBlockWidth/Height`, `kCfWedgeBits`,
   `kCfIiWeights1d`, `kRefinemvNeighbors`. The host-only tables are never
   referenced from device code. All integer constants used by the kernels are
   enumerators (`kCf*`, `kCompound*`) or function-local arrays (`mv_pair`,
   `area`, `dist`, `solved`).
   Host-only `static_assert`s compare every entry of `kCfWedgeBits`,
   `kCfWedgeBlockWidth/Height`, the codebook/LUTs and `kCfIiWeights1d` with
   `adv::tables::*` (guarded by `#if !defined(__CUDA_ARCH__)`, so the device
   pass never sees the host symbols). `adv::bsize_has_wedge` is no longer
   called: kernels and host validation use `cf_bsize_has_wedge` backed by
   `kCfWedgeBits`.
3. **No host-only library calls** — device code uses only scalar `<cstdint>`
   types; there are no `std::` containers or algorithms and no `static`
   mutable globals. `mc::kSubpelFilters` already carries the repo's
   `__device__ constexpr` switch; `mc::mc_*` and the `adv::*` rounding/scaling
   helpers are `AV2ENC_*_HD` and table-free.
4. **CPU oracle isolation** — `adv::run_compound_job` is `__host__ __device__`
   and its body reaches `adv::tables::kWedgeBits` through
   `bsize_has_wedge()`. A `.cu` cannot call it without nvcc device-compiling
   that body. The GPU test calls `test::run_compound_job_cpu()` from
   `compound_full_cpu_oracle.cpp` (host compiler), so the oracle is still
   exactly `adv::run_compound_job` while nvcc never sees it.

## Build and test

On the Linux + CUDA box:

```sh
./src/cuda/compound_full/build_and_test.sh        # sm_86, build + run
ARCH=sm_120 ./src/cuda/compound_full/build_and_test.sh
./src/cuda/compound_full/build_gpu_test.sh --no-run
```

Without a GPU or nvcc (macOS):

```sh
./src/cuda/compound_full/build_and_test.sh
./src/cuda/compound_full/build_emulation_test.sh
```

Local emulation result (this tree):

```text
arena: required=1999360 bytes, used=1999360, peak=1999360
blend checks: 3456
skip checks: 144
opfl checks: 30
refine checks: 342
compound_full_emulation: 4354 checks passed
```

The GPU test additionally:

- checks the arena contract: `required_device_bytes`, a one-byte-short arena
  returning `kArenaExhausted` with the exact required size, `used == required`,
  and two `mark()`/`rewind()` passes with no net growth or peak change;
- runs every family as a batch (`blend_batch_ws`, `skip_batch_ws`,
  `opfl_batch_ws`, `refine_search_batch_ws`, `refine_predict_batch_ws`) with
  descriptor arrays in the workspace;
- exercises `wedge_cache_build` on the device.

Inter-intra note: the blend consumes the inter prediction already in `dst`.
The test keeps the pre-launch contents in a separate `inter8`/`inter16` buffer
and passes a copy of that to the CPU/reference oracle; the downloaded GPU
output is never reused as the oracle input.

Skip note: the skip kernel writes only the block rectangle into the caller's
planes. The test zero-fills the three destination planes before each job (and
the shared batch pool) so bytes outside the block match the zero-initialised
CPU/reference planes. On the first mismatch it prints `plane/index/row/col/
gpu/cpu/ref0/ref1/in_block/block` for both 8-bit and 10-bit.

Remote command (as in the repo README; `/workspace` is not persistent):

```sh
# from the local repo
AV2ENC_HOST='<user>@<host>' AV2ENC_PORT='<port>' tools/sync_remote.sh

ssh -p <port> <user>@<host> \
  'cd /workspace/av2enc && ./src/cuda/compound_full/build_and_test.sh'
```

(No `AV2ENC_HOST` was configured in the local environment used to write this
module, and the `colab` SSH alias no longer resolves, so the on-device run is
the command above for the GPU host.)

## Limitations

- Reference scaling is handled by the shared §7.13.3.17 scaling helper for
  refine-MV; compound/skip still assume unscaled references, which is the M2
  baseline.
- OPFL/refine-MV CPU references are 8-bit only; the 10-bit paths are
  spec-generalised and checked against the independent/emulated reference as
  described above.
- One CUDA block per job is intentional for the component test. A production
  pipeline can tile a superblock into one launch; the per-sample bodies are
  already written for a thread stride.
