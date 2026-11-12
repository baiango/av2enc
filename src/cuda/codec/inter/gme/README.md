# CUDA global motion estimation (AV2 v1.0.0 §5.18.9, §7.12.2)

GPU port of the encoder-side global-motion component in
`src/encoder/inter/gme/`.  The module is self-contained: it adds no CMake
target and is not wired into M1/M2 yet.

## Scope

| Item | CUDA entry point |
|---|---|
| Correspondence extraction from block MVs | `estimate_blocks()`, `gme_extract_point_from_block()` |
| Correspondence extraction from per-block warps (centre + four radius offsets) | `estimate_warps()`, `gme_extract_points_from_warp()` |
| Direct point correspondences | `estimate_points()` |
| Deterministic integer least squares: TRANSLATION, ROTZOOM, AFFINE | `gme_fit_core()` |
| Median outlier rejection (up to `outlier_passes`, floor threshold) | `gme_kth_smallest()`, `gme_estimate_points_core()` |
| §5.18.9 quantisation grid and forced-integer-MV mode | `gme_quantize_model()` |
| AFFINE → ROTZOOM → IDENTITY degradation, scale/determinant/corner checks | `gme_finalize_gm_params()` |
| §7.13.3.21 shear validity, §7.13.3.23 model reduction | `gme_setup_shear_valid()`, `gme_reduce_warp_model_in_place()` |
| §7.12.2.2 projection, MV clamp, precision lowering | `gme_get_warp_motion_vector()` |
| `GLOBALMV` usability / warp selection | `gme_global_mv_for_block()` |
| Bilinear warp quality proxy (not the normative predictor) | `gme_warp_plane_bilinear()`, `gme_warp_plane_sse()` |
| Weighted quality / gain diagnostics | `gme_evaluate_quality()` |

There is no bit-depth dependency: the estimator consumes integer q8
correspondences and block MVs.  The image proxy is 8-bit, matching the CPU
helper.  Projection supports every block size (4×4 through 64×64, and larger)
and the full `MV_LOW..MV_UPP` / border-clamped MV range.

## Files

```text
src/cuda/gme/
  gme_device.cuh        shared host/device integer core (portable 128-bit,
                        fitting, outlier rejection, quantisation,
                        degradation, projection, shear, quality proxy)
  gme.h                 public CUDA API and arena contract
  gme_plan.cpp          host sizing / allocate_checked / mark/rewind /
                        stage / download (no kernels)
  gme.cu                extraction, fit, warp-plane and SSE kernels
  gme_emulation_test.cpp  no-nvcc host parity test
  gme_test.cu           real-GPU parity test
  build_and_test.sh     host-emulation build + run
  build_gpu_test.sh     real-GPU build + run
```

The public type names (`GmeParams`, `GmePoint`, `GmeBlockMv`,
`GmeWarpSample`, `GmeOptions`, `GmeEstimate`, `GmeDiag`) are POD mirrors of
the CPU `av2enc::inter::gme` types.  Diagnostics carry a reason *code* instead
of the CPU's `std::string`; `gme_reason_name()` returns the same text.

## API sketch

```cpp
#include "cuda/gme/gme.h"

av2enc::cuda::gme::GmeConfig config;
config.max_points = 8192;   // points / expanded warp points
config.max_blocks = 8192;   // block-MV descriptors per job
config.max_warps  = 2048;   // warp descriptors per job (5 points each)
config.max_jobs   = 2;      // independent references

const std::uint64_t need =
    av2enc::cuda::gme::required_device_bytes(config);

av2enc::cuda::Arena arena;
arena.init(need);                       // one arena for the whole pipeline
av2enc::cuda::gme::GmeBuffers buffers;
std::uint64_t reported = 0;
std::string error;
av2enc::cuda::gme::allocate_checked(arena, config, &buffers, &reported,
                                    &error);

av2enc::cuda::gme::GmeOptions options =
    av2enc::cuda::gme::gme_default_options();
options.model = av2enc::cuda::gme::kGmeFitAffine;

// Per reference: stage this frame's blocks/MVs, launch, download.
av2enc::cuda::gme::stage_blocks(buffers, /*job=*/0, host_blocks, block_count);
av2enc::cuda::gme::estimate_blocks(buffers, 0, block_count, options);
av2enc::cuda::gme::GmeEstimate estimate;
av2enc::cuda::gme::download_estimate(buffers, 0, &estimate);

if (estimate.diag.accepted && !estimate.diag.identity) {
  av2enc::cuda::gme::GmeGlobalMvUse use =
      av2enc::cuda::gme::gme_global_mv_for_block(
          estimate.params, /*ref=*/0, mi_col, mi_row, bw4, bh4,
          mi_cols, mi_rows);
  // use.use_warp selects the block warp; use.mv is the 1/8-pel centre MV.
}
```

`estimate_points()` / `estimate_blocks()` / `estimate_warps()` launch kernels
and do not synchronise.  `download_estimate()` synchronises.  The only copies
in the module are the explicit `stage_*` / `download_*` calls.

## Memory contract (single arena)

Every device byte comes from the single persistent `av2enc::cuda::Arena`.
The component contains no `cudaMalloc`, `cudaFree`, `cudaMallocHost`,
`cudaHostAlloc` or `cudaHostRegister`; the build scripts grep for them.

- `required_device_bytes(config)` is the exact persistent footprint,
  including 256-byte alignment padding:
  `points` + `active` + `residuals` + `keep` + `blocks` + `warps` +
  `results` + `control`, all `max_jobs` copies.
- `allocate_checked()` bump-allocates once.  A short arena returns
  `GmeStatus::kArenaTooSmall`, writes the exact requirement to
  `*required_bytes`, and leaves the bump pointer unchanged.
- `gme_mark()` / `gme_rewind()` expose the watermark for callers that want to
  bracket their own scratch.
- Launchers allocate nothing and never grow the arena.  Two consecutive
  launches leave `arena.used()` and `arena.peak_used()` flat.

Measured `required_device_bytes` (Arena::kAlignment = 256):

| points | blocks | warps | jobs | required bytes |
|---:|---:|---:|---:|---:|
| 8192 | 8192 | 2048 | 1 | 655,872 |
| 8192 | 8192 | 2048 | 2 | 1,311,488 |
| 8192 | 8192 | 2048 | 7 | 4,588,800 |
| 4096 | 4096 | 1024 | 1 | 328,192 |

`max_points` bounds both direct correspondences and the expanded warps
(5 points per valid warp).  If an extraction produces more points than a
caller-provided buffer, the GPU reports the CPU's "too many correspondences"
outcome instead of growing the arena.

## Bit-exact arithmetic

The CPU reference accumulates in `__int128`.  Device code cannot rely on a
128-bit integer type, so `gme_device.cuh` implements `GmeI128` as a
two's-complement `{lo, hi}` pair:

- `add`/`sub`/`neg`/`cmp`/`shl`/arithmetic `shr`;
- 64×64→128 and low-128 128×128 multiply;
- 129-bit shift-subtract division (exact for a full 128-bit denominator);
- the CPU's round-to-nearest `div_round_i128` and exact Q16 `ratio_q16`
  long division.

The median is the CPU's `nth_element(...)[size/2]` order statistic, computed
by binary bit selection over non-negative residuals (`gme_kth_smallest`), so
it is order-independent without a sort.  The outlier filter, refit loop,
finalisation, gain gate and diagnostics follow `gme.cpp` statement by
statement.

The fit runs one job per block with one active thread.  Integer sums are
associative, so a future block-parallel reduction can preserve the result;
the correctness-first single-thread fit keeps the accumulation and
degradation order identical to the shared core today.  Extraction and the
image proxy are data-parallel (one thread per block/warp/sample).

## Host-emulation parity

`./build_and_test.sh` compiles `gme_device.cuh`, `gme_plan.cpp` and the real
`Arena` with the host compiler against a tiny CUDA runtime stub, and compares
against `src/encoder/inter/gme/gme.cpp` (the CPU oracle).  No GPU or nvcc is
needed.

Coverage and result on this tree (`--quick` and full):

| Area | Cases | Result |
|---|---:|---|
| `GmeI128` vs host `__int128` (add/sub/mul/shift/div/ratio, high-bit divisors) | 10,500 iterations | 0 failures |
| §7.12.2.2 projection, all block sizes, precisions 4..7, full MV range | 349,440 + full-range | 0 failures |
| §5.18.9 finalisation / degradation variants | 24,080 | 0 failures |
| Fits: points/blocks/warps, models, noise, outliers, weights, options | 644 | 0 failures |
| Synthetic pan / zoom / rotate / combined | 4 | 0 failures |
| Real 720p pair 0/1, pair 0/10, known warp | 3 | 0 failures |
| Arena / one-byte-short / mark / rewind / staging | — | 0 failures |

```
checks: 1354622, failures: 0        # full run (includes pair 0/10)
checks: 1354598, failures: 0        # --quick
```

Synthetic proxy results match the CPU README exactly:

```text
synthetic pan              blocks=388 used=356 type=ROTZOOM gain=15.36 dB
synthetic zoom             blocks=382 used=243 type=ROTZOOM gain= 8.52 dB
synthetic rotate           blocks=389 used=319 type=ROTZOOM gain=11.33 dB
synthetic pan+zoom+rotate  blocks=384 used=320 type=ROTZOOM gain=13.11 dB
real 720p pair 0/1         blocks=727
real 720p pair 0/10        blocks=738
real 720p known warp       blocks=692
```

The block matcher used by the synthetic/real-frame tests is test-only and
CPU-side, exactly like the CPU reference's matcher; the module consumes the
resulting `GmeBlockMv` descriptors.  Image-matching motion estimation is the
separate `src/cuda/inter/me/` component.

`./build_gpu_test.sh` is the real-GPU test.  It stages the same inputs,
launches the actual kernels and compares the downloaded `GmeEstimate` with
`gme::estimate_points/block_mvs/warps` bit-for-bit on `GmParams` and the
integer diagnostics; floating diagnostics are checked to 1e-6.  It also runs
a device projection batch (all block sizes/precisions/full range), the device
bilinear/SSE proxy, two-job buffers, and the arena no-growth /
one-byte-short checks.  Its own `checks: N, failures: 0` line is the GPU
parity count.

## Device-visibility audit

The host-emulation test cannot see a host-only symbol referenced from device
code, so the shared header was audited separately:

1. Every free function in `gme_device.cuh` is `AV2ENC_GME_HD` (under nvcc:
   `__host__ __device__ inline`).  A scan for unannotated function definitions
   found **0**.
2. There are **0** namespace-scope `constexpr`/`const`/`static` objects.
   Constants are enumerators (`enum : int`) or `constexpr` functions.
3. No `std::` container, algorithm, string or I/O is used in device code;
   `<cstdint>`/`<cstddef>` are the only standard headers, plus
   `<cuda_runtime.h>` under `__CUDACC__` for device math.  The only `std::`
   spellings in the header are type aliases (`std::int32_t`, ...).
4. `gme_sqrt` / `gme_log10` select device math under `__CUDACC__` and the host
   compiler builtin otherwise; all other arithmetic is integer.
5. `gme.cu` and `gme_test.cu` were additionally parsed with a host-only clang
   CUDA front end (mocked `cuda_runtime.h`, no CUDA toolkit on the authoring
   host) to catch syntax/type errors in the kernel and launch code.  This is a
   syntax check, not a substitute for the nvcc device build in
   `build_gpu_test.sh`.

## Remote run

```sh
AV2ENC_HOST='<user>@<host>' AV2ENC_PORT='<port>' tools/sync_remote.sh
ssh -p <port> <user>@<host> \
  'cd /workspace/av2enc/src/cuda/gme && ./build_and_test.sh && ./build_gpu_test.sh'
```

`tools/sync_remote.sh` excludes `testdata/`; copy
`testdata/shine_1280x720_60f.y4m` to the remote box if you want the real-720p
cases (otherwise they print `skipped` and the synthetic cases still run).
Use `./build_gpu_test.sh --quick` to skip pair 0/10, and
`./build_gpu_test.sh --sanitize` for `compute-sanitizer --tool memcheck`.

## Limitations

- The fit is correctness-first single-thread per job.  It is a frame-level
  component; extraction and the image proxy are parallel.
- No M2 integration, no CMake edits, no bitstream change.
- The acceptance gate is the CPU correspondence-SSE gain heuristic, not a
  rate-aware RDO comparison.
- The bilinear warp/SSE proxy is an encoder-side quality measure, not the
  normative §7.13.3.19 predictor.
- `max_points` must cover the expanded warp count; an under-sized staging
  capacity is reported as "too many correspondences" rather than grown.
