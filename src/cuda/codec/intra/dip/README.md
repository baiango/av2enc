# AV2 DIP — CUDA predictor

Device implementation of AV2 v1.0.0 data-driven intra prediction
(§7.13.2.1, §7.13.2.3–§7.13.2.6), mirroring the CPU reference in
`src/encoder/dip/`.

## API

```cpp
#include "cuda/dip/dip.h"

av2enc::cuda::dip::DipBlock block;
block.recon = d_recon;          // reconstructed luma plane
block.recon_stride = stride;    // samples
block.x = x; block.y = y; block.w = w; block.h = h;
block.max_x = max_x; block.max_y = max_y;
block.have_left = have_left; block.have_above = have_above;
block.num4_above_right = ...; block.num4_below_left = ...;

av2enc::cuda::dip::DipParams params{};
params.mode = dip_mode;         // 0..5
params.transpose = dip_transpose;

av2enc::cuda::dip::dip_predict(block, params, d_pred, stream);
av2enc::cuda::dip::dip_predict_batch(d_jobs, job_count, stream);
```

`dip_predict` uses one CUDA block (128 threads) per prediction. `dip_predict_batch`
uses one CUDA block per `DipJob`; jobs must be independent and `d_jobs` must be
device memory. No entry point allocates, synchronises, or writes the
reconstructed plane.

Scope matches the M1 GPU pipeline: 8-bit luma, DIP-capable extents only
(`w > 4`, `h > 4`, `w*h >= 128`; 15 sizes). `dip_predict` validates the block and
parameters; `dip_predict_batch` follows the existing intra/MRL convention and
assumes the caller validated the jobs.

## Bit exactness

`dip_device.cuh` is the single device implementation:

1. §7.13.2.1 edge preparation with `numAboveNeeded = w + w/4`,
   `numLeftNeeded = h + h/4`;
2. §7.13.2.4 `dip_avg` over the same samples, 11 features, optional transpose;
3. §7.13.2.5 `(sum + 512) >> 10` with `Clip1` to 8 bits;
4. §7.13.2.6 horizontal then vertical resampling, integer division at the same
   points as the CPU code.

The weights come from the same generated `src/encoder/dip/dip_weights_data.inc`
as the CPU table; the device copy is `kDipWeightsDevice`.

## Tests

```
src/cuda/dip/build_and_test.sh      # GPU test if nvcc exists, else emulation
src/cuda/dip/build_gpu_test.sh      # nvcc + a CUDA device required
src/cuda/dip/build_emulation_test.sh
```

- `dip_test.cu` allocates a random reconstructed plane, runs the CPU reference
  and the CUDA kernel on the same descriptor for every DIP size, all 6 modes,
  both transpose values, all availability combinations, and partial
  right/bottom blocks, then repeats through the batch entry point. It checks
  byte-for-byte equality.
- `dip_emulation_test.cpp` compiles the same `dip_device.cuh` with CUDA keywords
  stubbed and compares it against the CPU reference, including real luma
  blocks from `testdata/shine_256x256_1f.y4m`. It needs no GPU or nvcc.

On the current macOS host there is no `nvcc`, so `build_and_test.sh` runs the
host-emulation parity test only:

```
dip_emulation_test: 3612 cases, 3613 checks, 0 failures
```

`ARCH=sm_86 ./build_gpu_test.sh` on the Linux/CUDA box reports:

```
dip_test: 3252 single cases, 3252 batch cases, 22771 checks, 0 failures
```

The arena workspace in `../workspace/` adds 319,488 CPU-exact samples through
`dip_predict_batch` (all 6 modes, both transpose values, all availability
combinations, seven DIP extents).

## Integration

Add a `use_dip && plane == 0` branch to the chosen-mode predictor at
`src/cuda/m1/still_gpu.cu:1615` and the search predictor at
`src/cuda/m1/search/luma_search.cu:429` (both currently call
`intra_modes::intra_predict_batch`). The M2 backend has the same dispatch point
at `src/cuda/m2/video_backend_cuda.cu:538`. Chroma must stay on the existing
intra path. The CPU encoder decision is in `src/encoder/dip/dip.h`
(`dip::decide`); the GPU module deliberately only provides prediction.

The sequence-header and block-syntax call sites are listed in
`src/encoder/dip/README.md`. A `dav2d` decoder gate is blocked by the hardcoded
`enable_dip = 0` writers and the missing `dip_mode_info` writer in the core, as
documented there.
