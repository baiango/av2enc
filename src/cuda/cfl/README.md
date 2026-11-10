# CfL CUDA kernel (AV2 §7.13.5)

CUDA port of the CPU reference in `src/encoder/cfl/`. The kernel includes
`cfl_common.h`, so both paths use the same integer helpers and the same POD
block descriptor. There is no floating point in CfL, so bit-exactness reduces
to matching the luma-average collection, the edge replication, and the
derived-alpha correlation.

## API

```cpp
#include "cuda/cfl/cfl.h"

// block.chroma holds the DC prediction on input, CfL prediction on output.
av2enc::cuda::cfl_predict(block, alpha_q8);            // explicit
int* d_alpha = nullptr;   // cudaMalloc(&d_alpha, sizeof(int))
av2enc::cuda::cfl_predict_derived(block, d_alpha);     // CFL_DERIVED_ALPHA

// Batched forms: one CUDA block per job. Device-resident descriptors.
av2enc::cuda::cfl_predict_batch(d_jobs, job_count);
av2enc::cuda::cfl_predict_derived_batch(d_blocks, d_alpha_out, job_count);
```

All entry points are asynchronous on `stream` (default stream when null), do
no allocation, and do not synchronize. `alpha_q8` is
`CflAlpha * 32` for explicit signalling; derived alpha is computed in the
kernel from the reconstructed chroma above/left neighbours.

## Mapping

* One CUDA block (128 threads) per chroma transform block.
* Thread 0 collects `lumaAvg`, `lumaAbove[]`, `lumaLeft[]` and (derived mode)
  `implicitAlpha`; `w + h <= 128` taps.
* All threads then evaluate `pred = Clip1(dc + Round2Signed(alpha*(L-lumaAvg),
  11))`, with `L` computed inline (9 taps) so no per-block scratch buffer is
  allocated.
* Shared memory: 2 × 64 ints + 2 scalars (520 B). No `cudaMalloc`, no arenas.
* 8-bit 4:2:0 is the supported encoder format. `sub_x`/`sub_y` are still read
  from the descriptor so a future 4:2:2/4:4:4 port does not need a new ABI.

## Build / run

On the Linux + CUDA box (no `nvcc` on the macOS host):

```sh
./src/cuda/cfl/build_cfl_test.sh                 # sm_86, build + run
ARCH=sm_120 ./src/cuda/cfl/build_cfl_test.sh     # secondary target
./src/cuda/cfl/build_cfl_test.sh --no-run
```

The script compiles `cfl.cu`, `cfl_test.cu`, and the CPU reference
`../../encoder/cfl/cfl.cpp` into one binary, so the test compares the actual
CPU oracle with the kernel.

## Test coverage

`cfl_test.cu` asserts byte equality (`0` mismatches) for:

* explicit alpha `{+8, -5, 0} × 8` eighths, block sizes 4/8/16/32, positions
  covering top-left, interior, superblock-top, and frame edges;
* `cfl_ds_filter_index` 0, 1, 2, and 3 (3 behaves as 0);
* partial right/bottom blocks where the coded grid is smaller than the coded
  block (luma-average replication + padded-plane reads);
* 300 randomised geometries with arbitrary Q8 alpha in [-511, 511];
* derived alpha, including the computed `implicitAlpha` itself, compared
  against `cfl::derive_implicit_alpha`;
* batch launches of 96 jobs for both explicit and derived modes, including
  the per-job `alpha_out`.

Status: verified on the RTX 3060 (`sm_86`, CUDA 12.8):

```
cfl_cuda_test: 1022200 samples, 412 derived alphas, all exact
compute-sanitizer memcheck: 0 errors
compute-sanitizer racecheck: 0 hazards
```

The first box run reported `MISMATCH [derived_alpha] expected 58 got 0`. That
was an API/writeback bug, not arithmetic: `cfl_predict_derived()` had no alpha
output parameter, so `cfl_test.cu` read an uninitialised `cudaMalloc` int
(which happened to be 0) while the kernel used the correct alpha internally
for the prediction (prediction samples already matched). The single-block
entry point now takes `int* d_alpha_out`; the batch entry point already used
`alpha_out + blockIdx.x`. The CPU reference is normative and unchanged.

## Integration notes

* The caller must pass the padded-64 reconstructed luma bounds and keep luma
  reconstruction for the full coded block (including padding), exactly as the
  CPU reference documents.
* `have_above` / `have_left` are `AvailUChroma` / `AvailLChroma` at chroma-mi
  granularity. Do not derive them from `y != 0`.
* For batched RDO, descriptors must not alias the same chroma plane within one
  launch; the kernel writes `chroma` in place.
