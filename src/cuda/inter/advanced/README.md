# Advanced inter — CUDA kernels

CUDA side of workstream 13 (`src/encoder/inter/advanced/`). Product target is
Linux + CUDA `sm_86` (8 GB Ampere, `compat8g`); this directory is component code
only, with no integration into `src/cuda/m0_pipeline.*` or the root CMake build.

Bit-exactness contract: the kernels call the same `AV2ENC_ADV_HD` inline helpers
as the CPU reference. There is no second transcription of the arithmetic, no
floating point, and no tolerance in the tests.

## Kernels

| Kernel | Entry point | Work decomposition | Shared memory |
|---|---|---|---|
| Compound / inter-intra | `compound_batch`, `compound_one` | 1 block per job, threads stride over samples | none |
| Block warp (§7.13.3.19) | `warp_batch` | 1 block per job, 8x8 sections | 15×8 `int32` = 480 B |
| Extended warp (§7.13.3.20) | `ext_warp_batch` | 1 block per job, 4x4 sections | 64×4 `int32` = 1 KiB |
| BAWP (§7.13.3.25) | `bawp_batch` | 1 block per job, thread 0 fits, then parallel apply | one `BawpFit` |
| OBMC experiment | `obmc_batch` | 1 block per job, threads stride over samples | none |

Reference-list selection (§7.7/§7.8) and GOP scheduling stay on the host: they run
once per frame, not per block, so a kernel would only add launch latency.

## Job structs

All job structs are small, trivially copyable structs of PODs (`WarpJob`,
`BawpGpuJob`, `ObmcGpuJob`, and `inter::adv::CompoundJob`). Pointers inside them
must be **device** pointers. Batch entry points take a device array of jobs and
launch one block per element; `job_count == 0` is a no-op. There is no allocation
and no host round trip inside a launch, so the encoder can point the structs at
`src/cuda/arena.*` buffers.

Warp jobs write `int16_t` predictions:

* `compound == 0` → clipped integer samples (`0..255`), as in §7.13.3.14's
  `Clip1(Preds[0][i][j])`;
* `compound == 1` → 1/16 sample precision (`sample << 4`), to be fed to
  `compound_batch` as `Preds[]`.

## Build and test

```sh
./build_advanced_test.sh                 # sm_86, build + run
./build_advanced_test.sh --no-run        # build only
ARCH=sm_120 ./build_advanced_test.sh     # secondary target
```

The script compiles `advanced.cu`, `advanced_test.cu` and the CPU reference
(`../../../encoder/inter/advanced/*.cpp`) with
`-Xcompiler -Wall,-Wextra,-Wconversion,-Wshadow`. It refuses to run without
`nvcc`; the CPU reference builds on macOS via
`../../../encoder/inter/advanced/build_and_test.sh`.

Expected output on a 3060:

```
device: NVIDIA GeForce RTX 3060 (cc 8.6, 28 SMs)
advanced_cuda_test: 1199 checks passed
timing: compound wedge 64x64 (200 blocks/sample pairs) ... ms total, ... us/block, ... Gsample/s
```

## What the test covers

* Compound: average (4 CWP weights, including the negative same-side weight),
  implicit mask, wedge (3 indices × 2 signs), diffwtd (2 mask types), across
  8x8/16x16/32x32/64x64/8x32/32x8/4x4/16x8 — output **and** mask arrays compared
  byte for byte.
* Inter-intra: 4 modes and wedge-inter-intra.
* Chroma compound: a luma wedge job produces the mask on the device, then a
  chroma job consumes it (the mask-subsampling path).
* Chroma warp: plane 1 with 4:2:0 subsampling, block and extended paths, both
  precision modes.
* Block warp: identity, integer translation and a random affine model, in both
  single-reference and compound precision.
* Extended warp: identity, translation, invalid shear, random affine models, and
  a half-size (scaled) reference; both precision modes.
* BAWP: 24 random jobs including chroma (where `BawpAlpha` comes from the caller),
  random `AvailU/AvailL`, explicit and fitted alpha; output and `BawpFit`
  (`alpha`, `beta`, `count`) compared.
* OBMC: overlap 0/2/4/8.
* Batch path: 8 jobs in one launch against the CPU reference.

The `timing:` line is a `cudaEvent`-timed loop of 200 single-job launches. It is
printed for convenience; no throughput claim is made here.

## Verification without a GPU (development host)

The development Mac has no CUDA device. Two checks were run locally:

1. **Host syntax/type check.** Both `.cu` files compile as C++20 with the CUDA
   qualifiers and launches stubbed out (`__global__`/`__device__`/`__shared__`
   empty, `<<<...>>>` removed). This catches API/typing mistakes but proves
   nothing about nvcc codegen.
2. **Single-thread kernel emulation.** The launches were rewritten to call each
   kernel with `blockDim = 1`, and a stub `cuda_runtime.h` mapped device memory
   to host memory. The full test then ran on the host and reproduced the CPU
   reference exactly (1199 checks). Because it runs one thread, it validates the
   arithmetic and the section/loop structure, not the shared-memory
   parallelisation. The barriers are reviewed by hand: every `__syncthreads()`
   is reached uniformly (the `continue` guards depend only on block-uniform job
   data), the intermediate buffer is written before the barrier and read after,
   and the next section's barrier happens before the buffer is reused.

Neither check is a substitute for the real gate. **Before relying on these
kernels**: build on the Linux box, run `build_advanced_test.sh`, then wire them
behind the CPU reference and compare reconstructed frames.

## Rules followed

* No `cudaMalloc` in the component; the test allocates, the encoder will use
  `src/cuda/arena.*`.
* No Blackwell-only instructions, no TMA/clusters/`wgmma`/`tcgen05`, no FP4.
* Integer-only arithmetic; no `-fmad` sensitivity to worry about.
* One launch per job batch, no synchronization inside the sample loops other than
  the shared-memory barriers described above.
* Fixed `sm_86` arch in the component script; the fat binary is the integration
  agent's concern.
