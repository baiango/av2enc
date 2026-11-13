# CUDA chroma-format support (`src/cuda/chroma_formats/`)

Status: **GPU pixel path for 4:2:2, 4:4:4 and 4:0:0 at 8 and 10 bit.**
The 8-bit product still path now consumes it for `--still-gpu --chroma 422`
through `cuda::still_gpu::make_chroma_gpu_executor()`: the CPU §8.2 writer keeps
the syntax/entropy loop and this component runs the per-block prediction,
residual, transform, quantisation, inverse and reconstruction stages.  The
10-bit product formats use the separate `high_bitdepth_gpu` stage.

CPU reference: `src/encoder/format/chroma/` (geometry, Subsampled_Size,
transform shapes, 8-bit arithmetic).  The 10-bit dequant/inverse oracle is
`src/cuda/m1/hbd/hbd_cpu.*`, the project's verified high-bit-depth reference.

Normative clauses:

| Area | Clause |
|---|---|
| Chroma formats / bit depths | §6.4.1 Table 6.2 / Table 6.3 |
| Visible chroma extent | §7.21.2 |
| Coded frame size (`MiCols`/`MiRows`) | §5.18.4.4 |
| ChromaMiSize / Subsampled_Size | §5.20.7.26, §5.20.3.2 |
| Chroma tx size / `txSzCtx` | §5.20.7.25, §5.20.7.27 |
| Adjusted_Tx_Size, Transform_Shift, residual duplication | §9.2, §7.15.4.1, §7.15.4 |
| DC prediction / Div_Lut | §7.13.2.1, §7.13.2.10, §7.13.3.22 |
| Dequant clip / inverse clips | §7.14.3, §7.15.2.1 |
| Monochrome conditionals | §5.4.3, §5.4.5, §5.4.8, §5.20.5.3 (geometry/legality only) |

## What is implemented

1. **Plane geometry** — `make_frame_geometry(width, height, format)` returns
   visible / coded / 64-aligned padded extents for luma and both chroma planes.
   `plane_origin_x/y()` converts a luma MI coordinate to a plane sample origin
   (`(mi >> sub) * 4`).
2. **Chroma residual shapes** — `chroma_residual_size()` is the CUDA twin of
   `chroma::chroma_mi_size()` and the normative `Subsampled_Size` table,
   including its `BLOCK_INVALID` entries (e.g. 4x8 luma in 4:2:2) and the
   Monochrome "no chroma plane" answer.
3. **Transform shapes for the extra sizes** — `make_tx_shape()` builds nominal
   size, `Adjusted_Tx_Size` (`32x64`/`64x64` -> `32x32`), duplication factors,
   `txSzCtx`, `dq_denom`, the *nominal* `Transform_Shift` (`TX_32X64` is
   `(6,12)`, not the adjusted square's `(6,13)`), the odd-rect pre-scale and
   the host-computed binary64 `scale_w`/`scale_h`/`quant_factor`.
4. **Forward/inverse transforms + quant/dequant** — the shared device bodies
   in `chroma_formats_device.cuh`:
   * forward DCT: binary64 with explicit `__dadd_rn`/`__dmul_rn`, same loop
     order as `chroma::forward_dct`;
   * quantisation: `(int)lround(coeff * quant_factor)`;
   * dequantisation: `(abs(q)*528 & 0xFFFFFF) >> 3 / dq_denom` with the
     `±(1 << (7 + BitDepth))` clip;
   * inverse: §7.15.4.1 integer matrix, int64 accumulation, row clip
     `±(1 << (BitDepth + 7))`, column clip `±(1 << BitDepth)`, `Round2(·2896,
     12)` pre-scale for odd rectangles, and §7.15.4 duplication.
5. **Prediction / residual / copy** — `dc_predict_dev()`,
   `make_residual_dev()` (source minus constant prediction, averaged over the
   `upscale_x * upscale_y` cells the decoder duplicates), `reconstruct_block_dev()`
   (`Clip1(pred + residual)` duplicated into the plane) and
   `copy_rect_dev()` (depth-clipping copy).  All formats and both depths use
   the same bodies; only the clip bound changes.
6. **Arena-backed workspace and batch launchers** — `Workspace` owns the
   source, prediction and reconstruction planes plus per-block work arrays;
   `predict`, `make_residuals`, `forward`, `dequantize`, `inverse`,
   `reconstruct` and `copy_planes` stage descriptors in arena scratch, launch,
   synchronise and rewind.

### Residual storage type (forward vs inverse)

The two residual stages use different containers, and the job API follows the
data flow:

| Stage | Value range | Type |
|---|---|---|
| forward residual (source - pred, averaged) | `-(2^BitDepth-1) .. 2^BitDepth-1` | `int16_t` |
| dequantized levels (inverse input) | `±(1 << (7 + BitDepth))` (10-bit: ±131072) | `int32_t` |
| inverse DCT row intermediate | `±(1 << (BitDepth + 7))` (10-bit: ±131072) | `int32_t` |
| inverse residual / reconstruct input | `[-2^BitDepth, 2^BitDepth-1]` (§7.15.2.1 column clip) | `int16_t` |
| levels / dequant | 32-bit | `int32_t` |

The §7.15.4 residual is 1 + BitDepth bits and both verified project inverse
paths (`txv::inverse_block`, `hbd::inverse_block`) emit `int16_t`, so
`InverseJob::residual_adj` and `ReconstructJob::residual_adj` are `int16_t`
and share the workspace buffer with the forward residual.  No int32 residual
buffer is allocated; `required_device_bytes` is unchanged by this decision.

## Memory contract (mandatory)

Every device byte comes from the single persistent
`av2enc::cuda::Arena` (`src/cuda/arena.h`).  The component contains no
`cudaMalloc`, `cudaFree` or `cudaMallocHost`; the build script greps for
call-shaped occurrences and fails if one appears.

* `required_device_bytes(config) = persistent_device_bytes(config) +
  scratch_device_bytes(config)`, every term rounded to `Arena::kAlignment`.
* `Workspace::init()` uses `Arena::allocate_checked()` for every buffer,
  records `Arena::mark()`, and restores it on any failure.
* A short arena returns `ChromaStatus::Code::kArenaTooSmall` with
  `required_bytes` set to the exact requirement; a bad config returns
  `kInvalidArgument` with `required_bytes == 0`.
* Each launcher brackets its descriptor staging with `mark()` / `rewind()`,
  so `arena.used()` is flat across back-to-back launches and `peak_used()`
  does not grow on the second launch.
* `max_blocks` bounds every batch; a larger count returns `kInvalidArgument`.

Persistent layout per configuration:

| Buffer | Elements | Element |
|---|---|---|
| `src[plane]`, `recon[plane]` | `padded_w * padded_h` per present plane | `uint16_t` |
| `pred` | `max_blocks * 4096` | `uint16_t` |
| `residual_adj` | `max_blocks * 1024` | `int16_t` |
| `coeff` | `max_blocks * 1024` | `double` |
| `levels`, `dequant` | `max_blocks * 1024` each | `int32_t` |

Scratch is the largest single job-descriptor array (`max_blocks * sizeof(job)`).
Measured values (64x64, `max_blocks = 16`, identical for 8- and 10-bit because
the sample container is `uint16_t` in both):

| Format | required | persistent | scratch |
|---|---:|---:|---:|
| 4:0:0 | 444,672 | 442,368 | 2,304 |
| 4:2:0 | 452,864 | 450,560 | 2,304 |
| 4:2:2 | 461,056 | 458,752 | 2,304 |
| 4:4:4 | 477,440 | 475,136 | 2,304 |

Larger frames scale with the plane storage only (e.g. 640x360 4:2:2:
2,394,368 bytes; 1920x1080 4:4:4: 25,495,808 bytes with `max_blocks = 16`).
The default `max_blocks = 16` is a workspace choice, not a spec limit; the
test uses `max_blocks = 4` (140,032 bytes for 64x64 4:2:2).

## Device-code audit (host emulation cannot catch this class)

nvcc is not installed on this host, so the pattern that broke three other
modules was audited by inspection and grep:

1. **Helpers called from `__global__`/`__device__`** — every free function in
   `chroma_formats_device.cuh` is declared `__device__` (most
   `__device__ __forceinline__`).  `grep -nE '^(static|inline|constexpr|const|__device__)'
   chroma_formats_device.cuh` lists only `__device__` functions plus one
   `static __device__ const int kDivLutDev[129]`.
2. **Namespace-scope data used in device code** — the only object is
   `kDivLutDev`, declared `static __device__ const`.  The DCT matrices come
   from `encoder/tx_variants/tx_tables_dev.cuh`, whose arrays are
   `__device__ const`.  Three constants that were originally namespace-scope
   `constexpr` (`kDivLutBits`, `kDivLutPrecBits`, `kDivLutNum`) are now
   function-local `constexpr` inside `approx_divide_dev`; the kernel
   `__shared__` arrays use the literal `1024`.  No host `constexpr` scalar or
   array is referenced from device code.
3. **No `std::` containers/algorithms in device code** — device code uses only
   the fixed-width `std::int*_t`/`std::uint*_t` type aliases.  `std::lround`
   appears only in the `#else` branch of `lround_dev`, which is compiled for
   host emulation; the device branch uses `::lround`.
4. **No mutable static globals** — there are none in the component.
   `kKernelThreads` and the job-validation helpers are host-only (`launch_*`
   thunks and `run_batch`, never called from a kernel).

After that audit the host-emulation build was rerun (it compiles the same
device bytes with `#define __device__`), and it passes.

## Parity

`build_and_test.sh` compiles the shared device header with the host compiler
and compares against the real CPU components.  It does not use a GPU and does
not require nvcc.

| Check | Coverage | Emulation checks |
|---|---|---:|
| Geometry | 4 formats x 8 frame sizes (64x64, 640x360, 100x100, 101x99, 65x63, 1x1, 128x64, 63x65), every field of all 3 planes | 1,472 |
| Subsampled_Size | all 29 BLOCK_SIZE entries x 4 formats, plus `is_chroma_offset_for_subsize` | 2,295 |
| Transform shapes | all 25 AV2 TX sizes: adjusted size, duplication, `txSzCtx`, shifts, `dq_denom`, bit-exact `quant_factor` | 2,723 |
| Transforms | all 25 sizes x 8/10-bit: forward coefficients bit-exact, levels, dequant, inverse (chroma module at 8-bit, hbd at 10-bit) | 70,585 |
| Pipelines | 364 cases = 4 formats x 2 depths x every present plane x 4 luma block sizes (64x64/32x32/32x16/32x8) x up to 4 positions; predict -> residual -> forward -> quant -> dequant -> inverse -> reconstruct, full padded plane compared | 1,510,393 |
| Depth-clipping copy | every present plane x 4 formats x 2 depths, out-of-range uint16 inputs, 17x13 (or plane extent) rectangle | 4,420 |
| Arena | required = persistent + scratch, aligned; two-cycle mark/rewind flat; one-byte-short `kArenaTooSmall` with exact size; invalid config | included |
| **Total** | | **1,514,866 checks, PASS** |

The 10-bit pipeline uses `chroma::forward_dct` + `quantize_coefficients`
(bit-depth independent) and `hbd::dequantize` + `hbd::inverse_block` with
`txv::make_tx_params`; at 8 bit the same stages are compared against
`chroma::dequantize_block` + `chroma::inverse_dct`.  This is the honest
mapping because the CPU chroma module is 8-bit-only by design; the 10-bit
clips (§7.14.3, §7.15.2.1) are the only arithmetic difference and are tested
explicitly.

`dequantize_level_dev()` applies the normative §7.14.3 clip at both depths.
`chroma::dequantize_level()` omits it; at 8 bit it is a no-op for every
residual magnitude the 8-bit DCT can produce, and the 25-size 8-bit sweep
pins that (any clipped value would have failed the bit-exact comparison).
The 10-bit sweep does exercise the clip.

The real-GPU test (`chroma_formats_test.cu`) runs the same 364 pipeline cases
through the actual kernels plus a GPU sweep of all 25 transform sizes x 2
depths, and repeats the arena contract with two real launches and a
one-byte-short arena.  Its check count is printed at runtime; it has **not**
been executed on this host (no nvcc, no CUDA device).

## Files

| File | Purpose |
|---|---|
| `chroma_formats.h` | Table 6.2, geometry, Subsampled_Size, `TxShape`, `Config`, `required_device_bytes`, `ChromaStatus`, `Workspace` (init is header-inline) |
| `chroma_formats.cpp` | Host twin of `chroma_format.cpp` / `chroma_residual.cpp` shape layer |
| `chroma_formats_device.cuh` | Shared `__device__` pixel bodies (DCT, quant/dequant, predict, residual, reconstruct, copy) |
| `chroma_formats.cu` | Kernels and arena-bracketed batch launchers |
| `chroma_formats_reference.h` | Host-only prediction/residual/reconstruction oracle used by both tests |
| `chroma_formats_emulation_test.cpp` | No-GPU/no-nvcc parity + arena test |
| `chroma_formats_test.cu` | Real-GPU parity + two-launch/no-growth + one-byte-short test |
| `build_and_test.sh` | Host-emulation build + run (`SANITIZE=1` for ASan/UBSan) |
| `build_gpu_test.sh` | nvcc `sm_86` build + run |

## Build and run

```sh
# Host emulation parity (this host, no GPU):
./src/cuda/chroma_formats/build_and_test.sh

# With sanitizers:
SANITIZE=1 ./src/cuda/chroma_formats/build_and_test.sh

# Real GPU (RTX 3060 / sm_86):
./src/cuda/chroma_formats/build_gpu_test.sh
ARCH=sm_120 ./src/cuda/chroma_formats/build_gpu_test.sh

# Remote box (Vast.ai; tar pipe because rsync is not installed there):
AV2ENC_HOST=<user>@<host> AV2ENC_PORT=<port> ./tools/sync_remote.sh
ssh -p <port> <user>@<host> \
  'cd /workspace/av2enc/src/cuda/chroma_formats && ./build_and_test.sh && ./build_gpu_test.sh'
```

## Limitations

* Not wired into M1/M2.  The launchers are the reusable seam (see
  `docs/gpu-coverage.md`, which lists this component as the largest remaining
  GPU-MISSING pixel family).
* The component implements lossy DCT only (`DCT_DCT`); ADST/FLDST and the
  identity types are `src/cuda/tx_variants/`' scope.  CfL/MHCCP predictors and
  chroma filters are separate tools.
* Streams are not written here: entropy/syntax stay CPU-side by design, so
  this module ends at reconstructed samples.
* `Config::max_blocks` is a hard batch bound; a larger count is rejected rather
  than silently growing the arena.
* `config.bit_depth` accepts 8 or 10 only.  AV2 v1.0.0 signals exactly those
  two depths (§6.4.1 Table 6.3); 12-bit is not signalable.
