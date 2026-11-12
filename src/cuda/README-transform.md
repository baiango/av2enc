# CUDA transform / quantization (`src/cuda/transform.*`)

Bit-exact CUDA port of the M0 transform arithmetic in `src/encoder/basic_encoder.cpp`
(`TileEncoder::forward_dct`, `TileEncoder::inverse_dct`, and the
transform/quantization block of `encode_residual_plane`).  The CPU path is the
reference and was not modified.

Files:

| File | Contents |
|---|---|
| `transform.h` | API (`TxParams`, stage kernels, 64x64 edge helpers) |
| `transform.cu` | kernels + host launchers; read-only DCT kernel constants |
| `transform_test.cu` | CPU reference copy + bit-exact comparison, edge cases, timing |
| `build_transform_test.sh` | `nvcc -arch=sm_86 -lineinfo -fmad=false` component build |
| `README-transform.md` | this file |

## Design / purity

Every entry point is a pure host function: device pointers + `cudaStream_t` in,
`cudaError_t` out.  Nothing is allocated, freed, or synchronized inside; there is
no mutable global state.  One CUDA kernel is launched per call (the composed
entry points launch two), and `nblocks` blocks are processed back-to-back in one
launch so per-block launch latency is amortized.  All launches go to the caller's
stream; the kernels never call `cudaDeviceSynchronize()`.

The only module-level data are the four `Dct_Kernel{n}` tables, duplicated as
`__device__ const` literals from `src/encoder/basic_tables.h` so the kernels are
self-contained.  They are immutable; the test verifies them implicitly (impulse
residuals exercise every table entry against the CPU reference).

## API

```cpp
namespace av2enc::cuda {

struct TxParams {
  int w, h;          // adjusted transform size (32x32 for a 64x64 luma block)
  int dq_denom;      // 1 << ((tx_pels > 256) + (tx_pels > 1024))
  int row_shift;
  int col_shift;
  double scale_w;    // 1 / (64 * sqrt(w))
  double scale_h;    // 1 / (64 * sqrt(h))
  double quant_factor;
};

std::optional<TxParams> make_tx_params(int coded_w, int coded_h, bool luma);
bool tx_size_supported(int w, int h);

cudaError_t forward_dct(const std::int16_t* d_residual, double* d_coeff,
                        const TxParams& p, int nblocks, cudaStream_t stream);
cudaError_t quantize(const double* d_coeff, std::int32_t* d_levels,
                     const TxParams& p, int nblocks, cudaStream_t stream);
cudaError_t dequantize(const std::int32_t* d_levels, std::int32_t* d_dequant,
                       const TxParams& p, int nblocks, cudaStream_t stream);
cudaError_t inverse_dct(const std::int32_t* d_dequant,
                        std::int16_t* d_residual, const TxParams& p,
                        int nblocks, cudaStream_t stream);

cudaError_t forward_transform(const std::int16_t* d_residual,
                              double* d_coeff_scratch,
                              std::int32_t* d_levels, const TxParams& p,
                              int nblocks, cudaStream_t stream);
cudaError_t inverse_transform(const std::int32_t* d_levels,
                              std::int32_t* d_dequant_scratch,
                              std::int16_t* d_residual, const TxParams& p,
                              int nblocks, cudaStream_t stream);

cudaError_t make_residual_64x64(const std::uint8_t* d_src, int src_stride,
                                int src_width, int src_height, int x0, int y0,
                                int pred, std::int16_t* d_residual,
                                cudaStream_t stream);
cudaError_t downsample_residual_64x64(const std::int16_t* d_residual64,
                                      std::int16_t* d_residual32,
                                      cudaStream_t stream);
cudaError_t reconstruct_64x64(const std::int16_t* d_residual, int pred,
                              std::uint8_t* d_dst, int dst_stride,
                              int dst_width, int dst_height, int x0, int y0,
                              cudaStream_t stream);
}
```

Conventions:

- Buffers are tightly packed, row-major, `b * w * h + y * w + x`.
- Residual values fit `int16_t`; in M0 they are source minus prediction in
  `[-255, 255]` (the 64x64 path averages four such values).
- `make_tx_params(coded_w, coded_h, luma)` mirrors the encoder: it applies the
  `64x64 luma -> adjusted 32x32, upscale 2x2` rule and derives `dq_denom` from
  `(upscale_x * tw) * (upscale_y * th)`.  Pass the **plane block dimensions**
  (chroma of a 64x64 block is 32x32, not 64x64).  Returns `nullopt` for sizes
  the CPU reference cannot transform (see Blockers).
- `forward_transform` / `inverse_transform` do not allocate; the caller supplies
  the intermediate scratch (`double` or `int32`, `nblocks*w*h` elements).
- Return value is `cudaErrorInvalidValue` for bad parameters and
  `cudaGetLastError()` after a valid launch.  No hidden synchronization.
- `reconstruct_64x64` skips writes outside `[0, dst_width) x [0, dst_height)`;
  pass the padded plane (64-aligned) to reproduce the encoder's padding writes.

## Arithmetic fidelity

The CPU reference computes with:

- forward DCT: two passes, `double` accumulation, `sum += kernel * sample`,
  then `sum * scale`; `Dct_Kernel{n}[j][i] = 64*sqrt(n)*B[j][i]`
- quantization: `(int)std::lround(coeff * quant_factor)` with
  `quant_factor = 8*dq_denom / (inv_scale*528)` and
  `inv_scale = (64*sqrt(w)/2^row_shift) * (64*sqrt(h)/2^col_shift)`
- dequantization: `(abs(q)*528) & 0xFFFFFF`, `Round2(., 3)`, `/ dq_denom`, sign
- inverse DCT: normative integer transform, `int64` accumulation,
  `Round2` with arithmetic shift, clip to `[-32768, 32767]` then `[-256, 255]`

The kernels replicate the exact same loop order per output coefficient.  Device
FMA contraction is prevented twice: the component build passes `-fmad=false`, and
the inner products use `__dmul_rn()` / `__dadd_rn()` explicitly (note: the
`__fmul_rn` / `__fadd_rn` intrinsics are the **float** versions and silently
down-convert `double`, so they must not be used here).  `std::sqrt` and
`quant_factor` are evaluated on the host in `make_tx_params()` with the same
expression as the CPU, so no device transcendental is involved.  The host CPU
reference is compiled with `-ffp-contract=off`.

Result: coefficients, levels, dequantized values and reconstructed residuals are
bit-identical, not just close.

## Test coverage

`transform_test.cu` contains a verbatim copy of the CPU reference functions
(`clip3`, `round2`, `transform_shifts`, `forward_dct`, `inverse_dct`,
quantize/dequantize, `source_sample`, 64x64 residual/reconstruction).  It checks:

- `TxParams` against the CPU expression for 16 block types (bitwise
  `quant_factor`), and rejection of the unsupported 64x32/32x64.
- Random and structured residuals for 11 transform sizes:
  32x32, 16x16, 32x16, 16x8, 32x8, 16x4, 16x32, 8x16, 8x8, 4x4, 4x16.
  Structured cases: zero, constants ±255/±1, ramps, checkerboard, and an impulse
  at every position (basis-vector coverage of all DCT table entries).
  Per size: 256 random blocks + all structured blocks (e.g. 1288 for 32x32).
- `quantize`/`dequantize` including `std::lround` ties
  (`k+0.5`, ±0.49999999999999994, ±0.50000000000000011, 200k random values) and
  dequantization of extreme levels (up to ±1e6, `dq_denom` 1/2/4).
- 64x64 luma edge path: two source planes (128x128 and 100x70, the latter with
  clipped right/bottom source reads), five block positions including clipped
  edges, five predictions (0/37/128/200/255).  Compares the 2x2-averaged residual
  from source and from a full 64x64 residual, coefficients, levels, dequant,
  inverse residual, padded-plane reconstruction and clipped visible
  reconstruction.
- Timing for 32x32 and 16x16 pipelines, single-block latency, and the 64x64
  edge helper.

The test fails (exit 1) on any mismatch and reports the mismatch count, the max
absolute difference, and the first differing element.

## Build and run

On the RTX 3060 box (CUDA 12.8):

```sh
ssh -p <port> <user>@<host>
cd /root/av2enc/src/cuda
./build_transform_test.sh
./transform_test
```

`CUDA_ARCH=sm_120 ./build_transform_test.sh` builds for the secondary target.
The script uses `nvcc -std=c++17 -arch=sm_86 -O3 -lineinfo -fmad=false`,
`-Xcompiler=-ffp-contract=off`, and `-Xptxas=-v` (resource usage).  `-Wpedantic`
is omitted because nvcc's generated stub files trip it with false positives; the
project warning set otherwise applies.

## Exact test output (RTX 3060, sm_86, CUDA 12.8, driver 12.8)

```text
av2enc CUDA transform test
device: NVIDIA GeForce RTX 3060 (sm_86, 28 SMs, 11.6 GiB)
CUDA runtime 12.8, driver 12.8
[params]
  16 block types checked, 3 rejected as expected
[32x32] blocks=1288  coeff[n=1318912 mismatch=0 max_abs=0]  levels[n=1318912 mismatch=0 max_abs=0]  dequant[n=1318912 mismatch=0 max_abs=0]  resid[n=1318912 mismatch=0 max_abs=0]  fused[n=1318912 mismatch=0 max_abs=0]  ifused[n=1318912 mismatch=0 max_abs=0]
[16x16] blocks=520  coeff[n=133120 mismatch=0 max_abs=0]  levels[n=133120 mismatch=0 max_abs=0]  dequant[n=133120 mismatch=0 max_abs=0]  resid[n=133120 mismatch=0 max_abs=0]  fused[n=133120 mismatch=0 max_abs=0]  ifused[n=133120 mismatch=0 max_abs=0]
[32x16] blocks=776  coeff[n=397312 mismatch=0 max_abs=0]  levels[n=397312 mismatch=0 max_abs=0]  dequant[n=397312 mismatch=0 max_abs=0]  resid[n=397312 mismatch=0 max_abs=0]  fused[n=397312 mismatch=0 max_abs=0]  ifused[n=397312 mismatch=0 max_abs=0]
[16x8 ] blocks=392  coeff[n=50176 mismatch=0 max_abs=0]  levels[n=50176 mismatch=0 max_abs=0]  dequant[n=50176 mismatch=0 max_abs=0]  resid[n=50176 mismatch=0 max_abs=0]  fused[n=50176 mismatch=0 max_abs=0]  ifused[n=50176 mismatch=0 max_abs=0]
[32x8 ] blocks=520  coeff[n=133120 mismatch=0 max_abs=0]  levels[n=133120 mismatch=0 max_abs=0]  dequant[n=133120 mismatch=0 max_abs=0]  resid[n=133120 mismatch=0 max_abs=0]  fused[n=133120 mismatch=0 max_abs=0]  ifused[n=133120 mismatch=0 max_abs=0]
[16x4 ] blocks=328  coeff[n=20992 mismatch=0 max_abs=0]  levels[n=20992 mismatch=0 max_abs=0]  dequant[n=20992 mismatch=0 max_abs=0]  resid[n=20992 mismatch=0 max_abs=0]  fused[n=20992 mismatch=0 max_abs=0]  ifused[n=20992 mismatch=0 max_abs=0]
[16x32] blocks=776  coeff[n=397312 mismatch=0 max_abs=0]  levels[n=397312 mismatch=0 max_abs=0]  dequant[n=397312 mismatch=0 max_abs=0]  resid[n=397312 mismatch=0 max_abs=0]  fused[n=397312 mismatch=0 max_abs=0]  ifused[n=397312 mismatch=0 max_abs=0]
[ 8x16] blocks=392  coeff[n=50176 mismatch=0 max_abs=0]  levels[n=50176 mismatch=0 max_abs=0]  dequant[n=50176 mismatch=0 max_abs=0]  resid[n=50176 mismatch=0 max_abs=0]  fused[n=50176 mismatch=0 max_abs=0]  ifused[n=50176 mismatch=0 max_abs=0]
[ 8x8 ] blocks=328  coeff[n=20992 mismatch=0 max_abs=0]  levels[n=20992 mismatch=0 max_abs=0]  dequant[n=20992 mismatch=0 max_abs=0]  resid[n=20992 mismatch=0 max_abs=0]  fused[n=20992 mismatch=0 max_abs=0]  ifused[n=20992 mismatch=0 max_abs=0]
[ 4x4 ] blocks=280  coeff[n=4480 mismatch=0 max_abs=0]  levels[n=4480 mismatch=0 max_abs=0]  dequant[n=4480 mismatch=0 max_abs=0]  resid[n=4480 mismatch=0 max_abs=0]  fused[n=4480 mismatch=0 max_abs=0]  ifused[n=4480 mismatch=0 max_abs=0]
[ 4x16] blocks=328  coeff[n=20992 mismatch=0 max_abs=0]  levels[n=20992 mismatch=0 max_abs=0]  dequant[n=20992 mismatch=0 max_abs=0]  resid[n=20992 mismatch=0 max_abs=0]  fused[n=20992 mismatch=0 max_abs=0]  ifused[n=20992 mismatch=0 max_abs=0]
[lround]
  qf=1[n=220160 mismatch=0 max_abs=0]  qf=real[n=220160 mismatch=0 max_abs=0]
[dequant]
  levels[n=10016 mismatch=0 max_abs=0]  denom=2[n=10016 mismatch=0 max_abs=0]  denom=4[n=10016 mismatch=0 max_abs=0]
[64x64 edge]
 cases=50  build[n=51200 mismatch=0 max_abs=0]  ds[n=51200 mismatch=0 max_abs=0]  coeff[n=51200 mismatch=0 max_abs=0]  levels[n=51200 mismatch=0 max_abs=0]  dequant[n=51200 mismatch=0 max_abs=0]  inv[n=51200 mismatch=0 max_abs=0]  recon[n=819200 mismatch=0 max_abs=0]  vis[n=116920 mismatch=0 max_abs=0]

[timing]
  32x32 8192 blocks: fwd(dct+quant)=45.103 ms (5.506 us/block) | inv(dequant+idct)=4.035 ms (0.493 us/block) | both=49.362 ms (6.026 us/block, 0.2 Mblocks/s, 0.17 Gpel/s)
  16x16 32768 blocks: fwd(dct+quant)=22.550 ms (0.688 us/block) | inv(dequant+idct)=1.921 ms (0.059 us/block) | both=24.488 ms (0.747 us/block, 1.3 Mblocks/s, 0.34 Gpel/s)
  1x 32x32 pipeline: 265.724 us/iteration (2000 iterations)
  64x64 edge residual: 3.998 us/block (2000 iterations)

7723 checks, 0 failures
RESULT: PASS
```

Additional validation on the same box:

- `CUDA_LAUNCH_BLOCKING=1 ./transform_test` → `7723 checks, 0 failures`,
  `RESULT: PASS`.
- `compute-sanitizer --tool memcheck --launch-count 500 ./transform_test` →
  `ERROR SUMMARY: 0 errors` (all correctness launches checked).

## Timing

Stable reruns of the same binary (the box is shared, so numbers move ±10–30%):

| Case | Forward (DCT+quant) | Inverse (dequant+IDCT) | Both |
|---|---:|---:|---:|
| 32x32, 8192 blocks | 45.0–45.4 ms (5.49–5.54 µs/block) | 4.03–4.07 ms (0.49 µs/block) | 49.1–49.5 ms (6.0 µs/block) |
| 16x16, 32768 blocks | 22.3–22.6 ms (0.68–0.69 µs/block) | 1.92–1.96 ms (0.059–0.060 µs/block) | 24.4–24.8 ms (0.75 µs/block) |

Derived 1080p estimate (2040 luma 32x32 + 2040 chroma 16x16 blocks per frame):
`2040*6.0 µs + 2040*0.75 µs ≈ 13.8 ms/frame` for forward+quant+dequant+inverse,
about 1.4% of the 1 fps budget.  The forward DCT is the bottleneck: it is FP64
(1/64 rate on GA106) and accumulates serially per coefficient to stay bit-exact.
The integer inverse transform is ~10x cheaper.

Single-block latency (4 launches per iteration) is 170–275 µs, i.e. launch
dispatch dominates in this container; batch blocks per call and never do a host
round trip per block.

Kernel resources (ptxas, 32x32): forward 42 registers / 4096 B static shared +
8192 B dynamic; inverse 40 registers / 4096 B static + 4096 B dynamic.

## Blockers / notes for integration

1. **64x32 and 32x64 transforms are rejected.**  The CPU reference only adjusts
   `width == 64 && height == 64`; for 64x32/32x64 it calls `dct_kernel(64)`,
   which falls through to the 32-point table while the loops still run to 64,
   i.e. out-of-bounds reads.  Those sizes occur only in the diagnostic partition
   modes `BasicPartitionMode::k64x64VertNone` / `k64x64HorzNone`, not in the default
   `k32x32` mode or its edge path (which uses a full 64x64 block).  Fix the CPU
   reference first if those modes must be bit-exact.
2. **Chroma of a 64x64 block is 32x32** and must be passed as `(32, 32, false)`;
   `make_tx_params(64, 64, false)` is (correctly) rejected because there is no
   chroma 64x64 transform.
3. `forward_transform` / `inverse_transform` need caller-owned scratch; the
   transform layer never allocates (8 GB arena rule).
4. The 64x64 residual builder and reconstructor are single-block launches; at
   ~30 µs dispatch each, batch or fuse them in the integration pipeline rather
   than calling them per 64x64 superblock.
5. Do not recompute `quant_factor`, `scale_w`, `scale_h` or the shifts with a
   different expression: use `make_tx_params()` (or copy the exact CPU
   expression) or bit-exactness is lost even though `-fmad=false` is set.
6. The tests use host `cudaMalloc`/`memcpy` for harness convenience only; the
   kernels themselves take raw device pointers.
