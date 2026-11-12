# CUDA transform variants (`src/cuda/tx_variants/`)

Bit-exact CUDA port of `src/encoder/tx_variants/` for the M1 transform variants:
all 25 AV2 coded transform sizes (4x4 … 64x64, square and rectangular) and all
16 `TX_TYPE` values (DCT / ADST / IDTX / FLIPADST in either dimension).

The existing `src/cuda/transform.*` is **not modified**; this is a separate
component with its own test and build script. It is not wired into the encode
path and no bitstream claim is made here.

## Files

| File | Contents |
|---|---|
| `tx_variants.h` | device API (`TxParams` comes from the CPU header) |
| `tx_variants.cu` | kernels + host launchers |
| `tx_variants_test.cu` | CPU-vs-GPU bit-exact tests, impulses, mixed batch, timing |
| `build_tx_variants_test.sh` | `nvcc -arch=sm_86 -fmad=false` component build |
| `README.md` | this file |

## API

```cpp
namespace av2enc::cuda {

using txv::TxParams;

struct TxBlockJob {
  TxParams params;
  int32_t in_offset;   // adjusted-size residual / coeff / levels offset
  int32_t out_offset;  // coded-size residual offset (inverse output)
};

cudaError_t txv_forward_block(const int16_t* d_residual_adj, double* d_coeff,
                              const TxParams&, int nblocks, cudaStream_t);
cudaError_t txv_quantize(const double* d_coeff, int32_t* d_levels,
                         const TxParams&, int nblocks, cudaStream_t);
cudaError_t txv_dequantize(const int32_t* d_levels, int32_t* d_dequant,
                           const TxParams&, int nblocks, cudaStream_t);
cudaError_t txv_inverse_block(const int32_t* d_dequant,
                              int16_t* d_residual_coded, const TxParams&,
                              int nblocks, cudaStream_t);

cudaError_t txv_forward_transform(const int16_t* d_residual_adj,
                                  double* d_coeff_scratch, int32_t* d_levels,
                                  const TxParams&, int nblocks, cudaStream_t);
cudaError_t txv_inverse_transform(const int32_t* d_levels,
                                  int16_t* d_residual_coded, const TxParams&,
                                  int nblocks, cudaStream_t);

// Per-job parameters and offsets for mixed-size candidate batches.
cudaError_t txv_forward_block_multi(const int16_t*, double*,
                                    const TxBlockJob*, int nblocks,
                                    cudaStream_t);
cudaError_t txv_forward_transform_multi(const int16_t*, double*, int32_t*,
                                        const TxBlockJob*, int nblocks,
                                        cudaStream_t);
cudaError_t txv_inverse_block_multi(const int32_t*, int16_t*,
                                    const TxBlockJob*, int nblocks,
                                    cudaStream_t);
cudaError_t txv_inverse_transform_multi(const int32_t* d_levels, int16_t*,
                                        const TxBlockJob*, int nblocks,
                                        cudaStream_t);

cudaError_t txv_downsample_residual(const int16_t* d_coded, int16_t* d_adj,
                                    const TxParams&, int nblocks, cudaStream_t);
cudaError_t txv_upsample_residual(const int16_t* d_adj, int16_t* d_coded,
                                  const TxParams&, int nblocks, cudaStream_t);
}
```

Conventions:

- block b starts at `b*w*h` for adjusted-size inputs/outputs and at
  `b*coded_w*coded_h` for coded-size residuals. `w/h` are the adjusted
  transform dimensions and `coded_w/coded_h` the coded block dimensions.
- `TxParams` is the CPU POD; derive it with `txv::make_tx_params()` on the host
  and pass by value (uniform launch) or upload an array (multi launch).
- `_multi` launches 1024 threads and each block reads one `TxBlockJob` from
  `d_jobs[blockIdx.x]`.  A job carries `TxParams` plus explicit `in_offset`
  (adjusted-size buffers) and `out_offset` (coded-size output), because mixed
  block sizes do not have a uniform stride.  The caller builds these prefix
  offsets when assembling the batch.  `_multi` cannot validate device memory;
  every job's `TxParams` and offsets must be valid.  Mixed sizes/types are
  supported and tested.
- `txv_inverse_transform` needs no dequant scratch: the kernel dequantizes the
  levels once into shared memory, so the fused path is one kernel with no extra
  global allocation.
- All functions are pure: no allocation, no free, no synchronization, no
  mutable global state. Return value is `cudaErrorInvalidValue` for bad
  parameters and `cudaGetLastError()` otherwise.
- Buffers are caller-owned and must fit the arena/VRAM policy; the component
  itself never calls `cudaMalloc`.

## Kernel design

- one CUDA block per transform block, one thread per adjusted coefficient
  (`w*h` threads; 32x32 = 1024)
- row/column kernels staged in `__shared__ int16_t[1024]` each (4 KiB total)
- forward doubles staged in dynamic shared memory (8 KiB max)
- inverse dequantized coefficients staged in `__shared__ int32_t[1024]`
  (4 KiB) plus the integer intermediate in dynamic shared memory (4 KiB max)
- total shared footprint <= 16 KiB, well inside the 48 KiB baseline and the
  7,500 MB VRAM profile (no static allocation at all)
- only standard CUDA cores / shared memory / integer and FP64 arithmetic; no
  tensor cores, TMA, clusters, `wgmma`/`tcgen05`, or FP4
- `-fmad=false` and explicit `__dadd_rn` / `__dmul_rn` (not the float
  `__fadd_rn`/`__fmul_rn`, which silently down-convert doubles)

IDTX dimensions skip the matrix entirely; §7.15.2.1's length-32 rule is applied
by selecting `Dct_Kernel32` for ADST/FDST at `n == 32` (so 32x32 and the
adjusted 64x64 path alias DCT, as on the CPU).

## Bit-exactness

The test links the actual CPU implementation (`tx_variants.cpp`) and compares:

1. `forward_block` coefficients: exact double equality, every size × type,
   random + structured + impulse residuals.
2. `quantize` levels: exact int32 equality (`lround` ties included via the CPU
   fuzz input; device `lround` is the same round-half-away-from-zero).
3. Fused forward: both coefficients and levels.
4. `dequantize`: exact int32 equality.
5. `inverse_block` and fused `inverse_transform`: exact int16 equality, including
   the coded-size duplication for 64x64/64x32/… and the §7.15.4.1 2896
   pre-scale for odd-log2 rectangles.
6. `downsample_residual` / `upsample_residual`: exact int16 equality.
7. Mixed-size `_multi` batch: exact coefficients/levels and exact coded-size
   reconstruction against a per-block CPU run.

Differences would show up as a nonzero mismatch count with the first index and
max absolute difference. The test exits 1 on any mismatch. Because the CPU and
GPU accumulate in the same order with contraction disabled, the double forward
transform is expected to be bit-identical, not merely close.

## Build and run on the CUDA box

```sh
ssh -p <port> <user>@<host>
cd /root/av2enc/src/cuda/tx_variants
./build_tx_variants_test.sh          # sm_86
./tx_variants_cuda_test              # actually ${TMPDIR}/av2enc_tx_variants_build/... in the script

CUDA_ARCH=sm_120 ./build_tx_variants_test.sh   # secondary target
```

The script compiles the CPU reference with `g++ -O3 -ffp-contract=off`, then
links it into an `nvcc -std=c++17 -arch=sm_86 -O3 -lineinfo -fmad=false`
binary with `-Xptxas=-v` resource reporting. `-Wpedantic` is omitted for nvcc
stub-file noise; the project warning set otherwise applies.

Useful extra checks on the box:

```sh
CUDA_LAUNCH_BLOCKING=1 ./tx_variants_cuda_test
compute-sanitizer --tool memcheck --launch-count 500 ./tx_variants_cuda_test
```

## Timing

`tx_variants_test.cu` contains a GPU-only timing harness (CUDA events, 5
iterations) for:

| Case | Blocks |
|---|---:|
| 32x32 DCT | 8192 |
| 32x32 ADST_ADST | 8192 |
| 32x32 IDTX | 8192 |
| 16x16 DCT | 32768 |
| 8x8 DCT | 32768 |
| 64x64 (adjusted 32x32) FLIPADST_FLIPADST | 4096 |

Measured on the RTX 3060 (sm_86) box, CUDA 12.8, one run on a shared instance
(expect ±10–30% between runs):

| Case | Forward (tx+quant) | Inverse |
|---|---:|---:|
| 32x32 DCT | 2.82 µs/block | 0.27 µs/block |
| 32x32 ADST_ADST | 2.72 µs/block | 0.24 µs/block |
| 32x32 IDTX | 0.10 µs/block | 0.07 µs/block |
| 16x16 DCT | 0.32 µs/block | 0.03 µs/block |
| 8x8 DCT | 0.04 µs/block | 0.008 µs/block |
| 64x64 FLIPADST (adjusted 32x32) | 2.58 µs/block | 0.25 µs/block |

The DCT/ADST/FLIPADST sizes are dominated by FP64 forward accumulation; IDTX
skips the matrix and is ~25× cheaper. Per policy, the CPU reference is not
benchmarked or optimized: it is the correctness oracle.

## Verified on the box

The component was built and run on the RTX 3060 (sm_86) box with CUDA 12.8:

```sh
cd /workspace/av2enc/src/cuda/tx_variants
./build_tx_variants_test.sh
/tmp/av2enc_tx_variants_build/tx_variants_cuda_test
```

Exact output:

```text
av2enc CUDA transform-variants test
device: NVIDIA GeForce RTX 3060 (sm_86, 28 SMs, 11.6 GiB)

[timing] GPU only; the CPU path is a correctness oracle
  32x32 DCT_DCT            8192 blocks: fwd(tx+quant)=  23.127 ms ( 2.823 us/block) | inv=   2.186 ms ( 0.267 us/block)
  32x32 ADST_ADST          8192 blocks: fwd(tx+quant)=  22.290 ms ( 2.721 us/block) | inv=   2.000 ms ( 0.244 us/block)
  32x32 IDTX               8192 blocks: fwd(tx+quant)=   0.835 ms ( 0.102 us/block) | inv=   0.551 ms ( 0.067 us/block)
  16x16 DCT_DCT           32768 blocks: fwd(tx+quant)=  10.544 ms ( 0.322 us/block) | inv=   0.980 ms ( 0.030 us/block)
   8x8  DCT_DCT           32768 blocks: fwd(tx+quant)=   1.408 ms ( 0.043 us/block) | inv=   0.255 ms ( 0.008 us/block)
  32x32 FLIPADST_FLIPADST   4096 blocks: fwd(tx+quant)=  10.582 ms ( 2.583 us/block) | inv=   1.031 ms ( 0.252 us/block)

7203 checks, 0 failures
RESULT: PASS
```

Additional checks on the same box:

- `CUDA_LAUNCH_BLOCKING=1 ./tx_variants_cuda_test` → `7203 checks, 0 failures`,
  `RESULT: PASS`.
- `compute-sanitizer --tool memcheck --launch-count 500 ...` →
  `ERROR SUMMARY: 0 errors`.

The 7,203 checks cover all 25 sizes × 16 types with random/structured blocks,
impulse coverage, every transform stage (coefficients, levels, dequant,
reconstruction, upsample/downsample) and the mixed-size `_multi` batch.

## Integration notes

- **Batching is mandatory.** One block per transform is ~30 µs of dispatch in
  this container for a small launch. The API already takes `nblocks`; feed it
  thousands of blocks. Never call it per transform in the block loop.
- **Mixed candidates.** `_multi` exists so RDO can launch one kernel for a list
  of `{size, type}` candidates. Upload one `TxBlockJob` array per candidate
  batch (it is `nblocks * sizeof(TxBlockJob)`), not one per block. The explicit
  `in_offset` / `out_offset` fields are what make mixed sizes correct; build the
  prefix sums on the host as the test does.
- **Arena.** Use `src/cuda/arena.*` for long-lived coefficient/level/residual
  buffers. The component has no allocations and no hidden sync, so it composes
  with a CUDA graph.
- **64x64 and adjusted sizes.** `make_tx_params(64,64,...)` returns `w=h=32`;
  the inverse emits a 64x64 coded residual by 2x2 duplication, matching
  §7.15.4. `txv_downsample_residual` builds the adjusted residual when the
  caller already has a full coded-size residual.
- **Do not confuse with `src/cuda/transform.*`.** The M0 layer rejects
  64x32/32x64 and omits the odd-log2 2896 pre-scale. Do not mix the two
  implementations for one block, and do not re-pin the M0 MD5s from this
  component without the full `dav2d` + reconstruction gate.
- **Decoder gate.** This component only produces transform coefficients and
  residuals; a stream-level claim requires the integration pass to emit real
  `TX_TYPE`/`TX_SIZE`/`TX_PARTITION` syntax and pass
  `tools/validate_dav2d.sh`. Nothing here claims that.
