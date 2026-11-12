# M1 batched transform / quantize stage (`src/cuda/m1/tx/`)

One CUDA launch for a batch of independent M1 transform blocks:

```
source block + prediction  (or coded residual)
  -> coded-size residual                       (src - pred)
  -> §7.15.4 downsample to the adjusted size   (Round2, same as still_encoder.cpp)
  -> forward 2D transform                      (binary64, DCT/ADST/IDTX/FLIPADST)
  -> quantize                                  (lround(coeff * quant_factor))
  -> eob over the §7.15 zigzag scan            (same order as build_scan())
  -> dequantize                                (abs(q)*528 & 0xFFFFFF, Round2(.,3))
  -> inverse 2D transform                      (int64, normative clips, 2896 prescale)
  -> coded-size residual by §7.15.4 duplication
```

The numeric reference is `TileEncoder::compute_tx` (`src/encoder/m1/still_encoder.cpp`)
driven through `av2enc::txv::forward_transform` / `inverse_transform`
(`src/encoder/tx_variants/`). The GPU stage is bit-exact against it.

This component is **not wired into the encoder**. It exposes the stage-1 API and
its parity/timing gates; the orchestrator (agent 1) owns the block plan, arena,
stream and job construction.

## Files

| File | Contents |
|---|---|
| `still_tx.h` | device API: `StillTxJob`, `StillTxBuffers`, `still_tx_batch`, `still_tx_init_job` |
| `still_tx.cu` | one fused kernel + host launcher |
| `still_tx_test.cu` | CPU-vs-GPU bit-exact tests, real-frame blocks, GPU timing |
| `build_still_tx_test.sh` | `nvcc -arch=sm_86 -fmad=false` component build (private build dir) |

## API

```cpp
namespace av2enc::cuda {

using av2enc::txv::TxParams;

enum class StillTxInput : uint8_t {
  kResidual = 0,    // d_residual_in already holds int16 src-pred
  kSourcePred = 1,  // residual = source - prediction, computed in-kernel
};

struct StillTxJob {
  TxParams params;                 // coded/adjusted size, TX_TYPE, quantizer
  uint8_t input;                   // StillTxInput
  uint8_t src_packed;              // 1 = packed block, 0 = plane + origin
  int16_t src_x, src_y;            // plane mode origin
  int16_t src_width, src_height;   // plane mode visible bounds (edge clamp)
  int32_t src_offset, src_stride;
  int32_t pred_offset;             // packed coded-size prediction
  int32_t residual_in_offset;      // coded-size int16 residual
  int32_t residual_adj_offset;     // adjusted-size int16 (optional output)
  int32_t coeff_offset;            // adjusted-size double (optional output)
  int32_t levels_offset;           // adjusted-size int32 (required buffer)
  int32_t dequant_offset;          // adjusted-size int32 (optional output)
  int32_t residual_out_offset;     // coded-size int16 (optional output)
  int32_t eob_offset;              // int32 (optional output)
};

struct StillTxBuffers {
  const uint8_t* src; const uint8_t* pred; const int16_t* residual_in;
  int16_t* residual_adj; double* coeff; int32_t* levels;
  int32_t* dequant; int16_t* residual_out; int32_t* eob;
  int* error_count;               // optional device guard counter
};

bool still_tx_init_job(StillTxJob*, int coded_w, int coded_h,
                    TxType = TxType::kDctDct);

cudaError_t still_tx_batch(const StillTxJob* d_jobs, int nblocks,
                        const StillTxBuffers&, int smem_elems,
                        cudaStream_t);
}
```

Conventions:

- One CUDA block per job, `nblocks` jobs per launch. Jobs may mix all 25 coded
  sizes and all 16 TX types; every buffer index is an explicit element offset
  supplied by the caller, so mixed sizes need no uniform stride.
- `smem_elems` is the largest adjusted area (`w*h`) in the batch; pass `0` for
  the 32x32 maximum (18,432 B dynamic shared memory). Jobs that exceed the hint
  are rejected through `error_count` and `eob = -1` instead of writing past the
  shared allocation. The caller knows the batch, so a same-size group should
  pass its exact area.
- `levels` is required. `residual_adj`, `coeff`, `dequant`, `residual_out` and
  `eob` are optional and skipped when their pointer is null. Leaving
  `residual_out` null is the forward-only shape used by candidate search.
- `src_packed = 0` reads the source from a plane of `src_width x src_height`
  with `src_stride` and clamps each coordinate to the visible plane, exactly
  like `TileEncoder::source_sample`; this is the edge-block path.
- `TxParams` carries the quantizer state (`dq_denom`, `quant_factor`). M1 pins
  `base_q_idx = 90`; there is no per-block qindex in this stage.
- No allocation, free, synchronization, mutable global state or host callback.
  The stage is CUDA-graph capturable (tested) and uses only standard CUDA
  cores, shared memory and FP64/integer arithmetic.
- `-fmad=false` plus `__dadd_rn`/`__dmul_rn` (never the float
  `__fadd_rn`/`__fmul_rn`) keep the double forward transform bit-identical.

## Kernel design

One block, one transform block, up to 256 threads. Dynamic shared memory:

| Region | Bytes | Use |
|---|---:|---|
| `int16 s_adj` | 2E | adjusted residual |
| `double s_tmp` | 8E | forward intermediate, reused as `int32 s_inter` |
| `int32 s_dq` | 4E | levels, then dequantized coefficients |
| `int16 s_kw`, `s_kh` | 2 x 2 KiB | transform kernels |

E is the padded adjusted area (max 1024), so the 32x32 path uses 18,432 B. The
fused stage keeps coefficients, levels and dequantized values in shared memory:
the inverse never re-reads global memory and the forward coefficients are only
written when `coeff` is requested. The eob scan uses a closed form for
`build_scan()` (anti-diagonals in ascending order, rows descending inside a
diagonal), so no thread performs the 1024-step serial walk; the test cross-checks
the closed form against the CPU `build_scan` for every adjusted size.

## Bit-exactness

`still_tx_test.cu` links the real CPU reference with `-ffp-contract=off` and
compares, per job:

1. adjusted residual (downsample),
2. forward coefficients (exact double bytes, so `-0.0` is caught too),
3. quantized levels,
4. eob,
5. dequantized coefficients,
6. coded-size inverse residual (incl. the 64x64/16x64/64x16/32x8/8x32
   duplication and the odd-log2 2896 pre-scale).

Coverage:

- all 25 coded sizes x 16 TX types x 5 patterns (random, gradient, impulse,
  flat, small), residual input and packed source+prediction;
- forward-only batches;
- plane reads with stride and overhanging edge blocks (all 25 sizes);
- the smem-hint guard (a 32x32 job is rejected while a 4x4 job in the same
  launch still matches);
- real blocks from `tests/data/basic_64x64_1f.y4m` and
  `tests/data/basic_640x360_1f.y4m`: both planes, corners, interior, bottom/right
  overhang, square and rectangular sizes, DCT and selected ADST/IDTX types;
- CUDA graph capture + replay of the whole batch;
- the scan closed form against `build_scan()` for every adjusted size.

Expected result on the 3060 box: `34482 checks, 0 failures`, also under
`CUDA_LAUNCH_BLOCKING=1`; `compute-sanitizer --tool memcheck` reports 0 errors.

## Build and run

```sh
cd /workspace/av2enc/src/cuda/m1/tx
BUILD_DIR=/workspace/av2enc/build-m1-tx ./build_still_tx_test.sh
/workspace/av2enc/build-m1-tx/still_tx_cuda_test \
  /workspace/av2enc/tests/data/basic_64x64_1f.y4m \
  /workspace/av2enc/tests/data/basic_640x360_1f.y4m
```

`BUILD_DIR` defaults to `${TMPDIR:-/tmp}/av2enc_still_tx_build`; keep it out of the
source tree. `CUDA_ARCH=sm_120` builds the secondary target.

## Timing (RTX 3060, CUDA 12.8, shared instance)

GPU-only, one launch per batch, full fused stage (±5–10% run to run on the
shared instance):

| Case | Blocks | ms | us/block |
|---|---:|---:|---:|
| 4x4 DCT | 32768 | 1.65 | 0.050 |
| 8x8 DCT | 32768 | 2.61 | 0.080 |
| 16x16 DCT | 16384 | 5.85 | 0.357 |
| 32x32 DCT | 4096 | 10.38 | 2.535 |
| 64x64 (adjusted 32x32) DCT | 4096 | 10.39 | 2.536 |
| 16x64 DCT | 4096 | 3.81 | 0.929 |
| 64x16 DCT | 4096 | 4.13 | 1.009 |
| 32x8 DCT | 8192 | 3.63 | 0.443 |
| 8x32 DCT | 8192 | 3.13 | 0.382 |
| 32x32 ADST_ADST | 4096 | 10.38 | 2.535 |
| 32x32 IDTX | 4096 | 0.59 | 0.144 |
| 8x8 DCT, forward only | 32768 | 2.17 | 0.066 |
| 16x16 DCT, forward only | 16384 | 5.53 | 0.337 |
| **640x360 M1 TX distribution** | **654** | **0.53** | **0.815** |

The last row is one 640x360 frame's transform workload in a single launch: the
218 luma block sizes from `AV2ENC_STILL_TRACE=1 av2enc <y4m> --still`, plus both
chroma planes at half size. It is a lower bound for stage 1: block predictions
depend on prior reconstruction, so a real frame must launch per dependency
level (smaller batches, more launches). The 32x32/64x64 rows are bounded by
FP64 throughput (GA106 FP64 is 1/64 rate); IDTX is ~18x cheaper because it has
no matrix.

## Integration notes

- **Batch by size where you can.** Pass the group's exact area as
  `smem_elems`; a mixed batch pays the 32x32 shared footprint for every block.
- **Explicit offsets are mandatory.** Mixed sizes have different element
  counts; prefix-sum the offsets as the test's `Batch` class does.
- **Do not use `src/cuda/transform.*` for M1.** That layer rejects
  64x32/32x64 and omits the odd-log2 2896 pre-scale. This stage follows
  `src/encoder/tx_variants/`.
- **`AV2ENC_STILL_ZERO_DC`** is a CPU debug hook in `compute_tx`; the GPU stage has
  no equivalent. A debug run must zero the device level-0 and recompute eob.
- **Recon is not here.** The stage emits the coded-size inverse residual
  (`residual_out`); reconstruction/prediction is agent 5's `src/cuda/m1/recon/`.
- **Entropy needs `levels` + `eob` + the scan.** Scan tables live in the cost
  model, not here; the eob is the same index `code_tx` uses.

## Blockers / open items

- Not integrated, so no bitstream or `dav2d` claim. Parity is component-level.
- Stage-1 batches are limited by prediction dependencies; full-frame batching
  needs the wavefront scheduler. The stage itself is graph-safe.
- The scan is recomputed per block (closed form, `O(w+h)` per index). If eob
  cost shows up in profiles, a small constant-memory scan table can replace it.
- Only lossy transforms are covered; lossless IDTX / DPCM cumulative sums are
  outside the M1 path, same as `tx_variants`.
