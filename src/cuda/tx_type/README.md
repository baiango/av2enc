# CUDA intra transform-type candidate evaluation (`src/cuda/tx_type/`)

GPU counterpart of `src/encoder/tx_type/`: evaluate every legal intra `TX_TYPE`
for a batch of blocks on the device, reduce with the CPU cost model, and return
the chosen type plus per-candidate metrics.

It contains **no transform arithmetic of its own**. Forward transform +
quantization and dequantization + inverse transform are the bit-exact batch
kernels from `src/cuda/tx_variants/` (`txv_forward_transform_multi`,
`txv_inverse_transform_multi`). This layer adds the candidate fan-out, the
reduction and the decision.

## Pipeline (5 launches per batch)

```
1. duplicate_adj_kernel        block adjusted residual -> per-candidate slot
2. txv_forward_transform_multi forward + quantize             (tx_variants)
3. txv_inverse_transform_multi dequantize + inverse           (tx_variants)
4. metrics_kernel              SSE / nonzero / Σ|level| / eob==1 per candidate
5. decide_kernel               cost + deterministic tie-break per block
```

Launch 1 exists because the `tx_variants` `_multi` kernels key their output
offsets by the *input* offset, so candidates need distinct slots. Any mix of
sizes and types works in one batch.

## Arithmetic / decision contract

- Levels, intermediate `double` coefficients and the reconstructed residual are
  **bit-identical** to `txv::forward_transform()` / `txv::inverse_transform()`
  on the CPU (same kernels, `-fmad=false`, `-ffp-contract=off`).
- SSE is an exact `int64` sum of squared coded-domain errors, cast to `double`
  exactly like the CPU.
- The decision matches `txtype::choose_intra_tx_type()` when the same candidate
  list and cost model are used:

```
rate = coeff_bits * nonzero [+ type_bits when use_type_bits]
cost = sse + lambda * rate
```

`type_bits` is computed by `txtype::static_type_bits()` (the §9.3 default CDF
cost) **on the host** and uploaded per candidate, so both sides see the same
number. The device only supports this default model; a custom CPU `rate_fn`
cannot be uploaded — use the per-candidate metrics and re-cost on the host.

Tie-break, identical to the CPU: lowest cost, then lowest SSE, then lowest
`TX_TYPE` value. Non-codable candidates are skipped:

```
eob == 0  -> only DCT_DCT (block is skipped; the decoder resets to DCT_DCT)
eob == 1  -> only DCT_DCT (§5.20.8.2 `!is_inter && eob == 1`)
otherwise any type is legal
```

`eob` itself is not computed on the device; `eob_one` is
`(nonzero == 1 && level[0] != 0)`, which is exactly `eob == 1`. The precise
`eob` for coefficient coding comes from the levels on the host.

## Files

| File | Contents |
|---|---|
| `tx_type.h` | plan PODs, cost model, decision, batch entry point |
| `tx_type.cu` | kernels + host plan builder + batch launcher |
| `tx_type_test.cu` | CPU-vs-GPU parity test (328,694 checks) |
| `build_and_test.sh` | standalone build/run; sm_86, links the real CPU refs |

## API

```cpp
namespace av2enc::cuda {

struct TxTypeCandJob {          // one candidate
  TxBlockJob tx;                // TxParams + in_offset (adj slot) + out_offset
  std::int32_t adj_src, block, type, candidate;
  double type_bits;
};
struct TxTypeBlockJob { std::int32_t cand_begin, cand_count, coded, coded_w, coded_h; };
struct TxTypeCostModel { double lambda, coeff_bits; bool use_type_bits; };
struct TxTypeCandidateMetrics { double sse; std::int32_t nonzero; std::int64_t sum_abs;
                               std::int32_t eob_one; };
struct TxTypeDecision { std::int32_t type, candidate_index, nonzero, eob_one;
                        double cost, sse; };

struct TxTypeBlockRequest {
  std::int32_t coded_w, coded_h, mapped_mode, reduced_tx_set, dct_only;
  std::int32_t coded_offset, adj_offset;   // filled by tx_type_pack_layout()
};
struct TxTypePlan {
  std::vector<TxTypeBlockJob> blocks;
  std::vector<TxTypeCandJob> candidates;
  std::vector<TxBlockJob> tx_jobs;         // same order as candidates
  std::size_t adj_scratch_elems, coeff_scratch_elems,
              level_scratch_elems, recon_scratch_elems;
};

void tx_type_pack_layout(TxTypeBlockRequest*, int nblocks,
                         std::size_t* coded_elems, std::size_t* adj_elems);
bool tx_type_build_plan(const TxTypeBlockRequest*, int nblocks,
                        TxTypePlan*, std::string* error);

cudaError_t tx_type_choose_batch(
    const std::int16_t* d_residual_coded,       // packed coded residual
    const std::int16_t* d_residual_adj_block,   // one adjusted residual/block
    const TxTypeBlockJob* d_blocks, int nblocks,
    const TxTypeCandJob* d_candidates, int ncandidates,
    const TxBlockJob* d_tx_jobs,
    std::int16_t* d_adj_scratch,   // ncandidates * kTxTypeAdjElems   int16
    double* d_coeff_scratch,       // ncandidates * kTxTypeAdjElems   double
    std::int32_t* d_levels,        // ncandidates * kTxTypeAdjElems   int32
    std::int16_t* d_recon_scratch, // ncandidates * kTxTypeCodedElems int16
    TxTypeCandidateMetrics* d_metrics,   // ncandidates
    TxTypeDecision* d_decisions,         // nblocks
    const TxTypeCostModel& cost, cudaStream_t stream);
}
```

Buffer sizes for a batch: with `n = plan.candidates.size()`,

| Buffer | Elements | Type |
|---|---:|---|
| `d_adj_scratch` | `n * 1024` | int16 |
| `d_coeff_scratch` | `n * 1024` | double |
| `d_levels` | `n * 1024` | int32 |
| `d_recon_scratch` | `n * 4096` | int16 |
| `d_metrics` | `n` | `TxTypeCandidateMetrics` |

That is 22 KiB per candidate (2 + 8 + 4 + 8), plus 24 B of metrics. For a
640x360 16x16 grid (880 blocks × up to 8 candidates = 7,040 candidates) it is
~155 MB; put these in the device-sized arena, not in `cudaMalloc` per frame.
A batch can be split per superblock row to bound the peak. The test uses
`cudaMalloc` as one-shot scaffolding only.

The batch call never allocates, frees or synchronizes. Plan offsets are trusted
(validating them would need a sync), so build plans with
`tx_type_build_plan()`.

## Hardware / build

Target is the 8 GB Ampere class; the test was run on the project 3060
(12 GB, sm_86) over ssh:

```sh
cd src/cuda/tx_type
./build_and_test.sh                 # sm_86
CUDA_ARCH=sm_120 ./build_and_test.sh
```

The host reference objects are compiled by `g++ -std=c++20 -O3
-ffp-contract=off`; `nvcc` compiles the `.cu` files with `-fmad=false` and
`-lineinfo`, and links `src/cuda/tx_variants/tx_variants.cu`.

## Tests

```
av2enc CUDA intra transform-type test
device: NVIDIA GeForce RTX 3060 (sm_86, 11920 MiB)
  lambda=0                                     601 blocks   2617 candidates  (eob0 0, eob1 0, non-codable 0)
  ...
  10468 candidates compared across 4 models
  QP 90 lambda = 544.5
  64x64 mixed 64/16/8/4                        337 blocks   2353 candidates  (eob0 538, eob1 20, non-codable 478)
  640x360 16x16 mode=0                         880 blocks   6160 candidates  (eob0 4368, eob1 30, non-codable 3766)
  640x360 16x16 mode=3                         880 blocks   6160 candidates  (eob0 4368, eob1 30, non-codable 3766)
  640x360 16x16 mode=1                         880 blocks   6160 candidates  (eob0 4368, eob1 28, non-codable 3764)
  640x360 16x32 (IDTX legal)                   440 blocks   3520 candidates  (eob0 2358, eob1 1, non-codable 2065)
  640x360 32x16 (IDTX legal)                   440 blocks   3520 candidates  (eob0 2451, eob1 6, non-codable 2150)
  640x360 32x32 (DCTONLY)                      220 blocks    220 candidates  (eob0 141, eob1 2, non-codable 0)
  640x360 16x16 reduced=1                      880 blocks   1760 candidates  (eob0 1248, eob1 12, non-codable 628)

328694 checks, 0 failures
RESULT: PASS
```

Coverage:

- **synthetic, one mixed batch**: all 25 TX sizes × modes {0, 3, 9, 12} ×
  `reduced_tx_set` {0, 1, 3} × two residual kinds, plus a `dct_only` block —
  2,617 candidates, four cost models (`λ = 0`, `544.5`, `544.5` without type
  bits, the `rdo` placeholder λ). Every candidate is compared on:
  `double` coefficients, quantized levels, inverse residual, SSE, nonzero,
  `sum_abs`, `eob_one`; every block on type/cost/SSE/candidate index/nonzero.
- **real 64x64 frame**: mixed 64x64 + 16x16 + 8x8 + 4x4 blocks in one batch.
- **real 640x360 frame**: full 16x16 grid (880 blocks) for DC, D45 and V, the
  long-32 16x32 and 32x16 grids (the only intra sets where IDTX is legal), the
  DCTONLY 32x32 grid, and `reduced_tx_set = 1`.
- 478–3,766 non-codable candidates per batch exercise the `eob <= 1` rule
  (both `eob == 0` and `eob == 1` cases are present in every real-frame batch).
- determinism: two identical launches produce identical metrics and decisions.
- argument validation: null pointers, negative counts, empty batch, bad sizes,
  bad modes, bad `reduced_tx_set`.
- plan builder: packed offsets, candidate counts per set, scratch sizes,
  `type_bits` equal to `txtype::static_type_bits()`.

Runtime of the test binary on the 3060: ~3.4 s including the CPU reference
comparison (not a performance claim; this is a correctness gate).

## Integration notes

- Stage 2 of `docs/m1-gpu.md` (GPU candidate search) can call this directly per
  block batch and use `TxTypeDecision::type` plus the winning candidate's
  levels. To emit syntax, feed the type through
  `txtype::tx_type_symbol()` on the host; nothing here writes bits.
- The winner's levels are already in `d_levels` at
  `plan.candidates[block.cand_begin + decision.candidate_index].tx.in_offset`
  (row-major adjusted size). The CPU integration path is documented in
  `src/encoder/tx_type/README.md`.
- Stage 1 (bit-exact block coding) can keep using `src/cuda/m1/tx/`; this
  component is the decision layer, not a replacement for the normative
  transform kernels.
- The GPU path is not wired into `src/cuda/m1/` or `src/main.cpp`; no bitstream
  or MD5 claim.
