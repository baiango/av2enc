# CUDA DC/AC delta-Q quantizer (`src/cuda/codec/transform/delta_q/`)

Bit-exact device half of `src/codec/transform/delta_q/` (§7.14.2 quantizer
derivation and §7.14.4 forward/inverse quantization with per-plane DC and AC
deltas).

The CPU reference remains the oracle.  The device derives each plane's DC and
AC quantizers from the effective `qindex` plus the *resolved* (`DeltaQ* + Base*`) deltas,
then runs the same integer rounding and the same floating-point operation
order as `quantize_block()` / `dequantize_block()`.

| File | Contents |
|---|---|
| `deltaq_types.h` | CUDA-free `Job` / `Batch` PODs |
| `deltaq_device.cuh` | shared host/device core: `get_q`, q2, quantize, dequantize |
| `deltaq_gpu.h` | `quantize_batch()` / `dequantize_batch()` launchers |
| `deltaq_gpu.cu` | kernels (one thread per block) |
| `deltaq_test_common.h` | host-only case matrix shared by both tests |
| `deltaq_emulation_test.cpp` | host-compiler parity test (no nvcc, no GPU) |
| `deltaq_gpu_test.cu` | real-GPU parity + arena test |
| `build_and_test.sh` | emulation always; GPU test when nvcc exists |
| `build_gpu_test.sh` | `nvcc -arch=sm_86` real-GPU test |

## API

```cpp
#include "cuda/codec/transform/delta_q/deltaq_gpu.h"
namespace dq = av2enc::cuda::deltaq;

dq::Job job;
job.plane = 1;                 // Y / U / V
job.coded_w = 8;               // 64-coded sizes use the adjusted 32x32 block
job.coded_h = 8;
job.qindex = 90;              // §7.14.2 get_qindex() result (base when no delta-q)
job.delta_dc = -3;             // resolved DeltaQUDc + BaseUVDcDeltaQ
job.delta_ac = 4;              // resolved DeltaQUAc + BaseUVAcDeltaQ
job.tx_type = 0;               // IDTX (9) is TX_CLASS_2D
job.q2_offset = -1;            // derive q2 from the DC/AC quantizers
job.lossless = 0;
job.allow_tcq = 1;             // luma TX_CLASS_2D -> extra dqDenom shift
job.use_fsc = 0;
job.coefficient_scale = 8.0 / (2.0 * 528.0);
job.in_offset = 0;             // element offsets into the shared buffers
job.out_offset = 0;

dq::Batch batch;
batch.coeff = d_coeff;         // double, adjusted w*h per job
batch.levels = d_levels;       // int32 (dequantize input)
batch.out = d_levels;          // int32 output; may alias the input
batch.jobs = d_jobs;
batch.num_jobs = n;
batch.q2 = nullptr;            // or an uploaded prebuilt q2 matrix
batch.status = d_status;       // optional: 0 ok, -1 rejected

dq::quantize_batch(batch, stream);     // coeff -> levels
dq::dequantize_batch(batch, stream);   // levels -> dequant
```

The launchers allocate nothing, free nothing and never synchronize.  A job is
rejected through `status` for a bad plane, coded size (outside 1..64), TX type
(outside 0..15), qindex (outside 0..255), or a non-positive q2 entry; a
rejected job's output is left untouched.

`q2_offset >= 0` selects a prebuilt `w*h` q2 matrix in `Batch::q2`.  The host
builds it with the CPU `dcdeltaq::build_q2_matrix()`, which keeps the
quantizer-matrix search and its tables off the device while still exercising
the §7.14.4 `Round2(q * m, 5)` path.

## Implementing rules

- `get_q()` / `qlookup()` are transcribed from §7.14.2, including the
  `qindex == 0 && delta <= 0 -> 64` special case.
- `Round2(x, n) = (x + 2^(n-1)) >> n` with int64 accumulation.
- Quantization is `llround(coeff * factor / static_cast<double>(q2))` with
  `factor = coefficient_scale * 528 * (1 << extra_tcq_shift)`; the expression
  order matches the CPU and the device build uses `-fmad=false`.
- Dequantization is `((abs(level) * q2) & 0xFFFFFF)`, `Round2(., 3)`, divide by
  `1 << ((pels > 256) + (pels > 1024) + extra_tcq_shift)`, sign, clip to
  `[-32768, 32767]`.
- `extra_tcq_shift` applies only to luma, non-lossless, TX_CLASS_2D, non-FSC
  blocks.  IDTX (9) is TX_CLASS_2D; V_*/H_* (10..15) are 1-D and never get it.
- Lossless: quantize is `llround(coeff)` and dequantize copies the levels.

## Tests

### Host emulation (no nvcc, no GPU)

```sh
./build_and_test.sh
```

Compiles `deltaq_device.cuh` with the host compiler and compares every level
and dequantized coefficient with `dcdeltaq::quantize_block()` /
`dequantize_block()` over:

- 8 coded sizes (4x4, 8x8, 16x16, 32x32, 64x64, 4x8, 16x32, 32x8);
- all 3 planes;
- TX types 0, 3, 9 (IDTX), 10 (V_DCT), 15 (H_FLIPADST);
- TCQ and FSC on/off, `get_q` over qindex 0..255 and deltas -87..71;
- lossless passthrough, identity/default quantizer matrices, rejected jobs.

Observed on the development host (Apple clang):

```text
delta-q emulation: 496 cases
  compared 491 blocks (5 rejected, 3 lossless, 2 QM, 96 IDTX, 26 TCQ-shift)
delta-q emulation: 425761 checks, 0 failures
RESULT: PASS
```

### Real GPU (sm_86)

```sh
./build_gpu_test.sh
```

Runs the same matrix as one mixed batch (prefix offsets, one thread per block),
then checks the inverse path, in-place `levels -> dequant`, and determinism
across two launches.  Every device buffer is allocated from one
`av2enc::cuda::Arena`; the test uses no `cudaMalloc`.

## Arena contract

The component owns no persistent device state.  All buffers (`jobs`, `coeff`,
`levels`, `dequant`, `q2`, `status`) are caller-owned and in production must
come from the single `av2enc::cuda::Arena`.  The source tree contains no
`cudaMalloc`/`cudaFree`; `tools/check_arena.sh` passes and both build scripts
fail if one appears.

## Integration notes (product TX path)

- One batch call per transform partition batch; do not launch per block.
- Build job deltas with `dcdeltaq::resolve_delta_q()` (or
  `derive_quant().plane[p]`) and set `qindex` to the `get_qindex()` result
  (`derive_quant().qindex`), so the device sees the same per-plane totals and
  effective qindex the frame header/segment map produced.  The raw
  `base_q_idx` is only for coefficient CDF selection, not the quantizer.
- Per-plane `PlaneQuant::dc/ac` already includes the §7.14.2 DC/AC split; the
  device re-derives them from `qindex` and the deltas, so the caller only
  needs the header state, not the qlookup tables.
- `q2` matrices for QM blocks are built on the host; for non-QM blocks leave
  `q2_offset = -1` and the DC/AC quantizers alone define q2.
- The level -> entropy path is unchanged: this module stops at levels (and
  reconstructs dequant for the inverse transform).  TCQ/CCTX and parity hiding
  consume the same `levels`/`dequant` convention from
  `src/cuda/codec/transform/tcq/`.
