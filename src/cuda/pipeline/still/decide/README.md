# Still coefficient decisions (`src/cuda/pipeline/still/decide/`)

Device §7.16 rate model and the RDOQ / TCQ trellises used by `--still-gpu`.
Owner: the M1 orchestrator (`src/cuda/pipeline/still/still_gpu.cu` includes
this directory; nothing else links it).

## Files

| File | Contents |
|---|---|
| `decide_core.cuh` | host+device core: §8.3.1 CDF view, exact `-log2(p)` symbol cost, the two av2 model position builds, `rdoq::quantize` and `tcq::quantize` trellis ports, §7.14.4 dequant |
| `decide_tables.h` | host packing of the four `COEFF_CDF_Q_CTXS` slices into one device blob + the 32,769-entry `-log2(p)` table |
| `decide_core_test.cpp` | host parity test against the shipped CPU references |
| `build_and_test.sh` | builds and runs the parity test (no nvcc, no GPU) |

## Why a port instead of `cuda/codec/{rate/rdoq,transform/tcq}`

Those batch kernels take caller-built rate tables.  A table entry is the cost
of one `(position, state, level)` triple, so a table is
`area * states * (max_level + 1)` doubles.  At 640×360 QP 90 a single 32×32
luma block reaches |level| 1311 (bound 2687), i.e. ~350 MiB for one block, so a
table-driven stage cannot cover the product path.  `decide_core.cuh` reads the
same costs on demand in the DP.

The transform arithmetic is **not** reimplemented: forward/quantise stays in
`pipeline/still/tx/still_tx.cu`, and the inverse of the decided Quant levels
uses `cuda/codec/transform/variants` `txv_inverse_block_multi`.

## Exactness

The CPU models compute `-std::log2(p / 32768.0)` for an integer probability
`p` in `[1, 32768]`.  The device reads `log2p[p]` from a table the host fills
once with `std::log2`, so the device rate equals the CPU rate bit for bit;
everything else in the model is integer arithmetic.  The trellis ports keep the
CPU loop order, tie-break order and arithmetic order.

`decide_core_test.cpp` compares, on randomized blocks of every adjusted TX size
(luma and chroma, three magnitude scales):

* `dec_rdoq_coefficient_bits` / `dec_eob_bits` with
  `rdoq::av2::Model` for random `(position, level, is_eob)`;
* `dec_tcq_coefficient_bits` with `tcq::av2::Model` for random
  `(position, state, level, is_eob)`;
* the full RDOQ port with `rdoq::quantize` (levels, EOB, rate, distortion,
  cost);
* the full TCQ port with `tcq::quantize` (base levels, Quant levels, EOB, rate,
  distortion, cost).

Result: 504,281 checks, 0 failures (`WERROR=1 ./build_and_test.sh`).

## Product wiring (`--still-tcq`)

### TCQ (`--still-tcq`, luma)

Per luma block, on the frame stream, one block at a time:

```
still_tx (forward + scalar levels + scalar coeffs)
  -> dec_prepare_kernel   scan-order inputs + §7.16 frozen contexts
  -> dec_tcq_kernel       tcq::quantize port (one thread, exact tie-breaks)
  -> dec_apply_kernel     syntax base levels (raster), Quant levels -> §7.14.4
                          dequant with dqDenom << 1, scan EOB
  -> txv_inverse_block_multi -> packed residual -> recon_write
```

### RDOQ (`--still-rdoq`, every plane)

The same pipeline with `dec_rdoq_kernel` (the `rdoq::quantize` port), the
scalar levels as the frozen reference, and the frame `dqDenom` (no TCQ shift).
U and V run the stage after their own forward transform.

`dec_status` is downloaded with the frame (one entry per block and plane) and a
rejected block fails the frame explicitly (no silent fallback).  No coefficient
crosses to the host.

The trellis DP keeps its `kDecMaxArea`-sized EOB / suffix / bound tables in the
caller's arena (`DecTrellisScratch`) instead of the kernel's local frame.  The
driver reserves a kernel's local memory for the maximum block size, so a
~32 KiB/thread frame made the launch fail with `cudaErrorMemoryAllocation` on a
busy shared GPU even though the device had ~0.5 GiB free; with the scratch the
stage runs at any free-memory level the arena itself fits.

Not wired yet: `--still-tx-partition` and `--still-rdo-intra`; see
`STATUS.md` for the concrete blockers and the reverted RDO-intra work.  Parity hiding (`tcq::parity`) is not part of the still
reference encoder: `--still-tcq` uses `tcq::quantize` only, so the GPU matches
the CPU by not selecting the `kMinDistortion` alternative either.
