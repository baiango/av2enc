# CUDA AV2 secondary transform (IST)

Bit-exact GPU port of `src/encoder/tx_secondary/`.  The full spec/API/table
notes live in that directory’s `README.md`.

| File | Contents |
|---|---|
| `tx_secondary_gpu.h` | `TxSecondaryJob` + batch launcher API |
| `tx_secondary_gpu.cu` | one-thread-per-block kernels and launch plumbing |
| `tx_secondary_device.cuh` | pure integer host/device arithmetic core |
| `tx_secondary_test.cu` | real-GPU parity test (needs nvcc) |
| `tx_secondary_emulation_test.cpp` | host-compiler parity test of the exact device core |
| `build_and_test.sh` | builds/runs emulation; real GPU test when `nvcc` exists |

```cpp
av2enc::cuda::TxSecondaryJob job;
job.tx_size = tx_size;          // 0..24
job.plane_tx_type = plane_type; // 0 DCT_DCT or 3 ADST_ADST
job.kernel = kernel;            // resolved by txsec::kernel_for_syntax()
job.sec_tx_type = sec;          // 1..3
job.transpose = transpose;      // 0/1
job.bit_depth = 8;              // 8..12
job.in_offset = in_elem_offset;
job.out_offset = out_elem_offset;
av2enc::cuda::txsec_inverse_dequant(d_in, d_out, d_jobs, njobs, stream);
```

The launcher performs no allocation and no synchronization.  `d_in == d_out`
(in-place) is supported.  Each job’s input/output block is `w*h` `int32_t`
elements, tightly packed at its offset; `w = Min(32, Tx_Width)`,
`h = Min(32, Tx_Height)`.

Result on this macOS host (no nvcc): device-core emulation
**11,504,810 checks, 0 failures**.  The real GPU test must be run on the sm_86
Linux box; it has not been run here.
