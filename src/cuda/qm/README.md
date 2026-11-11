# CUDA quantizer-matrix dequantization

`qm_gpu.cu` is the device half of `src/encoder/qm/`. It calls the same
`av2enc::qm::core` integer helpers as the CPU reference, so the two paths are
bit-exact by construction; there is no floating point in §7.14.4.

## API

```cpp
#include "cuda/qm/qm_gpu.h"

// Caller uploads these once (cudaMemcpy or the arena):
//   kQuantizerMatrixBytes = 15*2*3600 = 108000
//   kUserQmBytes          = 15*3*3*8*8 = 8640   (optional)
av2enc::cuda::qm::Batch batch;
batch.levels = d_levels;         // int32 coefficients, adjusted-size raster
batch.dequant = d_dequant;       // int32 output
batch.jobs = d_jobs;             // one Job per block
batch.num_jobs = n;
batch.qtable = d_qtable;         // required
batch.user_qm = d_user_qm;       // optional
batch.status = d_status;         // optional: 0 ok, -1 rejected
av2enc::cuda::qm::dequantize_batch(batch, stream);
```

`dequantize_batch()` allocates nothing, frees nothing, and does not
synchronize. One thread runs one job (4x4 through 64x64 adjusted sizes). Jobs
are rejected through `status` for an invalid TX size, plane, QM level, or a
non-positive DC/AC quantizer.

`Job` carries `seg_qm_level`, `qm_level` (the plane's `qm_y/u/v[0]`),
`bit_depth` (the §7.14.4 Clip3 bound), `using_qmatrix`, `qm_data_present`,
`allow_tcq`, `lossless`, `use_fsc`.
`UserQmLevel` is exactly 576 bytes in the CPU header, so the flattened
`[level][t][plane][i][j]` upload matches `core::user_qm_index()`.

## Tests

`./build_and_test.sh` (`CUDA_ARCH=sm_86` by default) compares every
dequantized coefficient of the CPU reference and the GPU kernel for:

* all 25 TX sizes;
* levels 0..15 (15 = QM off);
* planes 0/1/2 (8-bit 4:2:0);
* TX types 0..15 (2-D vs IDTX/1-D gating);
* QM on/off, user-QM data present/absent, TCQ luma shift (including IDTX as
  TX_CLASS_2D), lossless, 8-bit and 10-bit clip bounds;
* invalid jobs (bad TX size, plane, or quantizer) reported as `status = -1`.

The test allocates with `cudaMalloc`; it is a component test, not a hot path.

**Not run in this environment:** the macOS host has no `nvcc`/CUDA toolkit, so
the `.cu` sources and parity test could not be compiled or executed here. The
shared scalar core they call is compiled and exercised by the 258k-check CPU
test in `src/encoder/qm/`. Run `./build_and_test.sh` on the sm_86 box before
using the kernel.

## Integration

The kernel is a drop-in replacement for the per-coefficient dequant loops in
`src/cuda/m1/tx/m1_tx.cu` (`dequant_dev`) and
`src/cuda/m1/hbd/hbd_tx.cu`, and for
`src/cuda/tx_variants/tx_variants.cu` (`dequant_dev` /
`txv_dequantize`). Pass the QM state from the frame plan; no core file is
modified by this component.
