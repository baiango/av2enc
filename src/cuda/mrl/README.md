# AV2 MRL — CUDA predictor

GPU mirror of `src/encoder/mrl/`: luma directional MRL for `mrl_index 0..3`
plus the optional secondary-line blend. Same integer arithmetic and the same
availability rules.

Status:

* CPU-vs-GPU parity test added (`mrl_test.cu`) for every directional mode and
  angle delta, all four MRL indices, secondary on/off, 4x4..64x64, frame/SB
  edges, and smooth-neighbour flags.
* Host emulation of the exact device header passes locally without a GPU
  (`build_emulation_test.sh`); it compares against the CPU reference. The
  on-device run still needs the Linux + CUDA box.

## Files

```
src/cuda/mrl/
  mrl.h                 public API
  mrl_device.cuh        shared device body (kernel + host emulation)
  mrl.cu                kernels + validation
  mrl_test.cu           CPU-vs-GPU parity test
  mrl_emulation_test.cpp  host-emulation test
  build_mrl_test.sh
  build_emulation_test.sh
  README.md
```

## API

```cpp
#include "cuda/mrl/mrl.h"
using namespace av2enc::cuda::mrl;

MrlBlock block;    // reconstructed luma plane + geometry
MrlParams params;  // mode, angle_delta, mrl_index, mrl_sec_index, flags
std::uint8_t* d_pred;

mrl_predict(block, params, d_pred, stream);       // one block
mrl_predict_batch(d_jobs, count, stream);         // MrlJob { block, params, pred }
```

Entry points are asynchronous, allocate nothing, and never synchronize.
`mrl_predict` validates `plane == 0`, directional mode, index/delta ranges,
secondary-line availability (not 4x4), and that the selected reference lines
are inside the plane. The batch form trusts each device-resident descriptor.

## Device behaviour

`mrl_run_block_device()`:

1. builds `AboveRow`, `LeftCol`, `AboveSecRow`, `LeftSecCol` with
   `numAboveNeeded = numLeftNeeded = w + h + 2*mrl_index`, using
   `aboveMrlIndex = 0` on an SB boundary and the full `mrl_index` otherwise;
2. fills the negative indices `-1 .. -1-mrl_index`, including the secondary
   arrays;
3. for `mrl_index == 0` only, applies the §7.13.2.14 corner filter and
   §7.13.2.18 edge filter with the §7.13.2.15/.16 smooth flags;
4. extends the primary edges for `(pAngle, mrl_index)` and the secondary edges
   for `(pAngle, 0)`;
5. predicts every sample; with `mrl_sec_index` the result is
   `(primary + secondary + 1) >> 1`.

`pAngle = Mode_To_Angle[mode] + angle_delta*ANGLE_STEP +
Mrl_Index_To_Delta[mrl_index]`, then `wide_angle_mapping`.

`mrl_device.cuh` is also compiled by the host-emulation test with CUDA
keywords stubbed and one thread, so the tested arithmetic is the arithmetic the
kernel runs.

## Build / test

On the Linux + CUDA box:

```sh
./src/cuda/mrl/build_mrl_test.sh                 # sm_86
ARCH=sm_120 ./src/cuda/mrl/build_mrl_test.sh
```

Without a GPU:

```sh
./src/cuda/mrl/build_emulation_test.sh
# mrl_emulation_test: PASS (16904 checks)
```

The emulation test covers the full mode/delta/index space on representative
geometry plus 8000 random blocks. The on-device test uses random planes and
compares single-block and batch results byte-for-byte with the CPU reference,
including partial right/bottom blocks (`{400,400}` / `{440,440}` on a
`max=472` plane) and SB-boundary rows.

`mrl_predict_batch` takes a **device** array of `MrlJob`. Stage descriptors in
host memory and `cudaMemcpy(..., cudaMemcpyHostToDevice)`; writing through a
`cudaMalloc` pointer from the host is a host SIGSEGV before any kernel launches
(this was the earlier test bug, now fixed).

## Integration / decoder gate

None yet. MRL needs `enable_mrls = 1`, per-block `mrl_index`/`mrl_sec_index`
syntax, an encoder search flag, and transform-type derivation updates in
`src/encoder/m1/`, which this workstream does not own. See
`src/encoder/mrl/README.md` for the exact list. Without those edits a minimal
stream cannot be produced, so no decoder result is claimed.

Quality: no end-to-end QP-90 delta is claimed because MRL is not signalled.
The module is a spec-tested predictor with an integration plan, not a
bitstream feature.
