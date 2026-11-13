# AV2 IntraBC — CUDA

Device half of AV2 intra block copy, mirroring `src/codec/intra/ibc/core/`.
The kernels share the host/device core `ibc_core.h` (plane scaling, geometry,
block copy, SAD), so the arithmetic cannot drift from the CPU reference.

## API

```cpp
#include "cuda/codec/intra/ibc/ibc.h"
namespace cuda = av2enc::cuda;

cuda::ibc_copy_block(frame, mi_row, mi_col, bsize, bv,
                     dst_y, dst_stride_y, dst_u, dst_stride_u,
                     dst_v, dst_stride_v, visible_w, visible_h, stream);
cuda::ibc_sad_batch(frame, mi_row, mi_col, bsize, d_bvs, n, d_sads, stream);
cuda::ibc_validate_global(geometry, mi_row, mi_col, bsize, d_bvs, n, d_valid,
                          stream);
```

`ibc_search(arena, ...)` is the allocating entry point: it bump-allocates one
packed `(SAD, candidate)` word from the caller's `Arena` and rewinds before
returning.  `ibc_required_device_bytes(config)` is its exact arena requirement,
and an undersized arena returns `IbcSearchStatus::kArenaTooSmall` with the byte
count.

Scope: 8-bit 4:2:0, integer BVs in 1/8 luma samples, block sizes 8x8 through
64x64 (rectangles included).  Local four-buffer validity is a CPU concern; the
device validates the global §6.19.7.12 rule (`allow_global`).  No hidden
synchronization, no global state.

## Tests

```sh
cd src/cuda/codec/intra/ibc
./build_and_test.sh                 # sm_86, needs nvcc + GPU
GPU_ARCH=sm_120 ./build_and_test.sh
```

Verified on the RTX 3060 (`sm_86`, CUDA 13.0):

```
device: NVIDIA GeForce RTX 3060 (sm_86)
ibc_gpu_test: 36 checks, 0 failures
```

The arena workspace (`../workspace/`) adds a workspace-vs-module equality check
for SAD, global validation and block copy over a 16x16 block with 15 candidate
BVs.

## Integration

The still path must set `allow_screen_content_tools` and write
`palette_mode_info()`/IBC syntax on every eligible block, and reset the block
contexts after a `skip_flag = 1` IBC block (see `docs/negative-results.md`).
The component supplies prediction/SAD/validation only.
