# In-loop restoration — CUDA kernels (CCSO + Wiener NS)

CUDA port of `src/encoder/filters/restoration/`.  The kernels call the exact
same `__host__ __device__` core as the CPU reference (`ccso_core.h`,
`wiener_core.h`), so device output is **bit-exact** with the CPU for identical
planes, unit grids and coefficients.

Scope: 8-bit 4:2:0, `sm_86` primary.  No allocations, no hidden
synchronisation, no global state.  No bitstream integration; nothing here has
been gated with `dav2d`.  The stage-level wrapper
(`src/cuda/m1/restoration/still_restoration.*`) owns the CCSO -> §7.20 ordering
and the real-recon CPU-vs-GPU parity harness; these kernels are its building
blocks.

## Files

```text
src/cuda/restoration/
  ccso.h                CCSO launcher API
  ccso.cu               one thread per sample per plane
  wiener.h              loop restoration launcher API
  wiener.cu             copy step (cudaMemcpy2DAsync) + one thread per 4x4 block
  device_buffer.h       RAII cudaMalloc buffer for the tests
  ccso_test.cu          CPU-vs-GPU bit-exactness, validation, 720p timing
  wiener_test.cu        CPU-vs-GPU bit-exactness, validation, 720p timing
  build_and_test.sh     standalone nvcc build/run (no root CMake changes)
  README.md             this file
```

## API

```cpp
namespace av2enc::cuda {

cudaError_t ccso_frame(const restoration::CcsoCoreFrame& frame,
                       cudaStream_t stream = nullptr);

cudaError_t loop_restoration_frame(
    const restoration::RestorationFrame& frame,
    const restoration::LoopRestorationParams& params,
    cudaStream_t stream = nullptr);
}
```

### Device residency

Every pointer must be a device pointer.  For CCSO that includes
`plane[p].offsets`: `CcsoFrameParams` stores the offset table inline (host
memory), so a CUDA caller builds `CcsoCoreFrame` directly and points
`plane[p].offsets` at a device copy of the 576-entry `int16_t` table
(`ccso_set_offsets()` produces the host table; copy it once).  The
`CcsoBlks`, BruModes, segment, lossless and tile grids must also be
device-resident.

For loop restoration, `RestorationFrame` and `LoopRestorationParams` contain
only pointers, so they can be filled with device pointers directly; the
launcher calls `loop_restoration_to_core()` on the host to assemble the kernel
argument.

## Kernel design

- **CCSO**: one thread per sample of one plane.  A linear index is split across
  the three planes using per-plane sample counts.  Each thread re-derives its
  CCSO unit, the superblock-aligned `y2/x2` used for the BruModes test, the
  band and the two edge classes, then applies one clipped offset.  `CurrFrame`
  is read-only and `CdefFrame` is written at the thread's own sample, so there
  are no cross-thread hazards and no synchronisation.  One launch per frame.
- **Loop restoration**: `LrFrame := CdefFrame` with three `cudaMemcpy2DAsync`
  calls (only `width` bytes per row, guard columns untouched), then one kernel
  launch with one thread per 4x4 luma MI block per plane — the same loop
  structure as §7.20.  Each thread computes the §7.20.1 block context (unit
  index, stripe bounds, tile bounds, LrType) and runs the 32/12-tap Wiener
  filter for its block.  Every block writes a disjoint set of output samples;
  reads come from `CurrFrame`/`CdefFrame`, never from `LrFrame`.
- Launch geometry: 256 threads/block, grid computed from the work item count;
  no hardcoded SM count.

## Validation behaviour

`cudaErrorInvalidValue` for bad geometry, null/missing buffers, short strides,
aliasing `LrFrame`/`CdefFrame`, missing metadata, out-of-range parameters,
non-conforming `LoopRestorationSize` (§6.17.7.7), or a missing frame-filter class
map.  For a frame-level Wiener filter the caller supplies either
`filter_class_grid` + `subclass_lookup` or the legacy pre-looked-up
`subclass_grid`; see the CPU README.  `cudaErrorNotSupported` for
`gdf_frame_enable != 0` and for `RESTORE_PC_WIENER`.

For `RESTORE_SWITCHABLE` the per-unit `LrType` grid is device memory, so the
launcher cannot verify that no unit selects PC-Wiener; such a unit is left as a
copy of `CdefFrame`.  Callers must not emit PC-Wiener units until §7.20.4 is
implemented.

## Build and test

```sh
src/cuda/restoration/build_and_test.sh                 # sm_86
ARCH=sm_120 src/cuda/restoration/build_and_test.sh     # secondary target
```

The script compiles `restoration.cpp` with the host compiler, then builds
`ccso_test` and `wiener_test` with nvcc, linking the host object.  Both print
`PASS: N checks, 0 failures` and a 1280x720 timing line:

```text
  timing: 1280x720 CCSO 0.xxx ms/frame (xx.x MPix/s, 100 iters)
  timing: 1280x720 Wiener NS 0.xxx ms/frame (xx.x MPix/s, 50 iters)
```

Timing is measured with CUDA events on the kernel path only (upload/download
excluded).  Record the numbers from the RTX 3060 box in the component report;
they are not yet measured on this host (macOS has no CUDA toolchain), so no GPU
number is claimed here.

## Test coverage (CPU vs GPU)

- CCSO: 21 edge-class × extension-filter combinations, 6 `bo_only` band counts,
  all 16 `(scale, quant)` pairs, per-plane unit grids, lossless, BRU,
  tile clipping, odd geometries (5×3 … 21×11 MI), 24 randomized frames, plus
  guard-column and validation checks.
- Wiener: per-unit filters, zero coefficients, chroma-from-luma for
  `cfl_ds_filter_index` 0–3, stripe sourcing, two-class frame filters, mixed
  unit types on odd geometries, tile clipping, lossless/BRU, 16 randomized
  frames, validation and 720p timing.

Fixtures must themselves be conforming, or the CPU oracle refuses them and
there is nothing for the device output to match: `LoopRestorationSize[0]` is
64/128/256 for luma (§5.18.7.11) and every internal tile boundary spacing must
be a multiple of `maxPlaneRuSize = max(LoopRestorationSize[0],
LoopRestorationSize[1] << max(SubsamplingX, SubsamplingY))` (§6.17.7.7
`check_ru_size`). `run_case()` therefore reports a fixture rejection once
instead of a launch error plus a bogus mismatch.
