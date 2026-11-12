# CDEF — CUDA kernel (AV2 §7.18)

CUDA implementation of the wave-2 CDEF component. It shares the exact
arithmetic with the CPU reference through
`src/encoder/filters/cdef/cdef_core.h` (the same `__host__ __device__` inline
functions), so the device output is **bit-exact** with the CPU for identical
planes and metadata.

Scope: 8-bit 4:2:0, MI-aligned planes, separate source and destination buffers.
No bitstream integration. No allocations, no hidden synchronisation, no global
state.

## Files

```text
src/cuda/cdef/
  cdef.h              device frame API
  cdef.cu             launch + validation + one kernel per frame
  cdef_test.cu        CPU-vs-GPU bit-exactness, validation, timing
  build_and_test.sh   standalone nvcc build/run (no root CMake changes)
  README.md           this file
```

## API

```cpp
namespace av2enc::cuda {

struct CdefDeviceFrame {
  const uint8_t* src[3];   // device CurrFrame
  uint8_t* dst[3];         // device CdefFrame (must not alias src)
  int stride[3];
  int mi_cols, mi_rows;
  int subsampling_x, subsampling_y;          // 1,1
  const uint8_t* skip;                       // device
  const uint8_t* lossless;                   // device
  const uint8_t* lossless_chroma;            // device, optional
  const int8_t* cdef_idx;                    // device
  int n64_cols, n64_rows;                    // 0 derives ceil(mi/16)
  const uint16_t* tile_col_starts; int num_tile_cols;   // device
  const uint16_t* tile_row_starts; int num_tile_rows;   // device
};

cudaError_t cdef_frame(const CdefDeviceFrame&,
                       const av2enc::cdef::CdefOptions&,
                       cudaStream_t stream = nullptr);
}
```

Metadata and tile arrays must already be device-resident. The options struct is
the same POD used by the CPU (`damping`, `num_strengths`, the strength array,
`on_skip_txfm`, `disable_loopfilters_across_tiles`, `bit_depth`).

Validation returns `cudaErrorInvalidValue` for bad geometry, null or aliasing
pointers, short strides, odd MI grids, unsupported bit depth/subsampling,
out-of-range options, and missing tile arrays. Tile and `cdef_idx` contents are
device memory and are not read on the host; the caller must keep them in range.
Launch failures are returned from `cudaGetLastError()`.

## Kernel design

One launch per frame; one CUDA block of 64 threads per 8×8 luma CDEF block:

```text
grid  = (mi_cols / 2) * (mi_rows / 2) blocks
block = 64 threads
```

Thread 0 runs the decision path once and publishes it through shared memory:

- `cdef_idx[r & ~15][c & ~15]`;
- the tile/frame filter region (§5.20.9.3);
- the skip and chroma-skip flags (§7.18.1);
- the direction search (§7.18.2) and the derived luma/chroma strengths,
  directions and damping values.

After `__syncthreads()` every thread copies its sample from `src` to `dst`
(the §7.18.1 copy step runs for every block). Threads `0..63` then filter the
8×8 luma block; threads `0..15` filter the 4×4 U and V blocks. Blocks whose
`cdef_idx` is `-1`, whose four MI cells are all skipped, or that are lossless
return after the copy.

CDEF is a pure function of the unfiltered source frame and blocks write
disjoint destination regions, so no inter-block synchronisation is needed; the
single `__syncthreads()` only orders the per-block decision against its own
workers.

The decision arithmetic (tile region, skip/lossless, strength/direction
derivation) comes from the shared `cdef_core.h` helpers
(`cdef_filter_region`, `cdef_block_skip_flags`, `cdef_luma_params`,
`cdef_chroma_params`), the same ones the CPU reference and the encoder-side
strength search (`src/encoder/filters/cdef/cdef_search.h`) call.

ptxas on `sm_86`: 184 registers, 64-byte stack frame (direction projections;
held in local memory, not the register file), 0 spill stores/loads, 56 bytes
shared memory, 648 bytes constant memory (the normative tables). Block-level
parallelism (`~32k` blocks at 1080p) hides the low per-block thread count; no
shared-memory tiling is required because each block touches only its own
samples plus a ±2-sample halo.

## Build / run

On the 3060 box:

```sh
cd src/cuda/cdef
./build_and_test.sh                 # sm_86, build + run
./build_and_test.sh --no-run        # build only
ARCH=sm_120 ./build_and_test.sh --no-run
CDEF_TEST_QUICK=1 ./build-cuda/cdef_test /workspace/av2enc
```

The script compiles the CPU reference with the host compiler, compiles
`cdef.cu` + `cdef_test.cu` with nvcc, and links them. It does not touch the root
`CMakeLists.txt` or any other component.

## Tests and bit-exactness

`cdef_test.cu` runs the CPU reference and the GPU kernel on the same host
buffers and compares **every byte of every plane, including guard columns and
rows**:

1. 11 synthetic patterns × 3 sizes (flat, vertical/horizontal/diagonal steps,
   ramps, checker, alternating, noise, soft edge, low-amplitude ripple):
   exact.
2. 40 random frames (4×4 … 82×50 luma MI) with random strengths, damping,
   tiles, skip/lossless cells and `cdef_idx`: exact on every frame; 51,047
   changed samples exercised in the full run.
3. Real reconstructed frame `decoded.y4m` 640×360: exact (181,145 samples
   changed, mean |delta| 1.0830, max 4); guards intact.
4. Validation cases return `cudaErrorInvalidValue`.
5. Timing (CUDA events, 100 iterations after warm-up).

Measured on RTX 3060 12 GB, CUDA 12.8, `sm_86` (the 8 GB `compat8g` target runs
the same kernel; no large allocations are involved):

```text
checks: 2089, failures: 0
nvcc warnings (-Wall -Wextra -Wconversion -Wshadow): 0
compute-sanitizer --tool memcheck (CDEF_TEST_QUICK=1): 0 errors
```

## Timing (measured)

One kernel launch per frame, CUDA events over 100 iterations, 640×360 real
reconstruction and a synthetic 1920×1080 frame:

```text
GPU 640x360  : ~0.260–0.285 ms/frame
GPU 1920x1080: ~2.52–2.70 ms/frame
```

For comparison, the CPU reference (correctness oracle, not a performance
target) is ~9–12 ms/frame at 640×360 and ~79–94 ms/frame at 1080p on the Mac
host, and ~35 ms / ~345 ms on the shared Vast.ai Xeon. The GPU kernel is well
inside the 1 fps 1080p budget; CDEF is not a speed risk for the current
milestone. `ncu` counters are blocked in the container; timing uses CUDA events
and Nsight Systems can see the single kernel per frame.

## Integration notes

1. **Placement.** After reconstruction and deblock, before CCSO and loop
   restoration. The output is the reference frame for later pictures.
2. **Metadata upload.** Build the skip/lossless grids and the per-64×64
   `cdef_idx` on the host from the partition/segment and strength-search
   decisions, upload once per frame into the persistent arena, and keep them
   device-resident. Do not allocate in the frame loop.
3. **Geometry.** Luma `mi_cols*4` × `mi_rows*4`, chroma `mi_cols*2` ×
   `mi_rows*2`, `mi_cols`/`mi_rows` even. The encoder's 64-aligned
   reconstruction planes satisfy this.
4. **Buffers.** `src` is the unfiltered reconstruction; `dst` is a separate
   CDEF buffer. Do not alias them. After the kernel, `dst` becomes the reference
   frame.
5. **Tiles.** Set `disable_loopfilters_across_tiles` from §5.4.10 and pass the
   device `MiColStarts`/`MiRowStarts`. Devices arrays are not validated
   host-side; the caller must keep them in range.
6. **Scheduling.** One launch per frame can be folded into the per-frame CUDA
   graph. There is no `cudaDeviceSynchronize`, no `cudaMalloc`, and no CPU
   dependency inside `cdef_frame()`.
7. **Not wired in.** `src/main.cpp`, the root `CMakeLists.txt`, the M0 still
   path and the bitstream modules are untouched. Integration is a separate
   pass.

## Limitations

- 8-bit 4:2:0 only; even MI grid.
- `cdef_idx`, skip and lossless contents are trusted device data (the CPU
  reference validates them; the GPU launch cannot without a host copy).
- No strength search on the device; the caller supplies the parameter set.
- The kernel intentionally uses the same scalar arithmetic as the CPU. It is
  launch/memory bound at these sizes; vectorisation was not necessary.
