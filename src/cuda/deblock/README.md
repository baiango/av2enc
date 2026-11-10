# Deblocking filter — CUDA kernel (AV2 §7.17)

CUDA implementation of the M1 deblocking filter.  It shares the exact arithmetic
with the CPU reference through `src/encoder/filters/deblock/deblock_core.h`
(the same `__host__ __device__` inline functions), so the device output is
**bit-exact** with the CPU for identical planes and metadata.

Scope: 8-bit 4:2:0, even luma MI grid (frame padded to a multiple of 8).  No
bitstream integration.  No allocations, no hidden synchronisation, no global
state.

## Files

```text
src/cuda/deblock/
  deblock.h              device frame API
  deblock.cu             launch + validation (2 kernels per frame)
  deblock_test.cu        CPU-vs-GPU bit-exactness, validation, timing
  build_and_test.sh      standalone nvcc build/run (no root CMake changes)
  README.md              this file
```

## API

```cpp
namespace av2enc::cuda {

struct DeblockDeviceFrame {
  uint8_t* plane[3];             // device pointers
  int stride[3];
  const deblock::DeblockMi* mi[3];  // device pointers, mi_cols*mi_rows each
  int mi_cols, mi_rows;
  int subsampling_x, subsampling_y;         // 1,1
  const uint16_t* tile_col_starts; int num_tile_cols;  // device
  const uint16_t* tile_row_starts; int num_tile_rows;  // device
};

cudaError_t deblock_frame(const DeblockDeviceFrame&,
                          const deblock::DeblockOptions&,
                          cudaStream_t stream = nullptr);
}
```

`DeblockDeviceFrame` is the device twin of `DeblockFrame`; metadata and tile
starts must already be device-resident.  The options struct is the same POD
used by the CPU (`apply_deblocking_filter[4]`, tile-disable flag,
`DfDeltaQ[4]`, chroma deltas, `allow_df_sub_pu`, `frame_is_intra`/`enable_sdp`,
`bit_depth`).

Validation returns `cudaErrorInvalidValue` for bad geometry, null pointers,
short strides, odd MI grids, unsupported bit depth/subsampling, or missing tile
arrays.  Tile contents are device memory and are not read on the host; the
caller must keep them in range.  Launch failures are returned from
`cudaGetLastError()`.

## Kernel design

Two launches per frame, one per pass, matching the normative
vertical-before-horizontal order:

```text
launch 1: pass = 0 (vertical boundaries), planes Y/U/V
launch 2: pass = 1 (horizontal boundaries), planes Y/U/V
```

One thread handles one `deblocking_filter_edge(plane, pass, row, col)` call,
i.e. one 4-sample segment of one edge.  A linear index is split across the three
planes using per-plane edge counts; plane 0 iterates every luma MI row/col and
chroma iterates every second row/col, exactly like §7.17.1.

This is safe without inter-thread synchronisation: §7.17.1 notes that any edge
order gives identical results provided vertical edges precede horizontal ones.
Independently, filtering only occurs at transform/sub-PU boundaries, so the
distance between filtered edges is at least the transform size and never less
than the chosen filter reach.  The randomized CPU-vs-GPU test verifies this on
120 frames with mixed block/TX sizes and tile layouts.

No shared memory.  ptxas reports 56 registers, a 152-byte stack frame
(`s`/`t`/`second_deriv` locals), 0 spill stores/loads, and 532 bytes of
constant memory (the normative tables).  256 threads/block; 1080p needs
~260k threads per pass.

## Build / run

On the 3060 box:

```sh
cd src/cuda/deblock
./build_and_test.sh                 # sm_86, build + run
./build_and_test.sh --no-run        # build only
ARCH=sm_120 ./build_and_test.sh --no-run
DEBLOCK_TEST_QUICK=1 ./build-cuda/deblock_test /path/to/repo
```

The script compiles the CPU reference with the host compiler, then compiles
`deblock.cu` + `deblock_test.cu` with nvcc and links them.  It does not touch
the root `CMakeLists.txt` or any other component.

## Tests and bit-exactness

`deblock_test.cu` compares full planes (including guard columns/rows) after a
CPU run and a GPU run:

1. 9 synthetic patterns × 3 sizes (flat, vertical/horizontal step, ramps,
   checker, alternating, noise, soft edge): exact.
2. 120 random frames (sizes 20×16 … 160×90) with random coding/TX blocks,
   chroma metadata on the luma MI grid, random tiles, lossless/skip cells,
   random `apply_deblocking_filter`, `DfDeltaQ`, chroma deltas, sub-PU flag:
   exact on every frame (245,163 changed samples exercised in the last run).
3. Real reconstructed frame `decoded.y4m` 640×360: exact, 2,495 samples changed,
   mean |delta| 0.0075, max 5; a second GPU run is byte-identical.
4. Argument validation cases (null plane, short stride, odd MI grid, missing
   tiles, 4:4:4, 10-bit) return `cudaErrorInvalidValue`.
5. Guard samples outside the MI-aligned plane are never written.

Measured on RTX 3060 12 GB, CUDA 12.8, `sm_86`:

```text
checks: 3853, failures: 0
compute-sanitizer --tool memcheck: 0 errors
nvcc warnings (-Wall -Wextra -Wconversion -Wshadow): 0
```

## Timing (measured)

1080p (1920×1080) synthetic frame, 32×32 coding blocks / 32×32 transforms,
two kernel launches, CUDA events over 200 iterations, four runs:

```text
GPU: 0.228 – 0.233 ms/frame   (~13.3 – 13.6 Gsample/s incl. chroma)
CPU reference: ~11 – 15 ms/frame (correctness oracle, not a perf target)
```

Run-to-run spread on the shared Vast.ai box is a few percent.  This is well
below the per-frame budget; deblock is not a speed risk for the 1 fps 1080p
target.  `ncu` counters are blocked in the container; timing uses CUDA events
and Nsight Systems can see the two kernels per frame.

## Integration notes

1. **Placement.** Run after reconstruction and before CDEF/CCSO/LR; the output
   is the reference frame for later pictures.  Normative order is
   deblock → CDEF → CCSO → loop restoration.
2. **Metadata.** Build `DeblockMi` grids on the host from the partition/TX/
   quant decisions (see the CPU README table), upload once per frame into the
   persistent arena, and keep the buffer device-resident.  Chroma grids are
   indexed on the luma MI grid with the chroma block's entry replicated over
   its luma extent; chroma `tx_size` is the chroma transform size.
3. **Geometry.** Frame planes must be MI-aligned: luma `mi_cols*4` ×
   `mi_rows*4`, chroma `mi_cols*2` × `mi_rows*2`, `mi_cols`/`mi_rows` even.
   The encoder's 64-aligned reconstruction planes satisfy this.
4. **Flags.** `apply_deblocking_filter[0..3]` from §5.18.5.2; M0 regression
   streams have all four cleared and must not call this.  Tile starts come from
   `MiColStarts`/`MiRowStarts`; set `disable_loopfilters_across_tiles` from
   §5.4.10.
5. **Scheduling.** Two launches per frame can be folded into the per-frame
   CUDA graph.  There is no `cudaDeviceSynchronize`, no `cudaMalloc`, and no
   dependency on the CPU inside `deblock_frame()`.
6. **Not wired in.** `src/main.cpp`, the root `CMakeLists.txt`, the M0 still
   path, and the bitstream modules are untouched.  Integration is a separate
   pass.

## Limitations

- 8-bit 4:2:0 only; even MI grid.
- BRU (`use_bru`) is inter-only and not implemented.
- `qindex` is the effective per-cell quantizer index; the caller folds in the
  segmentation delta when `delta_q_present == 0`.
- The kernel intentionally uses the same scalar arithmetic as the CPU; it is
  memory/launch bound, not compute bound, so no vectorisation was necessary.
