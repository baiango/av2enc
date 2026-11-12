# CUDA §7.13.3.15 sub-PU grid

GPU implementation of the AV2 v1.0.0 **save subpu size process**
(§7.13.3.15, p. 484) and the `find_tx_size` machinery it calls.

The grid is the metadata §7.17 deblocking reads (`SubPuSize` through
`filt_max_size`, `SubPuColBase`/`SubPuRowBase` through the sub-PU edge test) and
the state the §7.13.3.14 reference-MV path records for TIP/OPFL/refine-MV and
compound sub-PUs. The CPU reference is
`src/encoder/m1/missing_bits/save_subpu.cpp`.

This directory owns only itself. It does not edit the core, the root
`CMakeLists.txt`, `src/main.cpp`, or any other component.

## Files

```text
src/cuda/subpu/
  subpu_device.cuh          shared host/device core: TxSize, Tx_Width/Tx_Height,
                            find_tx_size, SubPuGridView, reset, save_subpu
  subpu.h                   public API, PODs, arena/sizing declarations
  subpu_plan.cpp            host-only sizing, allocate_checked, mark/rewind,
                            deallocate, downloads (no kernels)
  subpu.cu                  kernels + reset_grid / save_batch / save_one /
                            build_tx_table launchers
  subpu_emulation_test.cpp  no-nvcc parity vs the CPU reference (shared core)
  subpu_test.cu             real-GPU parity, arena, two-launch and short tests
  build_and_test.sh         host emulation (no GPU, no nvcc)
  build_gpu_test.sh         nvcc + RTX build/run
```

## Normative behaviour

```text
subX = (plane > 0) ? SubsamplingX : 0
subY = (plane > 0) ? SubsamplingY : 0
subPuSz = find_tx_size(w, h)
lumaRow = y >> (MI_SIZE_LOG2 - subY)
lumaCol = x >> (MI_SIZE_LOG2 - subX)
for (r = 0; r < h >> (MI_SIZE_LOG2 - subY); r++)
  for (c = 0; c < w >> (MI_SIZE_LOG2 - subX); c++) {
    SubPuColBase[plane > 0][lumaRow + r][lumaCol + c] = lumaCol
    SubPuRowBase[plane > 0][lumaRow + r][lumaCol + c] = lumaRow
    SubPuSize[plane > 0][lumaRow + r][lumaCol + c] = subPuSz
  }
```

`SubsamplingX` and `SubsamplingY` are independent; plane 0 forces both to 0.
Both storage planes are indexed on the **full luma MI grid**.

The §7.13.3.14 early return is preserved exactly:

```text
if (w == PuWidth && h == PuHeight) return;   // nothing written
```

It happens after the same argument validation as the CPU reference (plane range,
region positivity, PuWidth/PuHeight non-negative), so e.g. plane 3 with
`w == PuWidth` is still rejected.

`find_tx_size` is the literal linear search over `TX_SIZES_ALL`; odd sizes,
non-TX rectangles, `0x0` and negatives return `kTxInvalid`.

## Memory contract (single arena)

There is **no** `cudaMalloc` / `cudaFree` / `cudaMallocHost` in this component
(the build scripts grep for them at build time). Every device byte comes from
the caller's persistent `av2enc::cuda::Arena`:

```cpp
const subpu::SubPuConfig config{max_jobs, mi_rows, mi_cols};
const std::uint64_t bytes = subpu::required_device_bytes(config);

av2enc::cuda::Arena arena;
arena.init(bytes);                       // one arena for the process

subpu::SubPuBuffers buffers;
std::uint64_t reported = 0;
const auto status =
    subpu::allocate_checked(arena, config, &buffers, &reported, &error);
// status == SubPuStatus::kArenaTooSmall => *reported is the exact size to grow
```

`required_device_bytes()` is the exact, 256-byte-aligned footprint:

```text
2 * ( align(mi_rows*mi_cols * 1)     // SubPuSize
    + align(mi_rows*mi_cols * 4)     // SubPuColBase
    + align(mi_rows*mi_cols * 4))    // SubPuRowBase
+ align(max_jobs * sizeof(SubPuJob))
+ align(max_jobs * sizeof(SubPuResult))
+ align(25 * sizeof(SubPuTxEntry))
```

For the test configuration `max_jobs=1024`, `mi_rows=mi_cols=32`:

```text
required_device_bytes = 59904
```

`allocate_checked()` takes one `Arena::mark()`, bump-allocates exactly that
amount, and `rewind()`s on failure. A one-byte-short arena returns
`SubPuStatus::kArenaTooSmall` with the bump pointer unchanged. `mark()` /
`rewind()` are exposed for callers that want to bracket additional scratch.
`deallocate()` restores the allocation mark.

Launchers allocate nothing. `save_batch()` stages descriptors into the
persistent job array and synchronises on the stream, so two back-to-back
launches leave `used()` and `peak_used()` flat.

## API sketch

```cpp
struct SubPuJob { int plane, x, y, w, h;
                  int subsampling_x, subsampling_y;
                  int pu_width, pu_height; };
struct SubPuResult { int code; };    // one of SubPuCode

SubPuStatus reset_grid(const SubPuBuffers&, cudaStream_t = nullptr);
SubPuStatus save_batch(const SubPuBuffers&, const SubPuJob* host_jobs,
                       int count, SubPuResult* host_results = nullptr,
                       cudaStream_t = nullptr);
SubPuStatus save_one(const SubPuBuffers&, const SubPuJob&,
                     SubPuResult* = nullptr, cudaStream_t = nullptr);
SubPuStatus build_tx_table(const SubPuBuffers&, cudaStream_t = nullptr);
SubPuStatus download_tx_table(const SubPuBuffers&, SubPuTxEntry*, int capacity,
                              cudaStream_t = nullptr);
SubPuStatus download_grid(const SubPuBuffers&, SubPuHostGrid*,
                          cudaStream_t = nullptr);
```

`SubPuResult::code` distinguishes the CPU reference's `false` reasons:
`kSubPuInvalidPlane`, `kSubPuInvalidRegion`, `kSubPuInvalidPuSize`,
`kSubPuNoTxSize`, `kSubPuRegionTooSmall`, `kSubPuOutOfBounds`.

Descriptors in one `save_batch()` must target disjoint MI cells (partition
leaves are disjoint); the kernel does not serialise overlapping writes.
`save_one()` is the safe form for arbitrary single invocations.

## Device-code audit

The module was audited against the three real-nvcc failure classes reported for
other CUDA modules:

1. **Every function reachable from a kernel is `__host__ __device__`.** The
   kernels call only `save_subpu`, `reset_grid_cell` and `make_tx_entry`, all
   `AV2ENC_SUBPU_HD inline`; those call `find_tx_size`, `tx_width`,
   `tx_height`, `grid_slot_ready`, all `AV2ENC_SUBPU_HD constexpr/inline`.
   Host-only helpers (`set_error`, `buffers_ready`, `grid_block_count`) are
   never called from device code.
2. **No namespace-scope host `constexpr`/`const` arrays are used in device
   code.** `Tx_Width`/`Tx_Height` live as *function-local* `constexpr` arrays
   inside the `__host__ __device__` accessors. All scalars used by kernels
   (`kMiSizeLog2`, `kMiSize`, `kTxSizeCount`, `kTxInvalid`,
   `kSubPuPlaneCount`) are unscoped **enumerators**, not variables. The module
   builds with nvcc **without** `--expt-relaxed-constexpr`; a clean sm_86
   build therefore proves no device symbol is missing.
3. **No `std::` containers or algorithms in device code.** The header uses only
   `<cstdint>` types; `<string>`/`<vector>` appear only in host declarations.
4. **No `static` mutable globals** shared between host and device.

## Tests

### Host emulation (`build_and_test.sh`, no GPU/nvcc)

Compiles `subpu_device.cuh` with the host compiler, links
`subpu_plan.cpp` + the real `Arena` against a tiny host CUDA runtime stub, and
compares cell-for-cell with `save_subpu.cpp`.

```text
$ ./build_and_test.sh
exhaustive cases: 14700
early-return cases: 615
rejection cases: 22
checks: 31474, failures: 0
```

### Real GPU (`build_gpu_test.sh`, nvcc + RTX)

```text
$ ./build_gpu_test.sh
device: NVIDIA GeForce RTX 3060 (sm_86)
config: max_jobs=1024 mi=32x32 required_device_bytes=59904
exhaustive cases: 14700, mismatches: 0
early-return cases: 615
rejection cases: 19
small-grid bounds cases: 6
batch jobs: 256, two-launch used/peak flat: 59904/59904
checks: 77986, failures: 0
```

`compute-sanitizer --tool memcheck`: **0 errors**.

Coverage:

- all 25 `TX_SIZES_ALL` shapes × 3 planes × 4 independent `(subX, subY)`
  combinations × 49 origins `(0,4,...,64)`, bit-exact on every cell;
- 615 early-return cases with a sentinel-prefilled grid: nothing written.
  This includes odd/non-TX extents that equal the PU size, proving the early
  return precedes `find_tx_size` exactly like the CPU reference;
- odd / non-TX shapes, invalid plane/origin/size, and bounds rejection with
  the same code and no writes;
- a 4×4 MI grid for exact-fit and just-past-the-edge rejection;
- a 256-job disjoint batch vs the CPU decode-order grid;
- exact allocation, one-byte-short `kArenaTooSmall`, deallocate, and
  two-launch no-growth (`used()` and `peak_used()` unchanged).

## Remote command

The RTX 3060 box was reached with the address from `vastai ssh-url`:

```sh
ssh -p 1531 root@142.127.68.223
cd /workspace/av2enc
./src/cuda/subpu/build_gpu_test.sh
```

From the Mac, syncing only this directory:

```sh
COPYFILE_DISABLE=1 tar czf - -C . src/cuda/subpu |
  ssh -p 1531 root@142.127.68.223 'tar xzf - -C /workspace/av2enc'
```

## Limitations

- The grid has two storage slots (`plane > 0`), matching the CPU reference.
  Callers that need U and V separated must keep separate buffer sets (or map
  the slots themselves); the normative decoder also keys on `plane > 0`.
- `save_batch()` does not serialise overlapping descriptors; use `save_one()`
  or disjoint jobs.
- This is metadata, not pixels: the launchers are sized for frame-level grids
  (4K 4:2:0 is ~9.3 MB of grid arrays) and do not attempt to fuse with the
  deblock kernel.
