# GDF filter — CUDA (§7.20.5)

CUDA implementation of the AV2 GDF filter, the §7.20.1 table-index derivation
and the §5.20.10.3 per-unit `use_gdf` application.  It shares the exact
integer core with the CPU reference (`src/encoder/filters/spec_tables/gdf.cpp`)
through `gdf_device.cuh` and the normative generated tables
(`encoder/filters/spec_tables/gdf_tables.h`), so device output is **bit-exact**
with the CPU for identical planes and parameters.

Scope: luma only (as the specification defines GDF), bit depth 8/10/12,
`GdfBlkSize` 64 and 128 (§5.18.7.9 unit-sized and SB-sized grids) plus smaller
even blocks for testing, and every `refDstIdx` x `qpIdx` combination.  A unit
that spans a 64-line stripe is split into its §7.20.2 stripe regions (stripes
are offset upwards by 8 luma samples) by the grid path.  The filter decision
stays on the CPU; only the pixels move to the GPU.  No bitstream integration.

## Files

```text
src/cuda/gdf/
  gdf_types.h             POD descriptions (source, §7.20.1 inputs, job, grid)
  gdf_device.cuh          shared __host__ __device__ §7.20.1/.2/.5 core
  gdf.h                   host launcher API + Arena requirement
  gdf.cu                  kernels and launchers
  gdf_emulation_test.cpp  host parity test (no GPU, no nvcc)
  gdf_test.cu             real CUDA CPU-vs-GPU parity + arena + timing test
  build_and_test.sh       host-emulation build/run
  build_gpu_test.sh       nvcc build/run (CUDA device required)
  README.md               this file
```

## API

```cpp
namespace av2enc::cuda::gdf {

struct GdfConfig { int max_units = 1; };

std::size_t required_device_bytes(const GdfConfig&);

struct GdfDeviceBuffers {        // persistent, owned by one GdfConfig
  GdfUnitJob* jobs;              // caller-filled batch descriptors (device)
  std::uint8_t* use_gdf;         // §5.20.10.3 decision grid (device)
  GdfUnitJob* grid_jobs;         // scratch: per-region job expansion (device)
  GdfTables tables;              // device copies of the §9.8 tables
  int max_units;                 // job slots
};

Status gdf_buffers_init(Arena&, const GdfConfig&, GdfDeviceBuffers*,
                        std::size_t* required_bytes, std::string* error);

Status gdf_upload_jobs(const GdfDeviceBuffers&, const GdfUnitJob* host_jobs,
                       int count, cudaStream_t, std::string* error);
Status gdf_upload_use_gdf(const GdfDeviceBuffers&, const std::uint8_t* host_mask,
                          int count, cudaStream_t, std::string* error);

Status gdf_derive_indices(const GdfIndexParams* d_params,
                          GdfTableIndices* d_out, int count,
                          cudaStream_t, std::string* error);

Status gdf_apply_units(const GdfFrameSource& src, const GdfUnitJob* d_jobs,
                       int job_count, int max_unit_size,
                       const GdfTables&, cudaStream_t, std::string* error);

Status gdf_apply_grid(const GdfFrameGrid& frame,
                      const GdfDeviceBuffers&, cudaStream_t,
                      std::string* error);
}
```

`GdfFrameSource` is the device twin of §7.20.2's `LumaSource` (CurrFrame,
CdefFrame, visible luma bounds, 64-line stripe bounds).  `GdfUnitJob` carries
the LrFrame plane/stride, the region origin and even extent, `qp_idx`,
`ref_dst_idx`, `pix_scale` (`GdfPixScale`), `bit_depth`, `use_gdf`, and
optional per-job `stripe_start_y`/`stripe_end_y` (negative = use the frame
stripe).  Per-job stripes are what the CPU decision's `GdfStripeRegion` list
needs for units that cross a stripe boundary.
`GdfFrameGrid` is the `GdfBlks` form: unit size, `cols`/`rows`, padded luma
`width`/`height`, `mi_rows` for the last-unit-row clamp, the frame-level
§7.20.1 inputs, and the LrFrame plane.  `GdfTables` is a POD view of device
table pointers filled by `gdf_buffers_init`.

### Arena contract (mandatory)

Every device byte comes from the caller's persistent `av2enc::cuda::Arena`:

- `gdf_buffers_init()` is the only allocating entry point.  It computes
  `required_device_bytes(config)`, takes one `Arena::mark()`, then requests
  every buffer (jobs, mask, grid scratch, six table blocks) with
  `Arena::allocate_checked`.  On any failure it rewinds to the mark, so
  `used()` is unchanged, and returns `Status::kArenaExhausted` with
  `*required_bytes` and the error text equal to the exact requirement.  There
  is no `remaining()` pre-check: 256-byte alignment can make a reservation fail
  after an earlier buffer succeeded, which is exactly the case the rewind
  handles.
- `jobs`, `use_gdf` and `grid_jobs` are allocated once (256-byte aligned, via
  `Arena::allocate_array`) and reused every frame.
- The normative §9.8 tables (alpha, weight, bias, intra/inter error, coords)
  are copied into arena device memory once by `gdf_buffers_init`.  The kernels
  receive a `GdfTables` descriptor of device pointers, so no namespace-scope
  host `constexpr` object is referenced from device code (see the audit below).
- The launchers allocate nothing, copy nothing and never synchronise.  They
  contain only `<<<...>>>` launches plus `cudaGetLastError()`.
- `gdf_upload_jobs` / `gdf_upload_use_gdf` are setup-time H2D copies and are
  not called by the launchers.
- The one-byte-short and two-launch mark/rewind tests live in `gdf_test.cu`.
  The one-byte-short setup fills the arena, rewinds to the exact bump offset
  that leaves `required_device_bytes(config) - 1` bytes (allocations are
  256-byte rounded), and then checks `kArenaExhausted`, the exact reported
  requirement, an empty output struct, and `used()` unchanged.

`required_device_bytes` = 135,680 B of device tables + two aligned
`GdfUnitJob` arrays (batch + grid scratch) + one aligned `uint8_t` mask array:

| max_units (job slots) | required bytes |
|---:|---:|
| 1 | 136,448 |
| 256 | 164,608 |
| 1,024 | 251,392 |
| 2,048 | 367,104 |
| 8,192 | 1,061,376 |
| 65,536 | 7,541,248 |

(`GdfUnitJob` is 56 bytes; each buffer is rounded up to 256 bytes.)  A grid
launch uses `gdf_grid_slots(cols, rows) = cols*rows*3` job slots because each
GdfBlks unit expands into up to three stripe regions.

## Kernel design

One CUDA block per unit (grid: per stripe region), 256 threads:

1. The block computes, per 2x2 output cell, the four 4x4 sums of the
   directional gradients (§7.20.5 `grad_sum`) and the 2-bit direction class.
   Only these sums are kept in shared memory — four `uint16_t` shifted sums and
   one `uint8_t` class per cell.  At `128x128` that is 36,864 bytes, under the
   default 48 KiB dynamic shared-memory limit (no opt-in needed).
2. `__syncthreads()`.
3. One thread per output sample runs the 22 taps (18 spatial + 4 gradient),
   using the cached cell sums for the gradient taps, then writes
   `Clip1(LrFrame + delta)`.

Holding the 4x4 sums (not the full `(w+2)x(h+2)` gradient plane) is exact: the
CPU reference stores the full plane but the class decision and all four
gradient taps read the same 4x4 window at the cell origin.  The emulation test
checks this against the CPU reference for 83M samples.

There is no inter-block ordering requirement: all source reads come from
CurrFrame/CdefFrame, never from LrFrame, and every output sample is written
exactly once.  A `use_gdf == 0` job returns before any work.

`gdf_apply_grid` is two launches on the same stream.  A prepare kernel walks
the GdfBlks grid, reproduces the CPU decision's `GdfUnitRegions` decomposition
(stripe offset -8, last-unit-row clamp to `MiRows-1`), derives §7.20.1
`qpIdx`/`refDstIdx` on the device, applies the CPU `use_gdf` mask and writes up
to three region jobs per enabled unit into `grid_jobs`; the unit kernel then
filters them.  No synchronisation or host round trip is needed between the
launches, and unused slots are skipped.  The region math is checked against an
independent transcription of `GdfUnitRegions` in the emulation test.

## Build and run

Host-emulation parity (no GPU, no nvcc):

```sh
src/cuda/gdf/build_and_test.sh
src/cuda/gdf/build_and_test.sh --asan
```

Real CUDA test (RTX 3060 target):

```sh
src/cuda/gdf/build_gpu_test.sh
ARCH=sm_120 src/cuda/gdf/build_gpu_test.sh --no-run
src/cuda/gdf/build_gpu_test.sh --no-run    # build only
```

Neither script edits the root CMake build or any other component.  The GPU
script compiles the CPU reference with the host compiler, then compiles
`gdf.cu`, `gdf_test.cu` and `src/cuda/arena.cpp` with nvcc and links them.

## Tests

`build_and_test.sh` compiles the shared device core on the host and compares it
with `ApplyGdfFilter` sample for sample:

- §7.20.1 derivation: exhaustive intra/inter x 8/10/12-bit x `qc` 0..3 x
  `base_q_idx` 0..255 plus 4,000 random reference arrays; all six
  `refDstIdx` and all six `qpIdx` outcomes observed.
- §7.20.1/§7.20.2 stripe-region decomposition (`gdf_unit_regions`) against an
  independent transcription of the CPU `GdfUnitRegions`, for 32/64/128 units
  over 128..1088-line frames and 3,000 random layouts; max 3 regions.
- §7.20.2 sample sourcing at plane and stripe boundaries.
- all 36 `refDstIdx` x `qpIdx` combinations x 3 bit depths x `GdfPixScale`
  1..4, and a unit-size sweep 2, 4, 8, 16, 32, 64, 128.
- plane-edge units, stripe boundaries on/around the unit, and 260 random
  blocks with random extents.
- all four direction classes observed.

`build_gpu_test.sh` runs the same matrices as one batched GPU launch per
matrix (plus the grid path), then compares the full LrFrame plane:

- 2,048 §7.20.1 descriptors on the device vs `GdfBlockTableIndices`;
- 432 jobs: 3 bit depths x 6 ref x 6 qp x 4 `GdfPixScale`, twice with
  different plane/stripe extents;
- full-range 10/12-bit sources (72 jobs), 128x128 and 128x64 units (108 jobs),
  300 random blocks with mixed bit depths and 5% disabled by `use_gdf`;
- `GdfBlks` grids 32/64/128 x 8/10/12-bit with all-on, random and all-off
  masks (27 launches), 320x192 frame (last unit row/column clipped) and the
  region decomposition exercised, including 128x128 units that span stripes;
- argument/status validation and the arena one-byte-short path;
- 1080p (1920x1088) grid timing with CUDA events.

### Measured

Host emulation on macOS arm64 (Apple clang 21, `-O2`):

```text
§7.20.1  refDstIdx counts 0..5 = 5027,218,213,642,1106,2938
         qpIdx counts    0..5 = 2015,2136,2549,2528,548,368
§7.20.1 stripe regions: max 3 per unit, exact vs CPU
filter:  82,948,096 samples compared, 0 differed inside units
classes: c0=127067 c1=129827 c2=146388 c3=322240
PASS:    82,982,969 checks, 0 failures
```

`--asan` (AddressSanitizer + UBSan) is also clean.  The host arithmetic avoids
left-shifting negative values (the shift is written as a multiply), so UBSan
does not report the CPU reference's implementation-defined pattern.

The GPU test prints its own sample/check counts, `required_device_bytes`,
arena peak and the 1080p timing line.  Those numbers require the RTX 3060 box;
this host has no CUDA toolchain or GPU, so no GPU measurement is claimed here.

## Device-code audit (host-only symbol check)

Real nvcc builds reject host-only symbols referenced from kernels even when the
host-emulation test passes, so this module was audited for that class of bug:

- Every free function reachable from a `__global__`/`__device__` function is
  `AV2ENC_GDF_HD` (`__host__ __device__ __forceinline__` under nvcc, plain
  `inline` in the host-emulation build): `gdf_alpha`, `gdf_weight`, `gdf_bias`,
  `gdf_intra_error`, `gdf_inter_error`, `gdf_clip3`, `gdf_clip1`,
  `gdf_round2`, `gdf_round2_signed`, `gdf_derive_table_indices`,
  `gdf_source_sample`, `gdf_gradient_sample`, `gdf_cell_sums`,
  `gdf_class_from_sums`, `gdf_taps_from_sums`, `gdf_filter_delta`,
  `gdf_unit_regions`, `gdf_filter_unit`, plus `valid_bit_depth` in `gdf.cu`.
  The only host-only helpers (`valid_source`, `valid_unit_size`,
  `align_up_bytes`, `table_device_bytes`, `set_error`) are called from host
  launchers only.
- `gdf_device.cuh` contains **no namespace-scope objects**.  There are no
  `constexpr`/`const` arrays or scalars in it at all, so nothing can be
  "undefined in device code".  The normative §9.8 data is uploaded into arena
  device memory once and reached through the `GdfTables` descriptor (device
  pointers passed as a kernel argument).  All compile-time bounds are unscoped
  enum constants (`kGdfNum*`, `kGdfDir*`, `kGdfMaxRegionsPerUnit`, ...), which
  need no storage.
- No `std::` container or algorithm is used in device code; only types such as
  `std::uint16_t`/`std::int64_t`/`std::ptrdiff_t`.
- No `static` mutable global is shared between host and device.  The only
  `static` tokens in the module are `static_cast`/`static_assert`-style casts;
  state lives in kernel parameters, shared memory and the arena.

The audit was performed by grep over `gdf_device.cuh` and `gdf.cu`
(`constexpr`, `static`, namespace-scope `const`, and every kernel call graph)
plus a CUDA-mode syntax check of both translation units with the host/device
qualifiers enforced (`__host__` functions cannot be called from `__global__`).

## Integration notes

1. **`GdfBlkSize`.** `gdf_unit_matches_sb_size` is sequence syntax
   (`gdf_stream/gdf_syntax.cpp`, `DeriveGdfBlkSize`).  §5.18.7.9 gives
   `max(Block_Width[SbSize], 128)`, so the result is 128 for SB 128 and for
   SB 64 with the flag clear and tile starts not aligned to 16 MI units; it is
   64 for SB 64 with the flag set, or with the `a & 16` tile case.  The CUDA
   core accepts 64 and 128 (and smaller even blocks for tests).  Pass the
   derived value as `GdfFrameGrid::unit_size` and `cols`/`rows` from
   `GdfUnitCols`/`GdfUnitRows`.
2. **Decision and regions.** Build the `GdfBlks` `use_gdf` grid on the CPU
   (one byte per unit, row-major), copy it once per frame with
   `gdf_upload_use_gdf`, and call `gdf_apply_grid` (it performs the stripe
   decomposition).  If the CPU decision already produced `GdfStripeRegion`s,
   submit them through `gdf_apply_units` with per-job stripe bounds instead.
   Disabled units/regions are skipped on the GPU.
3. **Bit depth.** 8/10/12 are implemented with the spec's own shifts.  v1.0.0
   signalable streams are 8- and 10-bit; 12-bit is kept for the reference and
   future work.
4. **Placement.** Run after CDEF/CCSO and before loop restoration, following
   the §7.20 ordering.  `LrFrame` must not alias CurrFrame/CdefFrame.
5. **Device-code audit.** Every helper reachable from a kernel is
   `AV2ENC_GDF_HD` (`__host__ __device__ __forceinline__`).  The device header
   contains no namespace-scope objects at all: all §9.8 data is reached through
   the `GdfTables` device pointers uploaded once by `gdf_buffers_init`, and all
   compile-time bounds are unscoped enum constants (which need no storage).
   There are no `std::` containers/algorithms and no mutable globals in device
   code, and no host `constexpr`/`const` array or scalar is indexed from a
   kernel.
6. **Not wired in.** `src/main.cpp`, the root `CMakeLists.txt`, the M1/M2
   encode paths and the bitstream modules are untouched.

## Limitations

- Luma only; the specification does not define GDF for chroma.
- The unit grid is expanded by a prepare launch; a 128x128 grid therefore has
  two launches, not one.  Both are allocation-free and synchronisation-free.
- Per-job descriptors are device memory, so the launcher cannot host-validate
  them.  The kernel skips malformed jobs (odd/oversized extents, out-of-range
  `qpIdx`/`refDstIdx`/bit depth/`GdfPixScale`, null LrFrame) instead of
  reading out of bounds.
- Units must not overlap; overlapping units would race on LrFrame writes.  The
  §5.18.7.9 grid is disjoint by construction.
