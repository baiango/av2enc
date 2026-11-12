# CUDA temporal motion field (§7.9, §7.12.2, §7.22)

GPU port of the temporal motion-field tools in
`src/encoder/inter/temporal/`.  The module is self-contained: it adds no
CMake target and is not wired into M1/M2 yet.

## Scope

| Spec | What runs on the GPU | Entry point |
| --- | --- | --- |
| §7.22 | `mvMethod` 0/1 block storage, 8x8 subsampled-grid writes, TIP candidate path, wedge allow-list, past/future reordering, compression | `store_blocks()` |
| §7.9 | `get_mv_projection`, clamp, sampled positions, block-position checks, projection jobs, trajectory intersection | `project()` |
| §7.9.2 / §7.10.5 | trajectory gap fill (`fill_traj_mv`) | `traj_fill()` |
| §7.12.2.7/.8 | temporal candidate scan / weighted append | `merge_stacks()` |
| §7.12.2.19 | one-move-to-front stack sort | `merge_stacks()` |
| §7.12.2.21 / §5.20.2.2 | bank reset, hit budget, ring update, MVP appends | `apply_bank_ops()`, `merge_stacks()` |

`mvMethod == 1` adds the §7.22 `MvDeltas` term, including the `n == 4`
`Round2Signed(sum, 3)` path and the `Round2Signed(delta, 1)` path.  Warp and
global MVs are supplied by the caller in the block descriptor (`warp_mv`,
`global_mv`) so this module does not duplicate the warp estimator.

Resolution range is unrestricted (any `MiCols`/`MiRows`, odd counts and
edge-overhanging blocks included).  There is no bit-depth dependency: MVs,
order hints and masks are integer metadata.

## Shared device header

`motion_field_device.cuh` is a `__host__ __device__` transcription of:

- `temporal_math.h` (projection, compression, TMVP helpers, sampled position),
- `motion_field.cpp` (`projection()`, `check_traj_intersect()`, `fill_traj_mv`,
  `temporal_sample/scan`, `get_tip_cand`, `store_motion_field_entry`),
- `ref_mv_bank.cpp` (reset/count/update/fill, `rmb` predicates).

It is compiled by nvcc for the kernels, by nvcc host-side in the GPU test and
by the host compiler in the emulation test.  The CPU reference and the GPU
therefore run the same integer arithmetic; only loop placement differs.

Kernel placement follows the spec's ordering constraints:

- §7.9 projection: **one thread per projection job**, jobs launched in process
  order.  The replace rule reads the field written by earlier source cells, so
  a cell-parallel launch would not be bit-exact.
- §7.9 trajectory fill: one thread per `(rf, y8, x8)` lattice anchor.  Anchors
  are read-only and the three interpolated neighbours are not anchors, so the
  CPU order is preserved without races.
- §7.22 storage: an 8x8 cell can be covered by two odd-origin 4x4 blocks, so
  storage runs two passes: `atomicMax` of the descriptor index per cell, then
  one thread per descriptor writes only the cells it owns (last descriptor
  wins, matching the CPU call order).  The stamp pass applies the same
  `i8/j8 in [0, 7]` predicate as `store_motion_field_entry()`: cells outside
  the 8x8 sub-block index are not written by the CPU helper and must not be
  stamped either, otherwise a >16 MI block would shadow the previous real
  writer.
- §7.12.2 stack generation: one thread per block, per-block stack state.
  Temporal sampling is bounds-checked against the 8x8 field extents: an odd
  `MiRows` has a trailing 4x4 row with no field cell, and reading it would be
  one row past the device allocation.  The test zero-pads its CPU field planes
  for the same row so the oracle reads a defined value.
- §7.12.2.21 bank update/reset: one thread total, normative ring order.

## Memory contract (single arena)

- All device memory comes from `av2enc::cuda::Arena`.  The module contains no
  `cudaMalloc`/`cudaFree`/`cudaMallocHost`.
- `required_device_bytes(MfConfig)` returns the exact persistent need,
  including 256-byte alignment padding.  Example values:

  | resolution | `mi_cols x mi_rows` | refs | `max_blocks` | required bytes |
  | --- | --- | ---: | ---: | ---: |
  | 64x64 | 16x16 | 2 | 300 | 159,232 |
  | 640x360 | 160x90 | 2 | 14,408 | 4,037,120 |
  | 1920x1080 | 480x270 | 2 | 129,608 | 35,688,192 |
  | 1920x1080 | 480x270 | 7 | 129,608 | 39,576,064 |

- `allocate_checked(arena, config, &buffers, &required, &error)` allocates
  once.  If the arena is short it returns `MfStatus::kArenaTooSmall`, writes
  the exact need to `required`, and leaves the bump pointer untouched.
- `motion_field_mark()` / `motion_field_rewind()` expose the arena watermark.
- Launchers allocate nothing, copy nothing and never synchronise.  The only
  copies are the explicit `upload_*` setup calls.
- `zero_field()` reproduces the CPU driver's initialisation (field and, when
  configured, the trajectory arrays).

## Tests

`./build_and_test.sh` — host emulation, no GPU/nvcc.  Compiles the shared
header with a CUDA stub and compares against the CPU sources.

`./build_gpu_test.sh` — real GPU test (`sm_86` default).  Uses one arena and
checks:

- storage parity over randomized block partitions (`mvMethod` 0/1, TIP, warp,
  global, wedge/hole masks, edge overhang),
- projection and trajectory parity over 4 hole patterns, `ProjStep` 1/2,
  64x64/128x128 SB, reference distances 0..6 and `RESTRICTED_OH`,
- bank update/reset/count and MVP appends,
- merge-stack scan + bank appends + §7.12.2.19 sort,
- resolutions 4x4 to 1920x1080,
- `required_device_bytes`/`allocate_checked`, one-byte-short
  `kArenaTooSmall` with the exact size and an unchanged bump pointer,
- two consecutive launches without growth in `used()` or `peak_used()`,
- first-differing-cell diagnostics for storage (cell, list/comp, expected, got,
  owning descriptor) on failure;
- `--sanitize` runs the test under `compute-sanitizer --tool memcheck`.
- arena returns to zero after every case (`mark`/`rewind`).

`--quick` skips the 1080p case.

Host-emulation result on this tree: **2,101,875 checks, 0 failures**.

## Device-symbol audit

The host-emulation test cannot detect a host-only symbol referenced from
device code, so the shared header was audited separately (no `nvcc` on the
authoring host):

1. Every free function in `motion_field_device.cuh` is `AV2ENC_MF_HD`
   (`__host__ __device__ __forceinline__` under nvcc).  A scan of all
   definitions found **0** unannotated functions.  `min/max` are local
   `mf_min`/`mf_max`, not `std::min`/`std::max`.
2. The two runtime-indexed namespace-scope tables used by the CPU headers
   (`Div_Mult`, `Tip_Weighting_Factor`) are **not** referenced from device
   code: they are exposed as `mf_div_mult()` and `mf_tip_weighting_factor()`
   switch functions.  The 17 scalar §9.2 constants (`kRefNone`, `kMvLow`,
   `kMaxRefMvStackSize`, ...) are likewise exposed as `AV2ENC_MF_HD constexpr`
   accessors (`mfkRefNone()`, ...) with `static_assert`s against the CPU
   header.  A scan of the header found **0** namespace-scope
   `constexpr`/`const`/`static` objects in this module and **0** `t::k*`
   references outside those `static_assert`s.
3. No `std::` container or algorithm is used in device code; the header
   includes only `<cstddef>`, `<cstdint>` and the POD temporal types.
4. No static mutable globals are shared between host and device.

## Remote run

```sh
AV2ENC_HOST=<user>@<host> AV2ENC_PORT=<port> tools/sync_remote.sh
ssh -p <port> <user>@<host> \
  'cd /workspace/av2enc/src/cuda/motion_field && ./build_and_test.sh && ./build_gpu_test.sh'
```
