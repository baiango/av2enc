# Warp stack — CUDA kernels

CUDA side of `src/encoder/inter/warp_stack/` and the warp half of
`src/encoder/inter/advanced/`. Product target is Linux + CUDA `sm_86` (8 GB
Ampere); `sm_120` targets the RTX 5090. This directory is component code and
does not edit the CPU builder, the root CMake build or the encoder path.

## What is implemented

| Piece | Entry point | Work decomposition |
|---|---|---|
| §7.13.3.19 block warp (8×8 sections) | `predict_launch` / `Workspace::run_predict` | 1 CUDA block per job, 128 threads |
| §7.13.3.20 extended warp (4×4 sections) | same, `path = 2` | 1 CUDA block per job, 128 threads |
| §7.12.2.3/.4 corner points | `corner_launch` | 1 thread per job |
| §7.12.3 find warp samples | `sample_launch` | 1 thread per job |
| §7.13.3.23 least-squares model | `estimate_launch` | 1 thread per job |

* Arbitrary 1..64 sample block extents, 8-bit **and** 10-bit reference planes.
* The block→extended fallback is exactly `adv::warp_needs_extended()`:
  `!setup_shear(params).valid || w < 8 || h < 8 || is_scaled_ref(...)`.
  An explicit block-path job with an invalid shear writes **nothing** and
  reports `sections == 0`, matching `adv::warp_predict()`.
* 6-parameter models are consumed directly. Reduced/clamped models come from
  `reduce_warp_model()` + `get_warp_translation()` (shared `advanced` header);
  corner-derived and least-squares-derived models use the same device code and
  can be fed straight back into `WarpPredictJob::params`.
* The scaled-reference branch of the extended warp is included (shared code
  with the CPU reference), although the M2 baseline does not use it.

The device header is **shared with emulation**:
`warp_stack_device.cuh` is compiled by nvcc for the kernels and by the host
compiler for `warp_stack_host_ref.cpp` / the emulation test. There is no second
transcription of the arithmetic, so parity is structural rather than a
tolerance.

## API

```cpp
namespace av2enc::cuda::warpstack {

struct WarpPredictJob {          // all ref/dst pointers are device pointers
  inter::adv::WarpParams params;
  const void* ref;               // uint8_t* (8-bit) or uint16_t* (10-bit)
  int ref_stride, ref_max_x, ref_max_y;
  inter::adv::WarpBlockGeometry geom;   // geom.bit_depth selects 8/10
  int compound;                  // 0 -> clip to bit depth; 1 -> 1/16 precision
  int path;                      // 0 auto, 1 block, 2 extended
  std::int16_t* dst;
  int dst_stride;
};

struct WarpPredictResult {       // per-job: path/sections/shear_valid/status
  int path, sections, shear_valid, status;
};

struct WarpCornerJob;            // grid + mi_row/col/bw4/bh4 + ref_frame0 + iter
struct WarpCornerResult {        // inter::warp::CornerResult + model_valid
  inter::warp::CornerResult corner;
  int model_valid;
};
struct WarpSampleJob;            // wraps inter::warp::WarpSampleParams
struct WarpEstimateJob;          // WarpSampleSet + mid_x/mid_y + block_mv

Status predict_launch(...);      // raw device-array launchers
Status corner_launch(...);
Status sample_launch(...);
Status estimate_launch(...);

class Workspace {                // persistent arena-backed batch tables
  Status create(Arena&, const WarpStackConfig&, std::string* error = nullptr);
  Status upload_predict/corner/sample/estimate(...);
  Status upload_mi_grid(..., int mi_cols, int mi_rows, ...);
  Status run_predict/corner/sample/estimate(int count, stream);
  Status submit_*  (...);        // upload + launch
  Status download_predict/corner/sample/estimate(...);  // for the CPU builder
};
}
```

`Workspace::create()` is the only place that allocates. Launchers
(`run_*`, `*_launch`) allocate nothing, copy nothing and never synchronise.
Upload/download helpers are separate copies. Use a stream for overlap.

### Stack-builder integration (CPU ordering stays CPU-side)

```cpp
Workspace ws;
ws.create(arena, config);                              // once, persistent
ws.upload_mi_grid(mi_host, mi_cols, mi_rows);          // once per frame
// Capture ws.grid() only AFTER the upload: before it, mi_cols/mi_rows are 0
// and a job built from it describes an empty grid.
for (int i = 0; i < n; ++i) ws.corner_jobs()[i].grid = ws.grid();
ws.submit_corner(corner_jobs, n);                      // batched grid reads
ws.download_corner(corner_results_host, n);
cudaStreamSynchronize(stream);
for (int i = 0; i < n; ++i) {
  if (corner_results_host[i].model_valid)
    inter::warp::insert_warp_candidate(corner_results_host[i].corner.wmmat, stack);
}
// ... scan/sort/clamp and the warp-bank tail stay in warp_stack.cpp ...
```

The sample table is consumed the same way: `submit_sample` fills
`WarpSampleSet[]`; `estimate` turns those into `WarpParams`, or the CPU calls
`estimate_warp_from_samples()` itself. Nothing reorders the stack on the GPU.

## Arena sizing

`required_device_bytes(config)` returns the exact number of bytes
`Workspace::create()` bump-allocates, including:

* prediction jobs + results (batch descriptors),
* corner jobs + `CornerResult` tables,
* sample jobs + `WarpSampleSet` tables,
* estimate jobs + `WarpParams` tables,
* an optional device `WarpMi` grid.

`plan_device_bytes(config)` exposes the per-table breakdown. Counts are
`max_*_jobs`; zero costs nothing. A workspace for the parity test
(2288 prediction jobs, 640 corner/sample jobs, 512 estimate jobs, 24×24 MI
grid) requires **753,152 bytes** (735.5 KiB). The local emulation run measures
`peak_used = 769,536` bytes with a 64 KiB slack arena (`capacity = 818,688`).
The GPU test's peak is about 19.5 MB (18.6 MiB) because it also holds the
2288-job destination slices; production callers only need
`required_device_bytes` plus their own planes.

An undersized arena returns `Status::kArenaExhausted` and
`Workspace::required_bytes()` reports the requirement; the caller grows the
arena and retries. Allocation is `Arena::allocate_array` /
`Arena::allocate_checked` only. There is no `cudaMalloc`, `cudaFree`,
`cudaMallocHost` or `cudaHostAlloc` in this directory; both build scripts grep
for those calls and fail if one appears.

## Tests

```sh
./build_and_test.sh              # nvcc present -> build_gpu_test.sh, else emulation
./build_emulation_test.sh        # no GPU/nvcc needed (macOS)
./build_gpu_test.sh              # sm_86, build + run
ARCH=sm_120 ./build_gpu_test.sh  # secondary target
```

### Emulation parity (local, `build_emulation_test.sh`)

**30,570 checks passed.** It compares the shared device header against:

* `adv::warp_predict` / `adv::warp_predict_ext` for every 8-bit case,
  block/extended/auto paths, both precision modes, **all** 4/8/16/32/64
  square and rectangular sizes, origin and edge positions, chroma subsampling;
* `adv::warp_needs_extended` and `setup_shear().valid` (invalid-shear
  rejection) — the block path writes no section and auto falls back to
  extended;
* `inter::warp::generate_points_from_corners` (800 randomized jobs),
  `find_warp_samples` (800) and `estimate_warp_from_samples` (500), including
  edge cells, top-superblock-row alignment and local-warp neighbours;
* a 10-bit reduction check: when all samples fit in 8 bits, the unclipped
  1/16-precision result is byte-identical to the 8-bit path;
* the real `Arena` with a 256-aligned host `cudaMalloc` shim: exact
  `required_device_bytes` consumption, two mark/rewind epochs with no net
  growth, and a `required - 1` arena returning `kArenaExhausted` with the
  requirement reported.

### GPU parity (`build_gpu_test.sh`)

The "CPU" side is `warp_stack_host_ref.cpp`, i.e. the same
`warp_stack_device.cuh` compiled by the host compiler. The test runs 2288
prediction jobs in one launch (8/10-bit, all sizes, origin/edge, single and
compound), compares every `std::int16_t` destination byte and every
`WarpPredictResult` field, and additionally compares GPU block/extended output
against `adv::warp_predict*` (including the invalid-shear no-write case). It
then runs 600 corner, 600 sample and 400 estimate jobs against the CPU stack
builder. Corner failures print the first eight mismatching jobs with the
differing field (`corners_found`, `all_mvs_same`, `negative_ref_pts`,
`corner_pts`, `corner_mvs`, `ref_pts` or `wmmat.p[i]`), the job geometry and the
grid shape. The emulation suite additionally exercises the exact
`Workspace::upload_mi_grid()` -> `Workspace::grid()` -> job path, so a
zero-sized grid capture fails locally too. All device buffers come from one `Arena`; the lifecycle checks assert
that `run_*` leaves `arena.used()` unchanged, that two mark/rewind epochs add
no net bytes and no new peak, and that a `required - 1` arena fails with
`kArenaExhausted`.

Expected output shape:

```
device: NVIDIA GeForce RTX 3060 (cc 8.6, 28 SMs)
warp_stack: required_device_bytes=753152 peak_used=19524096 capacity=... persistent_after=753152
warp_stack_gpu_test: 210 checks passed
```

## Remote command

From the Mac:

```sh
export AV2ENC_HOST=<user>@<host> AV2ENC_PORT=<port>
./tools/sync_remote.sh
ssh -p "$AV2ENC_PORT" "$AV2ENC_HOST" \
  'cd /workspace/av2enc/src/cuda/warp_stack && ./build_and_test.sh'
```

Or, explicitly on the Linux box:

```sh
cd /workspace/av2enc/src/cuda/warp_stack
./build_gpu_test.sh                 # sm_86
ARCH=sm_120 ./build_gpu_test.sh     # RTX 5090 smoke
```

## Rules followed

* No direct device allocation in this module; single persistent `Arena`.
* No allocation, copy or synchronisation inside a launcher.
* Integer-only arithmetic; no floating point and no `-fmad` sensitivity.
* No Blackwell-only instructions (no TMA, clusters, `wgmma`/`tcgen05`, FP4).
* Stack ordering, candidate insertion, sorting and clamping remain CPU-side.
* 8-bit and 10-bit share one device transcription; 8-bit is cross-checked
  against `advanced`, 10-bit is cross-checked GPU-vs-host on the full
  `uint16_t` range.

## Device-code audit (nvcc host/device boundary)

Host emulation cannot catch a host-only symbol referenced from a kernel, so
the module was audited for the `"undefined in device code"` class of bug:

1. **Call graph.** `predict_kernel`, `corner_kernel`, `sample_kernel` and
   `estimate_kernel` call only `__device__` or `__host__ __device__`
   functions. `ws_read`, the block/extended section helpers,
   `ws_job_valid` and `ws_predict_job_device` are `__device__` (they use
   `__syncthreads()` or are only reached from the kernel); the corner, sample
   and least-squares helpers are `__host__ __device__` because the emulation
   wrappers call them. Every `advanced` helper they reach (`setup_shear`,
   `reduce_warp_model`, `clip3`, `round2`, `motion_vector_scaling`,
   `div_lut_entry`, …) is already `AV2ENC_ADV_HD`.
2. **Namespace-scope arrays.** The only arrays indexed with runtime values in
   device code are the two §9.5 filter tables. They are **not** the host
   `inline constexpr` arrays: `warp_stack_device.cuh` carries
   `static __device__ constexpr` copies in the nvcc device pass
   (`__CUDA_ARCH__`) and `static constexpr` host copies otherwise. A host-only
   `static_assert` compares all 3,982 entries against
   `advanced/tables/warp_filters.h`, so the copies cannot drift. No other
   namespace-scope array is used by a kernel.
3. **Scalars.** Namespace-scope `constexpr` scalars (`kWarpedModelPrecBits`,
   `kMiSize`, `kLsMvMax`, …) are used only as constant expressions, never
   odr-used or address-taken. The repo's existing CUDA modules use the same
   `inline constexpr` scalar pattern; the observed nvcc failures are specific
   to arrays that need storage.
4. **No host-only library in device code.** No `std::` container, algorithm,
   allocation, I/O or virtual call appears in a device function; `std::` is
   used only for integer/pointer type aliases.
5. **No mutable static/global state.** The only statics are the immutable
   constexpr filter tables. Job/result state lives in kernel parameters,
   shared memory, or caller/arena buffers.
