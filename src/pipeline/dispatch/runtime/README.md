# Multi-GPU runtime layer

Status: **implementation + mocked dispatch gate. This environment has no CUDA
runtime or NVIDIA device, so no multi-GPU hardware, bitstream, or scaling claim
is made.** The real CUDA gate is built and run only on a Linux + CUDA host with
`nvcc`; here the CPU-only mock gate is the evidence.

This directory is additive. It consumes `../planner.h`, the tile geometry, and
`../../cuda/arena.*`; it does not edit the planner, M1/M2 core, `src/main.cpp`,
or any `CMakeLists.txt`. Nothing here is wired into the encoder target yet.

## Files

| File | Purpose |
|---|---|
| `status.h` | CUDA-free `Status` enum and names. |
| `device_backend.h` | Device/stream/event/arena/pinned/copy abstraction. No CUDA types. |
| `runtime.h/.cpp` | Device selection, per-GPU contexts/arenas, frame dispatch, merge, reference exchange, fallback. |
| `dispatch.h/.cpp` | Deterministic tile ownership table and raster-order merge. |
| `reference.h/.cpp` | Reference-pool duplication, transfer schedule, tile-region copies. |
| `cuda_backend.h/.cpp` | Optional CUDA implementation of `DeviceBackend`. Compiled only when `nvcc` is present. |
| `gpu_gate.cpp` | Real single/2-GPU plumbing gate with deterministic pattern bytes. |
| `tests/mock_backend.h` | Mock devices, arenas, pinned memory, copies, failure injection. |
| `tests/test_worker.h` | Deterministic `FrameWorker` used by both gates. |
| `tests/runtime_test.cpp` | CPU-only test suite. |
| `build_and_test.sh` | Host build, mock tests, optional CUDA gate. |

## Build and test

```sh
cd src/multi_gpu/runtime
./build_and_test.sh
SANITIZE=1 WERROR=1 ./build_and_test.sh
REQUIRE_GPU=1 ./build_and_test.sh   # fail instead of skipping the real gate
CUDA_ARCH=sm_120 ./build_and_test.sh
```

The host build uses only `c++ -std=c++20`; it compiles `runtime.cpp`,
`dispatch.cpp`, `reference.cpp`, `tests/runtime_test.cpp`, `../planner.cpp`, and
`../../encoder/tiles/tile_layout.cpp`. It does not include CUDA headers.

Current result on macOS (no CUDA): **530 checks, 0 failures**, clean under
ASan+UBSan and `-Werror`. The script then prints an explicit hardware
disclaimer because `nvcc` is absent. The CUDA backend and gate were
syntax-checked against a local CUDA-API stub, but they have not been compiled by
`nvcc` or run on a device in this environment.

## API

```cpp
#include "multi_gpu/runtime/runtime.h"   // or src/multi_gpu/runtime/runtime.h

using namespace av2enc::multigpu::runtime;

RuntimeConfig config;
config.gpu_ordinals = {0, 1};            // physical CUDA ordinals
config.planner.gpu_count = 2;
config.width = 1920;
config.height = 1080;                    // auto layout: 8x2 for 2 GPUs
config.payload_stride_bytes = 1u << 20;  // max compressed bytes per tile
config.allow_drop_missing = false;       // startup-only fallback policies
config.allow_drop_failed = false;
config.require_homogeneous = true;

std::unique_ptr<DeviceBackend> backend =
    make_cuda_backend(&error);           // or a test/mock backend
Runtime runtime;
if (!is_ok(runtime.initialize(*backend, config, &error))) { /* report */ }

// Per frame: the worker writes one payload per local tile and its recon region.
FrameRequest request;
request.frame_index = frame;
request.source = host_source;            // full 4:2:0 frame
request.source_bytes = frame_bytes;
FrameResult result;
if (!is_ok(runtime.dispatch_frame(request, worker, result, &error))) {
  // result.valid == false, result.failed_slot names the GPU, no partial merge
}
// result.tile_payloads is raster TileNum order; result.host_tx_bytes is the
// per-frame source + payload + recon gather + reference fill traffic.
```

Lower-level pieces:

- `compute_arena_capacity(device, headroom, cap)` — device-derived arena size.
- `select_devices_and_plan(backend, config, selection, &error)` — enumeration,
  homogeneity, layout choice, planner call, per-GPU VRAM-fit check.
- `build_dispatch_table(plan, ordinals, &error)` — `by_tile` and `by_gpu` views.
- `merge_payloads(plan, payload_views, &merged, &error)` — strict raster order;
  duplicate, missing, sub-2-byte, or out-of-range payloads abort the frame.
- `plan_reference_schedule(plan, ordinals, payload_bytes_per_gpu)` — deterministic
  source upload, payload D2H, raster recon gather, and per-GPU reference fill
  operations, with host-traffic totals.
- `tile_region_bytes`, `enqueue_tile_region_copy`, `copy_tile_region_host` —
  Y/U/V 4:2:0 region copies with correct plane and row offsets.

### `FrameWorker`

```cpp
class FrameWorker {
 public:
  virtual Status encode(const GpuFrameContext& ctx, std::string* error) = 0;
};
```

`GpuFrameContext` exposes the GPU slot/ordinal, compute and copy streams, the
GPU-local tile list in raster order, device source/payload/recon/reference
pointers, the current reference slot index, the tile layout, and
`tile_sizes_out`. The worker enqueues work on `ctx.compute_stream` and writes
the compressed byte count for local tile `i` into `tile_sizes_out[i]`
(`2..payload_stride_bytes`). It must not synchronize the device or reorder the
tile list. Host buffers used by asynchronous copies must remain valid until the
compute-done event completes; the pattern worker keeps persistent staging for
that reason.

## Per-GPU arena sizing

There is no fixed preset. For every selected device:

```text
usable   = min(total_vram, free_vram)
derived  = floor_to_MiB(usable - 512 MiB)
capacity = vram_cap_bytes == 0 ? derived : floor_to_MiB(min(derived, cap))
```

A capacity below 1 MiB is invalid. The runtime then checks the planner's
per-GPU live estimate (`persistent + scratch + candidate + planner headroom`)
against that device's own capacity; it never uses a global pool and never
silently lowers the cap for one GPU. If `cudaMalloc`-equivalent `arena_init`
fails, the runtime retries with a halved request (`arena_init_attempts`,
default 2), as the single-GPU runtime does. If the shrunken arena no longer fits
the planned live set, or all retries fail, that GPU is failed or dropped by the
startup fallback policy. The arena is allocated from the device-derived
capacity, not from the model estimate; the estimate is a fit check.

## Dispatch and merge

The planner's assignment is converted to:

- `by_tile[TileNum]` → GPU slot + local index,
- `by_gpu[slot]` → raster-sorted tile indices (also copied into a contiguous
  array for `GpuFrameContext`).

Dispatch order is GPU slot order. Completion order is irrelevant. Payloads are
downloaded per GPU, then merged by iterating `TileNum = 0..N-1` and taking the
payload from its owner. `merge_payloads` rejects a duplicate or missing tile; it
never writes a partial frame. This is the same order `tile_group.h` needs.

## Reference duplication and transfer scheduling

Every selected GPU owns `reference_frames * frame_bytes` of persistent VRAM
(planner `FrameMemory`). Per frame:

1. Source upload: `G * frame_bytes` H2D (conservative full-frame mirror, one per
   GPU), double-buffered by frame parity.
2. Payload D2H: one copy per local tile after compute.
3. Recon gather D2H: each GPU's owned tile rectangles only, one frame total.
4. Host assembly: pinned full-frame buffer, one copy per tile rectangle.
5. Reference fill: own tiles D2D from recon into the next reference slot;
   missing tiles H2D from the host assembly. The default `refs=4` slots rotate;
   `refs=1` refills the single slot after compute completes.

`plan_reference_schedule()` reports deterministic ops and byte totals. The
payload rows are optional and the runtime's init-time schedule leaves them zero
because actual compressed sizes are known only after a worker encodes a frame.
D2D bytes are excluded from host traffic. The transfer time is analytic from the
planner's PCIe model, not measured.

## Stream/event overlap

- Two streams per GPU: compute and copy. There is no global copy stream.
- Source upload runs on each GPU's copy stream, records an upload event, and the
  compute stream waits on it.
- `prefetch_source(frame, data, bytes)` enqueues the next frame's H2D into the
  other parity buffer and records the event. `dispatch_frame` then waits on that
  event instead of re-uploading. Only one prefetch may be outstanding.
- Compute records `compute_done`; the copy stream waits on it before payload and
  recon downloads. Each GPU's copy stream overlaps other GPUs' copy streams.
- The reference fill is enqueued after assembly, records `fill_done`, and is
  **not** host-synchronized in `dispatch_frame`. The next frame's compute stream
  waits on `fill_done`, which is the hard dependency. `synchronize()` drains it
  for shutdown/reporting.
- `device_synchronize()` is used only during teardown. The per-frame path uses
  events and one event synchronization per GPU for the payload gather.

## Failure handling

All fallback is startup-only and deterministic in request order.

| Failure | Default | With policy |
|---|---|---|
| Requested ordinal not present | `kDeviceUnavailable` | `allow_drop_missing` drops it and re-plans |
| Query/context/arena failure | status with the device id | `allow_drop_failed` drops it and re-plans |
| Compute capability, SM count, or VRAM class mismatch (5% class band) | `kDeviceMismatch` | `allow_drop_failed` keeps the lowest ordinal and drops later mismatches |
| Per-GPU planned live set does not fit its device-derived arena | `kOutOfMemory` | `allow_drop_failed` drops it and re-plans |
| Per-GPU arena init or allocation fails after retries | `kOutOfMemory` / `kArenaExhausted` | `allow_drop_failed` drops it and re-plans |
| Worker/copy/tile-size failure mid-frame | `kDispatchFailed` | **never** falls back mid-frame; result is invalid and no partial merge is emitted |
| Full-frame pinned staging allocation fails | `kOutOfMemory` | hard failure (host RAM, not attributable to one GPU) |

Dropping below `min_gpus` fails. The runtime never silently shrinks the run and
never emits a partial frame. `Selection::dropped_ordinals`, warnings,
`RuntimeStats::device_drops`, and `RuntimeStats::fallback_replans` record what
happened.

## Test coverage (`tests/runtime_test.cpp`)

- Device-derived arena capacity: no 7,500 MiB preset, explicit cap, cap larger
  than derived, too-small device, tiny cap, 10 GB device.
- 1080p/2-GPU selection: 8x2, 16 tiles, 8 tiles per GPU, planner agreement.
- 1080p/4-GPU selection: 16x2, 32 tiles, 8 tiles per GPU; model-only status.
- Missing GPU: strict failure and `allow_drop_missing` re-plan to 1 GPU/4x2.
- Device mismatch: 12 GB vs 8 GB and CC 8.6 vs 8.9; strict failure and
  deterministic drop.
- Full 2-GPU 640x360 dispatch: source upload, payloads, recon assembly, reference
  fill, host-traffic accounting, per-GPU stream separation, no device-wide sync.
- Frame 1: rotating reference slot, previous-frame reference contents, payload
  and recon patterns, `synchronize()` draining the fill.
- Merge order: reversed completion order still produces raster order; duplicate
  and missing payloads abort without partial output.
- Prefetch: source uploaded once, compute waits the upload event, no duplicate
  source traffic, reference gating still correct.
- Worker failure and invalid tile sizes: whole-frame abort, `failed_slot`, no
  partial merge, recovery at the next frame boundary with all GPUs.
- Copy failure mid-frame: whole-frame abort, no partial merge, error-path
  stream drain, and no mid-stream GPU fallback.
- Arena OOM fallback: drop one GPU and re-plan; strict mode fails with
  `kOutOfMemory` and leaves no arenas/contexts behind.
- Per-GPU allocation exhaustion: drop the offending GPU and re-plan; strict
  mode fails without leaving a half-initialized runtime.
- Arena halving retry: the shrunken arena still covers the planned live set.
- Single-GPU path: 4x2, no H2D reference fill, correct host traffic.
- Region-copy geometry: host assembly and a mock H2D/D2H round-trip for every
  tile rectangle, including 640x360 and 1080p bottom-row/chroma offsets.
- 10-bit frame memory doubles and the tile-region byte sum matches exactly.
- Reference schedule: deterministic op order, bytes, ordinals, D2D/H2D split.
- Dispatch table: full coverage, no duplicates, raster-sorted GPU lists.
- Full-frame pinned staging failure: hard `kOutOfMemory` with arenas, streams,
  and contexts released.
- Shutdown: arenas, arena blocks, pinned allocations, streams, and contexts are
  all released after the worker and runtime go out of scope.

## Real CUDA gate

On the Linux + CUDA host:

```sh
cd src/multi_gpu/runtime
./build_and_test.sh
```

`gpu_gate` selects `{0,1}` when two devices are visible, otherwise `{0}`, and
runs two frames through the real runtime with the pattern worker. It verifies
merged payload bytes, assembled reconstruction bytes, and byte accounting. Exit
codes: `0` pass, `1` failure, `2` usage, `77` no CUDA device/runtime.
`REQUIRE_GPU=1` turns a skip into a failure.

This is a **plumbing gate**, not an encoder gate: no encoder kernels run, no
bitstream is produced, and no `dav2d` validation or scaling measurement is
possible from it. On a one-GPU host it is the single-GPU gate; the mock suite
remains the 2-GPU dispatch gate.

## Blockers and limitations

1. **No CUDA host in this environment.** `cuda_backend.cpp` and `gpu_gate.cpp`
   are written but were not compiled here (no `nvcc`, no NVIDIA device). They
   compile-only on the Linux + CUDA host; if that build reveals an API mismatch,
   fix it there before citing the gate.
2. **No encoder worker is wired.** `FrameWorker` is the integration seam. Until
   a real tile encoder writes payloads/recon, there is no bitstream, no MD5, and
   no `tools/validate_dav2d.sh` result for this layer.
3. **Region copies are row-wise.** Correctness first: a tile rectangle is
   copied per plane per row because `DeviceBackend` has only linear async
   copies. A 2D/pitched copy entry point (one copy per plane per tile) would cut
   the copy count substantially and should be added before performance work.
4. **Host reconstruction assembly is a full-frame copy** at the frame
   boundary. This follows `docs/multi-gpu.md` (the merge is not a hot path at
   1 fps) but is not a throughput implementation.
5. **Startup fallback changes the layout.** With `auto_layout`, dropping from 2
   GPUs to 1 changes 8x2 to 4x2, which changes the tile layout and therefore
   the bitstream. This is explicit in `Selection`/stats, but a caller that needs
   a fixed stream shape must use `auto_layout=false` and pre-validate capacity.
6. **Homogeneity is a proxy.** The check uses compute capability, SM count, and
   a 5% VRAM class band. It does not detect clock/thermal differences; consumer
   cards may still be stragglers.
7. **`payload_stride_bytes` is caller-sized.** The runtime rejects a tile size
   outside `2..stride`, but does not grow the buffer dynamically. The default
   is 1 MiB per tile; a worst-case compressed tile larger than that must raise
   the stride at initialization.
8. **No profiler integration.** The layer exposes `GpuRuntimeInfo` and
   `RuntimeStats`, but does not emit the `multi_gpu` JSON contract from
   `docs/multi-gpu.md` §8. Wiring per-GPU stages and barrier waits is a separate
   step.
9. **The planner remains a model.** `plan.measured` stays false. The runtime
   does not turn analytic transfer or scaling numbers into measurements.

Until the real gate and an encoder worker exist, describe this directory as
**runtime implementation + mocked dispatch gate**, not as a working multi-GPU
encoder.
