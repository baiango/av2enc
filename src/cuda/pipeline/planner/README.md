# CUDA pipeline glue: device-sized arena + per-GPU dispatch

Status: **host-only integration layer.** No `cuda_runtime.h`, no device query,
no allocation, no kernel. It exists so the planner and the CUDA runtime agree on
one arena size and one work split before any device code runs. The existing
`src/cuda/m1/plan/` module is host-only under `src/cuda/` for the same reason.

Files:

| File | Purpose |
|---|---|
| `pipeline_gpu.h` / `.cpp` | `compute_arena_budget()`, `GpuTileWork`, `GpuWorkerPlan`, `build_gpu_dispatch()` |
| `README.md` | this contract |

The standalone tests live in `src/encoder/pipeline/pipeline_test.cpp` and are
built by `src/encoder/pipeline/build_and_test.sh`.

## Arena policy

The project rule is a device-sized arena (`AGENTS.md`, `docs/vram.md`), not a
fixed preset:

```text
capacity = min(device_total, device_free) - headroom      (headroom default 512 MiB)
capacity = min(capacity, user_cap)                        (when user_cap != 0)
```

`compute_arena_budget(total, free, headroom, cap, budget, error)` implements it
and records `device_known` and `capped_by_request`. Rules:

- `total == 0 && free == 0` means "device unknown"; then an explicit `cap` is
  required, otherwise the call fails. The planner never guesses a device size.
- Supplying exactly one of total/free is an error.
- `min(total, free) <= headroom` is an error (nothing usable remains).
- A user cap larger than the device-derived capacity is ignored (the device is
  the real limit). A smaller cap wins and sets `capped_by_request`.
- The result is the value the orchestrator passes to
  `av2enc::cuda::Arena::init()` and to `build_gpu_dispatch()`.

`PipelinePlanner::configure()` calls the same function, overwrites the
multi-GPU planner's `vram_cap_mb`, and rejects the plan when a worker's live
estimate exceeds the capacity.

## Dispatch

```cpp
std::vector<av2enc::cuda::pipeline::GpuWorkerPlan> workers;
std::string error;
if (!av2enc::cuda::pipeline::build_gpu_dispatch(
        plan, budget.capacity_bytes, workers, &error)) { /* abort */ }

for (const auto& worker : workers) {
  // worker.gpu_index, worker.tiles (raster TileNum order),
  // worker.arena_capacity_bytes, worker.vram_live_mb, worker.vram_fits
}
```

`GpuTileWork` carries the validated `TileRect` bounds, superblock span, the
effective `qindex`, the segmentation `alt_q`/`segment_id`, and the
`delta_q_reduced` value to code at the tile's first superblock. The worker must
not recompute geometry or QP, and must not write a frame header.

`build_gpu_dispatch()` rejects:

- an invalid frame plan or an empty arena/tile/GPU list;
- a tile assigned to more than one GPU;
- a tile assigned to no GPU;
- a tile whose plan `gpu_index` disagrees with the assignment;
- out-of-range tile indices.

Every worker's tile list is in ascending `TileNum` order, matching the
bitstream merge order. GPU completion order is irrelevant; slot payloads by
`TileNum`.

## Ownership / wiring

The orchestrator, not this layer, owns:

- calling `Arena::init(budget.capacity_bytes)` per GPU;
- creating streams/events and the one per-frame barrier;
- uploading source regions and downloading tile payloads;
- merging payloads in raster order and writing the OBU.

This layer only makes those steps deterministic and size-consistent.
