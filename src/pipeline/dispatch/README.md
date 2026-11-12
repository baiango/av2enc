# Multi-GPU tile planner

Status: **planning model only. No multi-GPU hardware, no measured scaling, no
bitstream.** The encoder CUDA path is not wired yet, the single-GPU 1 fps
denominator is a product target rather than a measurement, and every timing or
scaling number in this component is analytic. Treat it as a design contract,
not a result.

The module owns only deterministic tile assignment, reference-duplication
accounting, host-transfer planning, and per-GPU `compat8g` VRAM estimates. It
does not create CUDA contexts, streams, events, or allocations, and it does not
write tile payloads or OBUs.

## Build and test

```sh
cd src/multi_gpu
./build_and_test.sh
SANITIZE=1 ./build_and_test.sh   # ASan + UBSan
WERROR=1 ./build_and_test.sh
```

The script compiles `planner.cpp`, `planner_test.cpp`, and the shared tile
geometry from `../encoder/tiles/tile_layout.cpp`. No CMake, no CUDA. It uses
`CXX` if set, otherwise `c++`, with the project warning set.

## API

```cpp
#include "src/multi_gpu/planner.h"
namespace mg = av2enc::multigpu;

const auto layout_result = av2enc::tiles::plan_tile_layout(1920, 1080, 8, 2);
mg::PlannerConfig config;
config.gpu_count = 2;                 // 1, 2, or 4 in the current target
config.assignment = mg::AssignmentPolicy::kBalancedLpt;
config.vram_cap_mb = 7500.0;          // per GPU, not pooled
config.reference_frames = 4;
config.lookahead_frames = 4;
config.adaptive_cdf = false;          // default bitstream is disable_cdf_update=1

mg::MultiGpuPlan plan;
std::string error;
if (!mg::plan_multi_gpu(layout_result.layout, config, plan, error)) {
  // invalid geometry/config, or fewer tiles than GPUs
}
for (const mg::GpuPlan& gpu : plan.gpus) {
  // gpu.tile_indices, gpu.tile_count, gpu.vram.live_mb, gpu.vram.fits
}
```

Lower-level entry points:

- `assign_tiles(layout, gpu_count, policy, assignment, error)` — assignment and
  load metrics only.
- `default_layout_for_gpus(gpu_count)` — 1 -> 4x2, 2 -> 8x2, 3+ -> 16x2.
- `frame_memory(layout, gpu_count, refs, bit_depth)` — frame size and
  duplicated reference pools.
- `estimate_vram(...)` — `persistent + scratch + candidate + headroom`.
- `plan_host_transfer(...)` — source upload, payload D2H, recon gather,
  reference fill, optional CDF gather.
- `pcie_effective_gbps(gen, width, efficiency)`.
- `estimate_scaling_model(...)` — analytic prior; `measured == false`.

## Deterministic assignment

The default is **LPT on tile pixel area**:

1. sort tiles by descending pixel area, then ascending tile index;
2. assign each tile to the GPU with the lowest current load, ties to the lowest
   GPU index;
3. sort each GPU's tile list in raster order.

This reproduces the documented default sets:

| Frame | GPUs | Layout | Assignment |
|---|---:|---|---|
| 1920x1080 | 1 | 4x2 | all 8 tiles |
| 1920x1080 | 2 | 8x2 | `0,2,4,6,8,10,12,14` / `1,3,5,7,9,11,13,15` |
| 1920x1080 | 4 | 16x2 | `4,8,12,14,16,20,24,28` / `5,9,13,15,17,21,25,29` / `0,2,6,10,18,22,26,30` / `1,3,7,11,19,23,27,31` |
| 3840x2160 | 2 | 8x2 | alternating columns, same as 1080p |
| 3840x2160 | 4 | 16x2 | `0,4,8,12,16,20,24,28` / `1,5,9,13,17,21,25,29` / `2,6,10,14,18,22,26,30` / `3,7,11,15,19,23,27,31` |

The 1080p 4-GPU sets look irregular because the taller bottom-row tiles are
placed first and then balanced. That is intentional and deterministic. The
bitstream merge order is always raster `TileNum`, so a GPU's set does not need
to be contiguous.

`AssignmentPolicy::kContiguousRanges` computes an optimal contiguous raster
partition by dynamic programming (minimise the maximum tile-area load). It is
supported up to 512 tiles. At 1080p/8x2/2 GPUs it splits at the tile row
boundary and is inherently uneven (512 px vs 568 px), while LPT balances the
rows by alternating columns.

`content_skew` adds a deterministic integer-hash cost prior for the analytic
load-imbalance predictor. It never changes the area-based assignment. It is a
modelling device, not measured content.

## VRAM model

Each GPU owns its own `compat8g` arena; VRAM is not pooled. The estimate follows
`docs/vram.md`:

```text
frame_bytes = width * height * 1.5 * (bit_depth > 8 ? 2 : 1)
persistent  = (refs + lookahead + 2) * frame_bytes
              + max(1 MiB, width * height / 4)
scratch     = local_tiles * (max_local_tile_pixels * 24 B + 2 MiB)
candidate   = frame_bytes * rdo_multiplier
headroom    = 512 MiB
live        = persistent + scratch + candidate + headroom
```

`local_tiles` and `max_local_tile_pixels` are that GPU's tiles, not the frame
total. Model estimates (MiB, refs=4, lookahead=4, RDO 1x, 512 MiB headroom):

| Frame | GPUs | Layout | Persistent | Scratch | Candidate | Live |
|---|---:|---|---:|---:|---:|---:|
| 1080p | 1 | 4x2 | 30.66 | 69.25 | 2.97 | 614.9 |
| 1080p | 2 | 8x2 | 30.66 | 42.62 | 2.97 | 588.2 |
| 1080p | 4 | 16x2 | 30.66 | 29.31 | 2.97 | 574.9 |
| 4K | 1 | 4x2 | 120.63 | 207.25 | 11.87 | 851.7 |
| 4K | 2 | 8x2 | 120.63 | 118.00 | 11.87 | 762.5 |
| 4K | 4 | 16x2 | 120.63 | 67.00 | 11.87 | 711.5 |

These are estimates, not allocations. A plan can be structurally valid with
`vram_fits == false`; callers must reject it. The cap is per GPU and the usual
default is 7,500 MB.

### Duplicated references

Every GPU needs a full reference pool because inter prediction can read any
region. Extra duplication beyond one shared copy is
`(G - 1) * refs * frame_bytes`.

| Frame | Frame size | Refs/GPU | Per-GPU pool | 2-GPU extra | 4-GPU extra |
|---|---:|---:|---:|---:|---:|
| 1080p | 2.97 | 4 | 11.87 | 11.87 | 35.60 |
| 4K | 11.87 | 4 | 47.46 | 47.46 | 142.38 |

All values MiB. This is a persistent VRAM cost; only the newly reconstructed
frame moves per frame.

## Host-transfer plan

Default assumptions: PCIe 4.0 x16, 85% efficiency (26.78 GB/s effective),
1 Mbps payload at 1 fps, no NVLink and no GeForce P2P. Traffic is host-mediated.

| Part | Tile mode |
|---|---|
| Source upload H2D | `G * frame_bytes` (conservative upper bound) |
| Tile payload D2H | encoded payload at the target fps |
| Recon gather D2H | one full frame when `G > 1` |
| Reference fill H2D | `(G - 1) * frame_bytes` when `G > 1` |
| CDF gather D2H | `min(8, tiles) * 46.7 KiB`, only with `adaptive_cdf` |

The default `disable_cdf_update=1` path has no CDF term and no cross-GPU CDF
handoff. At 1 fps the transfer is well under 1% of the frame budget even on a
narrow link; at higher `throughput` rates it becomes material and the real lane
split must be measured.

## Tests

`planner_test.cpp` covers:

- the documented 1080p/4K assignments above for 1/2/4 GPUs;
- full tile coverage, no empty GPU, area imbalance within 2% for LPT;
- contiguous DP expectations, including the inherently uneven 1080p row split;
- the six live-VRAM estimates above within 0.5 MiB, plus the breakdown sum and
  the 7,500 MB cap;
- reference-pool duplication for 2 and 4 GPUs and 10-bit doubling;
- host-transfer parts, phase counts, effective bandwidth, and sub-1% budget;
- analytic-model labelling (`measured == false`) and deterministic repeats;
- rejection of zero GPUs, zero references, 12-bit input, PCIe 2, invalid
  efficiency, invalid serial fraction, odd 4:2:0 dimensions, and fewer tiles
  than GPUs;
- a cap below the estimate being reported as `vram_fits == false`, not silently
  accepted.

All tests pass under ASan+UBSan.

## Integration plan

1. **One coordinator.** The coordinator owns the frame header, QP/quality
   index, rate control, tile-group assembly, and the single per-frame barrier.
   Workers own one GPU each: arena, streams, CUDA graph, tile set, local CDF
   state.
2. **Choose the layout once.** Use `docs/multi-gpu.md` defaults: 1 GPU -> 4x2,
   2 GPUs -> 8x2, 4 GPUs -> 16x2 (only after 16x2 passes `dav2d`). Reject
   `tiles < gpus`; never drop a GPU silently.
3. **Assign, then dispatch.** `plan_multi_gpu()` gives each worker its tile
   indices. Workers must not write a competing frame header or merge order.
4. **Per-GPU arena.** Allocate from the estimate plus headroom under the
   7,500 MB cap. If a GPU does not fit, reduce local tile batch/lookahead/refs,
   or fail with the device id. Do not pool VRAM across GPUs.
5. **One barrier per frame.** Use one CUDA event per GPU per frame plus the
   coordinator barrier, not `cudaDeviceSynchronize`. Measure
   `barrier_wait_ms`, `load_imbalance`, per-GPU stage means, and
   `peak_vram_mb`.
6. **Merge in raster order.** Use `src/encoder/tiles/tile_group.h`; slot each
   payload by `TileNum`. A GPU completion order is not a bitstream order.
7. **Reference exchange.** Gather each GPU's local reconstruction once,
   assemble the full frame on the host, and fill the missing regions on every
   GPU. The reference fill is a hard dependency for the next frame.
8. **Determinism.** Default CDF policy is per-tile reset with
   `disable_cdf_update=1`. One coordinator owns quality decisions. A
   2-GPU throughput-mode stream with the same layout/QP must match the
   1-GPU MD5. `adaptive_cdf` is a separate, unvalidated path with a
   cross-GPU first-8-tile average.
9. **Decoder gate.** `tools/validate_dav2d.sh` on every layout used, including
   16x2, before any scaling claim.

## Pending hardware and disclaimers

**No multi-GPU result exists.** The Vast.ai box and the local dev box each have
one RTX 3060. The development target is N x RTX 3060 12 GB cards, each running
the 7,500 MB `compat8g` cap (8 GB emulation). Final single-GPU performance/QBS
validation still requires an 8,192 MiB enumerated RTX 3060. No NVLink; GeForce
P2P is restricted.

Until the validation sequence in `docs/multi-gpu.md` §10 runs, cite this module
as **design/planning/simulation**. In particular:

- do not claim measured 1.8–1.95x scaling;
- do not claim per-GPU `peak_vram_mb` from these estimates;
- do not compare model timings across GPU models or clocks;
- do not use 16x2 in 4-GPU tile mode until `dav2d` verifies it;
- do not treat 12 GB cards under the cap as 8 GB hardware validation.

Other limitations: homogeneous 3060s assumed; content skew is synthetic;
no thermal/clock model (a throttled GPU is a straggler on every frame); PCIe
lane splitting is a configuration input, not probed; adaptive-CDF averaging
with more than eight tiles is unvalidated.

## CLI host contract (wave 3): `--gpu-plan <N>`

The product CLI exposes the planning contract as a dry run:

```sh
av2enc testdata/shine_1920x1080_10f.y4m --gpu-plan 2
av2enc tests/data/basic_640x360_1f.y4m --gpu-plan 2 --tiles 2x2
```

It reads the Y4M header for geometry/frame rate, picks the layout
(`--tiles CxR` when given, otherwise `default_layout_for_gpus(N)`), prints the
deterministic per-GPU tile sets, the host-transfer estimate and the per-GPU
VRAM estimate, and exits without writing a stream. `--vram-cap <MiB>` overrides
the planner's default compat8g emulation cap.

The dry run never creates a CUDA context, allocates device memory, queries a
device or claims a multi-GPU result; `measured=0` is printed explicitly.
`--gpu-plan` is rejected when combined with `--video`, `--still`, `--still-gpu`
or `--lossless`, and a grid the planner cannot represent is rejected by name.

The host contract also exposes a per-tile dispatch vector
(`MultiGpuPlan::dispatch`, `TileDispatchEntry`): one entry per tile in raster
order with `raster_slot`, `tile_index`, `gpu_index` and the superblock/pixel
rect. `raster_slot` is the tile-group merge position — workers write into that
slot and the coordinator merges in raster order, never completion order. The
dry run prints it as `dispatch: <slot>:<gpu> ...`.

Every scaling/transfer number carries an `Evidence` flag (`estimated`,
`claimed`, `measured`). The planner only produces `estimated` today; the flag
exists so a caller cannot promote a model number to a result without recording
a matching run.

Gate: `src/pipeline/dispatch/gate_host_plan.sh --bin build-int/av2enc` checks
the documented 1080p 1/2/4-GPU assignments, the per-tile dispatch map, the
640x360 2-GPU default, an explicit `--tiles` override, the
no-output/no-measurement/evidence properties and the rejection paths. Local
result: 21/21 cases passed.
