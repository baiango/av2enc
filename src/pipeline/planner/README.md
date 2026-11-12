# Frame pipeline: rate control + tile geometry + per-tile QP + GPU assignment

Status: **planning/integration component.** It produces decisions, not bits. It
does not write an OBU, call the entropy coder, allocate device memory, or change
any pinned stream. The M1 writer still emits one tile with
`segmentation_enabled = 0`, `delta_q_present = 0` and `base_q_idx = 90`; wiring
these decisions into a writer is described under
[Integration points](#integration-points).

This directory composes the three existing owners instead of duplicating them:

| Owner | What is consumed |
|---|---|
| `src/encoder/rc/` | CQP/CRF/VBR `QpDecision`, QP slices, segmentation params, per-tile delta-Q plans |
| `src/encoder/tiles/` | `TileLayout` from `plan_tile_layout()`, tile rectangles, tile-group sizing |
| `src/multi_gpu/` | deterministic LPT/contiguous assignment, reference duplication, host transfer, per-GPU VRAM |
| `src/cuda/pipeline/` | device-sized arena policy and the per-GPU dispatch PODs |

No AVM/SVT/clave/av2-turbo encoder code was consulted. Normative references are
the AV2 v1.0.0 spec sections cited by the owners: §5.18.6.1 (`base_q_idx`),
§5.18.7.2–7 (`tile_info`), §5.18.7.8 (`delta_q_params`), §5.19/§5.20.1 (tile
group payload), §5.20.2.1 (`CurrentQIndex = base_q_idx` per tile), §7.14.2
(`get_qindex`), §9.2 (level tile limits).

## API

```cpp
#include "encoder/pipeline/pipeline.h"
namespace p = av2enc::pipeline;

p::PipelineConfig config;
config.width = 1920;
config.height = 1080;
config.gpu.gpu_count = 2;                        // gpu_count lives in the multi-GPU config
config.gpu.reference_frames = 4;
config.gpu.lookahead_frames = 4;
config.device.total_bytes = 8192ull << 20;       // device-derived arena
config.device.free_bytes  = 7800ull << 20;
config.device.headroom_bytes = 512ull << 20;     // optional --vram-cap: device.cap_bytes
config.min_base_q_idx = 90;                      // current M1 writer: exactly 90
config.max_base_q_idx = 90;
config.rate_control.mode = av2enc::rc::RateControlMode::kCqp;
config.rate_control.cqp.base_q_idx = 90;

p::FrameStats stats;
stats.complexity = {/* width, height, residual_mad, residual_mse, intra_fraction */};
// Optional: one TileStats per tile for complexity-adaptive tile QP.
// stats.tiles[i].complexity = ...;

p::PipelinePlanner planner;
std::string error;
if (!planner.configure(config, &error)) { /* invalid config/layout/VRAM */ }

p::FramePipelinePlan plan;
if (!planner.plan_frame(stats, plan, &error)) { /* abort the encode */ }
// ... encode plan.layout.tile_count() tiles at plan.tiles[i].qindex ...
if (!planner.end_frame(actual_payload_bits, &error)) { /* ... */ }
```

`plan_frame()` opens exactly one rate-control frame; `end_frame(actual_bits)`
closes it and advances the VBR virtual buffer. Pass encoded **payload bits**, not
bytes and not OBU overhead.

For a planning pass or tests, `plan_sequence(config, stats, actual_bits, out)`
runs the loop. An empty `actual_bits` span feeds the model estimate back and is a
deterministic preview, not a substitute for real accounting.

### `FramePipelinePlan`

| Field | Meaning |
|---|---|
| `qp` / `base_q_idx` | `rc::RateController::begin_frame()` decision. `qp.coeff_cdf_qctx` and `qp.qp_slice` come from `base_q_idx` only. `qp.lambda_sse` is the RDO lambda. |
| `layout` | exact `av2enc::tiles::TileLayout`; never recompute starts |
| `tiles[i].rect` | visible source window and superblock span |
| `tiles[i].target_qindex` | policy request |
| `tiles[i].qindex` | reachable effective qindex (equals target unless delta-Q quantises it) |
| `tiles[i].gpu_index` | worker that owns the tile |
| `tiles[i].delta_plan` | tile-local per-superblock delta-Q plan for `kDeltaQ` (empty otherwise); `sb_x`/`sb_y` start at 0 inside the tile |
| `tiles[i].delta_q_reduced` | reduced delta at the tile's first superblock |
| `segmentation`, `segment_map` | frame-global `ALT_Q` plan for `kSegmentation` |
| `delta_q` | `delta_q_present` / `delta_q_res` for `kDeltaQ` |
| `gpu` | copied `multigpu::MultiGpuPlan`: assignment, reference duplication, transfer, per-GPU VRAM |
| `layout_reason` | why this preset was selected (auto mode) |

## Tile layouts

`LayoutPolicy::kExplicit` uses the requested preset and fails if it is not
representable or not decoder-verified. `kAuto` follows the documented per-GPU
defaults (1 → 4x2, 2 → 8x2, 3+ → 16x2 only when allowed, else 8x2) and may fall
back to a smaller representable layout; `layout_reason` records the choice.

`LayoutPolicy::decoder_verified` is the dav2d gate table. 1x1, 2x1, 2x2, 4x2 and
8x2 are on. **16x2 is off by default.** The gate in this directory passes on the
pinned decoder for all six layouts including 16x2 at 1920x1080; flipping
`decoder_verified[5] = true` (or `allow_unverified = true`) enables 16x2. Do not
enable it in a product default before that evidence is reviewed and committed.

## Per-tile QP

`TileQpMode::kUniform` leaves every tile at `base_q_idx` and disables both
signalling mechanisms. This is the only mode the current M1 writer can emit.

`TileQpMode::kDeltaQ` builds one `rc::DeltaQPlan` per tile. The decoder resets
`CurrentQIndex = base_q_idx` at every tile (§5.20.2.1), so the first superblock
codes the whole tile's delta and the remaining steps are zero. `delta_q_res`
quantises the reachable step to `2^res`; `TilePlan::qindex` reports the achieved
value, which can differ from `target_qindex` by up to `2^res - 1`.

`TileQpMode::kSegmentation` assigns each distinct qindex a segment and fills a
frame-global `SegmentationMap` on the 4x4 MI grid. It requires at most 16
distinct tile qindexes (`MAX_SEGMENTS`) and fails explicitly otherwise. It is
mutually exclusive with delta-Q: an `ALT_Q` value would stack on the delta-Q
`CurrentQIndex`.

Both non-uniform modes use the same deterministic policy:

```text
delta = round(complexity_strength * 24 * log2(sigma_tile / sigma_frame))
qindex = clamp(base_q_idx + clamp(delta, -max_abs_delta, +max_abs_delta), 1, 255)
```

Higher activity gets a coarser qindex. This is a **policy model**, not a measured
BD-rate result. With no per-tile `TileStats`, or with `complexity_strength = 0`,
every tile stays at `base_q_idx`.

**Trap:** the coefficient-CDF slice is selected from the frame `base_q_idx`, never
from a tile/delta qindex (`docs/negative-results.md`). `rc::QpDecision` already
enforces that; do not recompute it from `TilePlan::qindex`.

## Per-GPU VRAM and host transfer

`PipelineConfig::device` is the only VRAM authority:

```text
capacity = min(device_total, device_free) - headroom       (headroom default 512 MiB)
capacity = min(capacity, device.cap_bytes)                 (when cap_bytes != 0)
```

There is no fixed 7,500 MB preset. The planner overwrites the multi-GPU
`vram_cap_mb` with this capacity, so `require_vram_fit` rejects a plan when any
worker's live estimate exceeds what the runtime can actually allocate. Unknown
device capacity requires an explicit cap. `plan.gpu.memory` carries the
duplicated reference pools `(G-1) * refs * frame_bytes`; `plan.gpu.transfer`
is the host-transfer model.

`src/cuda/pipeline/` turns the frame plan into one `GpuWorkerPlan` per GPU:

```cpp
#include "cuda/pipeline/pipeline_gpu.h"
av2enc::cuda::pipeline::ArenaBudget budget;
av2enc::cuda::pipeline::compute_arena_budget(total, free, 512ull << 20, cap,
                                             budget, &error);
arena.init(budget.capacity_bytes);          // source/cuda/arena.h
std::vector<av2enc::cuda::pipeline::GpuWorkerPlan> workers;
av2enc::cuda::pipeline::build_gpu_dispatch(plan, budget.capacity_bytes,
                                           workers, &error);
```

Each worker gets its tiles in raster order, the effective qindex, the coded
delta, and the shared arena capacity. It must not recompute tile geometry or
write the frame header.

## Determinism

`FramePipelinePlan` is a pure function of `(config, FrameStats, actual-bits
history)`. No clock, RNG, device query, or hidden host state is read. Tile stats
may be listed raster-order (`tile_index = -1`) or permuted; decisions are keyed
by tile index. Two planners with the same history produce identical tile QPs,
delta plans, segment maps, assignment and VRAM fields.

Build with `-ffp-contract=off` when comparing across compilers. `log2` in the
tile-QP and VBR models can differ by a last ulp across libm implementations; the
final qindex is rounded deterministically.

## Tests

`pipeline_test.cpp` (5,973 checks, 0 failures; ASan+UBSan clean under Apple
clang 21 on the macOS dev host):

- `tile_qindex_delta()` direction/clamping, tile-QP config validation.
- layout policy table, auto per-GPU defaults, 16x2 refusal, small-frame auto
  fallback, explicit invalid layouts.
- 1x1/2x1/2x2/4x2/8x2 geometry on 1080p: tile counts, exact pixel and
  superblock coverage.
- delta-Q tile QP: per-tile targets, achieved qindex, first-step-only plan,
  per-tile `CurrentQIndex` reset, res quantisation, no-signal case when flat.
- segmentation: params, MI map coverage, segment ids, effective qindex,
  disabled-when-flat, >16-segment rejection.
- CQP/CRF/VBR direction, QP slices and lambda, writer-range rejection without
  wedging the controller.
- determinism across repeated runs, permuted stats, and assignment-policy
  changes; QP is independent of assignment.
- device arena formula (device-derived, user cap, unknown-device/cap, invalid
  inputs), per-GPU fit/rejection, reference duplication, dispatch coverage.

Run:

```sh
cd src/encoder/pipeline
./build_and_test.sh
SANITIZE=1 WERROR=1 ./build_and_test.sh
```

## dav2d tiled-still gate

`gate_dav2d_tiled.cpp` / `gate_dav2d_tiled.sh` builds a real multi-tile still
without editing any core file:

1. `plan_tile_layout()` fixes the geometry.
2. Each tile is cropped and encoded with the public
   `av2enc::still::encode_still_picture()` (tile-local prediction, CDF reset,
   context reset).
3. Each tile's entropy payload is extracted from its CLK OBU.
4. The harness writes the tiled `tile_info()`, a single tile-group OBU with
   `tile_start_and_end_present_flag = 0` and **interleaved**
   `le(TileSizeBytes) tile_size_minus_1` / payload pairs in raster order.
5. The stream is decoded with the pinned `dav2d`, and the decoded planes are
   compared byte-for-byte with the assembled per-tile reconstructions.

The harness self-checks its writers first: the 1x1 frame header must equal
`av2enc::still::write_frame_header()` and the `disable_loopfilters_across_tiles = 0`
sequence header must equal `av2enc::still::make_sequence_header_obu()`.

Local result with `f9dafbc` (`tools/bin/dav2d-macos`), 8-bit 4:2:0, QP 90,
deblocking off:

| Source | Layout | Tiles | Result |
|---|---|---:|---|
| `shine_256x256_1f.y4m` | 1x1 | 1 | PASS, 0 mismatches |
| `shine_256x256_1f.y4m` | 2x1 | 2 | PASS, 0 mismatches |
| `shine_256x256_1f.y4m` | 2x2 | 4 | PASS, 0 mismatches |
| `shine_256x256_1f.y4m` | 4x2 | 8 | PASS, 0 mismatches |
| `basic_640x360_1f.y4m` | 2x2 | 4 | PASS, 0 mismatches (bottom strips 192/168) |
| `basic_640x360_1f.y4m` | 4x2 | 8 | PASS, 0 mismatches (192/192/128/128 columns) |
| `shine_1920x1080_10f.y4m` | 4x2 | 8 | PASS, 0 mismatches |
| `shine_1920x1080_10f.y4m` | 8x2 | 16 | PASS, 0 mismatches |
| `shine_1920x1080_10f.y4m` | 16x2 | 32 | PASS, 0 mismatches |

Run:

```sh
./gate_dav2d_tiled.sh ../../../testdata/shine_1920x1080_10f.y4m 8x2
DAV2D=/path/to/dav2d OUT_DIR=/tmp/gate ./gate_dav2d_tiled.sh src.y4m 4x2
```

**Filter-boundary finding.** With M1 deblocking enabled (`--deblock`), the
pinned decoder's output differs by ±1–2 at the *shared* tile edge even though
the gate writes `disable_loopfilters_across_tiles = 1` (256x256 2x2: 40 Y, 19 U,
26 V samples; all within 6 luma samples of an interior edge and zero in the tile
interior). With deblocking off the match is exact. The gate therefore uses
deblock-off for its exactness check and treats deblock-on as edge-tolerant.
Before shipping a filtered multi-tile stream, either disable deblocking for tile
mode or add the coordinator cross-tile filter pass from `docs/multi-gpu.md §3.2`;
do not assume the per-tile reference matches the decoder at a shared edge.

## Integration points

These are the exact places the M1/M2 writer must change. The pipeline deliberately
does not edit them.

### 1. Frame-level QP

- `src/encoder/m1/still_headers.cpp`, `write_frame_header_syntax()`: writes
  `base_q_idx` (§5.18.6.1). Feed `FramePipelinePlan::base_q_idx`.
- `src/encoder/m1/still_encoder.cpp`, `encode_still_picture()`: validates
  `options.base_q_idx` and uses it for quantization, RDO lambda and the block
  `mi.qindex`. Set `StillOptions::base_q_idx` to the same value. **Current code
  rejects anything except 90** (coefficient-CDF qctx 0). Keep
  `PipelineConfig::min_base_q_idx = max_base_q_idx = 90` until qctx 1–3 CDF
  tables and their gate land.
- `src/encoder/m1/still_syntax.cpp`: `init_cdfs(cdfs_)` is the qctx-0 table set. Do
  not select a slice from a delta/tile qindex.
- RDO: pass `qp.lambda_sse` (or `base_q_idx`) into the mode decision. The value
  is only as good as the documented `sse_scale` placeholder.

### 2. Per-tile QP (M2 / a new tiled writer)

- `kDeltaQ`: `write_frame_header_syntax()` must stop hardcoding
  `delta_q_present = 0` and write `delta_q_res`; the block writer must emit
  `read_delta_qindex()` syntax at each tile's first superblock using
  `TilePlan::delta_plan` (`reduced_delta`, `qindex_before/after`). Do not carry
  state across tiles.
- `kSegmentation`: the frame header must write `segmentation_enabled`,
  `update_map`, `temporal_update = 0` and the `ALT_Q` feature data from
  `FramePipelinePlan::segmentation`; the MI walk must code `segment_id` from
  `segment_map` (4x4 grid, chroma co-located).
- Neither path exists in `still_syntax.cpp` today. Both need a decoder gate before
  they can be enabled by default.

### 3. Tile geometry and tile-group emission

- Use `plan.layout` / `plan.tiles[i].rect` for the source window, prediction
  availability, deblock/loop-filter extents, and per-tile block geometry. A
  second geometry computation is a bitstream bug, not a tuning choice.
- Per-tile entropy: `init_symbol` at the tile start, `exit_symbol` at the end,
  fresh CDF copy, `clear_above_context()` at tile start and
  `clear_left_context()` at each superblock row inside the tile
  (§5.20.2.1, §5.20.9.1). Every payload must be ≥ 2 bytes.
- The current `assemble_still_picture()` writes a 1x1 `tile_info` and one
  payload. A tiled writer must:
  1. write `uniform_tile_spacing_flag = 1` and the
     `increment_tile_cols_log2` / `increment_tile_rows_log2` run for
     `layout.cols_log2` / `layout.rows_log2` (bounded by
     `layout.max_log2_tile_*`, starting at the level minimum);
  2. write `tile_size_bytes_minus_1` (2 bits) when `NumTiles > 1`;
  3. write `tile_start_and_end_present_flag = 0` after the frame header when
     `NumTiles > 1`, then byte-align;
  4. write, in raster `TileNum` order, `le(TileSizeBytes) tile_size_minus_1`
     followed immediately by that tile's payload. The last tile's size is
     implied by the OBU length.
- `src/encoder/tiles/tile_group.cpp` returns sizing/offsets, but its
  `obu_payload_bytes` and `tile_payload_offsets` model treats all size fields as
  contiguous (`header + all size fields + all payloads`). §5.20.1 interleaves
  each size field with its payload. This does not affect `tile_size_bytes_for()`,
  but do not use those offsets to place bytes without fixing/reviewing that
  module.
- `FramePlan` in `still_syntax.h` is full-frame geometry with no tile fields. A
  tiled writer needs either a per-tile `FramePlan`/block list or an explicit
  tile origin, and must not start the superblock walk at `(0,0)` for every tile
  while using full-frame `MiCols`.
- `disable_loopfilters_across_tiles`: see the filter-boundary finding above.

### 4. Multi-GPU coordinator

- One coordinator owns `PipelinePlanner`, the frame header and the tile-group
  merge. Workers never run their own rate controller.
- Dispatch `src/cuda/pipeline/build_gpu_dispatch()`; each worker consumes
  `GpuWorkerPlan`. Merge by `TilePlan::index` (raster `TileNum`), never by
  completion order.
- Size each worker's `Arena` from `compute_arena_budget()`; the same capacity is
  already in the plan. Do not pool VRAM across GPUs.
- Reference exchange: gather each tile region once, assemble the full reference
  frame, fill the missing `(G-1)/G` on every GPU. `plan.gpu.memory` carries the
  duplication budget.
- Default CDF policy is per-tile reset with `disable_cdf_update = 1`. Adaptive
  CDFs need the coordinator's first-eight-tile average and a separate gate.

### 5. CLI / media

`PipelineConfig` intentionally has no CLI. The orchestrator maps
`--gpu-count`, `--vram-cap`, `--layout`, and the rate-control mode into the
config. `src/main.cpp` and the root `CMakeLists.txt` are not touched here;
`build_and_test.sh` is the integration build until the orchestrator adds the
sources.

## Known limitations

- The bitrate model, CRF mapping, and tile-QP redistribution are policy
  placeholders. No quality/BD-rate claim is made.
- The current M1 writer supports only `kUniform`, `base_q_idx = 90`, one tile.
  Everything else is planned but not emitted.
- `plan_sequence()` with empty actual bits is a preview, not measured rate
  control.
- The 16x2 gate passed locally but is left off by default pending review.
- Multi-GPU timing/scaling remains the analytic model from `src/multi_gpu/`.
- No 10-bit, 4:2:2, 4:4:4, inter, SVC, or adaptive-CDF path is modelled.
