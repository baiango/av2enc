# Multi-GPU design: tile-parallel and GOP-parallel

Status: **design plus analytic simulation. No multi-GPU result here is measured.**

The encoder CUDA path is not yet wired into the encode path, and there is no
multi-GPU host. The Vast.ai box and the local dev box each have one RTX 3060.
Everything below is a design contract and a planning model; the numbers come
from `bench/multi_gpu_plan/`, not from hardware. Treat the 1.8–1.95× target as a
goal to validate later, never as a result.

The M1+ tools this design relies on — inter frames, loop filters, RDO, and rate
control — are currently **DEFERRED** in `docs/negative-results.md`. This document
designs the multi-GPU layer ahead of them; it is not a claim that those tools or
this scheduler exist.

Hardware assumption for the development target:

- **N× RTX 3060 12 GB**, each running the `compat8g` profile with a **7,500 MB
  (MiB) cap**. This is 8 GB emulation on 12 GB cards, not actual 8 GB hardware.
- No NVLink. GeForce peer-to-peer is restricted, so all inter-GPU data is
  **host-mediated** through pinned memory.
- Final single-GPU performance/QBS validation still requires an RTX 3060
  enumerating **8192 MiB**. The 12 GB cards are a compatibility proxy and a
  multi-GPU development target only.

The two modes share one rule: **each GPU owns its own `compat8g` arena**. VRAM is
not pooled. A 2× 3060 machine has 2 × 7,500 MB of usable encoder budget, not
15,000 MB for one process.

---

## 1. Modes at a glance

| Item | Tile-parallel (default) | GOP-parallel |
|---|---|---|
| Parallel unit | independent tile | closed GOP / segment |
| Per-frame barrier | yes, one per frame | no; segment completion only |
| Reference frames | full frame duplicated on every GPU; new reconstruction distributed each frame | full frame duplicated per GPU, but no cross-GPU reference traffic |
| Bitstream effect | tile boundaries cost bytes; no forced keyframes | extra keyframe at each segment start |
| Scaling risk | straggler tile gates every frame | segment tail and keyframe bitrate penalty |
| Best use | `quality` mode; the 1080p 1 fps floor | `throughput` mode, offline batch, long clips |
| CLI default | `--multi-gpu tile` | opt-in with `--multi-gpu gop` |

The default is **tile + quality**: the aggregate target stays at 1 fps for
1080p, and extra GPUs buy RDO/search/reference quality until it saturates.
`throughput` mode instead pins the single-GPU quality level and raises fps.

---

## 2. Tile-parallel mode (default)

### 2.1 Tile geometry

AV2 uniform tile spacing uses 64×64 luma superblocks. The extra superblocks are
distributed to the **first** tiles, and the last tile extends to the end of the
frame, so the final partial superblock row lands in the bottom tile row. This is
the `uniform_spacing` rule in `docs/tiles.md` §5.18.7.5, not an even pixel split.
It matters for load balance.

| Resolution | Layout | Tiles | Tile columns (SB) | Tile rows (SB) | Tile size (luma px) |
|---|---:|---:|---|---|---|
| 1080p | 4×2 | 8 | 8, 8, 7, 7 | 8, 9 | 512/448 × 512/568 |
| 1080p | 8×2 | 16 | 4, 4, 4, 4, 4, 4, 3, 3 | 8, 9 | 256/192 × 512/568 |
| 1080p | 16×2 | 32 | 2×14, 1×2 | 8, 9 | 128/64 × 512/568 |
| 4K | 4×2 | 8 | 15, 15, 15, 15 | 17, 17 | 960 × 1088/1072 |
| 4K | 8×2 | 16 | 8×4, 7×4 | 17, 17 | 512/448 × 1088/1072 |
| 4K | 16×2 | 32 | 4×12, 3×4 | 17, 17 | 256/192 × 1088/1072 |

Why the rows differ:

- 1080p: `ceil(1080/64) = 17` superblock rows, `floor(1080/64) = 16` full rows.
  With two tile rows the starts are 0 and 8; the top row is 8×64 = 512 px, the
  bottom row is 8×64 + 56 = **568 px**. The bottom row is *taller*.
- 4K: `ceil(2160/64) = 34`, `floor = 33`. With two tile rows the starts are 0
  and 17; the top row is 17×64 = 1088 px, the bottom is 16×64 + 48 = **1072 px**.
  The top row is *taller*.

At 1080p, a naive contiguous two-way split at the row boundary is 240 vs 270 SB
(12.5% area difference). That is why assignment policy, not just tile count,
matters.

### 2.2 Assignment for 1, 2, and 4 GPUs

Default layouts keep **8 tiles per GPU**, matching the 1-GPU `compat8g` 4×2
profile. The entropy stage is one workgroup per tile, so each GPU then sees the
same eight entropy workgroups as the single-GPU baseline. More tiles per GPU
would improve balance but add bitrate overhead; fewer tiles risk an
entropy-latency-bound GPU.

| Resolution | GPUs | Layout | Tiles | Tiles/GPU | Per-GPU area (SB) | Assignment |
|---|---:|---|---:|---|---:|---|
| 1080p | 1 | 4×2 | 8 | 8 | 510 | all tiles |
| 1080p | 2 | 8×2 | 16 | 8 / 8 | 255 / 255 | balanced LPT |
| 1080p | 4 | 16×2 | 32 | 8 / 8 / 8 / 8 | 128 / 128 / 127 / 127 | balanced LPT |
| 4K | 1 | 4×2 | 8 | 8 | 2040 | all tiles |
| 4K | 2 | 8×2 | 16 | 8 / 8 | 1020 / 1020 | balanced LPT |
| 4K | 4 | 16×2 | 32 | 8 / 8 / 8 / 8 | 510 / 510 / 510 / 510 | balanced LPT |

Per-GPU area is nominal SB cells, counting the partial edge row as a full cell.
The planner uses actual pixels for load imbalance, which is why the 1080p
2-GPU area is 255/255 SB nominally but 253.1/253.1 SB in pixel-equivalent terms.

"Balanced LPT" means longest-processing-time-first on **tile pixel area**, with
ties broken by lower tile index and then lower GPU index. It is deterministic.
At 1080p/4K **8×2**, the equal-area tie-break produces alternating columns:

```text
GPU 0: tiles 0, 2, 4, 6, 8, 10, 12, 14   (columns 0,2,4,6 in both rows)
GPU 1: tiles 1, 3, 5, 7, 9, 11, 13, 15   (columns 1,3,5,7 in both rows)
```

At 4 GPUs/16×2 the default LPT sets are:

```text
1080p (32 tiles)                  4K (32 tiles)
GPU 0: 4,8,12,14,16,20,24,28      GPU 0: 0,4,8,12,16,20,24,28
GPU 1: 5,9,13,15,17,21,25,29      GPU 1: 1,5,9,13,17,21,25,29
GPU 2: 0,2,6,10,18,22,26,30       GPU 2: 2,6,10,14,18,22,26,30
GPU 3: 1,3,7,11,19,23,27,31       GPU 3: 3,7,11,15,19,23,27,31
```

The 1080p sets look irregular because the LPT tie-break fills the larger
bottom-row tiles first. That is fine: the sets are deterministic and balanced
by area, and the bitstream order is still 0..31. The planner emits the same
sets in `gpu_tile_indices`.

Together with the 2-GPU alternating-column sets, these are the default
assignments. The important property is the per-GPU area, not adjacency. Tiles
are independent, references are full frames, and the bitstream merge order is
always raster order, so a GPU does **not** need a contiguous tile range.

A contiguous-range policy is also supported by the planner. It is the simpler
implementation if ownership must be one raster range, but it is not free:

| Case | Balanced LPT | Contiguous DP |
|---|---:|---:|
| 1080p, 2 GPUs, 8×2 | 1.917× | 1.868× |
| 4K, 2 GPUs, 8×2 | 1.916× | 1.890× |
| 1080p, 4 GPUs, 16×2 | 3.514× | 3.508× |
| 4K, 4 GPUs, 16×2 | 3.531× | 3.461× |

The 1080p 2-GPU contiguous case splits the taller 568 px bottom row from the
512 px top row, so its area imbalance is 1.059 even with optimal dynamic
programming. Both policies still land inside the 1.8–1.95× target at 2 GPUs in
this model; contiguous is simply less forgiving.

At a fixed QP/quality index, the assignment does not change the bitstream; only
the tile layout and coding decisions do. In `quality` mode, assignment can
affect the adaptive quality index through timing, so compare quality at matched
bitrate rather than expecting identical bytes.

### 2.3 Other layouts

The default is not the only option. The model is nearly flat between adjacent
layouts; the decision should come from measured tile overhead and entropy
occupancy, not from this table alone.

| Resolution | 2 GPUs, 4×2 | 2 GPUs, 8×2 | 4 GPUs, 8×2 | 4 GPUs, 16×2 |
|---|---:|---:|---:|---:|
| 1080p | 1.904× | 1.917× | 3.563× | 3.514× |
| 4K | 1.904× | 1.916× | 3.514× | 3.531× |

Notes:

- 2 GPUs with 4×2 gives each GPU only 4 tiles (4 entropy workgroups). It saves
  tile overhead but risks latency-bound entropy. Use it only if measurement
  shows the SMs stay busy.
- 4 GPUs with 8×2 gives 4 tiles per GPU for the same reason, but less tile
  overhead. At 1080p the model even prefers it slightly. Measure the real
  entropy tail before choosing.
- **16×2 is not yet in the verified `dav2d` set.** The project reports 1×1,
  2×1, 2×2, 4×2, and 8×2 decoding with `dav2d` (`docs/tiles.md`); 16×2 is not
  in that set. Test a 16×2 stream before relying on 4-GPU tile mode. If it
  fails, use 8×2 with 4 tiles/GPU, or GOP mode.

Measured tile byte overhead on one 720p frame at QP 160 (project measurement,
`docs/architecture.md`): 2×1 +1.7%, 2×2 +2.7%, 4×2 +4.7%, 8×2 +5.1%. That is
byte overhead, not full BD-rate. 16×2 is unmeasured; expect it to be at least
8×2 and measure the real clip cost. Tile boundaries are a compression cost, so
do not add tiles just to occupy GPUs.

### 2.4 Per-frame pipeline and barrier

One process owns N worker threads, one CUDA context per GPU. The coordinator
thread owns rate control, the frame header, the OBU/IVF writer, and the pinned
staging buffers. Each worker owns its arena, streams, CUDA graph, tile set, and
local CDF state.

Tile-mode frame:

```text
coordinator: choose QP/quality index (one decision for the frame)
  -> upload source regions to each GPU (double-buffered with the previous frame)
  -> launch each GPU's frame graph for its tile set
  -> worker: intra/inter search -> RDO -> transform/quant -> recon
            -> tile-local filters -> symbol gen -> per-tile range coder
  -> worker: D2H tile payloads
            (first-8-tile CDF states only with --adaptive-cdf)
  -> BARRIER: wait for every worker's per-frame event
  -> coordinator: merge payloads in raster TileNum order; write one OBU
  -> coordinator: assemble the full reconstructed frame from tile regions
  -> coordinator: H2D the missing reference regions to every GPU
  -> coordinator: with --adaptive-cdf, average and broadcast the frame CDF
  -> next frame
```

The barrier is one per frame, not one per tile. Use one CUDA event per GPU per
frame (or a condition variable signalled after the worker's D2H completes), not
`cudaDeviceSynchronize`, which would sync unrelated streams. The profiler's
`end_frame()` synchronization is separate; do not hide one inside the other.

Timing model:

```text
T_gpu(g)   = input_g + compute_g + entropy_g + payload_d2h_g
T_frame    = max_g T_gpu(g) + merge + cdf_gather + ref_sync
barrier_wait(g) = max_h T_gpu(h) - T_gpu(g)
load_imbalance  = max_g T_gpu(g) / mean_g T_gpu(g)
```

The slowest GPU sets the frame. `barrier_wait_ms` is the direct measurement of
the straggler cost; `load_imbalance` normalizes it. The planner's area model is
a predictor, not a substitute for the profiler.

The reference exchange is a hard dependency for the next frame. Each GPU
reconstructs only its tiles, but every GPU needs the **full** reference frame
because inter prediction can point into a region coded by another tile/GPU.
Gather each GPU's reconstructed regions once (one frame total), assemble on the
host, and fill the missing `(G-1)/G` on each GPU. Do not re-upload the entire
reference pool; only the new reconstructed frame moves.

### 2.5 Tile-payload merge

Use one tile-group OBU with all tiles (`tile_start_and_end_present_flag = 0`).
The rules are in `docs/tiles.md` §3, §6, §8; the important coordinator points:

1. **Order is raster TileNum**, independent of GPU completion order. Write each
   payload into a slot indexed by TileNum; never append in completion order.
2. Encode tiles first, record each tile's byte count, then choose
   `TileSizeBytes = max(1, ceil((max_tile_size + 1) / 256))`, clamp to 4. The
   `tile_size_bytes_minus_1` field is 2 bits.
3. Write `tile_size_minus_1` for tiles 0..N-2; the last tile's size is implied
   by the OBU length. Include the per-tile size fields and byte-alignment bits
   in the OBU size (leb128 in Annex B).
4. Each tile payload is byte-aligned by `exit_symbol()` and must be at least
   2 bytes. If a tile is smaller, the frame is invalid; fail, do not pad.
5. One tile is one contiguous byte range. Never split a tile across OBUs.
6. If any worker fails or returns a short payload, do not write a partial frame.
   Abort with a clear error. Do not silently fall back to one GPU mid-stream.
7. Prefer a single pinned assembly buffer sized for the worst-case frame. The
   merge is a copy at 1 fps, not a hot path; correctness first.

The frame header is frame-global: QP, segmentation, filter parameters, and tile
info are decided once by the coordinator. A worker must not write a competing
frame header.

### 2.6 CDF state

**Default: `disable_cdf_update=1`, as in the current encoder and the pinned
MD5 policy (`docs/negative-results.md`).** Frame-level CDFs do not carry across
frames. Each tile still gets a fresh copy of the default tables at
`init_symbol` (§8.2.2), and tiles never chain CDFs. There is no frame-level
averaging and no cross-GPU CDF handoff, so determinism is trivial. Do not enable
adaptive CDFs by default; that changes the pinned MD5s.

**Opt-in `--adaptive-cdf`** (future, separately validated; not required for
multi-GPU). With `single_picture_header_flag = 1`, `enable_avg_cdf = 1` and
`avg_cdf_type = 1`, `exit_symbol()` writes the tile's final CDFs for the first
`min(8, NumTiles)` tiles, and `frame_end_update_cdf()` installs the average
(§8.2.4). That average is the next frame's starting CDF. In multi-GPU tile mode
the first eight tiles can live on several GPUs, so the coordinator must:

- gather the final CDF state for TileNum 0..min(8, NumTiles)-1 from whichever
  GPUs own them, by TileNum order;
- compute the same average the decoder computes, including the last-entry
  scaling rule;
- broadcast the resulting frame-level CDF to every GPU before the next frame.

The state is about 46.7 KiB per tile, so eight tiles are ~374 KiB. It is small
but a real synchronization point; budget it in the barrier, not in the pixel
path. Do not let each GPU average only its own tiles; that produces different
frame CDFs per GPU and different decisions on the next frame. Do not ship this
path until it is validated against the CPU reference and `dav2d`; the default
`disable_cdf_update=1` path is what the current bitstream guarantees.

---

## 3. Determinism

At a fixed quality index, the tile-parallel bitstream must be a function of
(source, QP/quality index, tile layout, coding decisions), not of GPU count,
assignment, or completion order. In `throughput` mode with the same tile layout
and quality index, a
2-GPU merged stream should be byte-identical (MD5) to a 1-GPU stream of the same
frames. Validate that before trusting any scaling number.

### 3.1 Tile independence

- **Per-tile CDF reset.** Each tile copies the frame-level CDF at `init_symbol`.
  Do not share a mutable CDF object between workgroups.
- **Raster tile order.** The merge order is TileNum 0..N-1. A per-tile slot
  array indexed by TileNum is the simplest way to make completion order
  irrelevant.
- **Tile-local prediction contexts.** `is_inside()` is tile-relative
  (§5.20.3.1); `clear_above_context()` runs at tile start and
  `clear_left_context()` at each superblock row inside the tile (§5.20.2.1).
  The encoder must reset its above/left context engines at the same boundaries.
  Do not borrow intra samples or neighbour contexts across a tile edge.
- **Inter prediction is not cross-tile prediction.** Motion compensation reads
  the reference frame, which is global and identical on every GPU. That is
  allowed. "Cross-tile prediction" here means borrowing current-frame samples or
  contexts across tile boundaries, which the spec does not allow and the
  encoder must not do.
- **No shared RDO state.** Candidate caches, rate-control state, and quality
  decisions are coordinator-owned. A per-GPU governor would change decisions
  with assignment.

### 3.2 Loop filters across tiles: off or explicit

Set `disable_loopfilters_across_tiles = 1` (§5.4.10). Then deblocking skips
vertical/horizontal tile edges (§7.17.2, §7.17.4), CDEF/CCSO/GDF block grids
align to tile boundaries (§7.18, §7.19), and loop restoration clamps to the
tile window (§7.20.1). Each GPU filters its own tile regions and the coordinator
can assemble the filtered reference from per-GPU regions. This is the default.

If the quality cost of tile-edge filtering is too high, the explicit path is:

1. gather every tile's **unfiltered** reconstruction to the coordinator;
2. assemble the full frame;
3. run the cross-tile filter pass on one GPU (or split it with halos);
4. broadcast the filtered reference frame to every GPU.

That adds a second barrier and a full-frame exchange per frame. It is legal and
deterministic if the filter pass is deterministic, but it serializes the filter
stage and must be measured against the BD-rate gain. Do not enable cross-tile
filters casually: tile payloads remain independently decodable up to
reconstruction, but the final filtered frame/reference requires the cross-tile
filter pass.

The bit is sequence-level, so the decision applies to the whole stream. Measure
the BD-rate cost of `disable_loopfilters_across_tiles=1` on real clips before
choosing.

### 3.3 Rate control and quality mode

- One coordinator owns the per-frame QP/quality index. Workers do not run
  independent rate controllers.
- CDF policy is global. The default is `disable_cdf_update=1`; there is no
  cross-frame handoff. If `--adaptive-cdf` is enabled, the coordinator owns the
  first-8-tile average and broadcasts it; a per-GPU average is forbidden.
- `throughput` mode pins the 1-GPU quality index. Tile layout, QP, search
  ranges, and reference count stay fixed. This is the mode for bit-exactness and
  scaling measurements.
- `quality` mode holds the aggregate target fps and raises the quality index
  using the extra GPUs' time. Output intentionally differs from a 1-GPU run at
  the same quality index; compare quality at matched bitrate, not MD5.
- If quality mode increases the tile count, charge the tile byte overhead
  against the quality budget. Do not add tiles just to occupy SMs.

### 3.4 What to validate

- 1-GPU 8×2 vs 2-GPU 8×2, `throughput`, same QP/quality index: merged MD5 must
  match.
- 1-GPU 16×2 vs 4-GPU 16×2: same test, after `dav2d` accepts 16×2.
- `dav2d` decode of every layout used, including 16×2.
- CDF average (only with `--adaptive-cdf`): compare the coordinator's computed
  next-frame CDF against the CPU reference and a 1-GPU run. This is the most
  likely source of silent divergence when adaptive CDFs are enabled.
- Tile-edge reconstruction: compare tile-edge pixels with the 1-GPU encode of
  the same tile layout. They must match exactly if decisions match.

---

## 4. Per-GPU VRAM budget

Each GPU gets its own `compat8g` arena and its own **7,500 MB** hard cap. The
cap is a process-level budget, not a hardware partition. The planner's live
estimate uses the `docs/vram.md` split:

```text
persistent = (refs + lookahead + 2) * frame_bytes
             + max(1 MiB, 64 B per 16x16 block)
scratch    = local_tiles * (max_local_tile_pixels * 24 B + 2 MiB)
candidate  = frame_bytes * rdo_multiplier
headroom   = 512 MiB
live       = persistent + scratch + candidate + headroom
```

`local_tiles` is the number of tiles on that GPU, not the frame's tile count.
Splitting the frame across GPUs reduces each GPU's scratch; it does not reduce
the duplicated reference pool.

Model estimates, 8-bit 4:2:0, refs=4, lookahead=4, RDO 1× (MiB):

| Resolution | GPUs | Layout | Tiles/GPU | Persistent | Scratch | Candidate | Headroom | Live | Cap |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1080p | 1 | 4×2 | 8 | 30.66 | 69.25 | 2.97 | 512 | 614.9 | 7500 |
| 1080p | 2 | 8×2 | 8 | 30.66 | 42.62 | 2.97 | 512 | 588.2 | 7500 |
| 1080p | 4 | 16×2 | 8 | 30.66 | 29.31 | 2.97 | 512 | 574.9 | 7500 |
| 4K | 1 | 4×2 | 8 | 120.63 | 207.25 | 11.87 | 512 | 851.8 | 7500 |
| 4K | 2 | 8×2 | 8 | 120.63 | 118.00 | 11.87 | 512 | 762.5 | 7500 |
| 4K | 4 | 16×2 | 8 | 120.63 | 67.00 | 11.87 | 512 | 711.5 | 7500 |

These are model estimates, not allocations. The live set is far below the cap;
the rest is reserve for precompute, larger lookahead/RDO caches, and allocator
fragmentation if a higher-quality configuration is selected. The encoder should
allocate from the estimate plus headroom, not blindly fill 7,500 MB. The
`vram_budget` test's 7,244 MB arena exists to prove the cap, not to define the
working set.

### 4.1 Duplicated references

Every GPU needs a full reference pool because inter prediction can read any
region of the reference frame. `refs=4` means `4 × frame_bytes` per GPU. The
**extra** duplication beyond one shared copy is `(G-1) × refs × frame_bytes`.

| Resolution | Frame | Refs/GPU | Per-GPU pool | 2-GPU total | 2-GPU extra | 4-GPU total | 4-GPU extra |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1080p | 2.97 | 4 | 11.87 | 23.73 | 11.87 | 47.46 | 35.60 |
| 4K | 11.87 | 4 | 47.46 | 94.92 | 47.46 | 189.84 | 142.38 |

All values MiB. Double them for 8 references. This is a VRAM cost, not a
per-frame PCIe cost. The reference pool is persistent; only the newly
reconstructed frame is distributed each frame.

### 4.2 What is not pooled

- A GPU cannot borrow scratch or reference space from another GPU. If a local
  tile batch does not fit, reduce the local tile count, lookahead, or reference
  count, then fail clearly if it still does not fit.
- The 7,500 MB cap applies on every GPU. Do not auto-select `auto12g` because
  the development cards are 12 GB; `compat8g` is the target profile.
- Pinned host staging buffers for payloads/reference assembly are host RAM, not
  VRAM. Size them once, and do not count them against the per-GPU cap.
- If a real 8 GB card is used in a multi-GPU host, the same 7,500 MB cap leaves
  ~500 MB for driver/context; that is tighter than on the 12 GB development
  cards. The multi-GPU dev target is explicitly 12 GB cards under the cap.

---

## 5. Host-mediated PCIe transfers

There is no NVLink and GeForce P2P is restricted. Do not build on
`cudaMemcpyPeerAsync`; probe `cudaDeviceCanAccessPeer` and fall back to host
staging. The coordinator uses pinned buffers and copy engines. D2H and H2D
phases are serialized at the coordinator unless explicitly overlapped with
compute.

Per-frame tile-mode traffic (upper bound for input: one full frame per GPU):

| Phase | 1080p 2 GPU | 4K 2 GPU | 1080p 4 GPU | 4K 4 GPU |
|---|---:|---:|---:|---:|
| Source upload, H2D | 5.93 | 23.73 | 11.87 | 47.46 |
| Tile payloads, D2H | 0.12 | 0.12 | 0.12 | 0.12 |
| Recon gather, D2H | 2.97 | 11.87 | 2.97 | 11.87 |
| Reference fill, H2D | 2.97 | 11.87 | 8.90 | 35.60 |
| CDF gather, D2H (opt-in) | 0.36 | 0.36 | 0.36 | 0.36 |
| **Total MiB/frame** | **12.35** | **47.94** | **24.21** | **95.41** |
| **ms @ PCIe 4.0 ×16** | **0.73** | **2.13** | **1.20** | **3.99** |
| **% of a 1 s frame** | **0.07%** | **0.21%** | **0.12%** | **0.40%** |

Assumptions: 1 Mbps encoded payload at 1 fps (0.12 MiB/frame; scale linearly
with bitrate), 8 × 46.7 KiB CDF states, and 85% link efficiency. The CDF gather
is only nonzero with the opt-in `--adaptive-cdf`; with the default
`disable_cdf_update=1`, subtract 0.36 MiB and the table becomes a conservative
upper bound. The source upload can be reduced to ~one frame total by uploading
only each GPU's tile regions instead of a full frame per GPU; the table is
conservative.

Effective bandwidth and the 4K/4-GPU transfer time:

| Link | Effective GB/s | 4K/4-GPU xfer ms | % of 1 s |
|---|---:|---:|---:|
| PCIe 3.0 ×4 | 3.3 | 30.1 | 3.0% |
| PCIe 3.0 ×8 | 6.7 | 15.1 | 1.5% |
| PCIe 3.0 ×16 | 13.4 | 7.5 | 0.75% |
| PCIe 4.0 ×4 | 6.7 | 15.1 | 1.5% |
| PCIe 4.0 ×8 | 13.4 | 7.5 | 0.75% |
| PCIe 4.0 ×16 | 26.8 | 4.0 | 0.40% |
| PCIe 5.0 ×16 | 53.6 | 2.0 | 0.20% |

At 1 fps, PCIe is negligible even on a ×4 link. At 10 fps in `throughput` mode,
a 4K/4-GPU ×4 link would spend ~30 ms of every 100 ms frame (about 30%); the
link becomes a serious bottleneck. Re-run the planner with the real gen/width before promising
throughput scaling. A consumer board typically splits a ×16 slot into ×8/×8 for
two GPUs and ×4/×4/×4/×4 for four; that is the realistic case to model.

Overlap rules:

- Double-buffer source upload: frame N+1 can upload while frame N computes.
- Payload D2H and recon gather can overlap with the next frame's intra/ME work
  only if the next frame does not need the new reference yet. Inter prediction
  does need it; be conservative at first.
- The reference fill is a hard dependency for the next frame. With
  `--adaptive-cdf`, the CDF gather/average is too; do not hide either behind a
  second barrier.
- Use one copy stream per GPU, not one global copy stream, to avoid serializing
  independent D2H/H2D phases. The coordinator still owns the assembly order.

---

## 6. GOP-parallel mode

### 6.1 Segment model

GOP mode splits the sequence into **closed GOPs**. Each GPU encodes whole GOPs;
there are no references across segment boundaries. Each segment starts with a
keyframe, and the coordinator concatenates the segments in frame order.

- No per-frame barrier; a segment-level completion count is enough.
- No per-frame reference exchange between GPUs. Each GPU keeps its own reference
  pool for its segment.
- Segment assignment: round-robin or LPT by estimated segment cost. Equal-length
  GOPs are easiest to balance; the keyframe is at the same position in every
  segment, so keyframe compute is balanced too.
- The last wave may be partial: with S segments and G GPUs, wall time is
  `ceil(S/G)` segment slots. Utilization is `S / (ceil(S/G) * G)`.

| Segments S | 2 GPUs | 4 GPUs |
|---:|---|---|
| 10 | 5/5 = 100% | 3/3/2/2 = 83% |
| 20 | 10/10 = 100% | 5/5/5/5 = 100% |
| 40 | 20/20 = 100% | 10/10/10/10 = 100% |

Short clips with a small number of long GOPs are the worst case for 4 GPUs.
For a long movie, the tail is amortized.

The simulated GOP model (240 frames, 24-frame GOP, PCIe 4.0 ×16) gives
1.929× at 2 GPUs and 3.098× at 4 GPUs. The 4-GPU number is below the 4-GPU tile
projection because of the 3/3/2/2 tail: 10 segments do not divide by four. With
480 frames/48-frame GOP the speedup is the same 3.098× for the same reason. Do
not read the GOP model as "GOP mode always scales better"; it scales better
only when the segment count is large enough to fill the last wave.

### 6.2 GOP boundary-keyframe bitrate penalty

Let:

- `K` = keyframe bits / average inter-frame bits at the same QP (roughly 4–10
  for a typical encoder; measure it).
- `N` = GOP length in frames.
- `M` = keyframe interval of the single-GPU baseline in frames.

For a closed-GOP segment, average bits/frame (in inter-frame units) is
`(K + N - 1) / N`. The single-GPU baseline is `(K + M - 1) / M`. The relative
penalty is:

```text
penalty = [(K + N - 1) / N] / [(K + M - 1) / M] - 1
```

Relative to an all-inter stream, the simpler form is `(K - 1) / N`.

For `K = 6`, `M = 240`:

| GOP length N | Penalty vs M=240 baseline | Penalty vs all-inter |
|---:|---:|---:|
| 24 | +18.4% | +20.8% |
| 48 | +8.2% | +10.4% |
| 120 | +2.0% | +4.2% |
| 240 | 0.0% | +2.1% |

For `K = 4` the 24-frame penalty is +11.1%; for `K = 10` it is +32.5%. The
penalty scales roughly linearly with `K`.

Important qualifications:

- If the single-GPU baseline already inserts a keyframe every N frames (a
  closed-GOP cadence), GOP-parallel adds **no** bits. The penalty is relative to
  a baseline with a longer keyframe interval or open GOPs. State which baseline
  you are comparing.
- The keyframe also costs encode time. Equal-length GOPs spread that cost
  evenly; uneven segment lengths can create a straggler.
- Segment boundaries reset rate control and lookahead. The first frames of a
  segment can have worse rate control than a continuous stream; the
  keyframe-only formula does not capture that. Measure the real bitrate/quality.
- If the QBS budget is tight, a 2–18% keyframe penalty is not free. Use longer
  GOPs, or prefer tile mode for quality work.

### 6.3 When to use GOP mode

Use GOP mode when:

- `throughput` matters more than the keyframe penalty (offline batch, long
  clips, many segments);
- tile mode cannot fill the GPUs (few tiles, huge frames, or a layout the
  decoder rejects);
- no per-frame synchronization is desired, for example when the coordinator CPU
  is the bottleneck.

Do not use GOP mode as the default for the 1080p 1 fps quality target. Forced
keyframes at every segment boundary are a direct bitrate cost, and the quality
gate compares at matched bitrate.

---

## 7. CLI design

Proposed flags. Only `--gpus`, `--multi-gpu`, and `--scale-mode` are required by
this design; the rest are the minimum needed to make them well-defined.

| Flag | Values / default | Meaning |
|---|---|---|
| `--gpus` | `0` (default), `0,1`, `0,1,2,3` | CUDA device ordinals. One device means single-GPU; the multi-GPU mode is logged as off. |
| `--multi-gpu` | `tile` (default), `gop` | Parallel unit. Ignored with one device. |
| `--scale-mode` | `quality` (default), `throughput` | Spend extra GPUs on quality at the target fps, or on fps at fixed quality. |
| `--tile-layout` | `CxR`, default auto | 1 GPU 4×2, 2 GPUs 8×2, 3–8 GPUs 16×2. Three GPUs need 16×2: 8×2 gives a 5/5/6 split and a poor barrier. |
| `--gop-length` | `120` (proposal) | Closed-GOP segment length for GOP mode. Longer reduces keyframe penalty; shorter improves segment-count load balance. The simulator defaults to 24 so a 240-frame demo has enough segments for 4 GPUs. |
| `--vram-limit-mb` | `7500` | Per-GPU cap. Applies to every device; not a total pool. |
| `--precompute` | `compat8g` | Keep the 8 GB-compatible profile; do not auto-select 12 GB profiles on 12 GB cards. |
| `--target-fps` | `1.0` | Aggregate target in `quality` mode. Not a cap in `throughput` mode. |
| `--target-resolution` | `1080p` | Resolution for the 1 fps floor. |

Examples:

```sh
# 1-GPU baseline at the 1 fps floor
av2enc in.y4m -o out.ivf --gpus 0 --multi-gpu tile --scale-mode quality \
  --tile-layout 4x2 --vram-limit-mb 7500 --target-fps 1.0

# 2 GPUs, tile-parallel, quality mode (the default multi-GPU setup)
av2enc in.y4m -o out.ivf --gpus 0,1 --multi-gpu tile --scale-mode quality \
  --tile-layout 8x2 --vram-limit-mb 7500 --target-fps 1.0

# 2 GPUs, throughput benchmark at a fixed quality level
av2enc in.y4m -o out.ivf --gpus 0,1 --multi-gpu tile --scale-mode throughput \
  --tile-layout 8x2 --vram-limit-mb 7500

# 4 GPUs, tile-parallel
av2enc in.y4m -o out.ivf --gpus 0,1,2,3 --multi-gpu tile --scale-mode quality \
  --tile-layout 16x2 --vram-limit-mb 7500 --target-fps 1.0

# 4 GPUs, GOP-parallel, throughput
av2enc in.y4m -o out.ivf --gpus 0,1,2,3 --multi-gpu gop --scale-mode throughput \
  --gop-length 120 --vram-limit-mb 7500

# Profiling (one process, per-GPU fields)
AV2ENC_PROFILE=/tmp/mgpu.jsonl AV2ENC_VRAM_LIMIT_MB=7500 \
  av2enc in.y4m -o out.ivf --gpus 0,1 --multi-gpu tile --scale-mode throughput
```

Semantics and validation:

- `quality` mode: the aggregate frame time is held at `--target-fps`. Extra GPU
  time raises the quality index (RDOQ, partitions, modes, search, references,
  multi-pass) in the order in `docs/architecture.md`. If quality saturates, the
  extra GPUs may idle; that is allowed. Do not raise fps in this mode.
- `throughput` mode: the quality index is pinned to the 1-GPU value. Tile
  layout, QP, search, and references stay fixed. The output should be
  bit-identical to a 1-GPU encode with the same layout and quality index. This
  is the scaling benchmark mode.
- `--multi-gpu tile` requires `tiles >= gpus`. If not, fail and suggest a
  larger layout or GOP mode. Do not silently drop a GPU.
- `--multi-gpu gop` requires a closed GOP and no references across segment
  boundaries. If the encoder adds open GOPs/B-frames later, force closed GOPs
  for this mode or reject it.
- `--tile-layout` must be representable at the resolution and satisfy the level
  constraints (`docs/tiles.md` §3): at least one superblock per tile, at most
  64×64 tiles, and the level-derived maximum tile width/area. A 16×2 stream at
  1080p is legal but not yet verified with `dav2d`.
- `--vram-limit-mb` is checked on every GPU before allocation. If a device
  cannot fit the arena plus headroom, fail with the device ID. Never silently
  reduce the cap on one GPU.
- Mixed GPU models/compute capabilities: warn. The scaling target assumes
  homogeneous 3060s; a 5070 in the pool changes the slowest-GPU calculus and
  the fatbin path. Do not mix 8 GB and 12 GB cards for target validation.
- `--gpus` order is the assignment order for ties. It does not affect the
  bitstream, only which physical GPU gets which tile set.

### 7.1 Host dispatch contract (`--gpu-plan`)

The host-only dry run also prints `dispatch: <raster_slot>:<gpu> ...`.
`MultiGpuPlan::dispatch` has one `TileDispatchEntry` per tile in raster order:
the merge slot (always raster `TileNum`), the deterministic LPT owner, and the
superblock/pixel rect. A worker writes its payload into that slot; the
coordinator merges slots in ascending order and never in completion order.

Every timing/transfer/scaling number carries an `Evidence` flag
(`estimated` / `claimed` / `measured`). The planner only emits `estimated`;
promoting a number to `claimed` or `measured` requires recording a matching
run (same clip, QP/quality index, layout and GPU set).

---

## 8. Profiler contract

The existing profiler (`docs/profiling.md`) already records per-stage CPU/CUDA
times, `peak_vram_mb`, `vram_capacity_mb`, and `vram_limit_mb`. Multi-GPU adds
one level: the same fields **per GPU**, plus coordinator/barrier fields.

### 8.1 Required fields

Top-level `multi_gpu` object:

| Field | Meaning |
|---|---|
| `mode` | `tile` or `gop` |
| `scale_mode` | `quality` or `throughput` |
| `gpu_ids` | Physical device ordinals actually used |
| `tile_layout` | `CxR`, or `gop` |
| `tile_count` | Total tiles in the frame (tile mode) |
| `tiles_per_gpu` | Array, one entry per GPU |
| `load_imbalance` | `max_gpu_frame_ms / mean_gpu_frame_ms` (time-based; the truth) |
| `load_imbalance_area` | Planner-style area predictor, for comparison |
| `barrier_wait_ms_max` | Worst per-GPU barrier wait in the run |
| `merge_ms` | Coordinator payload merge / OBU assembly time |
| `cdf_gather_ms` | CDF gather and frame-level average time; 0 with the default `disable_cdf_update=1` |
| `ref_sync_ms` | Recon gather + reference fill time |
| `host_tx_bytes`, `host_tx_ms` | Total host-mediated traffic and time per frame |
| `ref_dup_mb` | Per-GPU reference pool; total/extra duplication |
| `scaling_vs_1gpu` | Only when a matching 1-GPU baseline exists; otherwise `null` |
| `gop` | Segment length, segment count, segments/GPU, keyframe penalty (GOP mode) |

Per-GPU array (`gpus[]`):

| Field | Meaning |
|---|---|
| `id` | Device ordinal |
| `tiles`, `tile_area_sb` | Local tile count and area |
| `frame_ms` | Local compute + entropy + payload D2H; excludes barrier wait |
| `barrier_wait_ms` | Time from local completion to barrier release |
| `peak_vram_mb` | Device-wide used peak on that GPU |
| `vram_capacity_mb`, `vram_limit_mb` | Device capacity and the 7,500 MB cap (the cap wins) |
| `host_tx_bytes`, `host_tx_ms` | Local share of host traffic |
| `stages` | Existing stage array, with `cuda.total_ms/mean_ms/max_ms` per stage |

New stage names for the coordinator/worker boundaries (in addition to the
existing stage list): `tile_dispatch`, `payload_gather`, `cdf_gather`,
`ref_sync`, `barrier_wait`, `merge`, `segment_assemble`.

### 8.2 Example shape

Numbers below are illustrative, not measured:

```json
{
  "profiler": "av2enc",
  "multi_gpu": {
    "mode": "tile",
    "scale_mode": "throughput",
    "gpu_ids": [0, 1],
    "tile_layout": "8x2",
    "tile_count": 16,
    "tiles_per_gpu": [8, 8],
    "load_imbalance": 1.007,
    "barrier_wait_ms_max": 3.1,
    "merge_ms": 1.2,
    "host_tx_bytes": 12950000,
    "host_tx_ms": 0.73,
    "ref_dup_mb": 11.87,
    "scaling_vs_1gpu": 1.917
  },
  "gpus": [
    {
      "id": 0,
      "tiles": 8,
      "tile_area_sb": 255,
      "frame_ms": 517.3,
      "barrier_wait_ms": 3.1,
      "peak_vram_mb": 588.3,
      "vram_capacity_mb": 11920.75,
      "vram_limit_mb": 7500,
      "stages": [
        {"stage": "intra_search", "cuda": {
          "calls": 8, "total_ms": 120.4, "mean_ms": 15.05, "max_ms": 18.2}},
        {"stage": "entropy", "cuda": {
          "calls": 8, "total_ms": 44.1, "mean_ms": 5.51, "max_ms": 6.8}}
      ]
    }
  ]
}
```

### 8.3 Derived metrics and rules

- `load_imbalance` is time-based, not area-based. A GPU with equal tile area can
  still be slower because of content, clocks, or thermal state. Area imbalance
  is only the planner's prior.
- `barrier_wait_ms` is the most direct straggler metric. If one GPU's wait is
  consistently large, the frame is gated by that GPU; inspect its per-stage
  times and `peak_vram_mb`.
- `gpu_utilization = sum(busy_ms_g) / (wall_ms * G)`, where `busy_ms` excludes
  barrier wait and idle. Do not sum overlapping stage times; use the union of
  the GPU's stage intervals.
- `scaling_vs_1gpu` requires a 1-GPU run with the same clip, QP/quality index,
  and tile layout. In `quality` mode it is a **quality headroom** factor, not an
  fps result. In `throughput` mode it is an fps scaling factor.
- Per-GPU `peak_vram_mb` is device-wide and includes the CUDA context and any
  other process on that GPU. Use exclusive GPUs for a clean number.
- Do not compare absolute CUDA stage times across GPUs unless clocks are
  locked; consumer cards boost differently under multi-GPU heat/power load.
- The profiler must not add a per-tile synchronization. One CUDA event pair per
  GPU per frame, resolved at `end_frame()`, plus the coordinator barrier.
- If the implementation uses N processes instead of N threads, each process
  writes its own summary; the coordinator must merge them into one
  `multi_gpu` summary. Do not emit N unrelated summaries and call it one run.

---

## 9. Scaling model and the 1.8–1.95× target

The planner in `bench/multi_gpu_plan/` uses:

```text
T_1        = (pixels / 1e6) * 481 ms/MP   (calibrated so 1080p is ~1.00 s)
T_G        = s * T_1 + (1 - s) * B * T_1 / G + T_pcie(G)
speedup    = (T_1 + T_pcie(1)) / T_G
```

where `s = 0.035` is the non-parallel fraction (coordinator, launch, frame
header/rate control, merge, CDF), `B = max_gpu_cost / mean_gpu_cost` is the
content-aware load imbalance, and `T_pcie` is the host transfer. The 481 ms/MP
constant is a **planning assumption calibrated to the 1 fps product floor**, not
a measured 3060 rate. The CUDA encode path is not complete, so the denominator
is not a measured baseline either.

Simulated matrix (PCIe 4.0 ×16, refs=4, content skew ±15%, balanced assignment):

| Resolution | GPUs | Layout | Load imbalance | Work ms | Xfer ms | Speedup | Efficiency |
|---|---:|---|---:|---:|---:|---:|---:|
| 1080p | 1 | 4×2 | 1.000 | 997.6 | 0.22 | 1.000× | 1.000 |
| 1080p | 2 | 8×2 | 1.007 | 520.4 | 0.73 | **1.917×** | 0.959 |
| 1080p | 4 | 16×2 | 1.030 | 283.9 | 1.20 | 3.514× | 0.878 |
| 4K | 1 | 4×2 | 1.000 | 3990.2 | 0.57 | 1.000× | 1.000 |
| 4K | 2 | 8×2 | 1.008 | 2083.0 | 2.13 | **1.916×** | 0.958 |
| 4K | 4 | 16×2 | 1.025 | 1130.0 | 3.99 | 3.531× | 0.883 |

**2-GPU target: 1.8–1.95×.** The model gives 1.917× at 1080p and 1.916× at 4K,
inside the target. The contiguous-range policy gives 1.868×/1.890×, also inside.
The 4-GPU projections (3.51–3.53×) are not a committed target; coordinator,
tail, and PCIe effects grow, and 16×2 is not yet decoder-verified.

In `quality` mode these work times are not exposed as fps; the governor holds
the aggregate target and spends the difference on quality. In `throughput` mode
the same numbers are the fps scaling.

Sensitivity of the 2-GPU speedup to the non-parallel fraction `s` and the
straggler `B`:

| `s` \ `B` | 1.00 | 1.05 | 1.10 |
|---:|---:|---:|---:|
| 3.5% | 1.932 | 1.846 | 1.768 |
| 5.0% | 1.905 | 1.822 | 1.747 |
| 10.0% | 1.818 | 1.747 | 1.681 |

Cells below 1.8× miss the target. At `s = 3.5%`, the 2-GPU target survives up to
about **7–8%** load imbalance; at `s = 5%`, up to about 6%. The model's
`B = 1.007` is optimistic because it assumes tile cost is proportional to area
plus a synthetic hash. Real content can be worse.

### Top scaling risk

**Content-dependent tile cost: the frame barrier waits for the slowest GPU.**
Area-based assignment cannot see a detailed tile, a difficult partition, or a
GPU that is thermally throttled. A 10% straggler with 3.5% serial work drops
2-GPU speedup to ~1.77×, below the 1.8× floor, even though the area balance
looks perfect. At 4 GPUs the tail risk is larger because there are more tiles
and more chances for one GPU to lag.

Mitigations, in order:

1. Measure per-tile cost on a sample; use the previous frame's cost (content is
   temporally correlated) for LPT, not area alone.
2. Use finer tiles (16×2) to reduce the granularity of a straggler, and charge
   the tile byte overhead against the quality budget.
3. Overlap the barrier with the next frame's intra/ME work where the dependency
   allows. Do not overlap the reference fill (or the adaptive-CDF average, if
   enabled); they are hard dependencies.
4. For throughput, consider GOP mode: no per-frame barrier, at the cost of
   keyframes and segment-tail utilization.
5. Do not fix imbalance by lowering quality on the slow GPU. That changes the
   same-quality comparison and can break determinism.

Secondary risks:

- **Coordinator/host serialization.** The E5-2690 v4 is weak; N CUDA graphs,
  payload merges, and reference copies may exceed the 3.5% budget. Measure
  `merge_ms`, `cdf_gather_ms`, and `ref_sync_ms`; they are included in `s`.
- **PCIe in throughput mode.** At 1 fps, even ×4 is fine. At 10+ fps, 4K/4-GPU
  ×4 links can spend 30 ms/frame and cap scaling. Match the lane split to the
  target fps.
- **Serial entropy tail.** Each GPU has 8 tile entropy workgroups. The range
  coder is serial per tile; if the tiles are small and the workgroups are
  latency-bound, the GPU can idle at the end of a frame. Overlap tile entropy
  with the next frame, or use more tiles per GPU, and measure.
- **No measured baseline.** The single-GPU 1 fps denominator is a product target,
  not a result. The CUDA path is not wired yet. All scaling numbers are model
  estimates until the validation plan below is run.

---

## 10. Pending hardware and validation plan

No multi-GPU host is available. The Vast.ai box is a single 3060; the local dev
box is a single 12 GB 3060. There is no NVLink and no GeForce P2P. The multi-GPU
planner runs on any machine with Python 3.9+ and no CUDA:

```sh
cd bench/multi_gpu_plan
python3 multi_gpu_plan.py --self-test
python3 multi_gpu_plan.py --matrix
python3 multi_gpu_plan.py --matrix --markdown
python3 multi_gpu_plan.py --matrix --json /tmp/multi_gpu_plan.json
```

When 2/4-GPU hardware exists, run this sequence before making any scaling
claim:

1. **Single-GPU baseline.** 1080p, fixed QP, 120 frames, `compat8g`, 7,500 MB
   cap, 4×2 and 8×2 layouts, `throughput` at a fixed quality index. Record
   `summary.json`. Confirm `tools/validate_dav2d.sh` passes for both.
2. **2-GPU tile, throughput.** 8×2, same QP/quality index. Check the merged
   MD5 against the 1-GPU 8×2 run. If it differs, stop and fix determinism
   (adaptive-CDF order if enabled, rate control, filter boundaries, tile size
   fields).
3. **Measure.** `frame_ms`, `barrier_wait_ms`, `load_imbalance`, per-GPU stage
   means, `peak_vram_mb`, `host_tx_ms`. Compute speedup against the 1-GPU 8×2
   baseline. Target 1.8–1.95×; record whether it is met.
4. **2-GPU tile, quality.** Hold 1 fps aggregate. Compare CVVDP/bitrate against
   the 1-GPU quality run at matched bitrate. Extra GPUs should buy quality, not
   fps. If quality saturates, say so.
5. **4-GPU tile.** 16×2 after `dav2d` verifies it; otherwise 8×2 with 4 tiles
   per GPU. Repeat steps 2–3. Do not commit a 4-GPU target from the model.
6. **GOP mode.** 2/4 GPUs, closed GOPs of 120 and 240 frames. Measure bitrate
   vs the 1-GPU baseline with the same GOP cadence and vs a sparse-keyframe
   baseline. Check segment-tail utilization and `dav2d` on every segment.
7. **PCIe matrix.** Repeat the 4K/4-GPU case at ×16 and at the board's actual
   split (usually ×8 or ×4). Record `host_tx_ms`; compare against the 1 fps
   budget and the throughput target.
8. **Thermals/clocks.** Run a 10-minute soak. Log clocks/temperatures if
   available; consumer cards may throttle with 2/4 GPUs. A throttled GPU is a
   straggler on every frame.
9. **Acceptance.** 2-GPU tile speedup ≥1.8× at same quality; per-GPU
   `peak_vram_mb ≤ 7500`; throughput-mode MD5 matches 1-GPU same layout; all
   streams pass `tools/validate_dav2d.sh`; no tile-edge artifacts or CDF
   divergence.

Until that sequence passes, cite this document as **design/simulation**, not as
a measured multi-GPU result.

---

## 11. Open questions

1. **16×2 decoder support.** `dav2d` has not been verified on 16×2 at 1080p/4K.
   Test before the 4-GPU default. If it fails, use 8×2 (4 tiles/GPU) or GOP
   mode.
2. **Adaptive CDF averaging with >8 tiles (opt-in only).** `docs/tiles.md` §9
   already flags the `TileNum < 1<<numLog2` guard in `dav2d`. Multi-GPU adds
   cross-GPU gather and ordering. Validate against the CPU reference before
   `--adaptive-cdf` is trusted. The default `disable_cdf_update=1` path has no
   such handoff.
3. **Cross-tile filter BD-rate.** How much quality is lost with
   `disable_loopfilters_across_tiles=1`? If large, the explicit filter pass may
   be worth its second barrier.
4. **Keyframe ratio K.** Measure it on the real encoder; the GOP bitrate table
   is sensitive to it.
5. **Tile cost model.** Area + synthetic skew is a placeholder. Measure per-tile
   time and test previous-frame LPT on real clips.
6. **Host CPU cost.** The E5-2690 v4 runs at 38–43% scaling; N worker threads
   with N CUDA graphs may make the coordinator the bottleneck before the GPUs.
   Measure `merge_ms`/`ref_sync_ms` and the host wall time.
7. **PCIe lane split.** The planner accepts gen/width, but the real board's
   bifurcation is unknown. Use `nvidia-smi topo -m` / `lspci` and model the
   actual ×8/×4 case.
8. **Mixed GPUs.** The target is homogeneous 3060s. A 5070 in the pool would
   change clocks, bandwidth, and the slowest-GPU calculus; do not mix for target
   validation.
