# M2 video intra candidate generator

Batched device generator for the intra fallback of the M2 video coder.  The M2
pipeline currently hard-codes every intra leaf to DC_PRED
(`entry.y_mode = 0`); this module produces, for every leaf and every legal
candidate, the predictor plane, its SSE against the source, and optionally the
quantised TX plan.  The video orchestrator can consume it without touching the
existing predictor code.

```
src/cuda/pipeline/video/intra/
  video_intra.h                 API: jobs, results, Config, Workspace
  video_intra_host.h            host-only candidate expansion + §5.20.9.1 availability
  video_intra_device.cuh        fused device body (shared predictor + SSE reduce)
  video_intra.cu                kernels + arena workspace
  video_intra_test.cu           sm_86 CPU-parity, odd geometry, TX, arena contract
  video_intra_emulation_test.cpp  host-emulation parity (no GPU/nvcc)
  build_video_intra_test.sh     nvcc build/run
  build_emulation_test.sh       host-emulation build/run
  README.md                     this file
```

The module is a leaf: it owns no product flag and makes no bitstream claim of
its own.  The M2 orchestrator consumes it behind `--video-intra-full` (see the
update note below).

> **Update (integration/video-intra-wire):** the module is now consumed by the
> M2 product behind `--video-intra-full`.  `video_backend_cuda.cu` runs the
> two-batch flow below and `src/pipeline/video/video_intra_search.h` is the CPU
> reference twin.  See "Remaining gaps".

## No duplicated predictor math

Every candidate is predicted by
`src/cuda/codec/intra/modes/intra_predict_device.cuh` —
`intra_run_block_device()` — i.e. the exact code `intra_modes.cu` and the M1 GPU
chroma loop compile.  This module only fills `intra_modes::IntraBlock`, batches
one launch, and reduces SSE.  Luma directional IDIF, chroma bilinear
interpolation, the chroma > 32 extension rule, wide-angle mapping, the corner
filter and the §7.13.2.18 edge filter therefore cannot drift from the verified
still component.

## Modes

| Plane | Candidates |
|---|---|
| Luma | DC, V, H, D45, D135, D113, D157, D203, D67, SMOOTH, SMOOTH_V, SMOOTH_H, PAETH, then the six non-zero angle deltas (−3,−2,−1,+1,+2,+3) for each directional mode = **61** |
| Chroma (U, V) | **DC only**, one job per plane.  The shared predictor applies the chroma derivation; the full UV directional search is deliberately not wired (see `docs/negative-results.md`). |

Candidate order is the CPU/still search order
(`intra::build_mode_candidates(..., true)`), so a strict `<` SSE reduce
reproduces the same tie-break.  `BlockJob::mode_mask` /
`include_angle_deltas` select a subset (DC-only, no-directional, base-only).

## §5.20.9.1 availability

`derive_availability()` implements `is_inside()` plus the §7.13.2.1 extension
bounds:

- `have_left` / `have_above` are tile-relative: a leaf at the tile start has no
  neighbour even when the frame continues.
- `max_x` / `max_y` clamp to the visible plane/tile extent.  A leaf may extend
  past the right/bottom edge (odd partial row/column); the predictor clamps all
  reads and the source clamp is applied per sample.
- `num4_above_right` is bounded by the visible above row and by `w/4`
  (raster order reconstructs the above row).  `num4_below_left` defaults to 0
  for raster order; a caller replaying the §6.19.2.3 `BlockDecoded` map can pass
  `decoded_below_left4` / `decoded_above_right4` to cap the geometric bound.

## API contract (video orchestrator hand-off)

```cpp
namespace av2enc::cuda::video::intra {

struct AvailRequest { int x, y, w, h; int tile_x0, tile_y0, tile_x1, tile_y1;
                      int plane_w, plane_h; int decoded_above_right4 = -1,
                      decoded_below_left4 = -1; };
Availability derive_availability(const AvailRequest&);

struct BlockJob {
  const uint8_t* recon_plane; int recon_stride;    // reconstructed plane
  const uint8_t* source_plane; int source_stride;  // source plane
  int x, y, w, h; int plane;                       // 4..64, plane 0/1/2
  int max_x, max_y;                                // visible recon bounds
  int source_max_x, source_max_y;                  // visible source bounds
  int have_left, have_above;                       // §5.20.9.1
  int num4_above_right, num4_below_left;
  int filter_type_above, filter_type_left;         // §7.13.2.15/.16
  int enable_edge_filter;
  uint32_t mode_mask; int include_angle_deltas;
};

struct CandidateResult { int64_t sse; int32_t pred_offset; int32_t block;
                         uint8_t mode; int8_t angle_delta; uint8_t plane; };

struct Config { int max_blocks = 64; int max_candidates = 61;
                int max_block_area = 4096; bool enable_tx = false;
                int max_tx_area = 1024; };

class Workspace {
  Status init(Arena&, const Config& = {}, std::string* = nullptr,
              uint64_t* required_out = nullptr);
  Status evaluate(const BlockJob*, int count, const EvaluateOptions& = {},
                  cudaStream_t = nullptr, std::string* = nullptr);
  Status synchronize(cudaStream_t = nullptr);
  Status download_results(CandidateResult*, int capacity, cudaStream_t = nullptr) const;
  Status download_prediction(uint8_t*, size_t capacity, cudaStream_t = nullptr) const;
  Status download_tx(TxCandidateResult*, int capacity, cudaStream_t = nullptr);
  Status download_residual(int16_t*, size_t capacity, cudaStream_t = nullptr) const;
  Status download_best(BestResult*, int capacity, cudaStream_t = nullptr);
  int block_count(), candidate_total(), candidate_offset(i), candidate_count(i);
  const CandidateResult* results_device(); const uint8_t* prediction_device();
  const TxCandidateResult* tx_results_device(); const int32_t* levels_device();
};
```

Semantics:

- All pointers in `BlockJob` are **device** pointers.  `pred_offset` is an
  element offset into the packed prediction buffer; the plane is `w*h` bytes in
  raster order.
- `sse = Σ (source − prediction)²` over the coded `w*h`, with source
  coordinates clamped to `[0, source_max_x] × [0, source_max_y]`.
- `download_*` are blocking host transfers (not the batch synchronisation).
- One `evaluate()` = one candidate kernel launch for the whole batch; the
  optional TX plan adds one `still_tx_batch` launch.  The caller calls
  `synchronize()` once per batch, then downloads.
- Jobs must be mutually independent: no job may read reconstruction another job
  in the same call writes.  Batch a wavefront level, or the whole frame when
  the reconstruction inputs are frozen.  The predictor never writes the recon
  plane.

Per-frame use:

```cpp
vi::Config cfg;                 // size for the frame / wavefront level
cfg.max_blocks = leaves;        // number of BlockJob entries
vi::Workspace ws;
ws.init(arena, cfg);            // one arena for the process

std::vector<vi::BlockJob> jobs;
for (const Leaf& leaf : leaves) {
  vi::AvailRequest req;
  req.x = leaf.x; req.y = leaf.y; req.w = leaf.w; req.h = leaf.h;
  req.tile_x0 = tile.x0; req.tile_y0 = tile.y0;
  req.tile_x1 = tile.x1; req.tile_y1 = tile.y1;
  req.plane_w = plane.w; req.plane_h = plane.h;
  const vi::Availability a = vi::derive_availability(req);

  vi::BlockJob job;
  job.recon_plane = recon[leaf.plane];   // device pointer
  job.recon_stride = recon_stride[leaf.plane];
  job.source_plane = source[leaf.plane];
  job.source_stride = source_stride[leaf.plane];
  job.x = leaf.x; job.y = leaf.y; job.w = leaf.w; job.h = leaf.h;
  job.plane = leaf.plane;
  job.max_x = a.max_x; job.max_y = a.max_y;
  job.source_max_x = plane.w - 1; job.source_max_y = plane.h - 1;
  job.have_left = a.have_left; job.have_above = a.have_above;
  job.num4_above_right = a.num4_above_right;
  job.num4_below_left = a.num4_below_left;
  // filter_type_*, enable_edge_filter, mode_mask, include_angle_deltas ...
  jobs.push_back(job);
}
ws.evaluate(jobs.data(), static_cast<int>(jobs.size()), {}, stream);
ws.synchronize(stream);         // the one frame/level sync

std::vector<vi::CandidateResult> results(ws.candidate_total());
ws.download_results(results.data(), static_cast<int>(results.size()));
std::vector<std::uint8_t> pred(ws.prediction_elems());
ws.download_prediction(pred.data(), pred.size());
// results[c].pred_offset indexes pred[] (w*h packed bytes)
```

`download_best()` is the same flow reduced to the SSE-minimising candidate per
block with the CPU tie-break.

### Optional TX plan

`EvaluateOptions::run_tx` makes the candidate kernel also write the coded-size
residual (`source − pred`) and runs the shared
`cuda/pipeline/still/tx/still_tx.cu` batch over every candidate:

- `download_tx()` returns `{eob, levels_offset}` per candidate; the levels are
  `levels_device()[levels_offset .. levels_offset + adj_w*adj_h)`.
- `download_residual()` returns the coded-size residual buffer (same
  `pred_offset` layout).
- The reconstruction is `Clip1(pred + inverse residual)` and is left to the
  orchestrator (the still TX stage does not write recon either).
- `eob == -1` means the TX stage rejected the job (parameter/smem guard);
  otherwise 0 means all-zero levels.
- TX types default to `DCT_DCT`, the only type the current M2 syntax writer can
  signal.  `intra_full::Workspace::derive_chroma_type` (§5.20.7.29) supplies a
  chroma type when a later wave enables non-DCT chroma.

## Arena

`required_device_bytes(config)` is exact: staging + results + prediction +
(optional) TX residuals/levels/results, each rounded to `Arena::kAlignment`
exactly like `Arena::allocate_array`.  `init()` on an undersized arena returns
`Status::kArenaExhausted`, reports the requirement through `required_out` /
`error`, and restores the bump pointer (no residue).  A bad configuration
returns `kInvalidArgument` with requirement 0.  No allocation happens outside
the caller's arena; `tools/check_arena.sh` passes.

Sizing scales with `max_blocks × max_candidates × max_block_area`.  For a
whole 640x360 frame batch use `max_blocks = leaf_count`, `max_candidates = 61`,
`max_block_area = 4096`, or keep the peak down by batching a wavefront level.
TX roughly triples the reservation (16-bit residual + 32-bit levels at the
maximal adjusted area).

## Evidence (RTX 3060, sm_86, CUDA 13.0)

| Suite | Result |
|---|---|
| `./build_video_intra_test.sh` | PASS 1,092 checks / 8,415 candidates / 7,005,840 samples / 0 failures |
| `./build_emulation_test.sh` | PASS 5,781 checks (no GPU) |
| `compute-sanitizer --tool memcheck` | 0 errors |

The real-device suite compares every candidate's predictor bytes and SSE
against `src/codec/intra/modes/` for:

- all 61 luma candidates (13 base + 8 directional × 6 deltas) on 13 sizes
  (4x4 … 64x64 plus 8x16/16x8/64x8/8x64/16x64/64x16/32x8/8x32), on luma and
  both chroma planes;
- eight edge states per size: interior, top-left, left, top, right partial,
  bottom partial, bottom-right partial and a tile-local origin;
- 100x70 and 640x360 bottom/right partial leaves, including 640x360 blocks at
  `y = 320` whose 64x64 extent runs past row 359;
- 50x35 (4:2:0 of 100x70) chroma leaves plus 64x64 chroma jobs (the §7.13.2.1
  chroma > 32 extension rule);
- candidate masks, mixed-plane batches, the CPU tie-break, and the arena
  contract (`required_device_bytes` exact, one-byte-short `kArenaExhausted`,
  invalid config, flat `used()`).

The TX sweep compares levels and EOB against
`av2enc::txv::downsample_residual` + `forward_transform` (the same CPU reference
the still TX component pins) for every candidate of the 13 supported sizes.

## Build and run

```sh
cd src/cuda/pipeline/video/intra
./build_video_intra_test.sh              # sm_86, build + run
./build_video_intra_test.sh --no-run
./build_video_intra_test.sh --quick      # skip the TX sweep
ARCH=sm_120 ./build_video_intra_test.sh
./build_emulation_test.sh                # no nvcc/device
SANITIZE=1 ./build_emulation_test.sh     # ASan + UBSan
compute-sanitizer --tool memcheck ./build-video-intra/video_intra_test
```

Both scripts are self-contained and never touch the root CMake build.  The
module is not in the product source lists, so the product build is unchanged
until the orchestrator wires it.

## Remaining gaps

- **Wired into the M2 product behind `--video-intra-full`.** The orchestrator
  (`video_backend_cuda.cu` / `video_pipeline.cpp`) calls one search batch
  (luma 61 + U/V DC, no TX), `download_best`, then a second small batch for the
  winning mode group + DC chroma with `run_tx`. Availability comes from the
  orchestrator's §6.19.2.3 `BlockDecoded` replay (the module cannot see the
  coded stream), so `BlockJob.num4_above_right` / `num4_below_left` are the
  measured counts, not `derive_availability()`'s geometric default. Default
  streams keep the DC-only fallback and stay byte-identical.
- Chroma directional / CfL / MRL / DIP / IBP / palette / IBC candidates are out
  of scope for this wave (the M2 syntax writer cannot signal them).
- Non-DCT TX types need the M2 TX-type search and `intra_full`'s chroma
  derivation; only `DCT_DCT` is exercised here.
- One launch per wavefront level is the intended integration; CUDA-graph
  capture is future work. The product wiring is per-leaf (two small batches),
  which is correctness-first rather than throughput-optimal.
- 8-bit 4:2:0 only, matching the M2 video core.
