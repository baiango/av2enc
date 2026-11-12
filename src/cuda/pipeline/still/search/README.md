# CUDA M1 luma intra candidate search

Batched GPU evaluation of the 61 luma intra candidates for one M1 block, with
the CPU mode decision reproduced **exactly** (same candidate order, same SSE
metric, replace only on strictly smaller SSE).

Owner: luma-search agent. Files:

```text
src/cuda/m1/search/luma_search.h              API + candidate table + gates
src/cuda/m1/search/luma_search.cu             kernels (expand / predict / SSE / combine)
src/cuda/m1/search/luma_search_test.cu        CPU parity, batching, chunks, timing
src/cuda/m1/search/build_luma_search_test.sh  standalone build/run script
src/cuda/m1/search/README.md                  this file
```

Nothing outside this directory is modified. Prediction arithmetic is **not**
duplicated: every candidate is an `intra_modes::IntraJob` and prediction runs
through `src/cuda/intra_modes`.

## API

```cpp
namespace av2enc::cuda::still::search {

struct LumaSearchBlock {
  const uint8_t* recon_plane; int recon_stride;
  int x0, y0, width, height;
  int max_x, max_y;                 // last visible plane sample, inclusive
  int have_left, have_above;
  int num4_above_right, num4_below_left;
  int filter_type_above, filter_type_left;
  const uint8_t* source_plane; int source_stride;
  int source_max_x, source_max_y;
};

struct LumaSearchGates {
  unsigned flags = kGateNone;       // kGateDcOnly | kGateBaseOnly | kGateNoDirectional
  int enable_edge_filter = 1;
};
struct LumaSearchResult { int mode; int angle_delta; int reserved; int64_t sse; };
struct LumaSearchDebug   { int64_t* d_candidate_sse; int stride; };

struct LumaSearchLayout;   // set of arena offsets + capacities
struct LumaSearchScratch;  // device jobs / SSE / pred buffers

int  luma_search_candidate_count(const LumaSearchGates&);
int  luma_search_candidates(const LumaSearchGates&, LumaCandidate* out, int cap);
LumaSearchLayout luma_search_layout(int max_blocks, int max_block_area, int chunk);
LumaSearchScratch luma_search_make_scratch(void* base, const LumaSearchLayout&);

cudaError_t luma_search(const LumaSearchBlock* d_blocks, int block_count,
                        const LumaSearchGates&, const LumaSearchScratch&,
                        LumaSearchResult* d_results,
                        const LumaSearchDebug* debug = nullptr,
                        cudaStream_t stream = nullptr);
}
```

Semantics:

- `mode` is the `intra_modes::IntraMode` / AV2 `YMode` value. A directional
  candidate is `(mode, angle_delta)`; non-directional candidates use delta 0.
- `sse` is the sum of `(source - prediction)^2` over the coded `width*height`
  block, with source coordinates clamped to `[0, source_max_x] x
  [0, source_max_y]` (partial edge blocks read the last visible sample, exactly
  like the CPU `source_sample`).
- All pointers in `LumaSearchBlock` are **device** pointers. All blocks in one
  call must share in-flight `recon_plane` contents; no block may read
  reconstruction written by another block in the same call.
- `d_blocks` / `d_results` are device arrays. `debug` is an optional device
  array of per-candidate SSE values, one `stride`-sized row per block, in CPU
  candidate order (active candidates only). Null disables the writes.
- No allocation, no `cudaDeviceSynchronize`, no global state. One launch
  sequence per candidate chunk on the supplied stream.
- Scratch base must be 16-byte aligned (the arena is); one call at a time per
  scratch buffer.

## Candidate order and gates

The order is the CPU's `intra::build_mode_candidates(..., true)`:

1. all 13 base modes (DC, V, H, D45, D135, D113, D157, D203, D67, SMOOTH,
   SMOOTH_V, SMOOTH_H, PAETH) with `angle_delta = 0`;
2. for each directional mode V..D67, deltas `-3, -2, -1, +1, +2, +3`.

| Gate | Candidates | Mirrors |
|---|---:|---|
| none | 61 | full `--still-full-intra` search |
| `kGateBaseOnly` | 13 | `AV2ENC_STILL_BASE_ONLY` |
| `kGateNoDirectional` | 5 (DC + 3 smooth + Paeth) | `AV2ENC_STILL_NO_DIR` |
| `kGateDcOnly` | 1 | `luma_dc_only` / `force_dc` / `AV2ENC_STILL_LUMA_DC` |

`kGateDcOnly` returns `(mode 0, angle 0, sse 0)` without predicting, matching
`choose_luma_mode`'s early return. The debug array is not written in that case.
The forced-mode env knobs (`AV2ENC_STILL_FORCE_Y*`) are the orchestrator's concern;
this component always searches the gated list.

## Tie-break and chunked evaluation

The combine kernel visits active candidates in the order above and updates the
winner only when `sse < best_sse`. Starting `best_sse` is `INT64_MAX`, so the
first active candidate (DC) always sets a winner even at equal SSE. Integer SSE
is associative, so the parallel reduction is exact; only the *order of the
comparison* matters, and that is serialised in the combine kernel.

When scratch memory bounds the batch, the candidate list is split into
consecutive chunks of `candidates_per_chunk`. Each chunk runs
expand → predict → SSE → combine, and combine carries the running best forward.
Because the chunks are a contiguous partition of the same ordered list, the
result is identical to one 61-candidate pass; the test verifies
`chunk = 61` vs `chunk = 7` on real frames.

## Batching and wavefronts

One call evaluates `block_count x active_candidates` jobs. Jobs are independent
because search only reads the reconstruction plane; the predictor's batch entry
point requires exactly that. Therefore:

- one block: all 61 candidates are batched (the per-block launch-latency case);
- many independent blocks: batch them in the same call;
- dependency-ordered encoders: call once per 64x64-superblock wavefront level
  (blocks with equal `sbx + sby`), as the test's `-- timing` wavefront cases do.
  Do not batch blocks that need reconstruction produced by another block in the
  same call.

Launch sequence per chunk: `ls_expand_kernel`, `intra_predict_batch`,
`ls_sse_kernel`, `ls_combine_kernel` (4 launches). With the default scratch
chunk of 61 this is 4 launches per call, plus 1 for the DC-only shortcut.

## Scratch sizing

```cpp
layout = luma_search_layout(max_blocks, max_block_area, candidates_per_chunk);
size_t bytes = layout.total_bytes;
// arena slice -> scratch
LumaSearchScratch scratch = luma_search_make_scratch(base, layout);
```

`total_bytes = jobs (IntraJob) + jobs (int64 SSE) + jobs * max_block_area
(prediction)`, 16-byte aligned and rounded to 256 B. `max_block_area` is the
largest `width*height` in the batch (use 4096 for 64x64). The prediction buffer
uses a uniform per-job stride of `max_block_area`; mixed-size batches pay the
largest area per job. Reduce `candidates_per_chunk` to bound memory:

| batch | chunk 61 | chunk 16 | chunk 8 |
|---|---:|---:|---:|
| 17 x 64x64 (max level) | 4.2 MiB | 1.1 MiB | 0.6 MiB |
| 60 x 64x64 (640x360 SB grid) | 14.7 MiB | 3.9 MiB | 1.9 MiB |
| 510 x 64x64 (1080p SB grid) | 125 MiB | 33 MiB | 16.5 MiB |
| 900 x 32x8 | 20 MiB | 5.3 MiB | 2.6 MiB |

`chunk` does not change results. For a 1080p still, batching whole wavefront
levels keeps the peak at 4.2 MiB (largest level) while a whole-frame batch
needs 125 MiB; both are far below the device arena, but the level batching also
matches the reconstruction dependency order.

## Verification (RTX 3060 12 GB, sm_86, CUDA 12.8)

Built and run on the Vast.ai box:

```sh
cd src/cuda/m1/search
./build_luma_search_test.sh              # default media, full run
./build_luma_search_test.sh --quick      # reduced timing loop
./build_luma_search_test.sh --no-timing  # parity only
./build_luma_search_test.sh --timing-only --quick   # benchmark, no parity
```

The reference is the **real CPU intra module** (`src/encoder/intra_modes`)
driven through the exact candidate loop and strict-less-than tie-break of
`choose_luma_mode`. For every block the test compares the best mode, angle,
SSE, and the SSE of every active candidate. Media: `testdata/basic_64x64_1f.y4m`
and a real 640x360 crop (`testdata/shine_640x360_1f.y4m`); synthetic planes are
generated in the harness.

Result (2026-11-11):

```text
-- synthetic block matrix --   143 blocks / 156,960 samples per gate+edge case
  sizes 4x4,8x8,16x16,32x32,64x64,16x64,64x16,32x8,8x32,64x32,32x64,64x8,8x64
  positions top-left/top/left/interior/bottom-right-overflow, padded stride,
  num4 counts 0..8, filter flags 00/01/10/11, edge filter on/off, flat
  source==recon blocks that force SSE ties
  8 cases, 0 mode / 0 SSE / 0 candidate-SSE mismatches
-- real 64x64 frame --        27 blocks / 20,480 samples per case
  64x64,32x32,16x64,32x8,64x32 layouts, 8 cases, all exact
-- real 640x360 frame --      1,200 blocks / 721,920 samples per case
  SB wavefront levels (bottom 64x64 blocks extend past row 360),
  240 x 16x64, 900 x 32x8, 8 cases, all exact
-- chunk invariance --        1,217 blocks, chunk 61 vs 7, identical
```

Totals: 10,960 block decisions and best SSE values, 216,460 per-candidate SSE
values, 7,194,880 source samples compared — 0 mismatches. The real-frame cases
include `have_above=0`/`have_left=0` top and left SB edges, blocks crossing the
visible bottom edge, right-edge rectangles, and 16x64/32x8 shapes.

The underlying predictor was also re-checked against the CPU module with
`src/cuda/intra_modes/build_intra_modes_test.sh --cpu-parity`: 628,256 samples,
0 mismatches **with the edge filter enabled**. (The intra README's note about a
CPU edge-filter divergence is stale: `intra_modes.cpp` now snapshots the edge
row as §7.13.2.18 requires.)

`compute-sanitizer --tool memcheck` on `--quick --no-timing`:
**0 errors**.

## Timing (search only, RTX 3060, shared Vast.ai box)

`ms` is one `luma_search` call including expand, batched prediction, SSE,
combine. `Msample-eval/s` counts candidate-block pixels evaluated (predict +
SSE) per second. The first row is launch-latency bound: four launches for a
single 64x64 block.

| Case | blocks | ms/call | blk/s | Msample-eval/s | scratch |
|---|---:|---:|---:|---:|---:|
| 1 x 64x64 | 1 | 0.071 | 14.2 k | 3.5 k | 0.25 MiB |
| 640x360 SB grid, one batch | 60 | 1.21 | 49.7 k | 12.4 k | 14.7 MiB |
| 1080p SB grid, one batch | 510 | 9.8 | 51.9 k | 13.0 k | 125 MiB |
| 640x360 16x64, one batch | 240 | 1.37 | 175 k | 10.9 k | 16.1 MiB |
| 640x360 32x8, one batch | 900 | 2.51 | 359 k | 5.6 k | 20.1 MiB |
| 640x360 SB wavefront levels | 60 (15 levels) | 1.87 ms/frame | 32.0 k | — | 1.5 MiB/level |
| 1080p SB wavefront levels | 510 (46 levels) | 12.5 ms/frame | 40.9 k | — | 4.2 MiB/level |

The box was shared during this run: another tenant was at 100% GPU. An earlier
run with the GPU idle measured 0.61 ms (60 blocks) and 4.9 ms (510 blocks),
i.e. roughly 2x faster. Rerun with `--timing-only` on dedicated hardware before
quoting these.

Interpretation:

- The predictor itself is ~0.066 ms for a 1080p frame of 64x64 blocks
  (intra_modes README, idle measurement). 61 predictions plus SSE explains the
  few-millisecond search; once blocks are batched the search is
  predictor-throughput bound, not launch bound.
- Splitting a frame into wavefront levels costs 1.3–1.5x here (46 calls x 4
  launches, plus small levels). CUDA graphs (stage 3) should remove most of the
  per-level launch cost. Batch more blocks per call if the dependency order
  allows it.
- `32x8` blocks are occupancy bound in the shared predictor (128 threads per
  CUDA block regardless of area) and in the SSE kernel (same shape); a
  warps-per-block variant is future work in the scheduler component, not here.

## Blockers / notes for integration

1. **No recon/entropy wiring.** This component emits decisions and SSE only.
   The orchestrator must run it against final reconstruction per wavefront
   level, then feed the decision into transform/recon and the CPU entropy
   writer. It cannot produce a bitstream or pass `dav2d` alone.
2. **Neighbour context is caller-owned.** `have_left/have_above`, the
   `num4_above_right`/`num4_below_left` extension counts, and the neighbour
   smooth flags must be derived exactly as `still_encoder.cpp::count_top_right`,
   `count_bottom_left`, and `neighborhood_smooth` do. The search does not
   second-guess them; passing different counts changes the prediction and is a
   caller bug, not a search bug.
3. **Block validity is a precondition.** `width`/`height` must be 4/8/16/32/64;
   `max_x`/`max_y` are visible bounds; source coordinates are clamped.
   `intra_predict_batch` does not validate jobs, by design.
4. **DC-only has `sse = 0`.** That is the CPU shortcut's value, not a real
   distortion. Do not use it for rate-distortion accounting.
5. **Peak scratch scales with block x candidate x area.** Use wavefront-level
   batching (max 17 x 64x64 for 1080p SBs) or a smaller chunk. The component
   never allocates; the arena slice comes from the orchestrator.
6. **The `pred` buffer is fully materialised.** 61 predictions per block are
   written before SSE. A fused predict+SSE kernel would need to change
   `intra_modes` (or duplicate its arithmetic), which this component
   deliberately avoids; the SSE pass is memory bound and costs roughly the
   same as the predictor pass.
7. **No CUDA graph yet.** Four launches per call are visible to Nsight Systems;
   stage 3 can capture the per-level sequence.
8. **Build wiring needs `intra_modes.cu`.** The root `CMakeLists.txt` GLOB only
   covers `src/cuda/m1/**`; `search/luma_search.cu` calls into
   `src/cuda/intra_modes/intra_modes.cu`, so that source must be in the same
   link set when the search is wired (owner: CLI/build agent). The component's
   own `build_luma_search_test.sh` compiles both today.
9. **The orchestrator's `stage::search(GpuFrame)` hook is frame-level.** A
   correct dependent-block search must run once per wavefront level, interleaved
   with reconstruction; a whole-frame pass would read reconstruction that has not
   been produced yet. Wire `luma_search()` from the scheduler/level loop (or a
   whole-`GpuFrame` implementation only for blocks whose inputs are frozen), not
   as a single early pass.
