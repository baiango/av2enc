# CUDA wavefront / CUDA-graph scheduler (M0 DC-intra)

Ownership: `src/cuda/scheduler/` (wave 1, workstream 8). This module is
**additive**: it does not edit `src/cuda/basic_pipeline.*`, `src/main.cpp`, the root
`CMakeLists.txt`, `src/encoder/`, `src/entropy/`, `src/bitstream/`, `tests/`, or
another agent's files.

## Problem

The M0 CUDA pipeline walks coding blocks in raster order and issues one
component call per stage per block. At 640x360 that is **2,740 component calls**
(3,580 actual kernel launches, because `forward_transform` and
`inverse_transform` each launch two kernels) and ~26 ms of GPU time, almost all
of it launch latency (`docs/negative-results.md`: "one kernel per M0 stage …
2,740 launches and is launch-bound at ~34 ms/frame"). At 1080p the same scheme
extrapolates to **25,380 component calls / 33,180 kernels** and ~236 ms.

The obstacle is not the arithmetic but the DC predictor: block *i* reads the
reconstructed column at `x0-1` and row at `y0-1`, so blocks in a row are
serially dependent. The scheduler exploits the only available parallelism —
the anti-diagonal wavefront — and removes the per-block launch cost.

## Design

### Dependency graph

Every real plane block (`Y`, `U`, `V` are independent planes) becomes a node.
Node `B` gets an edge from every block that owns a cell in

* column `B.x-1`, rows `[B.y, B.y+B.h)` (left prediction samples), and
* row `B.y-1`, columns `[B.x, B.x+B.w)` (above prediction samples).

The plan builds an exact coverage map per plane (4-sample cells for luma,
2-sample cells for chroma, both aligned to the M0 geometry) from the coding
block list, so a 64x64 luma block correctly depends on *two* 32x32 blocks when
its neighbours are split. Out-of-plane samples are never read (the predictor
clamps), so no extra edges are needed at the picture edge.

Wavefront level = `1 + max(level of prerequisites)` (longest path over the DAG,
computed with Kahn's algorithm). All nodes in a level are independent, and every
edge crosses a level boundary; `PlanCounters::dependency_violations` is the
number of edges that violate that rule (0 in every test).

At 640x360 the chroma plane is the critical chain: 20 chroma block columns +
10 rows - 1 = 29, plus the bottom edge row = **30 levels**. At 1080p it is
**92 levels**. Luma finishes much earlier; the three planes overlap in the same
level sequence because they do not depend on each other.

### Batching

For each level:

| Work | Batching |
|---|---|
| prediction + residual | one `intra::dc_predict_residual_batch` for every independent plane block in the level |
| forward DCT + quantise | one `transform::forward_transform` per (level, transform class) |
| dequantise + inverse DCT | one `transform::inverse_transform` per (level, transform class) |
| reconstruction | one batched launcher per (level, transform class) |
| 64x64 luma residual downsample | one `transform::downsample_residual_64x64` per 64x64 block (edge blocks only) |
| canonical coefficient scatter | one kernel per frame |

A **transform class** is a group of blocks whose `make_tx_params()` inputs are
identical: for the M0 tree there are three classes — 32x32 (all 32x32 luma and
32x32 chroma), 16x16 (chroma of a 32x32 luma block), and 64x64-adjusted-32x32
(64x64 edge luma). U and V blocks of the same class share a batch, so a chroma
wave costs five kernels, not ten.

Jobs are ordered `(level, class, plane, y, x)`. Class buffers are indexed in
that same order, so each `(level, class)` group is a contiguous run in both the
job array and the class buffers, and the batched transform can be launched with
a plain `pointer + slot * pels` offset.

### Per-frame sequence and CUDA graph

```
upload source planes (pinned staging)
for level in 0..levels-1:
    dc_predict_residual_batch(level jobs)
    for each (level, class) group:
        [64x64 luma: downsample]
        forward_transform   (forward DCT, quantise)
        inverse_transform   (dequantise, inverse DCT)
        recon               (clip / 2x2 duplication)
scatter coefficients into the canonical per-block layout
download recon planes + coefficients + predictions
```

`ScheduledFrame::prepare()` allocates one persistent arena, uploads the device
job descriptors, runs the sequence once un-captured (CUDA module warm-up),
captures it with `cudaStreamBeginCapture`, instantiates the graph, and performs
one warm replay (so the one-off graph upload does not land in the first
measured frame). `run()` then does 3 H2D copies, `cudaGraphLaunch`, 5 D2H
copies, one synchronisation, and the host-side copies into the caller buffers.

Capture is best-effort: if the driver refuses, `graph_available()` is false and
`run()` falls back to the eager batched sequence (still ~9x fewer launches than
per-block). `graph_error()` carries the reason.

### Bit-exactness

No normative arithmetic is re-implemented. Prediction/residual, forward DCT,
quantisation, dequantisation, inverse DCT and the 64x64 downsample are the
shared components (`src/cuda/intra.*`, `src/cuda/transform.*`) called with the
same arguments as `basic_pipeline.cu`. Batching only changes *when* independent
invocations run, never their inputs.

The single piece of glue is `recon_batch_kernel` (in `scheduler.cu`): a batched
form of `intra::add_residual_clip_to_plane` / `transform::reconstruct_64x64`
(`clip1(pred + residual)`, with 2x2 duplication for the adjusted 64x64 path and
`preds[]` written from `pred_block[0]`). It exists because the intra component
has no batch entry point; the harness compares every reconstructed sample,
coefficient and predictor against the per-block path. If
`add_residual_clip_to_plane_batch` is added upstream this kernel can be deleted.

## API

```cpp
#include "cuda/scheduler/scheduler.h"

using namespace av2enc::cuda::sched;

// Once per frame layout (outside the per-frame path):
std::vector<CodingBlock> blocks = blocks_from(basic_blocks, block_count);  // adapter
FrameJob job;
job.y = ...; job.u = ...; job.v = ...;          // source planes
job.geometry = ...;                              // widths, padded sizes, mi grid
job.blocks = blocks.data();
job.block_count = block_count;
job.total_coeffs = total_coeffs;
job.recon_y = ...; job.recon_u = ...; job.recon_v = ...;  // padded planes
job.coeffs = ...; job.preds = ...;               // canonical M0 layout

ScheduledFrame encoder;
std::string error;
if (!encoder.prepare(job, /*device=*/0, error)) { /* report error */ }

RunOptions options;              // options.use_cuda_graph = true (default)
RunStats stats;
for (each frame) {
  job.y = frame.y; job.u = frame.u; job.v = frame.v;   // new source pointers
  if (!encoder.run(job, options, &stats, error)) { ... }
  // stats.kernel_ms, stats.submit_ms, stats.used_graph, stats.gpu_nodes,
  // stats.host_launches, stats.workspace_bytes
}
```

| Type | Purpose |
|---|---|
| `CodingBlock` | one M0 coding block (`mi_row/mi_col`, luma/chroma sizes, upscale, pels, canonical `coeff_offset`, `real_coeffs`); same fields as `detail::BlockDesc` |
| `blocks_from()` | template adapter for `detail::BlockDesc` (no `basic_pipeline.h` include needed) |
| `FrameGeometry` | visible/padded sizes and `mi_cols/mi_rows` |
| `FrameJob` | source planes, block list, canonical output buffers |
| `Plan` | dependency analysis, levels, groups, classes, counters; buildable on a host with no GPU |
| `PlanCounters` | jobs, levels, groups, classes, edges, violations, naive vs scheduled launch counts |
| `ScheduledFrame` | persistent device workspace + captured graph; `prepare` once, `run` per frame |
| `RunStats` | host launches, graph nodes, upload/kernel/download/submit ms, workspace bytes |

`Plan::build` is also usable standalone for inspection or a different backend:
`plan.jobs()` is level-major, `plan.groups()` is the `(level, class)` batching,
`plan.level_begin(level)` / `level_end(level)` delimit a wave.

`run()` rejects a frame whose block list (FNV-1a signature), block count,
geometry or `total_coeffs` differs from what `prepare()` built, so a layout
change requires a new `prepare()`.

## Results

Conditions: Vast.ai RTX 3060 12 GB (dev proxy for the 8 GB target), CUDA 12.8,
`sm_86`, 56-core shared Xeon, `-O3 --fmad=false`. Means after warm-up, 5–15
iterations. Run-to-run variation is ~±5–10% on this shared box.

Plan shape (validated by `scheduler_plan_test`):

| Picture | Coding blocks | Plane jobs | Wavefront levels | Classes | Groups | Dependency edges | Violations | M0 component calls | M0 kernels | Wavefront kernels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64x64 | 4 | 12 | 3 | 2 | 6 | 12 | 0 | 52 | 68 | 34 |
| 640x360 | 210 | 630 | 30 | 3 | 69 | 1,197 | 0 | 2,740 | 3,580 | 386 |
| 1280x720 | 900 | 2,700 | 62 | 3 | 143 | 5,271 | 0 | 11,720 | 15,320 | 798 |
| 1920x1080 | 1,950 | 5,850 | 92 | 3 | 213 | 11,511 | 0 | 25,380 | 33,180 | 1,188 |

Measured (CUDA-graph replay; "kernel" = event time around the GPU work,
"wall" = full `run()` call including upload/download/host copies):

| Picture | M0 kernel ms | graph kernel ms | M0 wall ms | graph wall ms | kernel speedup | wall speedup | Device workspace |
|---|---:|---:|---:|---:|---:|---:|---:|
| 64x64 | 0.52 | 0.33 | 0.57 | 0.37 | 1.6x | 1.6x | 0.16 MiB |
| 640x360 | 25.7 | 3.88 | 26.2 | 4.39 | 6.6x | 6.0x | 9.0 MiB |
| 1280x720 | 109.0 | 8.32 | 111.0 | 10.12 | 13.1x | 11.0x | 37.1 MiB |
| 1920x1080 | 236.1 | 15.79 | 241.5 | 20.75* | 15.0x | 11.6x* | 79.9 MiB |

\* 1080p wall is 20.7 ms in steady state (15 iterations). A 5-iteration run
reports 22–26 ms because the first frames overlap the GPU clock ramp; the kernel
window is stable at 15.8–15.9 ms.

Batch-only (no graph) shows the split: at 1080p the eager wavefront path is
16.6 ms kernel / 22.8 ms wall with 1,188 host calls and **2.7 ms of host
submission**; graph replay keeps the same 1,188 nodes but reduces host
submission to ~0.05 ms. At 640x360: eager 4.13 ms kernel, graph 3.88 ms.

Interpretation:

* The **launch count** drops 9.3x at 640x360 and 27.9x at 1080p. This is the
  metric the module was created for.
* At 640x360 the remaining 3.9 ms is the serial wavefront itself — 30 steps,
  roughly five dependent kernels each — plus the transform work. The 2,740-call
  host dispatch bottleneck documented in `docs/negative-results.md` is gone.
* At 1080p the whole M0 block stage costs ~21 ms/frame instead of ~242 ms. The
  product target is 1,000 ms/frame; entropy is still CPU-side and is not part of
  this measurement, so this only proves the block stage leaves ample headroom.
* 64x64 is too small to benefit much (34 vs 68 kernels); the fixed per-frame
  cost dominates.

## Correctness

The harness (`scheduler_harness.cpp`) runs the same block list through

1. the per-block M0 path (`detail::run_gpu_job` — the production code),
2. the wavefront plan executed eagerly, and
3. the wavefront plan replayed from the CUDA graph,

and requires **bit-exact** equality of reconstructed Y/U/V (padded planes),
all quantised coefficients and all three predictors, on 64x64, 640x360,
1280x720 and 1920x1080. It also re-runs the replay after all benchmarks
(steady-state reuse of the persistent device buffers) and re-checks.

It then runs `encode_still_picture_cuda` (the real M0 planner and entropy path)
and verifies that its reconstruction matches the same block list — this
validates the test geometry replication against `PartitionPlanner` — and that
its component-call count equals the plan's prediction (2,740 at 640x360).

Because the scheduler reproduces the exact coefficients and predictions that
feed the CPU entropy writer, the bitstream and decoded output are unchanged:
`build_scheduler_test.sh` runs `tools/validate_dav2d.sh` on the IVF written by
the production path and passes the pinned MD5s:

| Case | Decoded MD5 | Result |
|---|---|---|
| 64x64 | `7efc3162bc0a4785685eba291a8096fd` | ok |
| 640x360 | `56f11d4a63ff07f2356ef5064b84fe0e` | ok |

`compute-sanitizer --tool memcheck` on the 640x360 case: **0 errors**.
`PlanCounters::dependency_violations` is 0 for every geometry.

## Integration notes (for the M0 pipeline / integration agent)

`src/cuda/basic_pipeline.cpp` is intentionally untouched. To wire the scheduler in:

1. **Plan lifetime.** Build the `CodingBlock` list once (from
   `PartitionPlanner`), then `ScheduledFrame::prepare()` once per frame layout,
   device, and profile. Keep the `ScheduledFrame` in `PipelineRuntime` next to
   `Context`/`Arena`; `prepare()` is the only place that allocates (one arena,
   ~80 MiB at 1080p, well inside `compat8g`).
2. **Per frame.** Point `FrameJob::y/u/v` at the pinned-upload source (or at the
   existing device planes once a device-resident path exists), call `run()`, and
   feed the returned `coeffs`/`preds` to the existing CPU entropy writer
   unchanged. The layouts are the M0 layouts (`coeff_offset`, three `preds` per
   coding block), so no bitstream code changes.
3. **Timings.** `PipelineTimings::kernel_launches` should become
   `stats.host_launches` (1 with a graph) and a new `gpu_nodes` field should
   carry `stats.gpu_nodes`; otherwise the profiler will report one launch and
   hide the wavefront. Suggest logging `levels`, `groups`,
   `dependency_violations`, `used_graph` and `workspace_bytes` per frame.
4. **Arena/stream ownership.** `ScheduledFrame` owns a private `Context` +
   `Arena` because `basic_pipeline`'s runtime is file-local. Both arenas count
   against the 7,500 MB process cap; at 1080p that is ~80 MiB, so the M0
   pipeline's 7,244 MiB reservation still fits (7.3 GiB total). If the runtime
   agent exposes the shared `Arena`/stream, an overload of `prepare(job, arena,
   stream, ...)` is the natural next step (no changes to `run()`).
5. **Diagnostic modes.** Blocks with `real_coeffs == 0` are skipped exactly like
   the current GPU job; their reconstruction stays 128 and their coefficients
   stay 0. `64x32`/`32x64` coding blocks are rejected by the transform
   component in both paths (pre-existing limitation).
6. **Requested upstream change.** Add
   `intra::add_residual_clip_to_plane_batch(const ReconJob*, int)` so the local
   recon kernel can be removed; its arithmetic is already identical.

## Limitations

* The wavefront depth is inherent: per-row DC chains give `W/32 + H/32` luma
  levels and about `W/16 + H/16` chroma levels (92 at 1080p). Levels cannot be
  merged without changing the normative reconstruction order.
* Launch count is content-independent: M0 has no early-out, so the counts above
  hold for any frame of the same size.
* The measured host is the 12 GB dev proxy. The 8 GB 3060 has less bandwidth;
  these kernels are tiny and mostly compute/latency-bound, but the numbers
  still need validation on an 8192 MiB card (`tools/check_target_host.py`).
* Only the M0 `k32x32` tree is exercised end-to-end here. The plan itself is
  general (any exact tiling); the test geometry helper is `k32x32`-only.
* Entropy stays on the CPU (M0 policy). This module only makes the block stage
  launch-free.

## Files and build

| File | Purpose |
|---|---|
| `scheduler.h` | public API (no CUDA types in the header) |
| `scheduler.cpp` | host-only plan: coverage, prerequisite DAG, levels, classes, launch counters, validation |
| `scheduler.cu` | device workspace, batched launch sequence, graph capture/replay, batched recon + coefficient scatter |
| `scheduler_plan_test.cpp` | host-only plan/dependency/validation/launch-count tests |
| `scheduler_harness.cpp` | per-block vs eager vs graph benchmark, bit-exact comparison, production cross-check, IVF emit |
| `basic_geometry.h` | test-only `k32x32` block-list builder (validated against `PartitionPlanner`) |
| `build_scheduler_test.sh` | standalone build + test driver |

```sh
# Host-only (works on macOS; no CUDA needed)
src/cuda/scheduler/build_scheduler_test.sh --plan-only

# Full: plan test + GPU harness + dav2d MD5 gate (CUDA host)
src/cuda/scheduler/build_scheduler_test.sh

# Options: --quick (skip the generated 1080p case), --no-validate
# Env: CUDA_ARCH=sm_86, BUILD_DIR=..., DAV2D=..., CXX=..., NVCC=...
```

Exit codes: 0 pass, 1 failure, 2 usage, 77 CUDA/inputs unavailable.

The GPU harness can also be run alone:

```sh
./build-scheduler/scheduler_harness --y4m tests/data/basic_640x360_1f.y4m \
    --out-dir /tmp/out --iterations 10
./build-scheduler/scheduler_harness --width 1920 --height 1080 --iterations 15
```
