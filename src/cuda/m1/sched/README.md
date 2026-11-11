# M1 wavefront / CUDA-graph scheduler adapter

Ownership: `src/cuda/m1/sched/` (workstream 8). The module is additive: it
does not edit `src/cuda/m1/` root files, `src/encoder/m1/`, `src/main.cpp`,
`CMakeLists.txt`, `src/cuda/scheduler/`, or another workstream's directory.

## Problem

M1 codes a still as a partition tree of leaves down to 8x8, and every leaf has
a luma block plus U/V chroma (4:2:0). The CPU-shaped GPU pipeline would run
prediction, forward transform/quantise, inverse transform and reconstruction
per block in syntax order. For the tested 640x360 frame that is 218 leaves /
654 plane blocks and an estimated **3,736 component launches**; the fine-tree
1080p case is 3,133 leaves / 9,399 plane blocks and **43,161** per-block calls.

Adjacent blocks are not independent: intra prediction reads reconstructed
neighbours, and CfL reads reconstructed luma. The scheduler exposes the legal
parallelism instead of the syntax order:

* build the real coverage-based dependency graph (left column, above row,
  above-left corner, directional above-right / below-left extensions, and the
  chroma -> luma CfL region);
* assign each plane block a wavefront level `1 + max(level of prerequisites)`;
* dispatch each level as one batched sequence (`predict -> transform ->
  reconstruct`) and capture the whole frame into one CUDA graph.

## Dependency model

All planes use **4-sample coverage cells**, which matches the encoder's own
accounting: luma MI cells are 4 luma samples and `count_top_right` /
`count_bottom_left` in `m1_still.cpp` index chroma in 4-chroma-sample units
(the chroma map is `mi_cols/2` wide).

For every plane block:

| Edge source | Region |
|---|---|
| left column | `x0-1`, rows `[y0, y0+h)` |
| above row | `y0-1`, columns `[x0, x0+w)` |
| above-left corner | `(x0-1, y0-1)` |
| above-right extension | `y0-1`, `[x0+w, x0+w + 4*num4AboveRight)` |
| below-left extension | `x0-1`, `[y0+h, y0+h + 4*num4BelowLeft)` |
| chroma cross-plane | reconstructed luma over the co-located leaf plus the CfL filtered above/left taps (§7.13.5) |

`num4AboveRight` / `num4BelowLeft` are **replayed exactly** from the block list
in syntax order, using the same superblock-local decoded-map and border-preset
rules as `clear_block_decoded_flags` / `count_top_right` /
`count_bottom_left` in `m1_still.cpp`. This matters: the extension is not just
"one block up and right". At the top row of a superblock the above-right count
can span two superblocks (`border_w = min(2*SB, tile_remainder)`), and at a
below-left superblock edge the count is unbounded on the left column. Using a
generic neighbour edge would under- or over-constrain later blocks.

Chroma adds an edge from every luma block owning a sample in
`[2x-3, 2x+2w) x [max(sb_top-1, 2y-3), 2y+2h)`: the block itself for
`sample_luma`, plus the one filtered row/column above-left used by the derived
alpha. The region is deliberately expanded by the filter taps so it stays
correct for any `cfl_ds_filter_index`. This is a superset for a DC-only chroma
block and exact for CfL; extra edges never move a block into a *lower* level.

The graph is a DAG and CPU syntax order is one topological order (edges always
go from an earlier leaf, or from Y to U/V inside a leaf). The plan counts both:

* `dependency_violations` – edges whose prerequisite is not in a strictly
  lower level;
* `syntax_order_violations` – edges that would run backwards in CPU order.

Both are 0 for every tested geometry, and the GPU reconstruction is bit-exact,
which is the empirical check that no needed edge is missing.

### A finding: directional extensions deepen the wavefront

Without extensions the luma depth is the anti-diagonal `W/8 + H/8`; with the
exact `num4AboveRight` reads it is deeper because the above-right neighbour is
on the same anti-diagonal and therefore one level ahead. All-8x8 plans measure:

| Picture | Anti-diagonal | Exact-extension levels |
|---|---:|---:|
| 64x64 | 16 | 23 |
| 640x360 | 125 | 197 |
| 1920x1080 | 375 | 599 |

Real M1 trees have larger blocks and fewer leaves, so the tested frames are
shallower (640x360: 78 levels; 1080p: 324). The extension reads happen while
`intra_modes` builds the edge arrays even for DC, so they are real dependencies,
not a conservative over-approximation.

## Batching

`LevelBatch` is a view of one level: the contiguous job range, the
`(level, TX class)` spans, and counts of luma / chroma / CfL jobs. Jobs are
stored level-major, then TX class, then CPU syntax index, so:

* luma and non-CfL chroma share one heterogeneous `intra_predict_batch`
  (`IntraJob` carries its plane, modes, source and recon pointers);
* mixed transform sizes use one `txv_forward_transform_multi` and one
  `txv_inverse_transform_multi` per level;
* reconstruction is one kernel per level;
* a scaled TX class (64-wide/tall coded size) adds one uniform-params
  `txv_downsample_residual` per `(level, class)` span;
* CfL adds the DC-predict / stage-to-plane / derived-alpha / stage-back /
  residual chain for that level.

The plan's `scheduled_kernel_launches` matches the verification executor's
actual graph node count exactly in every tested case.

## API

Header-only plan (no CUDA types), `sched.h`:

```cpp
using namespace av2enc::cuda::m1::sched;

Plan plan = build_m1_plan(frame_plan);       // sched::FramePlan (geometry view)
const PlanCounters& c = plan.counters();     // levels, edges, violations, ...
LevelBatch batch = plan.level_batch(level);  // no allocation
```

The shared orchestrator contract (`src/cuda/m1/m1_gpu.h`) is the integration
entry point (`sched_bridge.h`):

```cpp
av2enc::cuda::m1gpu::FramePlan shared;        // plan stage output
av2enc::cuda::m1gpu::BlockDecision* decisions; // CPU plan (stage 1) or GPU search

Plan plan;
std::string error;
build_m1_plan(shared, decisions, &plan, error);
```

Execution (`sched_exec.h`, CUDA):

```cpp
LevelOps ops;      // predict / transform / reconstruct / optional post
ops.user = state;
ops.predict = ...; // enqueue only: no alloc, no sync, no host callbacks

std::string error;
if (!run_level(plan, level, ops, stream, error)) { ... }   // one level
run_levels_eager(plan, ops, stream, &stats, error);        // whole frame

LevelGraph graph;   // plan + callbacks must outlive it
graph.prepare(plan, ops, stream, error);  // warm-up, capture, instantiate
graph.run(ops, options, &stats, error);   // cudaGraphLaunch, or eager fallback
```

`LevelGraph::prepare` runs the sequence once eagerly (module warm-up), captures
it with `cudaStreamBeginCapture`, instantiates, and performs one warm replay.
Capture failure (driver refusal / `BeginCapture` error) leaves
`graph_available() == false`; `run()` then uses the eager batched sequence. A
callback failure *inside* capture invalidates the stream, so that case returns
an error instead of pretending the fallback is safe. The plan, callbacks and
all device buffers must outlive the graph; per-frame data changes must be
uploads into the same device pointers, not new allocations.

`run_level` performs `predict -> transform -> reconstruct` and calls the
optional `post` hook once per frame after the last level. It does not allocate,
synchronize, or inspect device data.

## Verification executor

`sched_ref_bench.cu` implements the stage callbacks with the shared components:

| Stage | Component |
|---|---|
| intra prediction + residual | `cuda/intra_modes` (`intra_predict_batch`) |
| CfL | `cuda/cfl` (`cfl_predict_derived_batch`) |
| forward / inverse TX | `cuda/tx_variants` (`_multi`, `--fmad=false`) |
| reconstruction | local `clip1(pred + residual)` kernel |
| CfL residual | local `source(clamped) - CfL pred` kernel |

The two local kernels are the only arithmetic glue; they apply exactly the
expressions of `m1_still.cpp`'s `compute_tx`/`write_recon` and the CfL path.
The executor is verification-only (file name `*_bench.cu` is excluded from the
product glob); the orchestrator supplies its own `LevelOps`.

## Test method

`sched_bench`:

1. reads the first Y4M frame and runs the CPU M1 encoder with
   `AV2ENC_M1_TRACE=1`, `enable_deblock=false`; stderr is captured in-process
   and parsed into per-leaf geometry/modes;
2. checks the trace leaf count against `M1Result::leaves`;
3. builds the plan from those decisions and requires
   `dependency_violations == 0` and `syntax_order_violations == 0`;
4. runs the reference executor through the CUDA graph and eagerly, then
   compares every visible Y/U/V sample of the padded reconstruction against
   `M1Result::recon_*` (bit-exact, no tolerance);
5. reports best-of-N kernel time (CUDA events around the block sequence),
   graph nodes, host submissions, workspace and the plan counters.

Deblock is disabled on both sides; deblocking happens after intra prediction
and does not change the reconstruction the block pipeline produces. The CPU
encoder is the oracle for the same decisions; the harness never reimplements
mode search.

## Results

Host: Vast.ai RTX 3060 12 GB (8 GB target proxy), CUDA 12.8, `sm_86`, shared
56-core Xeon. `-O3 --fmad=false`, best of 5-6 iterations after warm-up.
Run-to-run variation on this box is ~±5-10%.

Real CPU M1 decisions (default DC luma + CfL chroma; deblock off):

| Picture | Leaves | Plane jobs | Levels | Classes | Groups | Edges | Nodes | Graph kernel | Eager kernel | Host submit (graph) | Exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 64x64 | 11 | 33 | 10 | 11 | 21 | 123 | 80 | 0.53 ms | 0.59 ms | 0.003 ms | yes |
| 640x360 | 218 | 654 | 78 | 20 | 344 | 4,021 | 689 | 9.15 ms | 9.39 ms | 0.007 ms | yes |
| 1920x1080* | 3,133 | 9,399 | 324 | 18 | 2,799 | 59,675 | 3,149 | 36.9 ms | 39.1 ms | 0.012 ms | yes |

\* generated 1080p gradient+noise frame (3.1 MB/frame luma+chroma), real CPU
tree. It is a high-detail frame (3133 leaves) and therefore a fine-tree upper
bound, not a natural-content measurement.

Full directional intra (`M1Options::luma_dc_only = false`) gives the same
bit-exact result and exercises directional prediction, the smooth-neighbour
filter type and the extension edges:

| Picture | Leaves | Plane jobs | Levels | Nodes | Graph kernel | Exact |
|---|---:|---:|---:|---:|---:|---|
| 64x64 | 11 | 33 | 10 | 70 | 0.49 ms | yes |
| 640x360 | 218 | 654 | 78 | 694 | 9.28 ms | yes |

Synthetic all-8x8 plans (worst-case job/level count for the picture size):

| Picture | Blocks | Jobs | Levels | CfL | Nodes | Graph kernel | Eager kernel |
|---|---:|---:|---:|---:|---:|---:|---:|
| 640x360 | 3,600 | 10,800 | 197 | half | 1,278 | 5.30 ms | 6.08 ms |
| 1920x1080 | 32,400 | 97,200 | 599 | half | 3,891 | 25.6 ms | 28.0 ms |
| 1920x1080 | 32,400 | 97,200 | 599 | all | 5,385 | 32.1 ms | 35.4 ms |

Interpretation:

* Launch count: 3,736 per-block calls collapse to **689 graph nodes** at
  640x360 (5.4x) and 43,161 to **3,149** at 1080p (13.7x); host submission for
  the frame drops from 1.9 ms / 26.8 ms (eager, one host call per kernel) to
  ~0.01 ms (one `cudaGraphLaunch`).
* The remaining time is not launch *submission* but the serial wavefront:
  78 dependent levels at 640x360 and 324 at 1080p, each with a
  predict -> TX -> recon chain and, where needed, the five-kernel CfL chain.
  The reference executor is component-faithful, not fused; a production
  `LevelOps` that fuses the CfL staging or the TX pair will cut nodes and time.
* 1080p block stage is **37 ms** on the 3060 with real M1 decisions and
  **26-32 ms** for the synthetic all-8x8 shapes, far below the 1000 ms/frame
  product target (entropy is CPU-side and not included). The M0 scheduler
  reaches 15.8 ms at 1080p with 92 levels and three planes; M1's finer tree
  (324 levels) and variable transforms explain the difference.
* The 64x64 case is too small to amortise the graph; its fixed cost dominates.

## Limitations

* The verification executor runs the per-level stage components separately;
  it does not fuse prediction+residual+TX or the CfL staging. Numbers are a
  valid launch/dependency measurement, not a tuned kernel result.
* Deblock and coefficient scatter are `post` hooks in the API but are not part
  of the verified path (deblock is off in the CPU oracle; entropy stays CPU).
* The plan assumes the leaf list is in **CPU syntax order** (superblock raster,
  then depth-first). The 4x4 leaf / `has_chroma == false` path is implemented
  but unreachable at the M1 default 8-pixel floor, so it is untested against
  the CPU.
* The plan uses the CPU's exact availability replay, which includes
  above-right reads that DC prediction does not mathematically need; an
  implementation that shrinks the edge arrays for DC can expose a shallower
  wavefront, but the shared `intra_modes` component reads them today.
* Measurements are on the 12 GB dev proxy. The 8 GB 3060 has less bandwidth;
  these kernels are small and mostly latency-bound, but the numbers still need
  validation on an enumerated 8192 MiB card.
* Windows is unsupported; macOS runs only the host plan test.

## Files and build

| File | Purpose |
|---|---|
| `sched.h` | host plan PODs, `Plan`, `LevelBatch`, `build_m1_plan` |
| `sched.cpp` | coverage, exact availability replay, dependency levels, spans |
| `sched_bridge.h/.cpp` | `m1gpu::FramePlan` + `BlockDecision` adapter |
| `sched_exec.h` | `LevelOps`, `run_level`, `run_levels_eager`, `LevelGraph` |
| `sched_exec.cu` | eager dispatch, CUDA-graph capture/replay, eager fallback |
| `ref_exec.h`, `sched_ref_bench.cu` | verification stage executor (excluded from product glob) |
| `sched_plan_test.cpp` | host-only plan tests (also runs on macOS) |
| `sched_bench.cpp` | CPU-vs-GPU bit-exact harness, benchmark, 1080p projection |
| `build_sched_test.sh` | standalone build + test driver |

```sh
# Host-only plan tests (no CUDA; works on macOS)
src/cuda/m1/sched/build_sched_test.sh --plan-only

# Full: plan + GPU bit-exact harness + synthetic 1080p projection
src/cuda/m1/sched/build_sched_test.sh

# Options: --quick (2 projection iterations), --no-projection,
#          --full-intra, --iterations N, --y4m FILE (repeatable)
# Env: CUDA_ARCH=sm_86, BUILD_DIR=..., CXX=..., NVCC=..., CUDA_INCLUDE=...
```

Exit codes: 0 pass, 1 failure, 2 usage, 77 CUDA/inputs unavailable.

To reproduce the 1080p real-tree result, generate a 1920x1080 4:2:0 Y4M frame
(any high-detail content) and pass it:

```sh
build-m1-sched/out/sched_bench --y4m path/to/1080p_1f.y4m \
    --no-projection --iterations 5
```
