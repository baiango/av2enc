# Profiling — av2enc

The encoder is **CUDA/Linux-first**. Profiling work happens on the Vast.ai
RTX 3060 box; macOS is not an encoder target.

This document covers two things:

1. The **built-in profiler** in `src/core/telemetry/`: low-overhead stage timers, CUDA
   event timings, optional NVTX ranges, JSONL events, a JSON summary, and VRAM
   accounting. It is the regression tool because it always works, even when
   hardware counters are blocked.
2. **External profilers** for deep dives: Nsight Systems on Linux, plus the
   limitations of Nsight Compute, `gprof`, `perf`, and
   `-finstrument-functions`. macOS `xctrace` is a short, untested note.

Measured 2026-11-11 on the Vast.ai RTX 3060 box (Ubuntu 24.04 container, CUDA
12.8, driver 570.181, GCC 13.3, Nsight Systems 2025.1.3).

## TL;DR

```sh
# Rebuild the encoder, then:
AV2ENC_PROFILE=/tmp/run.jsonl ./build/av2enc input.y4m -o out.ivf --frames 60

# Files:
#   /tmp/run.jsonl                 per-event JSONL
#   /tmp/run.jsonl.summary.json    aggregate JSON, written on close
# The summary always records precompute_profile=auto and vram_limit_mb=0.

jq '.stages[] | select(.cpu.calls > 0)' /tmp/run.jsonl.summary.json
jq 'select(.type == "frame") | .wall_ms' /tmp/run.jsonl

# Prove 8 GB compatibility while running on the 12 GB card:
AV2ENC_PROFILE=/tmp/run.jsonl ./build/av2enc ...
```

Nsight Systems timeline with NVTX stage ranges:

```sh
AV2ENC_PROFILE_NVTX=1 \
nsys profile --trace=cuda,nvtx,osrt --sample=cpu --force-overwrite true \
  -o /tmp/av2enc-run ./build/av2enc input.y4m -o out.ivf --frames 10
nsys stats --report nvtx_sum,cuda_gpu_kern_sum,osrt_sum \
  /tmp/av2enc-run.nsys-rep
```

## Why a built-in profiler

The Vast.ai container cannot use the usual deep-profiling tools reliably:

| Tool | Status on the 3060 box | Consequence |
|---|---|---|
| `ncu` (Nsight Compute) | Present, but GPU performance counters are blocked: `ERR_NVGPUCTRPERM` | Per-kernel occupancy, memory throughput, and stall reasons are unavailable. This is an unprivileged-container restriction; do not try to change driver permissions from inside. |
| `nsys` (Nsight Systems) | Present at `/usr/local/bin/nsys`, 2025.1.3, working | Use for timelines, NVTX, CUDA API, kernel durations, OS runtime. |
| `perf` / eBPF | Usually blocked by the container | Treat as unavailable unless `perf stat true` proves otherwise. |
| `compute-sanitizer` | Working | Use for memory errors, not performance. |

The built-in profiler measures the pipeline on every run, with no external tool
and no special build. It answers the regression question: *which stage got
slower?* Nsight Systems answers the deep-dive question: *why is that kernel
slow?*

## Enabling the built-in profiler

| Environment variable | Meaning |
|---|---|
| `AV2ENC_PROFILE=<path>` | Write per-event JSONL to `<path>`. On close, write `<path>.summary.json`. The path must be writable. |
| `AV2ENC_PROFILE_NVTX=1` | Emit NVTX ranges. Requires compiling `profiler.cpp` with `AV2ENC_HAVE_NVTX=1` and linking `nvToolsExt`. Can be used without `AV2ENC_PROFILE` for an NVTX-only run. |
| `AV2ENC_VRAM_LIMIT_MB=<MiB>` | Optional VRAM accounting budget. Default **0** = no fixed limit (the arena is device-sized); a non-zero value enables the over-budget warning. |

Truthy values are `1`, `true`, `yes`, and `on`. With neither variable set, the
profiler is disabled: every scope is a no-op branch, no JSON is written, and no
CUDA queries are made.

`AV2ENC_PROFILE=<path>` enables JSON logging. `AV2ENC_PROFILE_NVTX=1` alone
enables NVTX ranges and in-memory statistics but no JSONL file; this is useful
under `nsys`, where the JSON overhead is unnecessary.

If the JSONL path cannot be opened, `open_from_env()` returns `false` and fills
in an error string. Treat that as a warning, not an encoder failure.

## Output format

### JSONL event log

One JSON object per line. CPU and CUDA events use the same schema:

```json
{"type":"event","seq":0,"frame":0,"stage":"input_decode","kind":"cpu","thread":14910412935604673722,"at_ms":0.108459,"dur_ms":0.217208}
{"type":"event","seq":1,"frame":0,"stage":"transform_quant","kind":"cuda","thread":14910412935604673722,"at_ms":0.301002,"dur_ms":0.006144}
```

| Field | Meaning |
|---|---|
| `type` | `"event"` for a stage sample. A `"frame"` record is also written by `end_frame()`. |
| `seq` | Log order. CUDA events are written when resolved at `end_frame()`, so `seq` is not device execution order. |
| `frame` | Frame index passed to `begin_frame()`. |
| `stage` | Registered stage name. |
| `kind` | `"cpu"` for `CpuScope`, `"cuda"` for `CudaScope`. CPU and CUDA samples are aggregated separately. |
| `thread` | Hash of `std::thread::id`. Stable within a process; not an OS TID. |
| `at_ms` | Milliseconds since profiler activation. For CPU, the scope start. For CUDA, the **host launch time**, not the device start time. |
| `dur_ms` | Scope duration in milliseconds, six fractional digits (1 ns resolution for CPU; CUDA event resolution is coarser, about 0.5 µs). |

A frame record is emitted for each `begin_frame()`/`end_frame()` pair:

```json
{"type":"frame","frame":0,"wall_ms":1.902183}
```

`wall_ms` is the host wall time from `begin_frame()` to `end_frame()`, including
the profiler's CUDA event flush. It is the frame-level number for rate and
latency tracking.

### JSON summary

Written to `<path>.summary.json` on `close()` (also called by the destructor):

```json
{
  "profiler": "av2enc",
  "version": 1,
  "jsonl": "/tmp/run.jsonl",
  "precompute_profile": "auto",
  "vram_limit_mb": 0,
  "wall_ms": 123.456789,
  "frames": 60,
  "events": 780,
  "cuda_events_dropped": 0,
  "cuda_errors": 0,
  "write_error": false,
  "peak_vram_mb": 842.13,
  "vram_capacity_mb": 11920.75,
  "vram_free_mb": 11078.62,
  "threads": [
    {"id": 123456789, "cpu_calls": 720, "cuda_calls": 60}
  ],
  "stages": [
    {
      "stage": "intra_search",
      "cpu": {"calls": 60, "total_ms": 12.3, "min_ms": 0.18, "max_ms": 0.31, "mean_ms": 0.205},
      "cuda": {"calls": 60, "total_ms": 42.1, "min_ms": 0.65, "max_ms": 0.82, "mean_ms": 0.701}
    }
  ]
}
```

Notes:

- `stages` includes every registered stage, even ones with zero calls, so the
  schema is stable.
- `cpu.total_ms` can exceed `wall_ms`: `total` and parent scopes overlap nested
  children. Never sum stage totals and call it frame time.
- `mean_ms` is integer-nanosecond total divided by calls.
- `cuda_events_dropped` counts `CudaScope` intervals lost because the pending
  event pool filled up. It stays zero if `end_frame()` is called every frame.
- `precompute_profile` and `vram_limit_mb` are configuration fields and are
  always recorded. The default label is `auto`; there is no fixed VRAM cap.
- The measured VRAM fields (`peak_vram_mb`, `vram_capacity_mb`, `vram_free_mb`)
  are present only when a CUDA device is visible. They are omitted in CPU-only
  builds and on hosts without a CUDA device; they are never written as zeros.

### Reading it with jq

```sh
# Per-stage CPU and CUDA totals
jq -r '.stages[] | [.stage, .cpu.calls, .cpu.total_ms, .cuda.calls, .cuda.total_ms] | @tsv' \
  /tmp/run.jsonl.summary.json

# Frame wall times
jq -r 'select(.type=="frame") | [.frame, .wall_ms] | @tsv' /tmp/run.jsonl

# Slowest individual CPU events
jq -s 'map(select(.type=="event" and .kind=="cpu")) | sort_by(.dur_ms) | reverse | .[:20]' \
  /tmp/run.jsonl

# Total CPU time by stage from the event stream (overlaps included)
jq -s 'map(select(.type=="event" and .kind=="cpu")) | group_by(.stage) |
       map({stage: .[0].stage, calls: length, ms: (map(.dur_ms) | add)})' \
  /tmp/run.jsonl
```

## Stage list

Use these built-in constants; they are pre-registered and need no lookup:

| `StageId` constant | JSON stage | Scope | What it covers |
|---|---|---|---|
| `stages::kInputDecode` | `input_decode` | CPU (`input/decode`) | Y4M read, NVDEC submission/wait, pinned upload, color conversion, frame-pool handoff. |
| `stages::kTileSchedule` | `tile_schedule` | CPU or CUDA | Tile/frame scheduling, work-queue setup, launch bookkeeping. Not the block search itself. |
| `stages::kIntraSearch` | `intra_search` | CUDA (and CPU launch) | Intra prediction + SATD/SAD search, mode candidates. |
| `stages::kInterSearch` | `inter_search` | CUDA (and CPU launch) | Motion estimation, reference fetch, inter candidates. |
| `stages::kRdo` | `rdo` | CUDA (and CPU launch) | RDO cost evaluation and mode decision. |
| `stages::kTransformQuant` | `transform_quant` | CUDA | Forward transform, quantization, coefficient staging. |
| `stages::kRecon` | `recon` | CUDA | Inverse quant/transform and reconstruction into the reference frame. |
| `stages::kLoopFilters` | `loop_filters` | CUDA | Deblock, CDEF, CCSO, GDF, Wiener/restoration as they are added. |
| `stages::kEntropy` | `entropy` | CUDA (tile range coder) or CPU reference | Symbol generation and normative AV2 §8.2 range coding. |
| `stages::kBitstream` | `bitstream` | CPU | OBU headers, tile payload assembly, IVF muxing, file write. |
| `stages::kHostSync` | `host_sync` | CPU | Explicit `cudaStreamSynchronize`, event waits, D2H copies, host/device handoff. The profiler's own `end_frame()` sync is not part of this stage. |
| `stages::kTotal` | `total` | CPU | Whole-frame encode work. Parent scope; do not add it to other stages. |

Rules of thumb:

- Put `CudaScope` around a **stage group**, not around each kernel. One event
  pair records the whole group. The profiler resolves all pending events once
  per frame, so per-kernel scopes would multiply event traffic without adding
  useful timeline data. Nsight Systems already gives per-kernel durations.
- Use `CpuScope` around the host-side launch/control portions. On the E5 this is
  often the bottleneck even when the GPU is busy.
- It is fine to use the same stage name for a CPU scope and a CUDA scope. The
  summary separates `cpu` and `cuda` samples.
- Nest CPU scopes only when the nesting is meaningful. Nested totals overlap in
  the summary.
- `total` should enclose the frame's encode work. The `{"type":"frame"}`
  record measures the full `begin_frame()`/`end_frame()` interval, including
  the profiler flush.

Custom stages are allowed up to 64 total stages (including the 12 built-ins),
with names truncated to 47 characters:

```cpp
const av2enc::profile::StageId kMyStage =
    profiler.stage("my_stage");   // register during init/warm-up
...
av2enc::profile::CpuScope scope(profiler, kMyStage);
```

The string overload `CpuScope(profiler, "my_stage")` also works, but it performs
a small linear lookup on every construction. Prefer a cached `StageId` in hot
code.

## Integration API

Add `src/core/telemetry/profiler.cpp` to the encoder target and include
`"core/telemetry/profiler.h"`. The header has no CUDA dependency; `CudaScope` takes the
stream as `void*` so CPU-only translation units can include it.

```cpp
#include "core/telemetry/profiler.h"

using namespace av2enc::profile;

// One profiler for the process, created before the encoder starts work.
Profiler profiler;
std::string profile_error;
if (!profiler.open_from_env(&profile_error) && !profile_error.empty()) {
  std::cerr << "av2enc: profiler disabled: " << profile_error << "\n";
}
// open_from_env() returning false with an empty error simply means
// AV2ENC_PROFILE was not set. That is not an error.

// Optional summary label. The default is auto.
profiler.set_precompute_profile("auto");

// Optional: register custom stages once.
const StageId kMyStage = profiler.stage("my_stage");

for (std::uint64_t frame = 0; frame < frame_count; ++frame) {
  profiler.begin_frame(frame);

  {
    CpuScope scope(profiler, stages::kInputDecode);
    decode_or_upload_input(frame);
  }

  {
    // stream is a cudaStream_t; it converts implicitly to void*.
    // Pass nullptr for the default stream.
    CudaScope scope(profiler, stages::kIntraSearch, stream);
    launch_intra_search(frame, stream);
    launch_intra_cost_reduction(frame, stream);
  }

  {
    CudaScope scope(profiler, stages::kEntropy, stream);
    launch_tile_entropy(frame, stream);
  }

  {
    CpuScope scope(profiler, stages::kBitstream);
    write_ivf_frame(frame);
  }

  // Resolves CUDA events and synchronizes each used stream at most once.
  // This is the only per-frame profiler sync.
  profiler.end_frame();

  // Optional: after large persistent allocations, update the VRAM peak.
  profiler.sample_vram();
}

// Writes <path>.summary.json, flushes and closes the JSONL.
profiler.close();

// Human-readable table on stderr/stdout if desired.
profiler.print_summary(std::cerr);
```

Lifecycle rules:

- Create the profiler before encoding and keep it alive until all scopes have
  been destroyed. Do not call `close()` while a scope or worker thread is still
  recording.
- Call `begin_frame()` and `end_frame()` from the launch/control thread. Stage
  scopes may run on worker threads; the statistics and event writer are
  mutex-protected.
- `end_frame()` resolves pending CUDA events. If it is not called, up to 512
  CUDA intervals are queued; after that new CUDA scopes are dropped and the
  summary reports `cuda_events_dropped > 0`.
- Do **not** construct `CudaScope` while a CUDA graph is being captured. Wrap
  the graph launch itself instead.
- `CudaScope` records a start event and an end event around the launches, then
  returns immediately. The duration is resolved at `end_frame()`.
- The first CUDA scope creates the event pool (1024 events). Do a warm-up frame
  before measurement; this is a one-time setup cost, not a hot-path allocation.
- Stage registration and JSON formatting use fixed-size storage. There is no
  heap allocation in the steady-state hot path.
- The profiler is not a hard VRAM allocator. `AV2ENC_VRAM_LIMIT_MB` accounts and
  warns; enforcement belongs in the allocator/workspace layer.

### CMake wiring (not yet done in the root CMakeLists)

The root `CMakeLists.txt` is owned by the scaffold work and does not currently
build `src/core/telemetry/`. When wiring it in, add `src/core/telemetry/profiler.cpp` to
`av2enc_core` (or a small `av2enc_profile` library), and when the CUDA backend is
enabled:

```cmake
# C++ source; CUDA headers are needed only because AV2ENC_HAVE_CUDA=1.
find_package(CUDAToolkit REQUIRED)
target_sources(av2enc_core PRIVATE src/core/telemetry/profiler.cpp)
target_include_directories(av2enc_core PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_compile_definitions(av2enc_core PUBLIC AV2ENC_HAVE_CUDA=1)
target_link_libraries(av2enc_core PUBLIC CUDA::cudart)

# Optional NVTX:
target_compile_definitions(av2enc_core PUBLIC AV2ENC_HAVE_NVTX=1)
target_link_libraries(av2enc_core PUBLIC nvToolsExt)
```

When CUDA is not enabled, compile `profiler.cpp` with `AV2ENC_HAVE_CUDA=0` (or
leave the macro undefined). `CudaScope` then becomes a no-op and no CUDA headers
or libraries are required.

### Standalone demo

The demo has no CMake dependency:

```sh
src/core/telemetry/build_demo.sh
./src/core/telemetry/build/profile_demo              # CPU scopes + JSONL verification
./src/core/telemetry/build/profile_demo_cuda         # adds a trivial CUDA kernel
./src/core/telemetry/build/profile_demo --benchmark  # overhead measurement
```

`build_demo.sh` builds with `g++ -std=c++20`, strict warnings, and optionally
with `nvcc` when available (`sm_86`; override with `CUDA_ARCH=...`). Use
`WITH_CUDA=0` to skip the CUDA demo. NVTX is linked automatically when the CUDA
toolkit's `nvtx3` header and `libnvToolsExt` are present.

The demo verifies that:

- every JSONL line is a JSON object;
- event records contain `stage`, `frame`, `thread`, and `dur_ms`;
- frame records contain `wall_ms`;
- the expected number of events and all three frames are present;
- the summary contains all stages;
- VRAM fields are present when a CUDA device was used and omitted when it was
  not.

## Overhead

Measured on the Vast.ai RTX 3060 box (Xeon E5-2690 v4, 2026-11-11) with
`profile_demo --benchmark`:

| Operation | Cost |
|---|---:|
| Disabled `CpuScope` | 7–11 ns per scope |
| Enabled `CpuScope` (clock, stats, JSON line) | 0.69–0.94 µs per event |
| 12 CPU scopes/frame at 60 fps | 0.49–0.68 ms per second (0.05–0.07% of a 16.67 ms frame) |
| `cudaMemGetInfo` VRAM sample | ≈9–11 µs per sample |

Interpretation:

- The disabled path is a branch plus an atomic load. It is safe to leave scopes
  in production builds.
- The enabled CPU cost is dominated by two clock reads, the stats lock, number
  formatting, and the buffered JSONL write. At tens of stage samples per frame
  this is far below frame time.
- CUDA event timing is cheap to record but adds one stream synchronization per
  frame at `end_frame()`. Do not put a `CudaScope` around every kernel.
- `cudaMemGetInfo` costs about 11 µs, once per frame plus each explicit
  `sample_vram()`. That is about 0.07% of a 16.67 ms frame at 60 fps.
- CUDA event elapsed time has roughly 0.5 µs resolution. Use stage timers for
  regression and pipeline attribution, not for timing sub-microsecond kernels.
- NVTX adds annotation overhead and is intended for profiling runs, not for
  shipped benchmark runs.

Profiling overhead is measured relative to frame time. The encoder's target is
GPU-resident; the E5-side scope overhead matters because the CPU is the
launch/control bottleneck. A dozen CPU scopes per frame is acceptable.

## Precompute profile and VRAM accounting

### Focus: device-sized arena

The primary target is **8 GB Ampere (`sm_86`)**. The profiler defaults to the
**`auto`** label and records it in every summary:

```json
"precompute_profile": "auto",
"vram_limit_mb": 0
```

`set_precompute_profile()` is only a summary label. The runtime sizes the
arena from the device; `compat*`/`auto*g` names are offline planning estimates,
not encoder presets. macOS is not an encoder target.

### VRAM budget

The profiler tracks device-wide VRAM for reporting. It is not the allocator:
the runtime sizes the arena from the device (`min(total, free) - 512 MiB`).

- `AV2ENC_VRAM_LIMIT_MB` defaults to **0** (no warning). Set a value only for
  an explicit budget check.
- At `open()`, `cudaMemGetInfo()` records total capacity and free VRAM.
- `end_frame()` and `close()` sample `cudaMemGetInfo()` again and update the
  peak used value. Call `profiler.sample_vram()` manually after large
  persistent allocations if the peak could occur between frame boundaries.
- `peak_vram_mb` is **device-wide used** = `capacity - free`, not just the
  encoder's private allocations. It includes the CUDA context, runtime
  allocations, and anything else on the device. In the demo, the CUDA context
  alone accounts for roughly 127 MiB.
- MB means MiB (1024 × 1024 bytes).

Summary fields:

| Field | Always present? | Meaning |
|---|---|---|
| `precompute_profile` | yes | Label, default `auto`. |
| `vram_limit_mb` | yes | `AV2ENC_VRAM_LIMIT_MB` value, default 0 (unlimited). |
| `peak_vram_mb` | with CUDA device | Highest sampled used VRAM. |
| `vram_capacity_mb` | with CUDA device | Device total, e.g. `11920.75` on the 12 GB RTX 3060. |
| `vram_free_mb` | with CUDA device | Free VRAM at the last sample (normally close). |

If CUDA is absent, the three **measured** VRAM fields are omitted from the
summary and the VRAM line is omitted from `print_summary()`. They are never
reported as zeros. `precompute_profile` and `vram_limit_mb` are configuration
fields and are always recorded.

If sampled use exceeds the limit, the profiler prints one warning to stderr:

```text
av2enc profile: VRAM use 8123 MB exceeds AV2ENC_VRAM_LIMIT_MB=7500 (device capacity 11920 MB)
```

8 GB compatibility must be validated on an 8192 MiB card. The 12 GB dev box
sizes the arena from its own device and is a proxy, not proof. Do not look at `vram_capacity_mb` and assume the encoder may
use 11.9 GB: set `AV2ENC_VRAM_LIMIT_MB` explicitly if you want that warning.
The profiler does not fail the encode when the limit is exceeded; a hard budget
must be enforced by the allocator/workspace layer. For now, the gate is:
`peak_vram_mb` is recorded on every test configuration (no fixed target).

Caveats:

- Device-wide `cudaMemGetInfo()` includes other processes on a shared GPU. The
  Vast.ai container normally has exclusive use of the 3060.
- Sampling at frame boundaries can miss a short-lived allocation inside a
  frame. Call `sample_vram()` immediately after large allocations.
- The profiler's own CUDA event pool is negligible (1024 event objects), but it
  is included in the device-wide accounting.

Example:

```sh
AV2ENC_PROFILE=/tmp/run.jsonl \
  ./build/av2enc input.y4m -o out.ivf --frames 60
jq '{precompute_profile, vram_limit_mb, peak_vram_mb,
     vram_capacity_mb, vram_free_mb}' /tmp/run.jsonl.summary.json
```

## Nsight Systems (Linux, preferred external profiler)

Nsight Systems is installed on the box and works. Use the built-in NVTX ranges
to put stage names on the timeline.

```sh
AV2ENC_PROFILE_NVTX=1 \
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=cpu \
  --force-overwrite true \
  -o /tmp/av2enc-run \
  ./build/av2enc input.y4m -o out.ivf --frames 10

nsys stats --report nvtx_sum,cuda_gpu_kern_sum,osrt_sum \
  /tmp/av2enc-run.nsys-rep
```

What you get:

- `nvtx_sum`: our `av2enc.cpu:*` and `av2enc.cuda:*` ranges with counts and
  durations.
- `cuda_gpu_kern_sum`: actual per-kernel device time.
- `osrt_sum`: host-side syscall/library hotspots (file I/O, allocations,
  synchronization).
- The `.nsys-rep` opens in Nsight Systems UI on a workstation. Copy it off the
  box: `/workspace` is not persistent, and recycle/destroy wipes it.

Our NVTX domains are:

| Domain | Range names | Color |
|---|---|---|
| `av2enc.cpu` | Stage names from `CpuScope` | blue |
| `av2enc.cuda` | Stage names from `CudaScope` | green |

`CudaScope` ranges bracket the **host launch region**; actual kernel work
appears on the CUDA rows. Use both together: the NVTX range tells you which
pipeline stage issued the launches, the CUDA row tells you what the GPU ran.

NVTX-only mode (no JSONL file) is available with just
`AV2ENC_PROFILE_NVTX=1`. The profiler still records in-memory statistics, so
`print_summary()` works if the encoder chooses to call it.

If `nsys` is missing on a future instance, install a version that supports the
installed CUDA stack. The Ubuntu multiverse candidate is from 2022 and predates
CUDA 12.8; do not rely on it. `nsys --version` should report a 2024+ release.

### Nsight Compute is blocked

`ncu` is installed, but this unprivileged container cannot access GPU
performance counters:

```text
ERR_NVGPUCTRPERM: Unable to access the performance counters on device 0.
```

This is not fixable from inside the container. Do not change drivers or sysctls.
If occupancy/stall/memory-throughput counters become essential, move the run to
a VM or bare-metal instance with counter permission. Until then, use the
built-in stage timers plus Nsight Systems timelines and the working
`compute-sanitizer` (for correctness, not performance).

## Linux CPU profilers

### gprof

Useful for a first look at host-side call costs when Nsight Systems is not
available.

```sh
# Configure a profiling build (out-of-tree):
cmake -G Ninja -S . -B build-gprof \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_FLAGS="-pg -g" \
  -DCMAKE_EXE_LINKER_FLAGS="-pg"
cmake --build build-gprof
./build-gprof/av2enc input.y4m -o out.ivf --frames 10
gprof ./build-gprof/av2enc gmon.out > gprof.txt
```

Caveats:

- Instrumentation overhead; not for absolute frame-time regression.
- Poor visibility into inlined code and multithreaded workers. GPU time is
  invisible.
- `gmon.out` is written at process exit; long-running encodes must finish or be
  terminated cleanly.
- Use gprof for function-level CPU attribution, stage timers for pipeline
  attribution, and Nsight Systems for GPU/CPU correlation.

### perf

Likely blocked in the Vast.ai container because kernel performance events are
not permitted. Check before relying on it:

```sh
perf stat true
cat /proc/sys/kernel/perf_event_paranoid
```

If it works on a different host:

```sh
cmake -G Ninja -S . -B build-perf \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_FLAGS="-g -fno-omit-frame-pointer"
cmake --build build-perf
perf record -g --call-graph dwarf ./build-perf/av2enc input.y4m -o out.ivf --frames 10
perf report
```

If `perf stat true` fails with a permission error, stop; the container policy is
the limit, not the encoder.

### `-finstrument-functions`

For function-level traces when timeline tools are unavailable, compile the host
code with:

```sh
-DCMAKE_CXX_FLAGS="-finstrument-functions -g -O2 -fno-omit-frame-pointer"
```

Provide `__cyg_profile_func_enter` and `__cyg_profile_func_exit`, marked
`__attribute__((no_instrument_function))`, and write fixed-size binary records
with `__builtin_return_address(0)` plus a thread-local sequence number. Resolve
addresses offline with `addr2line -e ./build/av2enc -f -C` or
`llvm-symbolizer`.

Caveats:

- Very high overhead (every function entry/exit), and it can inhibit some
  inlining. Never use it for frame-time regression.
- It instruments host functions only, not CUDA device code.
- Exclude allocator/logging noise with
  `-finstrument-functions-exclude-function-list=...` if needed.
- Hooks must not recurse into themselves.

## macOS `xctrace` (optional, untested for this encoder)

macOS is not an encoder target and there is no CUDA there. `xctrace` is only a
convenience for host-side scalar/reference code, not for latency on the 3060.
This path has not been validated for av2enc:

```sh
xctrace record --template 'Time Profiler' \
  --output /tmp/av2enc.trace \
  --launch -- ./build/av2enc input.y4m -o out.ivf --frames 10
open /tmp/av2enc.trace
```

For a quick text sample of a running process, `sample <pid> 10 -file
/tmp/av2enc-sample.txt` also works. Do not use either to judge GPU stage costs.

## Choosing the right tool

| Question | Tool |
|---|---|
| Did stage X regress run-over-run? | Built-in stage timers (`AV2ENC_PROFILE`) |
| Which frame stage dominates wall time? | Built-in `summary.json` + JSONL |
| What is peak VRAM? | Built-in VRAM accounting |
| Where is the gap between two kernels? | Nsight Systems timeline + NVTX |
| Which kernel is slow, and how long does it run? | `nsys` `cuda_gpu_kern_sum` |
| Why is a kernel slow (stalls, occupancy, memory)? | Nsight Compute — **blocked here** |
| Which host function calls the most CPU? | `gprof`, `perf` (if permitted), `-finstrument-functions` |
| Is the kernel writing out of bounds? | `compute-sanitizer --tool memcheck` |

Workflow:

1. **Regression:** fixed clip, fixed QP, fixed frame count, same box and clocks.
   Run with `AV2ENC_PROFILE` (and `--vram-cap` if a budget is required). Record the commit,
   GPU/driver, resolution, frame count, and stage means/max plus `peak_vram_mb`.
   Compare stage means, not a single frame.
2. **Attribution:** use `nsys` with `AV2ENC_PROFILE_NVTX=1` to align stages with
   kernels and gaps. Check whether the CPU launch side or the GPU stage is the
   limit.
3. **Deep dive:** if counters are required, move to a permitted machine. Do not
   disable quality tools to make a stage faster; the quality gate still decides.
4. Always confirm the stream still decodes (`dav2d`) before accepting a speed
   result. A faster stage that breaks or degrades the bitstream is not a win.

## Files

| File | Purpose |
|---|---|
| `src/core/telemetry/profiler.h` | Public API: `Profiler`, `CpuScope`, `CudaScope`, stage ids, env-driven open. |
| `src/core/telemetry/profiler.cpp` | Implementation: fixed stage table, JSONL/summary, CUDA event pool, NVTX, VRAM. |
| `src/core/telemetry/profile_demo.cpp` | Standalone nested-scope and optional CUDA-kernel demo with output verification. |
| `src/core/telemetry/build_demo.sh` | g++/nvcc demo build without CMake. |
| `docs/profiling.md` | This document. |

## Pageable vs pinned transfers (measured 2026-11-14, RTX 3060)

`cudaMemcpyAsync` on a **pageable** `std::vector` is not asynchronous: a
device-to-host copy has to wait for the bytes (the driver cannot return before
the staging copy lands), so every "async" download drains the stream.  A
per-block decision loop then pays a full pipeline drain per copy — on the M2
720p path that was 184,947 syncs and 974 MB of host round trips for 60 frames,
62.6 % of the CUDA API time in `cudaMemcpyAsync`.

The fix is the sanctioned pinned staging provider (`av2enc::cuda::StagingBuffer`
in `src/cuda/context.cpp`): stage through pinned memory, queue the copy, and
collect several results behind **one** `cudaStreamSynchronize`.  Also size the
copies from the coded block, not from the maximum block (24 KiB per leaf even
for an 8x8 block is pure latency).  Same decisions, 4.95 -> 6.53 fps at 720p.

For kernel cost, the same rule applies to tiny grids: a per-candidate SSE that
scored one block per thread (4096 serial samples for a 64x64 candidate, ~50 us)
becomes ~2 us with one warp per candidate and a shuffle reduction (integer
sums are associative, so the result is unchanged).
