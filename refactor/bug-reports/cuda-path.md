# cuda-path bug scout

| id | severity | class | title | repro | evidence | files | status |
|---|---|---|---|---|---|---|---|
| SCOUT-CUDA-01 | med | doc | `--help`/README still call CDEF, CCSO and film-grain "CPU-only", but `--still-gpu` accepts and stages them | `build/av2enc --help \| grep -E 'still-(cdef\|ccso\|grain)'` | help prints "CPU-only" (lines 365-367, 372); `main.cpp` maps them into `GpuOptions` (3562-3572); `gpu_m3_options_supported()` only rejects wiener/chroma-lr/pc-wiener/gdf/rdoq/tx-type/rdo-intra/uv-full | `src/main.cpp:355-372,3562-3572`, `src/cuda/pipeline/still/still_gpu.h:299-365`, `src/cuda/pipeline/still/still_gpu.cu:1909-2019` | new |
| SCOUT-CUDA-02 | med | doc | m3 GPU gate docs list CDEF/CCSO/UV-full as unsupported/rejected at paths that no longer exist; film grain has no gate mode | `sed -n '190,215p' tools/m3_gpu/README.md` | table says CDEF `SKIP-REJECTED` "validate_options (src/cuda/m1/still_gpu.cu:1049)" and CCSO "not plumbed"; both now run on device; `ALL_FEATURES` has no `grain` | `tools/m3_gpu/README.md:180-213`, `tools/m3_gpu/RESULTS.md:113-124`, `tools/m3_gpu/verify_m3_gpu.sh:191-193` | new |
| SCOUT-CUDA-03 | med | suspected | `--still-gpu` does not imply `--still`; `--still --still-gpu --recon F` exits 0 and writes no `F` | `CUDA_VISIBLE_DEVICES= build-cuda/av2enc testdata/src_100x70.y4m -o /tmp/g.ivf --still --still-gpu --recon /tmp/g.y4m; echo $?; ls /tmp/g.y4m` | parser sets only `still_gpu` (`--still` sets `m1`); recon validation only fires when `!options.m1`, and the still-gpu branch has no recon write while the CPU branch does | `src/main.cpp:806-808,3376-3381,3544-3600,3727-3735`, `src/cuda/pipeline/still/still_gpu.h:465-478`, `README.md:90` | new |
| SCOUT-CUDA-04 | med | doc | documented stats-JSON attestation channel (`AV2ENC_GPU_STATS`/`AV2ENC_STILL_GPU_STATS`) has no producer in the encoder | `AV2ENC_GPU_STATS=/tmp/s.json build/av2enc testdata/src_100x70.y4m -o /tmp/s.ivf --still; ls /tmp/s.json` (and same on a CUDA build with `--still-gpu`) | `verify_gpu.sh` exports it and `stats_attests` reads it; `bench.py` reads kernel/launch/VRAM from it; `grep -rn AV2ENC_STILL_GPU_STATS src` → no writer (only `core/telemetry` env, different var) | `tools/still_gpu/verify_gpu.sh:97,732,867-880`, `tools/still_gpu/bench.py:14,485-539`, `tools/still_gpu/README.md:169-186`, `tools/m3_gpu/README.md:180-183` | new |
| SCOUT-CUDA-05 | med | doc | `--video` has no GPU product path: M2 CUDA backend is not in the build and the CLI always calls the CPU encoder | `grep -n 'pipeline/video' CMakeLists.txt; nm build/av2enc \| grep -c encode_video_sequence_gpu` | no `src/cuda/pipeline/video/*` sources in CMake, only CPU `video_gate`; `main.cpp` calls `av2enc::video::encode_video_sequence` in all three video paths; no `--video-gpu` option; docs admit gate-only | `CMakeLists.txt:427-431`, `src/main.cpp:2568,2621,2653`, `docs/gpu-coverage.md:25-27`, `src/cuda/pipeline/video/build_gpu_gate.sh` | new |
| SCOUT-CUDA-06 | low | doc | user-visible diagnostics and docs cite the removed `src/cuda/m1/` (and `src/cuda/m2/`) tree | `ls src/cuda/m1` (No such file); `build/av2enc ... --still-gpu` on a CUDA build whose CMake skipped the orchestrator prints the path | error text names `src/cuda/m1/still_gpu.cu`; README/docs/headers repeat `src/cuda/m1/{still_gpu,plan,search,tx,recon,chroma,cdef,restoration}` and `src/cuda/m2/` | `src/main.cpp:3003`, `README.md:89,95`, `docs/m1-gpu.md:4,77,101-108`, `docs/gpu-coverage.md:26`, `src/cuda/pipeline/still/still_gpu.h:13-14`, `src/pipeline/still/stream/still_stream_bridge.h:3,181`, `src/cuda/pipeline/still/plan/plan_gpu_bridge.h:2`, `src/cuda/pipeline/still/README.md:231-241` | new |
| SCOUT-CUDA-07 | low | confirmed | `AV2ENC_PROFILE` is documented as the encoder's built-in profiler but no product path instantiates it, so no JSONL/summary is written | `rm -f /tmp/p.jsonl*; AV2ENC_PROFILE=/tmp/p.jsonl build/av2enc testdata/src_100x70.y4m -o /tmp/p.ivf --still; ls -l /tmp/p.jsonl*` | `open_from_env`/`CpuScope` have no callers outside `core/telemetry`; README/docs present it as working (`AV2ENC_PROFILE=/tmp/run.jsonl ./build-cuda/av2enc ...`) incl. per-frame CUDA stage times and VRAM peak | `README.md:282-291`, `docs/profiling.md:21-45,240-257`, `src/core/telemetry/profile_demo.cpp:281` | new |
| SCOUT-CUDA-08 | low | doc | still-GPU arena is not the documented device-derived arena; `Arena::capacity_cap_bytes()` is ignored and the reservation is clamped to a 256 MiB floor | `sed -n '916-948p' src/cuda/pipeline/still/still_gpu.cu` | docs/AGENTS require `min(total,free) - 512 MiB` optionally bounded by `--vram-cap`; the orchestrator instead takes `max(256 MiB, 1.5*workingset+16 MiB)` and only *caps* by device/cap, so `arena_capacity_bytes` never reflects the device budget | `docs/cuda-interface.md:46-48,55-71`, `AGENTS.md` (VRAM arena), `src/cuda/pipeline/still/still_gpu.cu:118-123,916-948` | new |

## Details

### SCOUT-CUDA-01 — GPU-supported tools are still documented as CPU-only
Repro:
```sh
cd /Users/arin/work/av2enc && build/av2enc --help | grep -E -- '--still-(cdef|ccso|grain)'
```
Observed: `--still-cdef`, `--still-ccso`, `--still-ccso-edge` and `--still-grain` are labelled `(CPU-only)`, while `--still-gpu` accepts each of them (no entry in the `--still-gpu` rejection list at `src/main.cpp:3060-3108`, and `gpu_m3_options_supported()` deliberately keeps them accepted). The code has real device stages: `still_cdef_filter` / `ccso_frame` are invoked at `still_gpu.cu:1951,2013`, and grain is emitted through the copied CPU `StillConfig`.
Expected: `--help` (declared "the contract" in `README.md:101`) and `README.md:87-97` should not tell users these tools are CPU-only when the GPU path runs/passes them through.
Likely cause: the help table and README predate the CDEF/CCSO/grain landing; only the rejection list was updated.

### SCOUT-CUDA-02 — m3 GPU readiness table contradicts the code and its own gate
Repro:
```sh
cd /Users/arin/work/av2enc && sed -n '190,215p' tools/m3_gpu/README.md
```
Observed: CDEF is described as `SKIP-REJECTED` at `src/cuda/m1/still_gpu.cu:1049` (file does not exist; today's orchestrator accepts CDEF and applies §7.18 on device). CCSO is described as "not plumbed at all — GpuOptions has no CCSO field", although `GpuOptions::enable_ccso/ccso_bands_log2/ccso_edge` exist and are mapped. UV-full is called a "candidate for READY" although `gpu_m3_options_supported()` rejects it. `ALL_FEATURES`/`MODES` in the gate have no film-grain case at all, so `--still-gpu --still-grain` (accepted) is never gated.
Expected: the readiness table must match `gpu_m3_options_supported()` and every accepted M3 flag needs a feature mode (or a documented rejection).
Likely cause: docs were derived before the CDEF/CCSO landing; no grain row was added.

### SCOUT-CUDA-03 — `--still-gpu` does not actually imply `--still`
Repro (CUDA build; force CPU fallback so it runs on a CPU host):
```sh
CUDA_VISIBLE_DEVICES= build-cuda/av2enc testdata/src_100x70.y4m -o /tmp/g.ivf \
  --still --still-gpu --recon /tmp/g.y4m; echo rc=$?; ls -l /tmp/g.y4m
```
Observed (static): `--still-gpu` only sets `options.still_gpu`; `m1` stays false. `--recon` is allowed when `options.m1` is true (`main.cpp:3376`) and `--still --still-gpu` is never rejected as conflicting, so validation passes. The `options.still_gpu` encode branch ends at line 3600 without calling `write_still_recon_y4m`; the only write is in the CPU-still `else` branch (3727). Exit is 0 and the IVF is valid, but `/tmp/g.y4m` is never created.
Expected: either "implies --still" is implemented (write the recon dump, `GpuResult` already carries `recon_y/u/v`) or the redundant/conflicting flag combination is rejected. Also `--still-gpu --recon` alone exits 2 with an error that recommends `--still`, which then silently drops the dump.
Likely cause: the `--still-gpu` branch was written as a separate mode while the doc/validation kept the "implies --still" wording.

### SCOUT-CUDA-04 — stats-JSON attestation is documented but not implemented
Repro:
```sh
cd /Users/arin/work/av2enc && grep -rn "AV2ENC_STILL_GPU_STATS\|AV2ENC_GPU_STATS" src
# (no hits outside tools/docs)
AV2ENC_GPU_STATS=/tmp/s.json build/av2enc testdata/src_100x70.y4m -o /tmp/s.ivf --still; ls /tmp/s.json
```
Observed: `verify_gpu.sh` treats the stats file as the preferred GPU-execution attestation (`stats_attests`, line 867) and `bench.py` takes kernel/launch/VRAM numbers from it, but neither `src/main.cpp` nor `still_gpu.cu` writes it (`GpuStats` lives only in memory). Without `nsys`, a correct GPU run therefore ends as `SKIP`, and bench's kernel/launch/VRAM columns stay empty. `tools/m3_gpu/README.md:180-183` even states "The CLI currently does not write the stats file", contradicting `tools/still_gpu/README.md:169-186` which calls it the contract.
Expected: the encoder writes the documented JSON (`backend`, `frame_bytes_from_cpu`, `kernels`/`launches`, `vram_peak_bytes`) from `GpuStats`, or the gate/bench docs stop relying on it.
Likely cause: only the standalone `still_gpu_probe` implements `--stats`; the CLI glue was never added.

### SCOUT-CUDA-05 — `--video` is CPU-only although the CUDA M2 backend exists
Repro:
```sh
cd /Users/arin/work/av2enc && grep -n 'pipeline/video\|video_backend' CMakeLists.txt
nm build/av2enc | grep -c encode_video_sequence_gpu   # 0
build/av2enc --help | grep -i 'video.*gpu'             # empty
```
Observed: all three video dispatch sites call the CPU `av2enc::video::encode_video_sequence`; `src/cuda/pipeline/video/{video_backend_cuda.cu,video_pipeline.cpp}` are not in `CMakeLists.txt` (only the CPU `video_gate` target), and `docs/gpu-coverage.md:25-27` confirms `src/cuda/m2/` is "gate-only". `--video` is presented in the mode table/help without a CPU-only qualifier, while the product README headlines "GPU-resident encoder".
Expected: document `--video` as CPU-only in the product CLI (and drop the stale `src/cuda/m2/` wording), or add the CMake target/`--video-gpu` entry point and gate it.
Likely cause: the M2 device backend was built only for `build_gpu_gate.sh`; product wiring was deferred and the docs were not updated.

### SCOUT-CUDA-06 — stale `src/cuda/m1/` (and `m2/`) paths
Repro:
```sh
cd /Users/arin/work/av2enc && ls src/cuda/m1 src/cuda/m2   # both missing
grep -rn "src/cuda/m1\|cuda/m2" src/main.cpp docs/m1-gpu.md src/cuda/pipeline/still/README.md | head
```
Observed: the actual tree is `src/cuda/pipeline/still/` and `src/cuda/pipeline/video/`. `main.cpp:3003` emits a user-facing error naming `src/cuda/m1/still_gpu.cu`; the same stale root appears in the contract header, plan/stream bridge headers, `docs/m1-gpu.md` (ownership table and build commands) and `src/cuda/pipeline/still/README.md` ("BUILD_DIR=... src/cuda/m1/build_still_gpu_test.sh"). `docs/gpu-coverage.md:26` names `src/cuda/m2/`.
Expected: diagnostics and docs point at the real files (this is the same stale-path class as the `__has_include` probes already fixed in `main.cpp`/`still_gpu.cu`; those probes resolve, the text did not get updated).
Likely cause: directory move `src/cuda/m1|m2` -> `src/cuda/pipeline/{still,video}` with comments/docs left behind.

### SCOUT-CUDA-07 — documented profiler is not wired (confirmed)
Repro:
```sh
cd /Users/arin/work/av2enc && rm -f /tmp/p.jsonl*
AV2ENC_PROFILE=/tmp/p.jsonl build/av2enc testdata/src_100x70.y4m -o /tmp/p.ivf --still
ls -l /tmp/p.jsonl*        # "No such file"
```
Observed: exit 0, no `/tmp/p.jsonl` and no `.summary.json`. `Profiler::open_from_env()` and the `CpuScope`/`CudaScope` stages have no caller outside `src/core/telemetry` (only `profile_demo.cpp`). `README.md:282-291` and `docs/profiling.md:21-45` describe `AV2ENC_PROFILE` as the encoder's per-frame CPU/CUDA stage + VRAM logger (including for `./build-cuda/av2enc`).
Expected: main instantiates the profiler (per the `docs/profiling.md:240-257` recipe) or the docs mark it as harness-only.
Likely cause: profiler component landed with a demo, product wiring pending; the doc was written against the intended state.

### SCOUT-CUDA-08 — still-GPU arena sizing departs from the documented device-derived rule
Repro:
```sh
cd /Users/arin/work/av2enc && sed -n '916,948p' src/cuda/pipeline/still/still_gpu.cu
```
Observed: `desired = max(kArenaFloorBytes (256 MiB), 1.5*need_device + 16 MiB)`, then `min(desired, device_budget)` and `min(desired, --vram-cap)`. On any device larger than ~256 MiB the reservation is therefore not `min(total,free) - 512 MiB`; it is a fixed small multiple of the still working set. The still path also never consults `Arena::capacity_cap_bytes()` (the global set by `apply_vram_cap`), unlike `basic_pipeline.cu:383` and `video_backend_cuda.cu:1082`.
Expected: per `docs/cuda-interface.md:46-48` and the `AGENTS.md` non-negotiable, the persistent arena is sized from the device (optionally bounded), or the doc must state the still-specific working-set clamp. This is a policy/documentation mismatch, not a proven over/under-allocation (the 1.5x slack covers the still set).
Likely cause: deliberate "keep the arena small while other agents share the GPU" clamp added without updating the sizing contract.

## Notes
- `tools/check_arena.sh` on current `main`: PASS ("no raw CUDA allocations outside arena.cpp/context.cpp").
- All `__has_include` guards in `src/main.cpp` and `src/cuda/pipeline/still/still_gpu.cu` resolve to existing headers (verified); no stale probe remains there.
- No device was available, so no still-GPU functional claim is marked `confirmed`; parity-gated arithmetic bugs would fail loudly against the always-on CPU oracle (`verify_against_cpu` default true), which is why the findings above are the parity-blind surfaces (flags, attestation, diagnostics, docs).
