# M1 GPU verification and benchmark

Gate for the GPU-resident M1 path (`--still-gpu`, workstream 10 in
[`docs/m1-gpu.md`](../../docs/m1-gpu.md)). The GPU path is only "working" when
it reproduces the CPU M1 reference **bitstream exactly** — not when it looks
good in PSNR.

```
Y4M ──av2enc --still──────> cpu.ivf ──┐
                                    ├─ IVF frame-payload compare (bytes + MD5)  -> FAIL on any byte
Y4M ──av2enc --still-gpu──> gpu.ivf ──┘
        │                            ├─ dav2d decode both -> Y4M plane compare   -> FAIL on any sample
        └─ attestation ──────────────┘   (stats JSON / log marker / nsys kernels)
                 │
                 └─ optional recon parity: GPU StillGpuResult planes vs CPU StillResult.recon_*
```

The gate is **not** a quality metric. A stream can sit at 50+ dB with one block
desynced, and a CPU fallback under `--still-gpu` would produce identical bytes
while proving nothing. Hence the byte gate plus an execution attestation.

## Files

| File | Purpose |
|---|---|
| `verify_gpu.sh` | CPU-vs-GPU gate: frame bytes, MD5, `dav2d` decode, decoded-plane compare, recon parity, determinism, sanitizer, attestation, PASS/FAIL/SKIP. Also runs the option matrix. |
| `ivf_compare.py` | IVF frame-payload byte comparison (parses the 32-byte IVF header and 12-byte frame headers). |
| `plane_compare.py` | Bit-exact first-frame Y/U/V comparison; reads Y4M or `StillDUMP1` dumps. |
| `bench.py` | Wall/kernel/launch/VRAM benchmark at 64x64, 640x360 and 1080p, with proxy-hardware labelling. |
| `still_gpu_probe.cpp` | Optional probe: calls `encode_still_gpu` directly, writes an `StillDUMP1` recon dump and a stats JSON. Used for recon parity when the CLI exposes no dump. |
| `build_probe.sh` | Builds the probe against an existing `build-cuda/libav2enc_core.a`; never touches CMake or `src/`. |
| `remote.sh` | Syncs only `tools/still_gpu` to the 3060 box and runs `verify`/`bench` there. |

## Verdicts

| Exit | Verdict | Meaning |
|---:|---|---|
| 0 | `PASS` | GPU execution attested, frame bytes identical, both streams decode, decoded planes identical (and recon planes identical when a dump exists). |
| 1 | `FAIL` | Any byte/plane/decode difference, a GPU crash, a nondeterministic GPU encode, a sanitizer error, or a missing GPU stream that was not a stub. |
| 2 | setup | Missing encoder/decoder/ffmpeg/input; bad arguments. |
| 3 | `SKIP` | `--still-gpu` absent; blocked/stub path; CPU-reference staging; GPU exited 0 without an IVF; compute-sanitizer saw no CUDA work; or bytes matched but **GPU execution could not be attested** (possible CPU fallback). A SKIP is never a pass. |

`ALLOW_SKIP=1` (or `--allow-skip` for `bench.py`) maps `SKIP` to exit 0 for CI
that only wants "no regression", without hiding the verdict line.

## Usage

On the 3060 box (or any Linux+CUDA host):

```sh
# zero-argument smoke gate: generates a 640x360 testsrc2 frame
tools/still_gpu/verify_gpu.sh

# explicit input and recon parity
tools/still_gpu/verify_gpu.sh testdata/frame.y4m --recon

# native 1080p input (testsrc2 at 1920x1080), keep artifacts
tools/still_gpu/verify_gpu.sh --size 1920x1080 --keep

# option matrix (14 modes x 8 sizes), determinism, sanitizer
tools/still_gpu/verify_gpu.sh --matrix
tools/still_gpu/verify_gpu.sh --matrix --sizes 64x64,136x88 \
    --modes default,full-intra,no-cfl --determinism
tools/still_gpu/verify_gpu.sh --size 64x64 --sanitize --racecheck
```

From a laptop, using the prescribed tunnel:

```sh
AV2ENC_HOST='<user>@<host>' AV2ENC_PORT='<port>' \
    tools/still_gpu/remote.sh verify

AV2ENC_HOST='<user>@<host>' AV2ENC_PORT='<port>' \
    tools/still_gpu/remote.sh bench --sizes 64x64,640x360
```

`remote.sh` sends only `tools/still_gpu` by default so it cannot clobber another
agent's in-flight `src/` changes on the shared box. `FULL_SYNC=1` switches to
`tools/sync_remote.sh`.

`verify_gpu.sh` uses `$AV2ENC` / `$DAV2D` when set; otherwise
`build-cuda/av2enc` and the first of `tools/bin/dav2d`,
`tools/bin/dav2d-macos`.

Options worth knowing:

| Option | Effect |
|---|---|
| `--size WxH` | Generate a **native** testsrc2 input when no input file is given. |
| `--still-args "..."` | M1 flags for both encodes; mapped to `still_dump`/probe args automatically. |
| `--cpu-args "..."` / `--gpu-args "..."` | Extra CLI flags for one side only. |
| `--recon` | Also compare encoder-side reconstructions. |
| `--gpu-dump PATH` | GPU `StillDUMP1` file to compare (implies `--recon`). |
| `--gpu-dump-arg FLAG` | Append `FLAG <workdir>/gpu-recon.dump` to the GPU command. |
| `--require-recon` | `SKIP` when no GPU recon dump is available. |
| `--no-attestation` | Explicitly unsafe: PASS on identical bytes even with no GPU evidence. |
| `--no-nsys` | Do not fall back to Nsight Systems attestation (faster, more SKIPs). |
| `--determinism` | Encode twice with the CLI; byte mismatch fails with the exact offset. |
| `--determinism-probe` | Also encode twice in one process via `still_gpu_probe`. |
| `--sanitize` | Run the GPU encode under `compute-sanitizer --tool memcheck`. |
| `--racecheck` | Additionally run `--tool racecheck` at 64x64 (skipped elsewhere). |
| `--keep` | Keep IVF/log/dump artifacts. |

### Matrix mode

```sh
tools/still_gpu/verify_gpu.sh --matrix
```

Default dimensions: `64x64, 136x88, 640x360, 1280x720, 1920x1080` plus three
seeded random even sizes (`--seed`, `--random-count`). Default modes:
`default, full-intra, no-cfl, no-deblock, depth-0..4, depth-8, single, dc,
adaptive-cdf, vram-cap-64` (14 modes, 112 cases). Narrow with
`--sizes`/`--modes`.

Each case is an independent `verify_gpu.sh` run, so a failure cannot poison the
next case. Inputs are generated once per size natively and reused. The matrix
prints one progress line per case, a summary, **every failing case with the
exact byte offset / first sample / decoder error**, and grouped SKIP reasons.
Failure attribution (CPU-reference decode vs GPU decode vs byte diff vs
sanitizer) is printed so an invalid reference stream is not mistaken for a GPU
parity bug.

Runtime is bounded: `--matrix-budget SEC` (default 1800, 0 = unlimited) stops
starting new cases and reports the remainder as SKIP; `--max-cases N` refuses
an oversized matrix before it starts; `--timeout SEC` bounds each command.
Matrices above 40 cases print a warning; `--sanitize` above 8 cases warns
again because instrumentation is slow. Exit: all PASS → 0; any FAIL → 1; any
unverified SKIP → 3.

### Determinism

`--determinism` runs the GPU encode a second time (separate process, separate
IVF) and byte-compares both streams with MD5. A difference fails with the first
differing payload offset and a 16-byte context around it. The second run's
stats file is also required to attest, so a fallback second run cannot pass.

`--determinism-probe` additionally runs `still_gpu_probe --repeat 2`: two
`encode_still_gpu` calls in one process. Its stats JSON carries
`repeat_identical` and `repeat_first_diff_offset`. A CPU-reference probe is
ignored; an in-process difference fails the gate.

### Compute Sanitizer

`--sanitize` replaces the normal GPU run with
`compute-sanitizer --tool memcheck --error-exitcode 99`. The sanitized run's
IVF is the one byte-gated, decoded and plane-compared. Parsing:

* `ERROR SUMMARY: N errors` with `N > 0` → FAIL (errors are printed).
* `Target application terminated before first instrumented API call` → SKIP
  (no CUDA work; usually the CPU-reference staging path).
* no `ERROR SUMMARY` → FAIL (sanitizer did not complete).

Sanitizer output is **not** used as execution attestation: it can instrument
`cudaMalloc` without running a kernel. The normal stats/marker/nsys attestation
still applies.

`--racecheck` runs `--tool racecheck` as well, at 64x64 only. Any reported
error or warning fails the case; other sizes print `racecheck: skipped`. The
race run uses separate output paths so it never overwrites the gated IVF.

### Native inputs

`--size WxH` and the matrix generate testsrc2 at the exact requested size.
There is no 640x360→1080p upscale proxy any more, so 720p/1080p exercise real
bottom/right edge geometry. `bench.py` uses the same native generation.

## Execution attestation (contract)

`verify_gpu.sh` accepts any one of these as proof that `--still-gpu` actually ran
on the GPU. Without one it reports `SKIP`, because identical bytes from a CPU
fallback are indistinguishable from a working GPU path.

1. **Stats JSON (harness-only today).** The gate reads a JSON file at
   `$AV2ENC_GPU_STATS` (also exported as `$AV2ENC_STILL_GPU_STATS`) with either
   `"backend": "gpu"` or a positive `kernels`/`launches` count, **and**
   `"frame_bytes_from_cpu": false` (or the key absent). `bench.py` reads the
   same file for kernel time, launch count and the VRAM high-water mark. The
   **CLI does not write this file yet** (`still_gpu_probe --stats` is the only
   producer), so on a CLI run this channel is usually absent and the gate falls
   through to the log/nsys checks below. Treat it as the preferred format once
   the product glue lands, not as a currently emitted artifact.
2. **Log marker.** The GPU run prints a line matching `backend=gpu`,
   `backend: gpu`, `using the GPU`, `GPU path ran/active/enabled`, or a
   `stage ... gpu` list. The completion banner (`av2enc: M1 GPU still-picture
   encode complete`) is **not** evidence: every `--still-gpu` run prints it,
   including the CPU-reference staging path.
3. **Nsight Systems.** If neither exists, the gate re-runs the GPU encode under
   `nsys profile --trace=cuda` and requires at least one CUDA kernel instance.
   (`ncu` counters are blocked in this container; do not depend on them.)

A run that says `frame bytes came from the CPU M1 reference`, `stage hooks are
not linked`, or sets `frame_bytes_from_cpu: true` is a **staged CPU fallback**
and is reported `SKIP`, even when the bytes are identical and the exit code is
0. That is the current orchestrator behaviour while the stage hooks land
(`GpuResult::frame_bytes_from_cpu`, `GpuStageStatus`).

Do not infer "GPU" from a single stage. On the current tree the plan stage can
report `plan: gpu` (the host plan bridge runs) while every coding stage is
`cpu-reference` and `frame_bytes_from_cpu` is still true. `frame_bytes_from_cpu`
is the authoritative bitstream-origin flag; a stats file that claims
`backend: gpu` with `frame_bytes_from_cpu: true` is rejected.

Recommended stats schema (extra keys are ignored, all optional except one
attestation key):

```json
{
  "backend": "gpu",
  "frame_bytes_from_cpu": false,
  "kernels": 14,
  "launches": 14,
  "kernel_ns": 812345,
  "encode_ns": 1450000,
  "cuda_events": true,
  "arena_reserved_bytes": 11534336000,
  "arena_peak_bytes": 18234496,
  "device_peak_bytes": 11550000000,
  "stages": {"plan": "gpu", "search": "gpu", "tx": "gpu",
             "recon": "gpu", "chroma": "cpu-reference", "cfl": "gpu",
             "deblock": "gpu", "sched": "stub"}
}
```

`bench.py` labels the kernel-time source as `stats(cuda events)` when
`"cuda_events": true`, so CUDA-event timings are distinguishable from an
`nsys` sum. `GpuStats` currently exposes `kernel_launches`, `sync_count`,
`arena_*` and `device_*`; a `kernel_ns` field measured with CUDA events would
need to be added by the orchestrator (`GpuStats::total_ms` is host wall time,
not kernel time).

## Recon parity (contract)

`StillResult` already exposes `recon_y/u/v`; `tools/still_verify/still_dump` writes them
as `StillDUMP1`. For the GPU side, the orchestrator can expose its planes the same
way: when `AV2ENC_STILL_GPU_DUMP=<path>` is set, `--still-gpu` should write an
`StillDUMP1` file (same magic, JSON header, Y/U/V planes). `plane_compare.py`
then compares GPU planes to the CPU dump bit-exactly.

`verify_gpu.sh --recon` also accepts a CLI flag via
`--gpu-dump-arg --still-gpu-dump`. If no GPU dump is available, it can
build/use the optional probe:

```sh
tools/still_gpu/build_probe.sh          # needs a build-cuda lib with encode_still_gpu
tools/still_gpu/verify_gpu.sh in.y4m --recon
```

The probe calls `encode_still_gpu` with `allow_cpu_fallback=false`, writes the
same `StillDUMP1` format from `GpuResult::recon_y/u/v`, and records
`GpuStageStatus` plus `GpuStats` in JSON. With the current staged orchestrator
it exits 1 with `M1 GPU: required stage hooks are missing`; pass
`--probe-args --allow-cpu-fallback` to let it emit a CPU-reference dump for
plumbing tests (the gate ignores that dump because the backend is
`cpu-reference`). Probe output is only used when its stats say `backend: gpu`
and `frame_bytes_from_cpu: false`. Probe attestation is not used for the CLI
byte gate — the CLI run must prove itself.

If neither a dump nor a GPU probe is available it prints
`recon parity: fallback (no GPU recon dump)` and relies on the decoded-frame
compare — still a byte-exact check of the stream, just not of the encoder-side
recon buffers. `--require-recon` turns the absence into `SKIP`.

**Keep options in sync.** The CPU dump must be produced with the same options
as the CLI encode (depth, deblock, CfL, full-intra). A mismatch reconstructs a
different partition tree and looks like a transform bug; see
`tools/still_verify/README.md`. Use `--harness-args` to forward matching flags.

## Benchmark

```sh
tools/still_gpu/bench.py                       # 64x64, 640x360, 1080p
tools/still_gpu/bench.py --sizes 64x64 --iterations 1
tools/still_gpu/bench.py --cpu-baseline --json bench.json
```

Reported per row:

| Column | Meaning |
|---|---|
| `wall_ms` | Median of timed runs (`min_ms` is the best run). |
| `fps` | `1 / wall_ms` for a one-frame still. **Not** a video claim. |
| `kernel_ms` | GPU kernel time: stats/CUDA events if supplied, else the `nsys` kernel-sum. |
| `kern%` | `kernel_ms / wall_ms`; not concurrency-aware. |
| `launch` | CUDA launch API calls (`cudaLaunchKernel`, `cudaGraphLaunch`, …). |
| `kerns` | Kernel instances (a graph launch can contain many). |
| `vram_MiB` | High-water mark from stats when available; otherwise a sampled `nvidia-smi` delta. |
| `attest` | `gpu`, `oracle` (CPU row), or `none`. Rows without GPU evidence are `SKIP` and excluded from the fps verdict. |

CPU rows are opt-in (`--cpu-baseline`) and labelled oracle/reference only.
Per `AGENTS.md`, CPU-only mode is a correctness oracle, not a performance
target.

Inputs are generated natively (testsrc2 at the requested size); 1080p is no
longer an upscale proxy. Defaults are deliberately light (`--iterations 3`,
one warmup) because the box is shared with other agents.

## Hardware disclaimer

The current box enumerates an **RTX 3060 with 12288 MiB**, compute capability
8.6. The product target is the **8192 MiB** RTX 3060 (`sm_86`). The 12 GB card
with a device-sized arena is a development/proxy configuration, not 8 GB
validation (`docs/negative-results.md`, "Benchmark / hardware"). `bench.py`
prints `WARNING: PROXY HARDWARE` whenever the name, memory size or compute
capability differs from the target, and the gate does **not** claim the 1 fps
video target — that stays unclaimed until video is wired and measured on an
8192 MiB 3060.

Kernel time from `nsys` includes profiler perturbation; use it for relative
comparison, not as an absolute product number.

## Current status (2026-11-11)

The 3060 `build-cuda` now runs the integrated GPU pipeline: every stage reports
`gpu`, `frame_bytes_from_cpu` is false, and the gate passes at 64x64, 640x360
and native 1920x1080. All PASS rows are attested by Nsight Systems kernel
counts because the CLI does not yet write `AV2ENC_GPU_STATS`; the probe does.

Full matrix on the live path (`--matrix`, 14 modes x 8 sizes, 209 s):

```
result:   50 PASS, 62 FAIL, 0 SKIP
attribution: 35 CPU-reference encode failures, 27 CPU-reference decode failures
            0 byte differences, 0 GPU decode failures, 0 plane differences
```

Representative PASS (1920x1080, default mode, 14,421 kernels observed):

```
frame 0: payload 87928 B  md5 cpu=… gpu=…  IDENTICAL
RESULT: IVF FRAME BYTES IDENTICAL
kernels:  14421 (GPU work observed)
cpu decode: ok (threads=1)
gpu decode: ok (threads=1)
RESULT: PLANES BIT-EXACT (0 differing samples)
RESULT: PASS
```

`--recon` at 1920x1080 compares `GpuResult::recon_*` to a freshly rebuilt CPU
`still_dump` and is bit-exact (Y/U/V md5s match the decoded stream as well).
`--determinism`, `--determinism-probe`, `--sanitize` (memcheck 0 errors) and
`--racecheck` (0 hazards) pass at 64x64/640x360.

The 62 matrix failures are all in the **CPU reference**, before GPU parity can
be judged. They are real correctness bugs for the M1 encoder workstream, not
harness bugs:

| Size | PASS / FAIL | Cause |
|---|---|---|
| 64x64, 640x360, 1920x1080 | 13 / 1 | `depth-8`: `EOB multisize 0 is not supported by M1 transform sizes` |
| 532x368 | 11 / 3 | `depth-0`, `depth-8`, `single`: `partition decision failed … spec-forced partition conflicts` |
| 136x88 | 0 / 14 | `dav2d`: `Invalid argument`; depths 0–3/8/single fail in the encoder |
| 1280x720 | 0 / 14 | `dav2d`: `Invalid argument`; depth 0/1/8/single fail in the encoder |
| 416x208 | 0 / 14 | `dav2d`: `Invalid argument`; depth 0–2/8/single fail in the encoder |
| 518x322 | 0 / 14 | encoder: `spec-forced partition conflicts` (all modes) |

A 640x360/depth-4 case failed once with **identical payload MD5s** but three
decoded samples differing; it did not reproduce in 200 straight decodes or
20 gate runs. The plane gate now decodes both streams back-to-back with
`--threads 1` and, when payloads are identical but planes differ, retries the
pair once and reports `decoded YUV: IDENTICAL (decoder retry)`. The clean
matrix above ran with that behaviour (zero retries needed).

A build without the flag, a blocker, or a staged CPU-reference orchestrator
still takes the earlier paths: `RESULT: SKIP (GPU path not present …)`,
`RESULT: SKIP (GPU run reported an unimplemented/blocked path)`, or
`RESULT: SKIP (GPU run used CPU-reference staging …)`. A SKIP is never a pass.

## Limitations

* Only the first frame is compared. M1 is a still-picture path; the CLI clamps
  `--frames` to 1.
* IVF timestamps are reported but not part of the byte gate (they are not
  stream data).
* `plane_compare.py` supports 8-bit 4:2:0 only, matching M1 scope.
* The plane gate decodes both streams back-to-back with `DAV2D_THREADS`
  (default 1) and retries the pair once if identical payloads decode
  differently. The retry keeps a rare decoder inconsistency from being
  reported as a GPU parity failure; the byte gate remains authoritative.
* If the encoder emits a stats file only on a clean exit, a crashed GPU run is
  classified FAIL (stub text) or SKIP (no IVF), not PASS.
* `nsys` attestation only proves that *some* CUDA kernel ran; it cannot prove
  the encoder's kernels produced the bytes. Prefer the stats file or an output
  marker. A deliberately deceptive CPU-fallback shim that also launches an
  unrelated kernel would be attested; the gate is not adversarial, it is
  trying to stop the realistic accident (silent fallback that does no GPU
  work).
* Recon parity is only as strong as the dump: `StillGpuResult` must come from the
  same encode options as the CPU `StillResult` dump, and from the GPU run that
  produced the IVF being gated. A dump from a second encode is not cross-checked
  against the IVF beyond the option fields in the header.
* A matrix FAIL can come from the CPU reference itself (for example the
  geometry bug in the snapshot above). The summary attributes CPU-reference
  decode failures separately so they are not mistaken for GPU parity failures.
  Strictness is intentional: a case whose reference does not decode cannot
  pass.
* `compute-sanitizer` output is not execution attestation. Memcheck can
  instrument `cudaMalloc` without a single kernel; the stats/marker/nsys
  channels still decide whether the GPU path ran.
* Racecheck is only run at 64x64 (per the gate contract). Any hazard error or
  warning fails the case; it is intentionally strict and may need tuning once
  real kernels land.
* `--sanitize` is slow (instrumented execution) and is meant for small
  dimensions; the matrix warns when it is combined with many cases.
