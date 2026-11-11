# M1 GPU verification and benchmark

Gate for the GPU-resident M1 path (`--m1-gpu`, workstream 10 in
[`docs/m1-gpu.md`](../../docs/m1-gpu.md)). The GPU path is only "working" when
it reproduces the CPU M1 reference **bitstream exactly** — not when it looks
good in PSNR.

```
Y4M ──av2enc --m1──────> cpu.ivf ──┐
                                    ├─ IVF frame-payload compare (bytes + MD5)  -> FAIL on any byte
Y4M ──av2enc --m1-gpu──> gpu.ivf ──┘
        │                            ├─ dav2d decode both -> Y4M plane compare   -> FAIL on any sample
        └─ attestation ──────────────┘   (stats JSON / log marker / nsys kernels)
                 │
                 └─ optional recon parity: GPU M1GpuResult planes vs CPU M1Result.recon_*
```

The gate is **not** a quality metric. A stream can sit at 50+ dB with one block
desynced, and a CPU fallback under `--m1-gpu` would produce identical bytes
while proving nothing. Hence the byte gate plus an execution attestation.

## Files

| File | Purpose |
|---|---|
| `verify_gpu.sh` | CPU-vs-GPU gate: frame bytes, MD5, `dav2d` decode, decoded-plane compare, recon parity, attestation, PASS/FAIL/SKIP. |
| `ivf_compare.py` | IVF frame-payload byte comparison (parses the 32-byte IVF header and 12-byte frame headers). |
| `plane_compare.py` | Bit-exact first-frame Y/U/V comparison; reads Y4M or `M1DUMP1` dumps. |
| `bench.py` | Wall/kernel/launch/VRAM benchmark at 64x64, 640x360 and 1080p, with proxy-hardware labelling. |
| `m1_gpu_probe.cpp` | Optional probe: calls `encode_m1_gpu` directly, writes an `M1DUMP1` recon dump and a stats JSON. Used for recon parity when the CLI exposes no dump. |
| `build_probe.sh` | Builds the probe against an existing `build-cuda/libav2enc_core.a`; never touches CMake or `src/`. |
| `remote.sh` | Syncs only `tools/m1_gpu` to the 3060 box and runs `verify`/`bench` there. |

## Verdicts

| Exit | Verdict | Meaning |
|---:|---|---|
| 0 | `PASS` | GPU execution attested, frame bytes identical, both streams decode, decoded planes identical (and recon planes identical when a dump exists). |
| 1 | `FAIL` | Any byte/plane/decode difference, a GPU crash, or a missing GPU stream that was not a stub. |
| 2 | setup | Missing encoder/decoder/ffmpeg/input; bad arguments. |
| 3 | `SKIP` | `--m1-gpu` absent from the build; GPU run reported a stub/blocked path; GPU exited 0 without an IVF; or bytes matched but **GPU execution could not be attested** (possible CPU fallback). A SKIP is never a pass. |

`ALLOW_SKIP=1` (or `--allow-skip` for `bench.py`) maps `SKIP` to exit 0 for CI
that only wants "no regression", without hiding the verdict line.

## Usage

On the 3060 box (or any Linux+CUDA host):

```sh
# zero-argument smoke gate: generates a 640x360 testsrc2 frame
tools/m1_gpu/verify_gpu.sh

# explicit input and recon parity
tools/m1_gpu/verify_gpu.sh testdata/frame.y4m --recon

# 1080p, keep artifacts, JSON-ish tool output
tools/m1_gpu/verify_gpu.sh --size 1920x1080 --keep
```

From a laptop, using the prescribed tunnel:

```sh
AV2ENC_HOST='<user>@<host>' AV2ENC_PORT='<port>' \
    tools/m1_gpu/remote.sh verify

AV2ENC_HOST='<user>@<host>' AV2ENC_PORT='<port>' \
    tools/m1_gpu/remote.sh bench --sizes 64x64,640x360
```

`remote.sh` sends only `tools/m1_gpu` by default so it cannot clobber another
agent's in-flight `src/` changes on the shared box. `FULL_SYNC=1` switches to
`tools/sync_remote.sh`.

`verify_gpu.sh` uses `$AV2ENC` / `$DAV2D` when set; otherwise
`build-cuda/av2enc` and the first of `tools/bin/dav2d`,
`tools/bin/dav2d-macos`.

Options worth knowing:

| Option | Effect |
|---|---|
| `--size WxH` | Generate a testsrc2 input when no input file is given. |
| `--cpu-args "..."` / `--gpu-args "..."` | Extra CLI flags for one side only (`--m1-full-intra`, `--no-cfl`, …). |
| `--recon` | Also compare encoder-side reconstructions. |
| `--gpu-dump PATH` | GPU `M1DUMP1` file to compare (implies `--recon`). |
| `--gpu-dump-arg FLAG` | Append `FLAG <workdir>/gpu-recon.dump` to the GPU command. |
| `--require-recon` | `SKIP` when no GPU recon dump is available. |
| `--no-attestation` | Explicitly unsafe: PASS on identical bytes even with no GPU evidence. |
| `--no-nsys` | Do not fall back to Nsight Systems attestation (faster, more SKIPs). |
| `--keep` | Keep IVF/log/dump artifacts. |

## Execution attestation (contract)

`verify_gpu.sh` accepts any one of these as proof that `--m1-gpu` actually ran
on the GPU. Without one it reports `SKIP`, because identical bytes from a CPU
fallback are indistinguishable from a working GPU path.

1. **Stats JSON.** The process writes a JSON file at `$AV2ENC_GPU_STATS`
   (also exported as `$AV2ENC_M1_GPU_STATS`) with either
   `"backend": "gpu"` or a positive `kernels`/`launches` count, **and**
   `"frame_bytes_from_cpu": false` (or the key absent). This is the preferred
   channel; `bench.py` reads the same file for kernel time, launch count and
   the VRAM high-water mark.
2. **Log marker.** The GPU run prints a line matching `backend=gpu`,
   `backend: gpu`, `using the GPU`, `GPU path ran/active/enabled`, or a
   `stage ... gpu` list. The completion banner (`av2enc: M1 GPU still-picture
   encode complete`) is **not** evidence: every `--m1-gpu` run prints it,
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

`M1Result` already exposes `recon_y/u/v`; `tools/m1_verify/m1_dump` writes them
as `M1DUMP1`. For the GPU side, the orchestrator can expose its planes the same
way: when `AV2ENC_M1_GPU_DUMP=<path>` is set, `--m1-gpu` should write an
`M1DUMP1` file (same magic, JSON header, Y/U/V planes). `plane_compare.py`
then compares GPU planes to the CPU dump bit-exactly.

`verify_gpu.sh --recon` also accepts a CLI flag via
`--gpu-dump-arg --m1-gpu-dump`. If no GPU dump is available, it can
build/use the optional probe:

```sh
tools/m1_gpu/build_probe.sh          # needs a build-cuda lib with encode_m1_gpu
tools/m1_gpu/verify_gpu.sh in.y4m --recon
```

The probe calls `encode_m1_gpu` with `allow_cpu_fallback=false`, writes the
same `M1DUMP1` format from `GpuResult::recon_y/u/v`, and records
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
`tools/m1_verify/README.md`. Use `--harness-args` to forward matching flags.

## Benchmark

```sh
tools/m1_gpu/bench.py                       # 64x64, 640x360, 1080p
tools/m1_gpu/bench.py --sizes 64x64 --iterations 1
tools/m1_gpu/bench.py --cpu-baseline --json bench.json
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

Defaults are deliberately light (`--iterations 3`, one warmup) because the box
is shared with other agents.

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

The working tree now has `src/cuda/m1/` and the `--m1-gpu` CLI wiring, but the
orchestrator still stages a CPU reference while stage hooks are missing:
`GpuResult::frame_bytes_from_cpu` is true. The `build-cuda` binary on the 3060
box at the time of testing was older than the tree and rejected the flag
outright. Both are `SKIP`:

```
# blocked build
av2enc: error: --m1-gpu is not available in this build: the M1 GPU pipeline
(src/cuda/m1/m1_gpu.h) is not integrated yet. Use --m1 for the CPU M1 path.
RESULT: SKIP (GPU run reported an unimplemented/blocked path (exit 2))

# staged CPU-reference orchestrator (exit 0, byte-identical output)
av2enc: note: M1 GPU stage hooks are not linked yet; frame bytes came from
        the CPU M1 reference
RESULT: SKIP (GPU run used CPU-reference staging; frame bytes did not come
        from GPU stages)
```

A build without the flag takes the earlier exit: `RESULT: SKIP (GPU path not
present in this encoder build)`. All are the intended behaviour for "not wired
yet": no false pass. The CPU reference path itself is covered separately by
`tools/m1_verify/`.

The gate has also been exercised against local fakes: a silent CPU fallback
that claims `--m1-gpu` (SKIP: unattested), a real `--m1-gpu` blocker (SKIP), a
staged CPU-reference shim that prints the orchestrator's note (SKIP), a stub
that prints "not implemented" (SKIP), a corrupted GPU stream (FAIL at the byte
gate and at decode), recon mismatch (FAIL), and stats/nsys-attested runs (PASS,
including the real `nsys` kernel-count path on the 3060). Sample output is in
the handoff report, not committed.

## Limitations

* Only the first frame is compared. M1 is a still-picture path; the CLI clamps
  `--frames` to 1.
* IVF timestamps are reported but not part of the byte gate (they are not
  stream data).
* `plane_compare.py` supports 8-bit 4:2:0 only, matching M1 scope.
* If the encoder emits a stats file only on a clean exit, a crashed GPU run is
  classified FAIL (stub text) or SKIP (no IVF), not PASS.
* `nsys` attestation only proves that *some* CUDA kernel ran; it cannot prove
  the encoder's kernels produced the bytes. Prefer the stats file or an output
  marker. A deliberately deceptive CPU-fallback shim that also launches an
  unrelated kernel would be attested; the gate is not adversarial, it is
  trying to stop the realistic accident (silent fallback that does no GPU
  work).
* Recon parity is only as strong as the dump: `M1GpuResult` must come from the
  same encode options as the CPU `M1Result` dump, and from the GPU run that
  produced the IVF being gated. A dump from a second encode is not cross-checked
  against the IVF beyond the option fields in the header.
