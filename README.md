# av2enc — from-scratch GPU AV2 encoder

> **Vibe-coded:** This project is built with **DeepSeek V4.1 Flash 1017** in
> **OpenCode**, with AI subagents writing most of the code and a human directing
> the architecture and review. Treat it accordingly.

> **Agents:** read `AGENTS.md`, `docs/cuda-interface.md`, and
> `docs/negative-results.md` before coding. Do not repeat listed failed
> approaches.

> **Docs go stale in hours.** This is an active research codebase that changes
> fast. Documents, tables, and status notes can lag the code. When they
> disagree, the **code and tests are authoritative** — read the implementation.

## Build provenance

Token usage recorded by OpenCode for the AV2 project tree — the main session
plus all subagent sessions — as of **2026-11-13**:

| Counter | Tokens |
|---|---:|
| Fresh input | 138,310,364 |
| Output | 23,678,242 |
| Reasoning | 25,225,516 |
| **Input + output + reasoning** | **~187.2 M** |
| Cached context reads | 15,029,932,736 |
| **Including cache reads** | **~15.22 B** |

324 sessions (main + subagents). Cache reads dominate because every subagent
re-reads a long context; they are normally billed at a fraction of fresh input.
The figures keep growing while work continues, so treat them as a snapshot, not
a final total.

## Status (2026-11-11)

**First frames decode with `dav2d`, on both CPU and CUDA.** The CUDA Basic
path uses the runtime/transform/intra components and produces a bitstream
identical to the CPU reference.

| Case | IVF | MD5 | PSNR avg | CUDA time |
|---|---:|---|---:|---:|
| 64x64 | 1,138 B | `7efc3162bc0a4785685eba291a8096fd` | 42.13 dB | 1.34 ms |
| 640x360 | 19,751 B | `56f11d4a63ff07f2356ef5064b84fe0e` | 47.20 dB | ~34 ms |

Both are gated by `tests/basic_encode_smoke.sh`; the CUDA build passes 6/6
tests. The 640x360 CUDA path is launch-bound (2,740 launches) and keeps entropy
on the CPU. A byte-exact GPU tile entropy coder exists, but it is 34–46× slower
per tile at Basic tile counts, so CPU entropy remains the default.
Adaptive CDFs are opt-in (`--adaptive-cdf`); defaults stay bit-identical.
The wavefront/CUDA-graph scheduler component reaches ~4.4 ms at 640×360 and
~21 ms at 1080p, but is not wired into the CLI/entropy writer yet.

Partial edge superblocks follow the §5.20.3.2 implied-partition recursion, not
a blanket 64×64 `PARTITION_NONE`. Basic leaf coders cover 64×64, 32×32, 32×16
and 32×8 (64×32/32×64 are partition nodes only), with the normative rectangular
DC predictor (`Div_Lut`), rectangular pre-scale, and luma transform-type syntax.
If the boundary recursion needs a smaller block (for example 136×88 or 518×322,
which reach 16×16/8×32, or right-edge 16×32 cases),
`m0::geometry_supported()` refuses the geometry and the CLI prints a note and
uses the Still path rather than emitting a desynchronised stream.
`tests/basic_edge_smoke.sh` gates the five reported sizes plus 64×64, 640×360
and 1080p.

### Still (opt-in, `--still`)

Partition tree + full intra + variable transforms + CfL + deblock. It is
**bit-exact against `dav2d` on all planes** (verified per sample, not by PSNR):

| Still | Still IVF | Avg PSNR | Basic IVF | Basic avg |
|---|---:|---:|---:|---:|
| 64×64 | 936 B | 43.28 dB | 1,138 B | 42.13 dB |
| 640×360 | 13,355 B | 46.37 dB | 19,751 B | 47.20 dB |
| 1080p, native | 61,574 B | 47.73 dB | 83,228 B | 49.79 dB |

CPU Still accepts `--qp 1..255` (default 90). `base_q_idx` selects the §8.3.1
coefficient-CDF qctx slice (`<=90`, `<=140`, `<=190`, else) and the §7.14.2
`qlookup` quantiser step; all four slices are gated against `dav2d --strict 1`
with zero reconstruction differences. `--still-gpu`, Basic and the Still GPU TX
stage still pin 90.

Full directional intra (`--still-full-intra`) is also bit-exact but not a
rate/quality win at QP 90 yet, so it stays opt-in. On the native 1080p frame
Still is 26% smaller than Basic but 3.0 dB lower PSNR-Y; AVM is still 2.45×
smaller than Still at comparable CVVDP. Details, source clip, and the AVM
comparison: `docs/m1-results.md`.

### Still GPU (opt-in, `--still-gpu`)

`--still-gpu` selects the GPU-resident Still pipeline
(`src/cuda/pipeline/still/`, see `docs/m1-gpu.md`) and implies `--still`; Basic
remains the default. It takes the same `--no-cfl`, `--no-deblock`,
`--still-full-intra`, `--still-depth`, and `--vram-cap` options as the CPU Still
path. CDEF, CCSO and film-grain signalling have device stages and are accepted;
Wiener/chroma-LR/PC-Wiener/GDF, RDOQ, TX-type, RDO-intra, UV-full and the
screen-content/QM tools have no GPU stage and are rejected by name. The target
is byte-identical output to CPU Still. `--still` stays the CPU reference path.
CPU-only builds parse the flag but reject it with a clear error, and a CUDA
build without the orchestrator entry point
(`src/cuda/pipeline/still/still_gpu.cu`) reports the path as unavailable rather
than silently falling back. Still tools without a GPU stage are rejected by
name (exit 2), never silently dropped.

## CLI

`av2enc --help` is the contract; this is the summary. Every mode is opt-in
except Basic, and every mode writes an AV2 IVF (fourcc `AV02`).

| Mode | Flag | What it encodes |
|---|---|---|
| Basic | *(default)* | One key frame: 8-bit 4:2:0, 64x64 superblocks, 32x32 DCT, DC intra only, no filters |
| Still | `--still` | One frame with the partition tree, full intra modes, variable transform sizes, CfL and deblock |
| Still GPU | `--still-gpu` | The same still syntax with the block pipeline on the GPU; needs a CUDA build and implies `--still` |
| Video | `--video` | A key frame plus P frames, `--frames N` |
| Video GPU | `--video-gpu` | The same key + P sequence with the M2 device backend; implies `--video`, needs a CUDA build, never falls back to the CPU encoder |
| Lossless | `--lossless` | Q=0 lossless still picture (DPCM intra, 8x8 blocks); its own writer |

Product flags are grouped in `--help` as Basic, Still and Video; test-only
knobs (depth caps, lambda scales, reconstruction dumps, `--gpu-plan`) live in
the Debug section. Still tools default off, and a tool the selected mode does
not implement is rejected by name instead of being ignored.

Effort presets: `--effort N` (10..10000, log scale) and `--cpu-used 0..8` add
measured Still features in score order, taking a prefix that never exceeds the
budget. The ranking is opt-in, implies `--still`, and is the only path that
changes a stream by design; `--effort-plan` prints the selection without
encoding. Scores, the exact `--cpu-used` mapping and the host-specific
calibration caveat are in `docs/effort.md`.

Formats: Basic, Still and Video are 8-bit 4:2:0 by default. `--still` also
selects `--bit-depth 10` (`C420p10`/`C422p10`/`C444p10`/`Cmono10`,
`--qp 1..303`) or an 8-bit 4:2:2/4:4:4/4:0:0 `--chroma` input (pins `--qp 90`);
the input must already have that format. Video requires dimensions divisible by 8.

Video options: `--p-frames`, `--video-refs`, `--video-mv-precision`,
`--video-subpel-eighth`, `--video-inter-residual`, `--video-long-term`,
`--video-olk`, `--video-leading`, `--video-sef`, `--video-switch`,
`--video-ras`, and `--rc cqp|crf|vbr` with `--crf` / `--bitrate`. `--rc crf`
derives `base_q_idx` from `--crf` and rejects an explicit `--qp`; `--rc vbr`
uses `--qp` as its statistics-pass anchor and codes the controller's per-frame
`base_q_idx` plan. `--video-gpu` mirrors the flat single-reference P chain and
1/8/quarter/half MV precision; it rejects the reference/header modes it does not
implement by name. B frames and DRL reordering are rejected until they pass the
pinned decoder gate.

Usage and option-combination errors exit 2; encoder, decoder and I/O failures
exit 1. `--gpu-plan N` is a host-only multi-GPU dry run: it prints the tile
assignment and exits without writing a stream.

## Performance expectations

- **Product target:** Vast.ai Linux instance (RTX 3080 Ti production,
  RTX 5090 maximum speed), GPU-resident encoder. The host CPU is a weak,
  shared, downclocked Xeon; it is a launch/control processor, not a pixel
  engine.
- **CPU-only mode:** correctness reference and test oracle only. It is not
  optimized, has no SIMD/threading work, and may be far slower than the GPU path.
  Do not benchmark it as the product or use it to judge CUDA design.
- **macOS:** AVM/CVVDP scoring only; not an encoder target.
- The 1080p ≥1 fps floor is measured on the RTX 3060 12 GB development box;
  production/max-speed runs use the 3080 Ti and 5090.

## Hard constraint

No encoder code from AVM, SVT-AV1/AV2, clave, or any other encoder.
Implement from the **AV2 v1.0.0 specification**.

`avmdec` / `dav2d` may only be used as **black-box validators** (decode our
output, compare pixels). Do not read or copy their encoder-side code.

## Target

- Development / feature baseline: **RTX 3060 12 GB (`sm_86`)**.
- Production target: **RTX 3080 Ti 12 GB (`sm_86`)**.
- Maximum-speed target: **RTX 5090 32 GB (`sm_120`)**.
- 8 GB cards, RTX 4070 Super and RTX 5070 are **no longer targets**.
- Multi-GPU development: **N× RTX 3060 12 GB**, each with a device-sized arena.
- Product speed target: **≥1 encoded fps at 1080p on the RTX 3060 12 GB**.
  The 3080 Ti and 5090 raise throughput; surplus compute goes into quality
  (RDOQ, partitions, modes, search, references), never into more fps.
- Build: **Linux + CUDA 12.8 + C++20 + CMake/Ninja**. CPU is a reference/test
  path only, not the product.
- Fat binary: `sm_86;sm_120`. The feature baseline stays `sm_86`; the same
  kernels run on both, with no Blackwell-only instructions required (no TMA,
  clusters, FP4 or tensor-core paths).
- VRAM: **no fixed preset**. The arena reserves
  `min(device_total, device_free) - 512 MiB`, optionally bounded by
  `--vram-cap <MiB>`. Extra VRAM scales **precompute**, never compression-tool
  shortcuts.
- Encoder architecture: GPU-resident. CPU only launches work, handles OBU
  headers and file I/O; pixel work, mode decision, filters, and tile entropy
  stay on the GPU.

## VRAM arena sizing

There is no hard-coded 7,500 MB ceiling and no "compat" profile switch. At first
GPU use the encoder queries the device and reserves the largest block it can
obtain, leaving 512 MiB for the context and driver:

```text
capacity = min(total_vram, free_vram) - 512 MiB
```

`--vram-cap <MiB>` sets an optional upper bound; `0` (the default) means "take
the maximum". If another process takes memory between the query and the
reservation, the encoder halves the request and retries instead of failing.
CPU-only builds accept `--vram-cap` as a no-op.

The arena is sized from the device; the RTX 3060 12 GB is the development and
feature baseline, with the RTX 3080 Ti for production and the RTX 5090 for
maximum throughput.

## Remote box

```
ssh -p <port> <user>@<host>
/workspace/av2enc
```

`/workspace` is **not** a persistent volume. The local repo at
`$HOME/work/av2enc` is the source of truth; sync to the box before builds.
Disk is 16GB: keep test media tiny (<100MB total), prefer 64x64/256x256 frames.

## Platforms

- **Linux x86_64 + CUDA** — the only supported encoder target (Vast.ai RTX 3060/3080 Ti/5090,
  CUDA 12.8).
- **macOS arm64** — **not an encoder build target.** It is used only for the AVM
  baseline, AVM encoding, and CVVDP/MPS quality scoring. CPU reference tests may
  run there, but Mac encoder support is not promised.
- **Windows** — unsupported and untested. There is no Windows machine here, so
  if you want to run this on Windows: **good luck**. The C++20/CUDA sources are
  portable in principle, but the tooling is POSIX (`tools/*.sh`, SSH/tar sync).
  Ports and PRs are welcome; Windows is not a supported target.

## First milestone

A valid AV2 still-picture IVF:

- 8-bit, 4:2:0
- one tile
- 64x64 and 256x256
- key frame only
- DC intra prediction
- DCT 32x32
- fixed QP
- no deblock/CDEF/CCSO/GDF/loop restoration, no film grain, no SVC, no palette/CfL

Success = `avmdec` (or `dav2d`) decodes the stream and output is close to source.

## Directory ownership

| Path | Owner |
|---|---|
| `docs/spec-*`, `spec/tables/`, `tools/spec_extract/` | spec agent |
| `src/io/`, `src/bitstream/`, `src/main.cpp`, `tests/`, `CMakeLists.txt` | scaffold agent |
| `bench/hello_cuda/`, `docs/remote-env.md` | CUDA agent |
| `src/entropy/`, `src/cpu/`, `src/cuda/` | later agents |

Do not edit outside your area without coordinating.

## Rules

1. Spec-only. Cite section numbers. Mark unknowns instead of inventing syntax.
2. Generate tables with scripts; never hand-type CDF/transform tables.
3. Write a scalar CPU reference before every CUDA kernel; assert bit-exactness.
4. Do not GPU the entropy coder. It is serial per tile.
5. Pin the spec version and decoder commit. AV2 is still moving.

## Quality

`tools/quality.py` computes bitrate, PSNR, SSIM, and CVVDP (JOD) for an IVF.
It also records `decode_fps` and `decode_realtime_x` from `dav2d`. The AVM
reference baseline is in `bench/avm_baseline/README.md`; compare at matched
bitrate, and always confirm `dav2d` decoded the stream first. A fast-decode
profile is deferred until quality and encode speed are settled.

`tools/score_speed.py` reports speed vs CVVDP. `tools/score_tradeoff.py` enforces
the three-way budget: max CVVDP subject to bitrate <= R_max and speed >= S_min.
The planned 720p60 gate is CVVDP >= 9.0 at <= 500 kbps, with speed measured
against AVM on the same machine. The product target is **1080p at >=1 fps on the
RTX 3060 12 GB development box**; the 3080 Ti and 5090 are the production and
maximum-speed hosts. Validate with `tools/check_target_host.py`.

## Profiling

The encoder has a built-in profiler that works without GPU counter permissions:

```sh
AV2ENC_PROFILE=/tmp/run.jsonl ./build-cuda/av2enc ...
```

It logs per-frame CPU and CUDA stage times, JSONL plus a summary, and VRAM
peak/capacity. `AV2ENC_PROFILE_NVTX=1` adds NVTX ranges for Nsight Systems.
See `docs/profiling.md`.

## License

The encoder source is released under **The Unlicense**, a public-domain
dedication. See [LICENSE](LICENSE). Anyone is free to copy, modify, publish,
use, compile, sell, or distribute this software, in source or binary form, for
any purpose, commercial or non-commercial.

Third-party material keeps its own terms:

- **AV2 specification material**: the generated tables and constants under
  `spec/tables/` and `src/encoder/filters/spec_tables/` are derived from the
  AV2 Bitstream & Decoding Process Specification v1.0.0, copyright Alliance for
  Open Media. The specification states that its licensing information is at
  <https://www.aomedia.org/license/>.
- **Specification PDF and attachments**: not redistributed. The PDF must be
  obtained separately, and the normative attachment headers are fetched on
  demand and SHA-256 verified by the table generator.
- **Reference decoders**: `dav2d` and `avmdec` are AOM reference software used
  only as black-box validators. They are not redistributed; `tools/bin/` is
  ignored by git.
- **Test media**: fixtures under `tests/data/` are generated, and `testdata/`
  is local-only and not required by the build.
