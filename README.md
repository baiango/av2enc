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
plus all subagent sessions — as of **2026-11-14**:

| Counter | Tokens |
|---|---:|
| Fresh input | 154,965,814 |
| Output | 26,341,729 |
| Reasoning | 27,573,926 |
| **Input + output + reasoning** | **~208.9 M** |
| Cached context reads | 17,529,485,504 |
| **Including cache reads** | **~17.74 B** |

361 sessions (main + subagents). Cache reads dominate because every subagent
re-reads a long context; they are normally billed at a fraction of fresh input.
The figures keep growing while work continues, so treat them as a snapshot, not
a final total.

## Status (2026-11-13)

The encoder is feature-complete for the spec tools that the pinned decoders can
validate. Both product paths are byte-identical to the CPU reference where the
same decisions are used.

**Objectives**

- **CPU:** maximize CVVDP/JOD at **256x144 @ 30 fps, 30 kbps**. Quality is the
  KPI, not CPU speed.
- **GPU:** use the decisions calibrated at 144p30/30 kbps unchanged, and
  maximize **1280x720** encode throughput as a cross-encoder benchmark.

**Measured baselines** (Shine clip, 60 frames)

| Case | Bitrate | CVVDP | PSNR-Y |
|---|---:|---:|---:|
| CPU 144p30, QP 150 | 30.3 kbps | 8.354 JOD | 32.25 dB |
| AVM reference 144p30, CBR 23 | 30.6 kbps | 9.927 JOD | 59.22 dB |
| CPU/GPU 144p30 calibration | 30.3 kbps | drift 0 | drift 0 |
| GPU 720p, QP 150 (calibrated policy) | 268.8 kbps | 9.465 JOD | 40.83 dB |
| GPU 720p, QP 90 | 1943 kbps | 9.876 JOD | 51.76 dB |

GPU 720p speed on the RTX 3060: **4.15 fps at QP 150**, **5.92 fps at QP 90**
(60 frames). The CPU is the quality reference; GPU output is currently
byte-identical at the calibration point.

**Reference pins**

| Case | MD5 |
|---|---|
| Basic 64x64 | `7efc3162bc0a4785685eba291a8096fd` |
| Basic 640x360 | `56f11d4a63ff07f2356ef5064b84fe0e` |
| Still 64x64 | `0ef9825062a25c4d950ee924113c6f89` |
| Still 640x360 | `e61de628b0812b716b0efa7dcafcdd06` |
| Video 720p default | `c6d588647aacb2275dfaea41a00fb501` |

**Implemented surface**

- Still: partition tree, full directional intra, MRL/DIP/IBP, CfL (derived,
  explicit, downsampling), MHCCP, palette, IntraBC, IST, TX type/partition,
  IDTX, CCTX, TCQ, RDOQ, QM, DC/AC delta-Q, segmentation/per-segment Q,
  deblock, CDEF, CCSO, Wiener NS, PC-Wiener, chroma LR, GDF, film grain,
  content interpretation, SB128/256, extended/SDP/sub-8 partitions, multi-tile,
  lossless, and the 10-bit/chroma formats.
- Video: key/P/B frames, 1-7 references, quarter/half/eighth pel, compound
  AVERAGE/wedge/diffwtd, temporal layers, long-term/OLK/leading/SEF/SWITCH/RAS,
  MFH, timing, content interpretation, adaptive/average CDF, CRF/VBR,
  multi-tile, deblock, CDEF, CCSO, Wiener/chroma LR/GDF/PC-Wiener, BAWP,
  global motion/warp, CfL, MHCCP, QM, full intra search, TX search, and
  10-bit/chroma formats. TIP is CPU-only; morph, OPFL/refine-MV, inter-intra,
  sub-PU/temporal MV and video segmentation are still being completed.

**Tests**

- Default `ctest`: 10/10 in about 5 s.
- Product gates: `gate_video_gpu`, `gate_video_filters`, `gate_video_formats`,
  `gate_video_gpu_intra`, `gate_video_tx_search` all green on the RTX 3060.
- CUDA module batch: 100 suites, 98 pass; the two failures are the pre-existing
  `tx_partition` and still `recon` harnesses (test-only; the product paths pass).
- CPU baseline: 19 pinned streams, 18 gates, 30 features.
- Every accepted stream is checked with pinned `dav2d --strict 1` and decoded
  picture counts; AVM is used only where `dav2d` cannot parse a spec tool
  (user QM, CDF continuity, layered SVC).

## CLI

`av2enc --help` is the contract; this is the summary. Every mode is opt-in
except Basic, and every mode writes an AV2 IVF (fourcc `AV02`).

| Mode | Flag | What it encodes |
|---|---|---|
| Basic | *(default)* | One key frame: 8-bit 4:2:0, 64x64 superblocks, 32x32 DCT, DC intra only, no filters |
| Still | `--still` | One frame with the partition tree, full intra modes, variable transform sizes, CfL and deblock |
| Still GPU | `--still-gpu` | The same still syntax with the block pipeline on the GPU; needs a CUDA build and implies `--still` |
| Video | `--video` | Key/P/B sequence with the full CPU video tool set, `--frames N` |
| Video GPU | `--video-gpu` | The same sequence with the M2 device backend; implies `--video`, needs a CUDA build, never falls back to the CPU encoder |
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

Video options: `--p-frames`, `--b-frames`, `--video-refs 1..7`,
`--video-num-ref-frames`, `--video-mv-precision`, `--video-subpel-eighth`,
`--video-inter-residual`, `--video-tlayers`, `--video-compound`,
`--video-compound-wedge`, `--video-compound-diffwtd`, `--video-gme`,
`--video-warp`, `--video-tip` (CPU), `--video-bawp`, `--video-cfl`,
`--video-mhccp`, `--video-pc-wiener`, `--video-qm`, `--video-deblock`,
`--video-cdef`, `--video-ccso`, `--video-wiener`, `--video-chroma-lr`,
`--video-gdf`, `--video-intra-full`, `--video-tx-search`, `--tiles CxR`,
`--video-mfh`, `--video-timing`, `--avg-cdf`/`--adaptive-cdf`, and
`--rc cqp|crf|vbr` with `--crf` / `--bitrate`. `--video-gpu` supports the same
surface except the explicitly named CPU-only/decoder-gated modes (TIP on GPU,
morph, OPFL/refine-MV, inter-intra, sub-PU/temporal MV, segmentation,
implicit reference lists, DRL reorder, CDF continuity, SVC, user QM).

Usage and option-combination errors exit 2; encoder, decoder and I/O failures
exit 1. `--gpu-plan N` is a host-only multi-GPU dry run: it prints the tile
assignment and exits without writing a stream.

## Performance expectations

- **CPU mode objective:** maximize **CVVDP/JOD at 256×144 @ 30 fps and
  30 kbps**. The CPU is the quality reference; speed is not its KPI.
- **GPU mode objective:** use the decisions calibrated at **144p30 @ 30 kbps**
  (minimal drift: CVVDP ±0.05 JOD, bitrate ±2%, PSNR-Y ±0.1 dB) unchanged,
  then maximize **1280×720** encode throughput. The 720p run is a benchmark
  against other encoders, not a separate quality-tuning point. Production
  hardware is the RTX 3080 Ti; the RTX 5090 is the maximum-speed host.
- **Not targets:** the old 1080p ≥1 fps floor, 8 GB cards, RTX 4070 Super and
  RTX 5070.
- **macOS:** AVM/CVVDP scoring only; not an encoder target.

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
- GPU benchmark target: maximize **720p encode fps** on the 3060/3080 Ti/5090
  using the 144p30/30 kbps decisions; there is no 1080p speed floor anymore.
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

## Current capabilities

The encoder covers the spec tools that the pinned decoders validate, in both
still and video modes. The remaining gaps are explicit decoder or product
limits, not hidden fallbacks:

- **CPU completeness wave in progress:** morph, OPFL/refine-MV, inter-intra,
  sub-PU/temporal MV, video segmentation/delta-Q, 10-bit/chroma CDEF/CCSO/LR,
  metadata OBUs, SVC layers and implicit reference lists.
- **Decoder-gated with `dav2d`:** user-defined QM OBUs, cross-frame CDF carry,
  multi-layer SVC and implicit reference maps (AVM validates some of these).
- **Not signalable in AV2 v1.0.0:** 12-bit.
- 1-D intra types with negative H/V_DCT are rejected by the pinned decoder.

See `docs/cuda-gap-analysis.md` for the per-tool CUDA matrix and
`docs/negative-results.md` for every confirmed decoder/product limitation.

## Directory ownership

Read `AGENTS.md` for the current ownership map and the shared-root rules
(`src/main.cpp`, `CMakeLists.txt`, `refactor/baseline.json`, pipeline roots).
Agent worktrees and branches live outside the main checkout.

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
a three-way budget (max CVVDP subject to a bitrate cap and a speed floor).

Current gates:

- **CPU quality:** 256x144 @ 30 fps at **30 kbps**, maximize CVVDP. Baseline
  8.354 JOD; AVM reference 9.927 JOD at 30.6 kbps.
- **GPU calibration:** same 144p30/QP150 decisions; CVVDP ±0.05 JOD,
  bitrate ±2%, PSNR-Y ±0.1 dB drift. Current drift is zero.
- **GPU benchmark:** 1280x720 throughput with the calibrated decisions
  (4.15 fps at QP 150, 5.92 fps at QP 90 on the RTX 3060).

Validate a report's host with `tools/check_target_host.py` (RTX 3060 12 GB /
3080 Ti / 5090).

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
