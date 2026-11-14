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
| Fresh input | 161,190,385 |
| Output | 27,328,250 |
| Reasoning | 28,743,431 |
| **Input + output + reasoning** | **~217.3 M** |
| Cached context reads | 18,336,365,247 |
| **Including cache reads** | **~18.55 B** |

385 sessions (12 top-level + 373 subagents). Cache reads dominate because every
subagent re-reads a long context; they are normally billed at a fraction of
fresh input.
The figures keep growing while work continues, so treat them as a snapshot, not
a final total.

## Status (2026-11-14)

The CPU writer covers every AV2 tool that the pinned decoders (`dav2d`, plus
`avmdec` where `dav2d` cannot parse) can validate, including the cases
previously misdiagnosed as decoder limits (ref-MV bank,
`mv_traj` with >=3 refs, 8-bit 4:0:0 PC-Wiener, 4:2:2 16-pixel leaves, TIP
block references, arbitrary frame dimensions). The GPU path is the
decision-identical M2 mirror of the CPU and is byte-identical where the same
decisions are used.

**Objectives**

- **CPU:** maximize CVVDP/JOD at **480x240 (240p, 2:1) @ 30 fps, 100 kbps**.
  Quality is the KPI, not CPU speed.
- **GPU:** use the decisions calibrated at 240p30/100 kbps unchanged, and
  maximize **1440x720** encode throughput as a cross-encoder benchmark.

**Measured baselines** (NASA clip, 60 frames). The benchmark source is
[`First-8K-Video-from-Space_orig.mp4`](https://images-assets.nasa.gov/video/First-8K-Video-from-Space/First-8K-Video-from-Space~orig.mp4)
(8192x4096, 2:1, bt709). The **default** CPU point is QP 189 at 480x240/30;
the measured opt-in points are `quality` at QP 190 and `--nightly` at QP 117.
Old Shine rows are historical.

| Case | Device | Bitrate CPU / GPU | CVVDP | PSNR-Y | Encode |
|---|---|---:|---:|---:|---:|
| CPU 240p30, default QP 189 (NASA) | Apple M4 | 100.0 / - kbps | 4.802 JOD | 24.47 dB | 23.7 fps (1 thread) / 54.6 fps (8 threads) |
| CPU 240p30, `quality` QP 190 | Apple M4 | 100.6 / - kbps | 6.302 JOD | 27.58 dB | 23.4 fps (8 threads) |
| CPU 240p30, `--nightly` QP 117 | Apple M4 | 100.1 / - kbps | **9.547 JOD** | **40.87 dB** | 9.1 fps (8 threads) |
| AVM CBR (NASA) | Apple M4 | 92.1 / - kbps | 8.791 JOD | 37.78 dB | 0.071 fps |
| CPU/GPU parity, QP 189 | RTX 3080 Ti | 100.0 / 100.0 kbps | drift 0 | drift 0 | 49.4 fps |
| CPU/GPU parity, QP 189 | RTX 3060 | 100.0 / 100.0 kbps | drift 0 | drift 0 | 40.0 fps |
| GPU 1440x720 QP90 | RTX 3080 Ti | - / 7664 kbps | 9.704 JOD | 42.85 dB | 5.40 fps |
| GPU 1440x720 QP90 | RTX 3060 | - / 7664 kbps | 9.704 JOD | 42.85 dB | 4.88 fps |

AVM's CBR rate control undershot its request and produced 92.1 kbps at
0.071 fps, so the quality comparison is at 92.1 vs 100.0 kbps. `quality` and
`--nightly` are CPU-only; the default and GPU paths keep the historical
25,009-byte stream / MD5 `a1d0dba1259b20a4b628d4cd1e14c33c`. The `quality`
stream is 25,141 B / MD5 `3984677b9b665952984b07b7cf75c6ea`; the nightly QP117
stream is 25,022 B / MD5 `b7388311a0e13cd90c7e1c9a2883588e`. At the 100.2 kbps
point, nightly is +4.74 JOD over the default and +3.24 JOD over `quality`.

The QP 189 GPU bitrate matches the CPU exactly: both hosts produce the same
25,009-byte stream as the CPU reference (`100.0 / 100.0 kbps`, MD5
`a1d0dba1259b20a4b628d4cd1e14c33c`). Encode fps above is 60 frames divided by
the best-of-3 wall time on the M4 (host shared); the thread count is shown.
The same 60-frame 480x240 stream takes 1.22 s on the RTX 3080 Ti (49.4 fps) and
1.50 s on the RTX 3060 (40.0 fps), i.e. 2.1x and 1.7x the 1-thread CPU
reference speed for the same 100 kbps stream. The 1440x720 QP90 stream is byte-identical
on both GPUs and on the CPU reference (MD5
`f9b6e086aa7d9da78792cf74b9746741`, 958,023 bytes); the CPU encodes the same
60 frames at 1.36 fps on the M4.

The previous Shine 1280x720 measurements were 268.8 kbps / 9.465 JOD / 40.83 dB
at QP 150 and 1943 kbps / 9.876 JOD / 51.76 dB at QP 90. The CPU is the quality
reference; GPU output is byte-identical at the calibration point.

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
  global motion/warp, CfL, MHCCP, QM, full intra search, TX search, DIP/IBP,
  MRL/IST, TCQ/CCTX, SDP/DDT, palette/IBC, SB128/256, sub-8/shared chroma,
  grain (all schedules), lossless (formats, hierarchical-P/B, 4:0:0), SVC,
  arbitrary dimensions, and 10-bit/chroma formats. TIP block references and
  the CPU-only tools are rejected by `--video-gpu` by name.

**Tests**

- `tools/test.sh`: 10/10 unit tests in about 6 s; `ctest`: 23/23.
- CPU product gates: `tools/run_gates.sh --no-ctest` **32/32** on the final
  tree, 19 pinned streams, 30 feature checks.
- The 1440x720 throughput + calibration gate was green on the RTX 3060 and
  3080 Ti benchmark hosts; the earlier full GPU product matrix was green on the
  RTX 3060 before those boxes were retired. The old 100-suite CUDA batch was
  98/100 (pre-existing test-only `tx_partition` and still `recon` harnesses).
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
the input must already have that format. Video accepts every size the selected
chroma layout can represent: 4:2:0 needs even width/height, 4:2:2 needs an even
width, and 4:4:4/4:0:0 accept any size. The coded picture is the §5.18.4.4
MI-aligned extent (`MiCols = 2*((W+7)>>3)`), the encoder reconstructs that
padded frame and the output/`--video-recon` picture is cropped to the real
W x H. `--video-gpu` keeps a named CPU-only rejection for sizes that are not
multiples of 8 (the CUDA kernels assume an MI-aligned picture); use `--video`
for every arbitrary size.

Video options: `--p-frames`, `--b-frames`, `--video-refs 1..7`,
`--video-num-ref-frames`, `--video-mv-precision`, `--video-subpel-eighth`,
`--video-inter-residual`, `--video-tlayers`, `--video-compound`,
`--video-compound-wedge`, `--video-compound-diffwtd`, `--video-gme`,
`--video-warp`, `--video-tip` (CPU), `--video-tip-block-ref` (CPU),
`--video-bawp`, `--video-cfl`,
`--video-mhccp`, `--video-pc-wiener`, `--video-qm`, `--video-deblock`,
`--video-cdef`, `--video-ccso`, `--video-wiener`, `--video-chroma-lr`,
`--video-gdf`, `--video-intra-full`, `--video-tx-search`,
`--video-partition-rd`, `--video-rd-preset quality`, `--nightly`, `--tiles CxR`,
`--video-mfh`, `--video-timing`, `--avg-cdf`/`--adaptive-cdf`, and
`--rc cqp|crf|vbr` with `--crf` / `--bitrate`. `--video-gpu` supports the same
surface except the explicitly named CPU-only/decoder-gated modes (TIP block
references, morph, OPFL/refine-MV, inter-intra, sub-PU/temporal MV,
segmentation, implicit reference lists, DRL reorder, CDF continuity, grain,
lossless, SB128/256, non-/8 dimensions, SVC, user QM).

Usage and option-combination errors exit 2; encoder, decoder and I/O failures
exit 1. `--gpu-plan N` is a host-only multi-GPU dry run: it prints the tile
assignment and exits without writing a stream.

CPU workers: `--cpu-threads N` (default 1 = the historical serial path, 0 =
auto = `hardware_concurrency`) runs the independent tile encoders and the
batched NEWMV motion search of the CPU `--video` reference on N host threads.
The work is partitioned before the threads start and merged in index order, so
the IVF bytes and the decoder-visible reconstruction are identical at every N;
`tests/multithread_identity.sh` asserts 1/2/4 workers for a whole-frame and a
2x2-tiled P sequence, and `tools/refactor/verify_baseline.sh --check` pins the
default stream. Entropy coding, the coding-order partition walk and the loop
filters stay serial, and `--video-gpu` rejects the flag (the device backend has
its own scheduling).

Measured on the M4 (median of 3, host shared):

| Case | 1 thread | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| 480x240 QP189, 60 f | 24.6 fps | 1.45x | 1.84x | 2.20x |
| 1440x720 QP90, 60 f | 2.41 fps | 1.22x | 1.59x | 2.18x |

The default single-tile P chain parallelises the batched NEWMV search (about
70% of the 720p encode); tiles additionally parallelise the tile encoders.

CPU-mode development now runs on the M4 (and Modal for larger sweeps) instead
of Vast.ai: the tight RD loop needs fast local iteration, and GPU boxes are
tuned for CUDA validation rather than CPU quality experiments. GPU hosts are
used only when CUDA parity or the 1440x720 benchmark is the task.

## Performance expectations

- **CPU mode objective:** maximize **CVVDP/JOD at 480×240 (240p, 2:1) @ 30 fps
  and 100 kbps**. The CPU is the quality reference; speed is not its KPI.
- **GPU mode objective:** use the decisions calibrated at **240p30 @ 100 kbps**
  (minimal drift: CVVDP ±0.05 JOD, bitrate ±2%, PSNR-Y ±0.1 dB) unchanged,
  then maximize **1440×720** encode throughput. The 720p run is a benchmark
  against other encoders, not a separate quality-tuning point. Production
  hardware is the RTX 3080 Ti; the RTX 5090 is the maximum-speed host.
- **Not targets:** the old 1080p ≥1 fps floor, 8 GB cards, RTX 4070 Super and
  RTX 5070.
- **CPU development loop:** Apple M4 (this Mac) with `--cpu-threads` and
  `AV2ENC_CPU_FAST=ON` for the daily RD loop; **Modal**
  (https://modal.com/) for bursty parallel QP/thread/RD sweeps. CPU-mode
  development moved off Vast.ai because GPU-box turnaround is slow for tight
  RD iteration; GPU hosts remain for CUDA validation and the 1440x720
  benchmark.

## CPU build options (opt-in)

The default build is portable and byte-identical on every host (`-O3`, no host
tuning). Three cache options tune host code generation only; none of them
changes the bitstream, and all are OFF by default:

| Option | Effect |
|---|---|
| `AV2ENC_NATIVE=ON` | `-mcpu=native` (Apple clang/arm64) or `-march=native` (x86 Clang/GCC). Host-specific binary. |
| `AV2ENC_LTO=ON` | Full `-flto` for the host C++ targets (CPU-only builds; ignored on CUDA builds). |
| `AV2ENC_CPU_FAST=ON` | Aggregate: NATIVE + LTO + `-funroll-loops -fno-math-errno -fno-trapping-math`. |

```sh
cmake -S . -B build-int -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DAV2ENC_ENABLE_CUDA=OFF -DAV2ENC_ALLOW_CPU_ONLY=ON \
      -DAV2ENC_CPU_FAST=ON
cmake --build build-int -j8
tools/test.sh --build build-int
```

Measured on the M4 with the same source and interleaved runs (best of N wall
times; the fanless host was also under other load):

| Case | Default | `AV2ENC_CPU_FAST=ON` | Gain |
|---|---:|---:|---:|
| 480x240 QP189, 60 f | 2.49 s (24.1 fps) | 2.00 s (30.0 fps) | +24% |
| 1440x720 QP90, 60 f | 26.2 s (2.29 fps) | 19.6 s (3.05 fps) | +33% |

Both streams stay byte-identical (25,009 B / `a1d0dba1...` and 958,023 B /
`f9b6e086...`), the 19-stream pinned baseline passes with the tuned binary, and
the default build is unchanged (`verify_baseline.sh --check`: 19 streams, 25/25
ctest, 30 feature checks).

Combined with `--cpu-threads 8` on the M4 (interleaved, best of 2):

| Case | Default, 1 thread | `CPU_FAST`, 8 threads | Gain |
|---|---:|---:|---:|
| 480x240 QP189, 60 f | 20.8 fps | 63.7 fps | 3.06x |
| 1440x720 QP90, 60 f | 2.58 fps | 6.41 fps | 2.48x |

All four binaries produce the same IVF bytes (`a1d0dba1...` / `f9b6e086...`). Absolute times on this host move with load (the same
default binary ranged 2.5–3.5 s at 480x240 across the session), so the gain is
measured interleaved against the default build rather than against the older
15.5 fps row above.

The gain is almost entirely **full LTO**. CMake's `INTERPROCEDURAL_OPTIMIZATION`
picks `-flto=thin` for AppleClang, which measured ~18% *slower* than the default
build here, so the option passes plain `-flto` explicitly. `-mcpu=native` and
`-funroll-loops` were within noise on the M4 (kept because x86 hosts can
autovectorize with `-march=native`). No fast-math is ever enabled and
`-ffp-contract=off` stays in force.

**Portability:** `AV2ENC_NATIVE`/`AV2ENC_CPU_FAST` binaries require the build
CPU; use the default build for release artifacts and the CUDA product. LTO is
skipped with a warning on CUDA builds because the device link is driven by
`nvcc`. Details and the full measurement notes: `docs/cpu-build-options.md`.

## Hard constraint

No encoder code from AVM, SVT-AV1/AV2, clave, or any other encoder.
Implement from the **AV2 v1.0.0 specification**.

`avmdec` / `dav2d` may only be used as **black-box validators** (decode our
output, compare pixels). Do not read or copy their encoder-side code.

## Target

- CPU RD / quality development: **Apple M4 (10 cores)** with `--cpu-threads`
  and `AV2ENC_CPU_FAST=ON`; **Modal** (https://modal.com/) for parallel sweeps.
  This replaced Vast.ai for CPU-mode development.
- GPU development / feature baseline: **RTX 3060 12 GB (`sm_86`)**.
- Production target: **RTX 3080 Ti 12 GB (`sm_86`)**.
- Maximum-speed target: **RTX 5090 32 GB (`sm_120`)**.
- 8 GB cards, RTX 4070 Super and RTX 5070 are **no longer targets**.
- Multi-GPU development: **N× RTX 3060 12 GB**, each with a device-sized arena.
- GPU benchmark target: maximize **720p encode fps** on the 3060/3080 Ti/5090
  using the 240p30/100 kbps decisions; there is no 1080p speed floor anymore.
- Build: **Linux + CUDA 12.8 + C++20 + CMake/Ninja** for the GPU product;
  **macOS arm64 + Apple clang** for CPU-mode development. The CPU is the
  quality/RD reference and the development path; the GPU is the product.
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

- **Linux x86_64 + CUDA** — the encoder product target (GPU hosts such as
  Vast.ai RTX 3060/3080 Ti/5090, CUDA 12.8).
- **macOS arm64** — the **CPU-mode RD/quality development host** (Apple M4):
  CPU encoder builds, `--cpu-threads`, gates, and AVM/CVVDP scoring. **Modal**
  (https://modal.com/) supplies bursty parallel CPU sweeps when the M4 is busy.
  CPU-mode development deliberately moved off Vast.ai, whose GPU boxes are
  tuned for CUDA tests and too slow to turn around for the RD loop. Mac CUDA is
  not supported.
- **Windows** — unsupported and untested. There is no Windows machine here, so
  if you want to run this on Windows: **good luck**. The C++20/CUDA sources are
  portable in principle, but the tooling is POSIX (`tools/*.sh`, SSH/tar sync).
  Ports and PRs are welcome; Windows is not a supported target.

## Current capabilities

The CPU writer covers the spec tools the pinned decoders validate. The
remaining limits are explicit decoder or product limits, not hidden fallbacks:

- **CPU:** all tracked features are wired and gated; see
  `docs/cpu-feature-completion.md` (36 items) and `docs/negative-results.md`.
- **GPU:** M2 mirror of the 8-bit 4:2:0 CPU decisions. CPU-only tools (morph,
  OPFL/refine-MV, inter-intra, TIP block references, segmentation, implicit
  maps, DRL reorder, CDF continuity, SVC, user QM) are rejected by name.
- **Pinned `dav2d` limits:** user QM OBUs, cross-frame CDF carry, the nested
  decoder model and multi-layer SVC need `avmdec`; 1-D negative H/V_DCT is
  rejected. Cross-xlayer SVC prediction is spec-forbidden (§6.8.2).
- **Not signalable in AV2 v1.0.0:** 12-bit.

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
It also records `decode_fps` and `decode_realtime_x` from `dav2d`. The current
AVM reference point is the NASA 240p30 row in the baseline table above
(92.1 kbps, 8.791 JOD); the historical Shine 720p sweep is archived in
`bench/avm_baseline/README.md`. Compare at matched bitrate, and always confirm
`dav2d` decoded the stream first. A fast-decode
profile is deferred until quality and encode speed are settled.

`tools/score_speed.py` reports speed vs CVVDP. `tools/score_tradeoff.py` enforces
a three-way budget (max CVVDP subject to a bitrate cap and a speed floor).

Current gates:

- **CPU quality:** 480x240 (240p, 2:1) @ 30 fps at **100 kbps**, maximize CVVDP.
  NASA baseline: 4.802 JOD at 100.0 kbps, PSNR-Y 24.47 dB (QP 189).
  Stable opt-in preset: `--video-rd-preset quality` measures **6.302 JOD
  (+1.50 JOD), PSNR-Y 27.58 dB (+3.11 dB)** at QP190 / 100.56 kbps; the
  interpolated 100.04 kbps point is 6.273 JOD. The per-tool table and exact
  flags are in `docs/video-rd-preset.md`.
  Algorithmic nightly aggregate: **`--nightly` measures 9.547 JOD /
  40.87 dB PSNR-Y at 100.1 kbps (QP117)** — +4.74 JOD over the default and
  +3.24 JOD over the stable preset at essentially the same rate. It combines
  64-band/refit/priced-unit CCSO, bottom-edge inter and residual masking, the
  measured partition probe, a reference-intro key/first-P QP offset, luma
  inter RDOQ, CDEF, the BAWP carrier over inter residual and the calibrated
  leaf RD cost (`--video-leaf-rd`). It is CPU-only and
  rejects `--video-gpu`, so the default and GPU streams keep the historical CPU
  decisions (`a1d0dba1...`, 25,009 B). The measured tables, per-tool
  contributions and exact flags are in `docs/video-nightly.md`.
- **GPU calibration:** same 240p30 decisions at QP 189; CVVDP ±0.05 JOD,
  bitrate ±2%, PSNR-Y ±0.1 dB drift. The QP189 stream is byte-identical on both
  GPU hosts and the CPU (`a1d0dba1...`).
- **GPU benchmark:** 1440x720 QP90, 60 frames: 5.40 fps on the RTX 3080 Ti and
  4.88 fps on the RTX 3060; 7664 kbps, CVVDP 9.704 JOD, PSNR-Y 42.85 dB
  (stream identical to the CPU reference).

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
