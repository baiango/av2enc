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

Token usage recorded by OpenCode for the work on this checkout — all top-level
sessions and their subagents — as of **2026-11-20**:

| Counter | Tokens |
|---|---:|
| Fresh input | 310,437,762 |
| Output | 43,506,339 |
| Reasoning | 51,262,329 |
| **Input + output + reasoning** | **~405.2 M** |
| Cached context reads | 28,328,893,787 |
| **Including cache reads** | **~28.73 B** |

880 sessions (290 top-level + 590 subagents). The count spans the three
OpenCode project ids that touch the checkout (the parent workspace, the repo
root and the publish clone), because the work moved roots while the project was
re-scoped. Cache reads dominate because every subagent re-reads a long context;
they are normally billed at a fraction of fresh input. Measured from
`session_v2` (`tokens_input`, `tokens_output`, `tokens_reasoning`,
`tokens_cache_read`) in the OpenCode state database.
The figures keep growing while work continues, so treat them as a snapshot, not
a final total.

## Status (2026-11-18)

The CPU writer covers every AV2 tool that the pinned reference decoder
(`avmdec`; legacy conformance gates still exercise `dav2d`) can validate,
including the cases
previously misdiagnosed as decoder limits (ref-MV bank,
`mv_traj` with >=3 refs, 8-bit 4:0:0 PC-Wiener, 4:2:2 16-pixel leaves, TIP
block references, arbitrary frame dimensions). The GPU path is the
decision-identical M2 mirror of the CPU and produces a **byte-identical** stream.
The parity gate enforces exact equality (`cmp` on the IVF), not a tolerance.

**Objectives**

- **CPU:** maximize the composite KPI **K** (see below) at **480x240 (240p,
  2:1) @ 30 fps, 100 kbps**. Quality is the KPI, not CPU speed; K blends CVVDP
  (50%), Butteraugli (25%) and SSIMULACRA2 (25%). **The target to beat is
  SVT-AV1 v4.3.0 preset 0** — the mature AV1 quality reference — at matched bitrate. AVM is
  kept as a reproducible calibration point.
- **GPU:** use the decisions calibrated at 240p30/100 kbps unchanged, and
  maximize **1440x720** encode throughput as a cross-encoder benchmark.

**Measured baselines** (NASA clip, 60 frames). The benchmark source is
[`First-8K-Video-from-Space_orig.mp4`](https://images-assets.nasa.gov/video/First-8K-Video-from-Space/First-8K-Video-from-Space~orig.mp4)
(8192x4096, 2:1, bt709). The **default** CPU point is QP 189 at 480x240/30;
the measured opt-in points are `quality` at QP 190 and `--fast-decode` at QP 106
(the rate-matched 100 kbps point; QP 110 is the gate regression configuration).
Rows recorded on the retired unlicensed clip are historical.

| Case | Device | Bitrate CPU / GPU | CVVDP | Butteraugli | SSIMULACRA2 | **K** | PSNR-Y | Encode |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CPU 240p30, default QP 189 (NASA) | Apple M4 | 100.0 / - kbps | 4.802 JOD | 21.79 | -41.17 | 28.5 | 24.47 dB | 89.6 fps (1 thread) / 95.2 fps (8 threads) |
| CPU 240p30, `quality` QP 190 | Apple M4 | 100.6 / - kbps | 6.302 JOD | 19.73 | -4.07 | 42.5 | 27.58 dB | 50.9 fps (8 threads) |
| CPU 240p30, `--fast-decode` QP 106 | Apple M4 | 100.0 / - kbps | **9.733 JOD** | **3.42** | **80.84** | **82.2** | **44.67 dB** | 19.0 fps (8 threads) |
| CPU 240p30, nightly + `AV2ENC_EFF_PERC=1` | Apple M4 | 100.0 / - kbps | 9.771 JOD | 2.26 | 82.44 | **85.1** | 44.46 dB | 7.5 fps (8 threads) ‡ |
| AVM e6f99d3, QP 124 (NASA) | Apple M4 | 99.3 / - kbps | 9.862 JOD | 1.29 | 90.38 | **90.2** | 50.27 dB | 1.4 fps (8 threads) |
| SVT-AV1 4.3.0, CRF 42, preset 4 (NASA) | Apple M4 | 98.6 / - kbps | 9.740 JOD | 2.15 | 85.04 | 85.8 | 45.81 dB | 300 fps |
| SVT-AV1 4.3.0, CRF 36.5, preset 0 (NASA) | Apple M4 | 100.0 / - kbps | **9.828 JOD** | **1.55** | **88.89** | **88.8** | **48.88 dB** | 19.1 fps |
| rav1e 0.8.1, speed 2, Q120 (NASA) | Apple M4 | 100.1 / - kbps | 9.717 JOD | 2.71 | 81.67 | 83.7 | 42.81 dB | 12.8 fps |
| CPU/GPU parity, QP 189 | RTX 3080 Ti | 100.0 / 100.0 kbps | drift 0 | drift 0 | drift 0 | 28.5 | drift 0 | 49.4 fps |
| CPU/GPU parity, QP 189 | RTX 3060 | 100.0 / 100.0 kbps | drift 0 | drift 0 | drift 0 | 28.5 | drift 0 | 40.0 fps |
| GPU 1440x720 QP90 | RTX 3080 Ti | - / 7664 kbps | 9.704 JOD | - | - | - | 42.85 dB | 5.40 fps |
| GPU 1440x720 QP90 | RTX 3060 | - / 7664 kbps | 9.704 JOD | - | - | - | 42.85 dB | 4.88 fps |

† AVM is a clean upstream build of `e6f99d3`, with no local patch. Commit,
binary SHA-256, commands and stream MD5s: `bench/avm_baseline/README.md`.
‡ eff-perc fps is measured at the QP119 recipe (100.92 kbps); the 100.0 kbps
quality point is interpolated. This is not an iso-QP comparison: on the
eff-perc drift recipe the gate is ~2.2x slower (6.8 vs 14.6 fps at QP106;
7.6 vs 16.5 fps at QP119) because it re-scores top-N candidates and prices the
interval rate.

Butteraugli (lower better) and SSIMULACRA2 (higher better) are measured with
`tools/extra_metrics.py` on the decoded stream; `-` means not measured (the GPU
parity rows inherit their CPU row, and AVM is measured with the current binary). The GPU parity rows are exactly
zero drift, i.e. identical streams, so they inherit their CPU row's metrics.

Encode-speed figures use the `release-timing` preset (`-O3`, `AV2ENC_HARDEN=OFF`);
pins, gates and CI use the hardened `release` build. The two produce byte-
identical streams, so quality rows are unaffected by the choice.

### The KPI: composite K over three perceptual metrics

The project optimises a single number, **K**, built from three full-reference
metrics because they disagree in ways that matter. **CVVDP** (JOD, higher better)
is the display-model measure every `--fast-decode` decision was tuned against.
**Butteraugli** (distance, lower better) is a contrast-masking model that reacts
to low-frequency error. **SSIMULACRA2** (0-100, higher better; negative means
badly broken) is a learned multi-scale metric.

Each metric is mapped onto a common 0-100 quality axis with **fixed published
anchors** (not the min/max of the measured set, so scores stay comparable across
runs), then the three are combined:

```text
q_cvvdp = clamp(JOD / 10, 0, 1)            # JOD 10 is the top of the scale
q_ba    = 1 / (1 + Butteraugli / 3)        # BA 3 is "clearly visible"
q_ss2   = clamp((SSIMULACRA2 + 50) / 150, 0, 1)   # 100 = perfect, -50 = floor

K = 100 * (0.50*q_cvvdp + 0.25*q_ba + 0.25*q_ss2)
```

CVVDP carries half the weight because its `standard_fhd` display model matches
the deployment case and every RD decision was calibrated on it. The other two
carry a quarter each so that CVVDP's saturation near 10 JOD cannot hide real
quality differences - which it demonstrably does: on the NASA clip CVVDP ranks
SVT preset 0 only 1.2% ahead of us while Butteraugli says 2.7x and SSIMULACRA2
says 9.2 points.

```sh
tools/kpi.py --cvvdp 9.713 --butteraugli 4.14 --ssimulacra2 79.08
# or: tools/kpi.py --quality-json q.json --extra-json e.json
```

```sh
# CVVDP: tools/quality.py (AVM avmdec, Apple MPS, standard_fhd display model)
AVMDEC=tools/bin/avmdec tools/quality.py --ref <ref>.y4m --ivf <stream>.ivf

# Butteraugli + SSIMULACRA2: tools/extra_metrics.py (per frame, averaged)
# needs `butteraugli_main` (libjxl) and `ssimulacra2` on PATH
tools/extra_metrics.py --ref <ref>.y4m --test <decoded>.y4m
```

Quality scoring uses the AVM decoder; set `AVMDEC` when it is not in
`tools/bin`. All K/PSNR tables in this document are **4:2:0**. A 4:4:4 or
4:2:2 stream is a separate oracle: do not mix formats or interpolate across
them. `tools/quality.py` and `tools/extra_metrics.py` reject chroma mismatches.
Legacy recon gates still invoke `dav2d` and are being migrated.

| Stream | Rate | CVVDP | Butteraugli | SSIMULACRA2 | **K** |
|---|---:|---:|---:|---:|---:|
| default QP189 | 100.0 kbps | 4.802 | 21.79 | -41.17 | 28.5 |
| `quality` QP190 | 100.6 | 6.302 | 19.73 | -4.07 | 42.5 |
| **`--fast-decode` QP106** | **100.0** | **9.733** | **3.42** | **80.84** | **82.2** |
| **`--fast-decode` + `AV2ENC_EFF_PERC=1`** | 100.0 | 9.771 | 2.26 | 82.44 | **85.1** |
| AVM e6f99d3 QP124 | 99.3 | 9.862 | 1.29 | 90.38 | **90.2** |
| SVT-AV1 p4 CRF42 | 98.6 | 9.740 | 2.15 | 85.04 | 85.8 |
| SVT-AV1 p0 CRF36.5 | 100.0 | **9.828** | **1.55** | **88.89** | **88.8** |
| rav1e speed 2 Q120 | 100.1 | 9.717 | 2.71 | 81.67 | 83.7 |

One opt-in pipeline improves on `--fast-decode` without changing its defaults: the
perceptual RD core (`AV2ENC_EFF_PERC=1`) at **85.1 K** at 240p/100 kbps (it is
negative at 720p: **43.6 K** at 293.7 kbps against the 45.61 pin). It is
environment-gated, off by default, and not part of `--fast-decode`. The lpl36 refresh
(`AV2ENC_PIPE_LPL_REFRESH=1`) revives the frame-refresh allocator with a
first-party metric: **83.9 K** at 240p/100 kbps (at the 83.80 wall) and flat at
the 720p pin, so it is a metric experiment rather than a new best; see
`docs/pipe-lpl-refresh.md`.

With AVM and rav1e included the metrics no longer rank identically: CVVDP puts
av2enc slightly ahead of rav1e (9.733 vs 9.717), while Butteraugli and
SSIMULACRA2 put rav1e ahead (2.71 vs 3.42; 81.67 vs 80.84). K makes the
disagreement explicit instead of letting one metric hide it. CVVDP alone
puts us 0.097 JOD (1.0%) behind preset 0, which reads as "nearly equal"; K puts us
6.7 points behind (82.2 vs 88.8), driven by Butteraugli and SSIMULACRA2 seeing
sustained luma-detail error that CVVDP discounts at this display size. Track K,
and treat a change that improves K without moving CVVDP as a real quality win
rather than a neutral result: the QP-scaled lambdas did exactly that (+0.7 dB
PSNR, +1.1 K, +0.02 JOD).

**AVM measurements.** AVM is built clean from upstream commit `e6f99d3`
(2026-11-12) with
`--end-usage=q --cpu-used=6 --threads=8`; no local patch is applied. CBR
undershoots at low targets, so these are constant-QP runs; the exact command
and binary SHA-256 are in `bench/avm_baseline/README.md`. QP 122/123/124 measured
**102.4 / 97.9 / 99.3 kbps at 9.855 / 9.855 / 9.862 JOD, 1.30 / 1.33 / 1.29 BA,
90.44 / 90.24 / 90.38 SS2**; QP 124 lands at 99.3 kbps and interpolates to
100.0 at **90.2 K** (9.860 JOD, 1.29 BA, 90.39 SS2, 50.27 dB). The 240p
QP 122/123 and 720p QP 203 streams are byte-identical to the old local
diagnostic build, so the instrumentation did not affect encoding. The older
CBR row (99.5 kbps / 8.452 JOD)
and the retired-clip sweep are kept in `bench/avm_baseline/README.md` for
history; the binaries and modes are not directly comparable. `quality` and
`--fast-decode` are CPU-only; the default and GPU paths keep the historical
25,009-byte stream / MD5 `a1d0dba1259b20a4b628d4cd1e14c33c`. The `quality`
stream is 25,141 B / MD5 `3984677b9b665952984b07b7cf75c6ea`; the nightly QP110
stream is 24,113 B / MD5 `7f03eb6e82d6ba4cf6666c78285f945b` (96.5 kbps). At
iso-rate ~100 kbps, nightly is +4.91 JOD over the default and +3.41 JOD over
`quality`. The
aggregate now also carries adaptive coefficient CDFs (a −13.9% entropy saving at
identical decisions, which pays for a `-25` key offset instead of `-16`) and the
drift-triggered inter refresh. See `docs/video-nightly.md`.

On **real detail** the picture changes: with a 1440x720/60 clip derived from an
8192x4096 8K master at 300 kbps, SVT preset 0 is **0.92 JOD / 3.3 dB ahead**
(7.32 vs 8.24 JOD). The smooth NASA 720p upscale gives the opposite sign
(+0.16 JOD for av2enc), so 720p numbers must name their content. Method, clip
build and the full ladder: `bench/space8k_720p/README.md`.

**K at 1440x720/60, ~300 kbps (native-detail clip, 60 frames = 1 s):**

| Encoder | Setting | Rate | CVVDP | Butteraugli | SSIMULACRA2 | K | Encode |
|---|---|---:|---:|---:|---:|---:|---:|
| av2enc `--fast-decode` | QP217, key -57, leaf 250, ext-partitions | 292.6 kbps | 7.324 | 28.87 | -10.15 | **45.6** | 2.4 fps |
| av2enc `--fast-decode` + `AV2ENC_EFF_PERC=1` | QP225 | 293.7 kbps | 7.118 | 32.41 | -14.90 | 43.6 | 0.76 fps § |
| av2enc `--fast-decode` + `AV2ENC_PIPE_LPL_REFRESH=1` | QP216/217 interp. | 292.6 kbps | 7.339 | 29.78 | -9.78 | 45.7 | 0.53 fps ¶ |
| AVM e6f99d3† | QP203 | 293.3 | 8.130 | 22.03 | 22.83 | **55.8** | 0.17 fps |
| SVT-AV1 4.3.0 p0 | CRF63/62 interpolated to 292.6 | 292.6 | 8.244 | 20.89 | 23.21 | **56.6** | 1.7 fps |
| rav1e 0.8.1 speed 2 | Q209/210 interpolated to 292.6 | 292.6 | 7.788 | 20.12 | 14.90 | **53.0** | 2.3 fps |

§ eff-perc is 240p-tuned; the 720p point is negative (43.6 vs 45.6 K) and is
shown for completeness.
¶ lpl36 is flat at 720p (45.7 vs 45.6 K); a metric experiment, not a default.

Encode figures are 60 frames divided by wall time at 8 threads. AVM 720p takes
344.0 s (0.17 fps) at QP203.

† Clean upstream AVM `e6f99d3`; provenance and commands:
`bench/avm_baseline/README.md`.


SVT-AV1 v4.3.0 was measured at two presets, decoded with `dav1d`, and scored
with the same `tools/quality.py` CVVDP model. **Preset 4 / CRF 42** gives
98.6 kbps / 9.740 JOD / 45.81 dB at 300 fps. **Preset 0 / CRF 36.5** gives
**100.0 kbps / 9.828 JOD / 48.88 dB** at 19.1 fps. Preset 0 is the quality
reference; preset 4 is the speed reference. AV1 is not AV2, so these are
cross-encoder calibration points, not codec-parity claims. At this rate the
nightly AV2 stream is **−0.007 JOD / −1.14 dB** versus preset 4 and
**−0.095 JOD / −4.21 dB** versus preset 0.

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

The previous 1280x720 measurements were 268.8 kbps / 9.465 JOD / 40.83 dB
at QP 150 and 1943 kbps / 9.876 JOD / 51.76 dB at QP 90. The CPU is the quality
reference; GPU output is byte-identical at the calibration point.

**Reference pins**

| Case | MD5 |
|---|---|
| Basic 64x64 | `c9294fa48f70fc82a26c64908ddc0b12` |
| Basic 640x360 | `0d90c958b8dd5d12c6ab2eef2a2c19bf` |
| Still 64x64 | `2b31e378787363641c836fdf563bbd0e` |
| Still 640x360 | `8ffbe059cfeac3e6ee255f3bb7cf72b8` |
| Video 720p default | `65feaa8685e2d861a44a65632268ba92` |

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
- Every accepted stream is checked with the reference decoder (`avmdec`;
  legacy recon gates still use `dav2d`) and decoded picture counts.

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
`--video-partition-rd`, `--video-rd-preset quality`, `--fast-decode`, `--tiles CxR`,
`--video-mfh`, `--video-timing`, `--avg-cdf`/`--adaptive-cdf`,
`--video-cdf-per-frame-qp`, and
`--rc cqp|crf|vbr` with `--crf` / `--bitrate`. `--video-gpu` supports the same
surface except the explicitly named CPU-only/decoder-gated modes (TIP block
references, morph, OPFL/refine-MV, inter-intra, sub-PU/temporal MV,
segmentation, implicit reference lists, DRL reorder, CDF continuity,
per-frame-QP CDF reset, grain,
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

## SIMD (AVX2 / NEON)

Explicit SIMD is **open** as an optimization path: **AVX2** on x86_64 and
**NEON** on arm64. Host-side SIMD is much cheaper and faster to implement,
validate and iterate than a new CUDA kernel, so it is explicitly in scope for
the CPU reference and for shared pixel/transform/filter/ME/entropy helpers.

The order is: scalar reference first, then compiler autovectorization/LTO
(`-O3`, `AV2ENC_NATIVE`/`AV2ENC_CPU_FAST`), then explicit intrinsics where the
profile shows a hot loop. SIMD code must keep a scalar fallback, be
byte-identical across x86_64 and arm64, and keep the decoder/recon gates green.
Do not use SIMD to change an RD decision unless the change is a measured,
decoder-gated opt-in.

## Performance expectations

- **CPU mode objective:** maximize the composite **K** (CVVDP 50%, Butteraugli
  25%, SSIMULACRA2 25%) at **480×240 (240p, 2:1) @ 30 fps and 100 kbps**, against
  SVT-AV1 preset 0. The CPU is the quality reference; speed is not its KPI.
- **GPU mode objective:** use the decisions calibrated at **240p30 @ 100 kbps**
  unchanged, with **zero drift**: the GPU stream must be byte-identical to the
  CPU stream (the parity gate uses `cmp`, not a tolerance), then maximize
  **1440×720** encode throughput. The 720p run is a benchmark
  against other encoders, not a separate quality-tuning point. Production
  hardware is the RTX 3080 Ti; the RTX 5090 is the maximum-speed host.
- **Not targets:** the old 1080p ≥1 fps floor, 8 GB cards, RTX 4070 Super and
  RTX 5070.
- **SIMD:** AVX2 (x86_64) and NEON (arm64) are open optimization targets;
  explicit intrinsics are much cheaper and faster to iterate than new CUDA
  kernels. Keep the scalar reference and byte-identical outputs.
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

For the edit/compile loop there are presets: `cmake --preset dev` (`-O1` + PCH,
~20 s clean, ~13 s after a header edit) and `cmake --preset dev-cache` (`-O1` +
ccache, ~1 s rebuild of an unchanged tree, e.g. per-agent worktrees). Both are
bit-exact with `release`; quote timings and pins only from `release`. See
`docs/build-speed.md` for the measurements and why PCH and ccache are
alternatives rather than a stack.

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

`/workspace` is **not** a persistent volume. The local checkout is the source
of truth; sync it to the box before builds.
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
- **Legacy `dav2d` limits:** user QM OBUs, cross-frame CDF carry, the nested
  decoder model and multi-layer SVC need `avmdec`; 1-D negative H/V_DCT is
  rejected. Quality measurements now use `avmdec`; these gates are being
  migrated. Cross-xlayer SVC prediction is spec-forbidden (§6.8.2).
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
It also records `decode_fps` and `decode_realtime_x` from the decoder. The
cross-encoder **target is SVT-AV1 v4.3.0 preset 0**, measured on the same clip,
decoded with `dav1d` and scored with the same model: 100.0 kbps / 9.828 JOD /
48.88 dB. The AVM row in the baseline table is built clean from upstream
`e6f99d3` (see the note above): the current QP-mode measurement (90.2 K at
240p); the older CBR point and the archived retired-clip 720p sweep
are in `bench/avm_baseline/README.md`. Compare at matched bitrate, and
always confirm the decoder decoded the stream first. A fast-decode
profile is deferred until quality and encode speed are settled.

`tools/score_speed.py` reports speed vs CVVDP. `tools/score_tradeoff.py` enforces
a three-way budget (max CVVDP subject to a bitrate cap and a speed floor).
`tools/oracle.py` scores a candidate's QP ladder at the matched 100 kbps point
and reports segment ceilings; see `docs/oracle-search.md`.

Current gates:

- **CPU quality:** 480x240 (240p, 2:1) @ 30 fps at **100 kbps**, maximize CVVDP.
  NASA baseline: 4.802 JOD at 100.0 kbps, PSNR-Y 24.47 dB (QP 189).
  Stable opt-in preset: `--video-rd-preset quality` measures **6.302 JOD
  (+1.50 JOD), PSNR-Y 27.58 dB (+3.11 dB)** at QP190 / 100.56 kbps; the
  interpolated 100.04 kbps point is 6.273 JOD. The per-tool table and exact
  flags are in `docs/video-rd-preset.md`.
  Algorithmic nightly aggregate: **`--fast-decode` measures 9.733 JOD /
  44.67 dB PSNR-Y at 100.0 kbps (QP106), K = 82.2** — +4.91 JOD over the default
  and +3.41 JOD over the stable preset at essentially the same rate. It combines
  64-band/refit/priced-unit CCSO, bottom-edge inter and residual masking, the
  measured partition probe, a reference-intro key/first-P QP offset, luma
  inter RDOQ, CDEF, the BAWP carrier over inter residual, the calibrated
  leaf RD cost (`--video-leaf-rd`), 1/8-pel ME refinement, the RDOQ-tuned
  TX-search lambda, keyframe intra luma TX partitioning, adaptive coefficient
  CDFs and the drift-triggered inter refresh. It is CPU-only and
  rejects `--video-gpu`, so the default and GPU streams keep the historical CPU
  decisions (`a1d0dba1...`, 25,009 B). The measured tables, per-tool
  contributions and exact flags are in `docs/video-nightly.md`.
- **GPU calibration:** same 240p30 decisions at QP 189, with **zero** drift
  allowed: the gate compares the GPU IVF against the CPU stream with `cmp` and
  requires byte-identical output and decoded planes. The QP189 stream is
  byte-identical on both GPU hosts and the CPU (`a1d0dba1...`).
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
- **Reference decoders**: `avmdec` is the quality oracle; `dav2d` remains in
  legacy recon gates. Both are AOM reference software used only as black-box
  validators. They are not redistributed; `tools/bin/` is ignored by git.
- **Test media**: fixtures under `tests/data/` are generated, and `testdata/`
  is local-only and not required by the build.
