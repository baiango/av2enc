# av2enc — from-scratch GPU AV2 encoder

> **Vibe-coded:** built with **DeepSeek V4.1 Flash 1017** and **MiMo V2.6
> Flash** in **OpenCode**, with AI subagents writing most of the code and a human
> directing the architecture and review. Treat it accordingly.

> **Agents:** read `AGENTS.md`, `docs/cuda-interface.md` and
> `docs/negative-results.md` before coding. Do not repeat listed failed
> approaches.

> **Docs go stale in hours.** The code and tests are authoritative.

## What it is

A spec-only AV2 encoder. The CPU writer is the quality reference; the GPU path
is the product and must produce a **byte-identical** stream (the parity gate uses
`cmp`, not a tolerance). No encoder code from AVM, SVT-AV1/AV2 or clave;
`avmdec` and `dav2d` are black-box validators only.

## Objectives

- **CPU:** maximize the composite **K** at **480×240 @ 30 fps, 100 kbps**.
  The quality wall is **SVT-AV1 v4.3.0 preset 0 (88.8)**; AVM `e6f99d3`
  (**90.2**) is the calibration point.
- **GPU:** use the 240p decisions unchanged, byte-identically, and maximize
  **1440×720** encode throughput.
- **K:** `K = 100 * (0.50·JOD/10 + 0.25/(1+BA/3) + 0.25·(SS2+50)/150)`,
  with CVVDP (JOD), Butteraugli (BA, lower better) and SSIMULACRA2 (SS2).

## Current results (240p30, matched 100 kbps, NASA clip)

| config | kbps | JOD | BA | SS2 | K @100 |
|---|---:|---:|---:|---:|---:|
| `--fast-decode` QP106 (pin) | 100.0 | 9.733 | 3.42 | 80.84 | 82.2 |
| **`--video-rd-preset best`** QP112 (joint plan, opt-in research) | 100.0 | 9.883 | 1.99 | 86.45 | **87.19** |
| **`--nightly-b`** QP111 (chained B + compound + TIP + CWP depth + exact rate + joint plan, key offset −37, opt-in research) | 100.0 | 9.885 | 1.84 | 86.38 | **88.21** |
| AVM `e6f99d3` QP124 | 99.3 | 9.862 | 1.29 | 90.38 | **90.2** |
| SVT-AV1 p0 CRF36.5 | 100.0 | 9.828 | 1.55 | 88.89 | **88.8** |
| rav1e 0.8.1 speed 2 | 100.1 | 9.717 | 2.71 | 81.67 | 83.7 |

`best` now carries the **joint plan solve**: a KEEP at **87.19** (+0.68
bracket / +0.92 fit over the 86.51/86.30 non-joint best), but a **research
gate** — the serial raster walk costs ~12 min per 60-frame encode, so
`--fast-decode` remains the shipped fast path.  `AV2ENC_RD_JOINT_P=0` restores
the 86.33 lead stack (a LEAD, not a KEEP: +0.097 over the flat aggregate,
inside the adjacent-QP ripple).  `--nightly-b` is **88.207** (+2.45 over flat,
+0.286 over the 87.284 exact-rate arm) and also a 12-min research gate;
research-only because its 60-frame stream is `avmdec`-exact while the pinned
`dav2d` has a §7.21.6 drain defect (17/60).  AVERAGE compound, the TIP-output
composition on the dead-end leaves, the CWP NEAR_NEARMV depth pair, the exact
compound-rate probe and the joint solve are on by default in `--nightly-b`
(the depth arm needs the TIP field guard); `AV2ENC_RD_JOINT_P=0` reproduces the
87.284 exact-rate arm, `AV2ENC_COMPOUND_EXACT_RATE=0` the 87.023 depth arm,
`--video-compound-depth-off` the 86.805 TIP arm, `AV2ENC_TIP_LEAF=0` with the
depth off the 86.61 compound arm, and `--video-compound-off` the 85.47
single-prediction arm. The compound NEWMV modes are exact but +0.054 and stay
off.

**720p (1440×720/60, ~293 kbps):** av2enc **45.6**, AVM **55.8**, SVT **56.6**,
rav1e **53.0**. The 720p gap is content-dependent; see
`bench/space8k_720p/README.md`.

## Quick start

```sh
cmake --preset release
cmake --build --preset release -j8

# rate-matched 240p point
build-release/av2enc testdata/nasa_480x240_30fps_60f.y4m -o /tmp/fd.ivf \
  --video --frames 60 --qp 106 --cpu-threads 8 --fast-decode

# best measured flat point (joint plan KEEP, 87.19; ~12 min)
build-release/av2enc testdata/nasa_480x240_30fps_60f.y4m -o /tmp/best.ivf \
  --video --frames 60 --qp 112 --cpu-threads 8 --video-rd-preset best

# chained B research path (88.207; avmdec-validated, ~9 min)
build-release/av2enc testdata/nasa_480x240_30fps_60f.y4m -o /tmp/b.ivf \
  --video --frames 60 --qp 119 --cpu-threads 8 --video-gops 7 --nightly-b
```

### Presets

| flag | meaning |
|---|---|
| *(default)* | historical key-only/basic decisions, byte-pinned |
| `--video-rd-preset quality` | measured still/keyframe preset |
| **`--fast-decode`** | measured CPU RD aggregate (was `--nightly`; the old name is an alias) |
| **`--video-rd-preset best`** | flat best stack (**87.19** joint plan KEEP, ~12 min): EFF_PERC, head/tail TX, driver, key −39, drift 1.5/3250 + joint solve; `AV2ENC_RD_JOINT_P=0` → 86.33 LEAD |
| **`--nightly-b`** | chained single-key B + compound + TIP + CWP depth + exact rate + joint plan, opt-in research (**88.207**, ~9 min; `AV2ENC_RD_JOINT_P=0` → 87.284, `AV2ENC_COMPOUND_EXACT_RATE=0` → 87.023, `--video-compound-depth-off` → 86.805, `AV2ENC_TIP_LEAF=0` with depth off → 86.61, `--video-compound-off` → 85.47) |

### Pins

| stream | bytes | MD5 |
|---|---:|---|
| default QP189 | 25,009 | `a1d0dba1259b20a4b628d4cd1e14c33c` |
| quality QP190 | 25,141 | `3984677b9b665952984b07b7cf75c6ea` |
| fast-decode QP106 | 25,008 | `0639b43630abb454f7be5ac494d2efb4` |
| fast-decode QP110 | 23,690 | `1b99127a4cd1d386fa5600fb63c57a83` |
| 720p QP217 (key −57, leaf 250, ext partitions) | 36,572 | `c4ba28138d8f6569d5e08721a66d755c` |
| `--video-rd-preset best` joint QP112 | 25,007 | `9a36aeb7c02733342e4ec5db300e4d6d` |
| `--nightly-b` 16 f joint (QP116, gops 2) | 19,723 | `505430de620af361a18b42503543dbda` |
| `--nightly-b` 60 f joint (QP118, gops 7) | 25,093 | `34103c1f26ca7225ebaf74f67e771175` |
| `--nightly-b` 60 f joint (QP119, gops 7) | 24,845 | `40de8b687d929652a312407764227398` |
| joint-off control: `best` QP116 | 26,334 | `fba63446670b43888c3ac7b280ad44dc` |
| joint-off control: `--nightly-b` 16 f exact (QP116) | 19,774 | `51b664515a897a3b19be4131790867fa` |
| joint-off control: `--nightly-b` 60 f exact (QP121) | 25,163 | `dfa6b1aadd565a2adf47c7e186c1a973` |
| basic 64×64 / 640×360 | — | `c9294fa4…` / `0d90c958…` |
| still 64×64 / 640×360 | — | `2b31e378…` / `8ffbe059…` |

```sh
tools/validate_dav2d.sh /tmp/fd.ivf testdata/nasa_480x240_30fps_60f.y4m
python3 tools/oracle.py --ref testdata/nasa_480x240_30fps_60f.y4m \
  --ivf fd@106=/tmp/fd.ivf --target-kbps 100
```

## Capabilities

- **CPU:** the AV2 tool surface the pinned decoders validate — partition/TX
  trees, intra modes, CfL/MHCCP, palette/IBC, 1–7 references, B/temporal layers,
  compound, BAWP, global motion/warp, TIP, loop filters (deblock, CDEF, CCSO,
  Wiener/PC-Wiener, chroma LR, GDF), RDOQ/TCQ/CCTX, QM, SDP, SB128/256,
  multi-tile, lossless, 10-bit/chroma. Full matrix:
  `docs/cpu-feature-completion.md`.
- **GPU:** decision-identical M2 mirror of the 8-bit 4:2:0 CPU path. CPU-only
  tools are rejected by name. Matrix: `docs/cuda-gap-analysis.md`.
- **Decoder pins:** `dav2d` f9dafbc is the shipped conformance gate; `avmdec`
  is the quality oracle. The chained B path is `avmdec`-only until the drain
  defect is fixed.
- **Not signalable:** 12-bit. **Not supported:** Windows, macOS CUDA, 8 GB
  cards, RTX 5070. The RTX 4070 Super (12 GB, `sm_86`) is a supported CUDA
  target again: the encoder is compute-bound rather than memory-bound, so the
  narrower 192-bit bus costs less than the bandwidth figure implies.

## Performance

- `--cpu-threads 8` is **2.2×** the 1-thread 240p encode and **2.18×** at 720p;
  the stream is byte-identical at every N.
- `AV2ENC_CPU_FAST=ON` (native + LTO) is +24–33 %; `AV2ENC_LTO` alone is most
  of it. Default builds stay portable and byte-identical.
- SIMD (AVX2/NEON) is open for the scalar reference; keep a scalar fallback and
  byte-identical outputs.

## Targets

- CPU RD: Apple M4 with `--cpu-threads` and `AV2ENC_CPU_FAST=ON`; Modal for
  parallel sweeps. GPU hosts are for CUDA validation and the 720p benchmark.
- GPU: RTX 3060 12 GB (feature baseline), RTX 3080 Ti (production), RTX 5090
  (maximum speed), RTX 4070 Super (supported secondary `sm_86` target; the
  workload is compute-bound, so its smaller bus is not disqualifying). Linux +
  CUDA 12.8 for the GPU product; macOS arm64 for CPU.
- **Metal (speculative, not committed):** no Metal backend exists today and
  macOS is not an encoder build target. It may be picked up if the CUDA
  structure turns out to port more easily than expected, or if it pays for
  itself by speeding up development — fast local GPU iteration on the M4
  rather than renting a CUDA host. Any such backend would still have to match
  the CPU reference bit-for-bit.
- VRAM arena: `min(total, free) − 512 MiB`, optionally `--vram-cap <MiB>`.
  No fixed preset.

## Rules

1. Spec-only; cite section numbers. Mark unknowns instead of inventing syntax.
2. Generate tables with scripts; never hand-type CDF/transform tables.
3. Scalar CPU reference before every CUDA kernel; assert bit-exactness.
4. Do not GPU the entropy coder (serial per tile).
5. Pin the spec version and decoder commit.

Read `AGENTS.md` for ownership and the shared-root rules; worktrees and branches
live outside the main checkout. `docs/negative-results.md` is the list of
measured dead ends.

## Quality and gates

- `tools/quality.py` — CVVDP/JOD, PSNR, SSIM (`avmdec` via `AVMDEC`).
- `tools/extra_metrics.py` — Butteraugli and SSIMULACRA2 per frame.
- `tools/kpi.py` — the composite K.
- `tools/oracle.py` — matched-rate K and segment ceilings.
- `ctest` 14/14 and `gate_video_nightly.sh`, `gate_video_tlayers_chain.sh`,
  `gate_video_schedule_rc.sh`, `gate_video_tlayer_refs.sh` pass on the release
  build.

## Build provenance

Token usage recorded by OpenCode for the work on this checkout — all top-level
sessions and their subagents — as of **2026-11-24**:

| Counter | Tokens |
|---|---:|
| Fresh input | 351,990,144 |
| Output | 49,825,141 |
| Reasoning | 59,749,303 |
| **Input + output + reasoning** | **~461.6 M** |
| Cached context reads | 33,525,690,715 |
| **Including cache reads** | **~33.53 B** |

1010 sessions (375 top-level + 635 subagents) across the four OpenCode project
ids that touch the checkout (the parent workspace, the shared work root that
owns the lane worktrees, the repo root and the publish clone). Cache reads are
billed at a fraction of fresh input and dominate because every subagent re-reads
a long context. Measured from `session_v2` in the OpenCode state database;
figures keep growing while work continues.

## License

The encoder source is **The Unlicense** (public domain). See `LICENSE`.

Third-party material keeps its own terms: AV2 specification-derived tables
under `spec/tables/` and `src/encoder/filters/spec_tables/` (AOM; see
<https://www.aomedia.org/license/>); the specification PDF is not redistributed;
`avmdec`/`dav2d` are AOM reference software used only as black-box validators
and are not redistributed (`tools/bin/` is ignored); `testdata/` is local-only
and the fixtures under `tests/data/` are generated.
