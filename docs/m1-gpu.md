# M1 GPU path — contract and workstreams

Goal: move the M1 still-picture encoder from the CPU path in
`src/pipeline/still/still_encoder.cpp` to a GPU-resident pipeline, while keeping the
bitstream **byte-identical** to the CPU M1 reference for the same options.

M1 GPU scope is unchanged: 8-bit 4:2:0, one tile, one keyframe, QP 90 (the
CPU M1 path supports QP 1..255 with the §8.3.1 qctx slice; the GPU TX batch
still bakes qlookup(90)), partition
tree + intra + variable TX + CfL + deblock. Entropy stays on the CPU until the
GPU coder is proven at these tile counts.

## Policy: correctness before performance

Do not optimize the GPU path until the integrated correctness matrix is green
(byte identity vs CPU M1, decoded-plane identity, memory safety, determinism,
all supported options, edge/odd sizes). The unoptimized pipeline is workable:

| Still (parity check on) | Wall |
|---|---:|
| 64x64 | 0.38 s |
| 640x360 | 0.49 s |
| 1080p | 1.36 s |

It is currently slower than the CPU M1 path, which is expected: per-block
ordered launches plus an in-process CPU parity encode. That is acceptable for
testing, so scheduler/wavefront optimization is deferred until correctness is
signed off.

## Correctness matrix (verified 2026-11-11)

`tools/still_gpu/verify_gpu.sh --matrix` → **112/112 PASS** (14 modes x 8 sizes):

- sizes: 64x64, 136x88, 640x360, 1280x720, native 1920x1080, 416x208, 518x322, 532x368
- modes: default, full-intra, no-cfl, no-deblock, depth 0/1/2/3/4/8, single, dc, adaptive-cdf, vram-cap
- each PASS: GPU frame bytes identical to CPU M1, `dav2d --strict 1` decode, Y/U/V 0 differing samples, Nsight-attested kernels
- determinism: repeat encodes and two in-process encodes identical
- memcheck/racecheck: 0 errors/hazards

This pass required fixing the partition allowed-mask test, boundary-forced partitions,
the min-block floor, and the mandatory `df_delta_q_present` header bits. M0 has a
separate legacy edge-tree defect for some partial sizes; M1 is the correct path.

## Staged plan

1. **Stage 1 — GPU block coding.** Host decides the tree and modes (existing
   CPU logic); GPU performs prediction, transform/quant, inverse transform,
   reconstruction, and deblock; CPU writes the same syntax. Bitstream must be
   byte-identical to CPU M1. This proves the kernels and data marshalling.
2. **Stage 2 — GPU candidate search.** Batch the 61 luma candidates (and CfL
   alphas) per block on the GPU, with the same SSE metric and deterministic
   tie-breaking as `choose_luma_mode`. Bitstream still identical.
3. **Stage 3 — wavefront/CUDA graphs.** Batch independent blocks per dependency
   level and replay with the CUDA graph scheduler. Target: 1080p still in the
   same order of magnitude as the M0 scheduler, then video.

Stages 1 and 2 are the deliverable of this wave; stage 3 is wired when the
scheduler adapter lands.

## Data flow

```
Y4M frame (host, 8-bit 4:2:0)
  -> FramePlan (host): SB tree, leaf blocks, geometry, syntax order
  -> per block: source + reconstructed neighbours
       -> luma search (GPU, stage 2)
       -> residual + forward DCT + quantize (GPU)
       -> inverse quant + inverse DCT + add prediction (GPU)
       -> recon planes (GPU)
  -> chroma: DC/CfL prediction, same TX path
  -> deblock (GPU)
  -> download: recon planes, per-block mode info, coefficient levels
  -> CPU entropy writer (existing §8.2 syntax)
  -> IVF/OBU frame
```

## Shared PODs (`src/cuda/pipeline/still/still_gpu.h` (namespace `av2enc::cuda::still_gpu`), owner: orchestrator)

All stages exchange plain PODs; device pointers are explicit. No stage allocates
or synchronizes; the orchestrator owns the arena and stream.

- `FramePlan`: width/height, padded strides, `mi_cols/mi_rows`, block list in
  syntax order, per-block chroma residual size.
- `BlockPlan`: `mi_row`, `mi_col`, luma w/h, chroma w/h, TX w/h, chroma TX w/h,
  coefficient offsets, `has_chroma`, neighbour availability bits.
- `BlockDecision`: luma mode + angle delta, UV mode / is_cfl + alpha, TX type,
  eob per plane, SSE.
- `GpuFrame`: host input planes, host output recon/coeff/pred buffers, block
  array, counts.
- `GpuOptions`: `base_q_idx`, `enable_cfl`, `enable_deblock`,
  `enable_edge_filter`, `luma_dc_only`, `min_block_size`, `max_partition_depth`.

`bool encode_still_gpu(const StillPicture&, const GpuOptions&, GpuResult&,
std::string&)` is the public entry point. It must produce the same frame bytes
as `av2enc::still::encode_still_picture` when the CPU path is in the same mode.

## Ownership (do not edit outside your directory)

| # | Area | Directory | Deliverable |
|---:|---|---|---|
| 1 | Contract + orchestrator | `src/cuda/pipeline/still/` root files | `still_gpu.h`, `still_gpu.cu`, stage wiring |
| 2 | Frame/block plan | `src/cuda/pipeline/still/plan/` | tree + leaf geometry + syntax order |
| 3 | Luma candidate search | `src/cuda/pipeline/still/search/` | batched 61-candidate SSE, parity with CPU |
| 4 | Transform/quant | `src/cuda/pipeline/still/tx/` | batched fwd/inv TX, levels, eob |
| 5 | Reconstruction/deblock | `src/cuda/pipeline/still/recon/` | recon writes, availability, deblock |
| 6 | CfL | `src/cuda/pipeline/still/cfl/` | GPU CfL + derived alpha, CPU parity |
| 7 | Entropy bridge | `src/pipeline/still/stream/` | result stream -> CPU syntax |
| 8 | Scheduler adapter | `src/cuda/pipeline/still/sched/` | dependency levels, graphs |
| 9 | CLI + build | `src/main.cpp`, `CMakeLists.txt` | `--still-gpu` (default off), link |
| 10 | Verification/bench | `tools/still_gpu/` | CPU-vs-GPU bytes, `dav2d`, fps |

Only agent 9 edits `src/main.cpp` or `CMakeLists.txt`. Only agent 1 edits
`src/cuda/pipeline/still/` root files. Everyone else stays in their subdirectory.

## Gates

- Per component: CPU-vs-GPU bit-exact tests on synthetic blocks, then real
  64x64 and 640x360 frames.
- Stage 1: `--still-gpu` frame bytes == CPU `--still` frame bytes for the same input;
  `dav2d` decodes; recon planes match `StillResult.recon_*` byte-for-byte.
- Stage 2: same bitstream after replacing CPU mode search with GPU search.
- Performance: report 1080p still wall/kernel time and VRAM; do not claim the
  1 fps video target until video is wired and measured on an 8192 MiB 3060.
- No new VRAM preset: reuse the device-sized arena and `--vram-cap`.

## Non-negotiables

- Spec-only; no AVM/SVT/clave encoder code.
- `sm_86` baseline; no Blackwell-only instructions.
- No `cudaMalloc` in hot paths; use the arena; one sync per frame at most.
- Do not edit another workstream's files. Do not commit; report and leave changes.
