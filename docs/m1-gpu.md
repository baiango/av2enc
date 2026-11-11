# M1 GPU path — contract and workstreams

Goal: move the M1 still-picture encoder from the CPU path in
`src/encoder/m1/m1_still.cpp` to a GPU-resident pipeline, while keeping the
bitstream **byte-identical** to the CPU M1 reference for the same options.

M1 scope is unchanged: 8-bit 4:2:0, one tile, one keyframe, QP 90, partition
tree + intra + variable TX + CfL + deblock. Entropy stays on the CPU until the
GPU coder is proven at these tile counts.

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

## Shared PODs (`src/cuda/m1/m1_gpu.h`, owner: orchestrator)

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

`bool encode_m1_gpu(const StillPicture&, const GpuOptions&, GpuResult&,
std::string&)` is the public entry point. It must produce the same frame bytes
as `m1::encode_m1_still_picture` when the CPU path is in the same mode.

## Ownership (do not edit outside your directory)

| # | Area | Directory | Deliverable |
|---:|---|---|---|
| 1 | Contract + orchestrator | `src/cuda/m1/` root files | `m1_gpu.h`, `m1_gpu.cu`, stage wiring |
| 2 | Frame/block plan | `src/cuda/m1/plan/` | tree + leaf geometry + syntax order |
| 3 | Luma candidate search | `src/cuda/m1/search/` | batched 61-candidate SSE, parity with CPU |
| 4 | Transform/quant | `src/cuda/m1/tx/` | batched fwd/inv TX, levels, eob |
| 5 | Reconstruction/deblock | `src/cuda/m1/recon/` | recon writes, availability, deblock |
| 6 | CfL | `src/cuda/m1/cfl/` | GPU CfL + derived alpha, CPU parity |
| 7 | Entropy bridge | `src/cuda/m1/entropy/`, `src/encoder/m1/stream/` | result stream -> CPU syntax |
| 8 | Scheduler adapter | `src/cuda/m1/sched/` | dependency levels, graphs |
| 9 | CLI + build | `src/main.cpp`, `CMakeLists.txt` | `--m1-gpu` (default off), link |
| 10 | Verification/bench | `tools/m1_gpu/` | CPU-vs-GPU bytes, `dav2d`, fps |

Only agent 9 edits `src/main.cpp` or `CMakeLists.txt`. Only agent 1 edits
`src/cuda/m1/` root files. Everyone else stays in their subdirectory.

## Gates

- Per component: CPU-vs-GPU bit-exact tests on synthetic blocks, then real
  64x64 and 640x360 frames.
- Stage 1: `--m1-gpu` frame bytes == CPU `--m1` frame bytes for the same input;
  `dav2d` decodes; recon planes match `M1Result.recon_*` byte-for-byte.
- Stage 2: same bitstream after replacing CPU mode search with GPU search.
- Performance: report 1080p still wall/kernel time and VRAM; do not claim the
  1 fps video target until video is wired and measured on an 8192 MiB 3060.
- No new VRAM preset: reuse the device-sized arena and `--vram-cap`.

## Non-negotiables

- Spec-only; no AVM/SVT/clave encoder code.
- `sm_86` baseline; no Blackwell-only instructions.
- No `cudaMalloc` in hot paths; use the arena; one sync per frame at most.
- Do not edit another workstream's files. Do not commit; report and leave changes.
