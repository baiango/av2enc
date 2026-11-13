# CUDA backend for the format video core

`video_hbd_cuda.cu` implements `av2enc::video::hbd::Backend` on the device for
the 10-bit / chroma `--video-gpu` path.  It is selected automatically when
`VideoOptions::format` is not 8-bit 4:2:0; the historical 8-bit pipeline in
`src/cuda/pipeline/video/video_backend_cuda.cu` is untouched.

## Design

The orchestration (`src/pipeline/video/hbd/video_hbd.cpp`) is shared with the
CPU reference: partition tree, MV stack, candidate comparison, syntax and the
range coder all run on the host.  The device owns the pixels:

| Stage | Kernel / module |
|---|---|
| source upload, reference upload, recon storage | `cudaMemcpy2DAsync` (arena buffers) |
| DC intra prediction (§7.13.2.10) | `dc_predict_kernel`, one thread per block |
| translational MC (§7.13.3.18) | `mc_kernel` calling `hbd_mc_sample` |
| MC SSE | `sse_kernel` |
| motion estimation | `me_kernel` calling `hbd_me_search` |
| forward quantise / dequantise / inverse | `cuda::hbd::hbd_tx_batch` (the verified 10-bit M1 stage) |
| reconstruction | host `clip1(pred + residual)` (identical to the CPU backend) |

`video_hbd_ops.h` is the shared host/device arithmetic, so the two backends are
bit-identical by construction (including the subpel filter downgrade, the
edge-clamped DC predictor and the 1/8-pel ME output).

## Allocation

One `Arena` per backend.  Every device buffer is bump-allocated at the first
frame and reused; the backend never calls `cudaMalloc`.  When no external arena
is supplied it sizes one from `min(total, free) - 512 MiB`, honouring
`Arena::capacity_cap_bytes()` (`--vram-cap`).  `tools/check_arena.sh` passes.

## Gate

`src/cuda/pipeline/video/gate_video_formats.sh` (registered as CTest
`gate_video_formats`) runs the CPU reference and this backend on the same 360p
source for all seven formats, requires identical units and reconstruction, and
requires `dav2d --strict 1` to output the expected four pictures with output
identical to the encoder reconstruction.  See
`src/pipeline/video/hbd/README.md` for the pinned MD5 table.
