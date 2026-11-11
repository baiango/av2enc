# CUDA M2 P-frame baseline

GPU path for the M2 P-frame encoder. It is a decision-identical mirror of
`src/encoder/m2/`: same partition tree, same MV-stack derivation, same
candidate ordering/tie-breaking, same transform and same §8.2 range coder. The
output is **byte-identical** to the CPU M2 encoder for the same input and
options, and `dav2d` decodes both to the same reconstruction.

The CPU directory is not modified and remains the reference. This tree owns
only `src/cuda/m2/`.

## What runs where

| Stage | Owner |
|---|---|
| Partition block-variance cost | `m2_variance_kernel` (this tree) |
| Integer + half/quarter subpel ME | `src/cuda/inter/me` (`me_search_batch`) |
| Single-ref §7.13.3 MC | `src/cuda/inter/mc` (`mc_predict_batch`) |
| Luma SSE against source | `m2_sse_kernel` (this tree) |
| DC intra prediction + residual | `src/cuda/intra_modes` (`intra_predict_batch`) |
| Residual downsample / forward / quant | `src/cuda/tx_variants` (`txv_*_multi`) |
| Inverse transform | `src/cuda/tx_variants` (`txv_inverse_transform_multi`) |
| Reconstruction `Clip1(pred + residual)` | `m2_recon_kernel` (this tree) |
| Reconstruction-plane writes / reference upload | `m2_backend_cuda.cu` |
| Partition recursion, MV stack, candidate comparison, EOB scan | host (CPU decision helpers) |
| OBU/IVF assembly and §8.2 entropy | host (`src/encoder/m2/`) |

The split is deliberate. Pixel work, ME/MC, transform and reconstruction are
device-resident; the decisions are the CPU decisions by construction, which is
what "byte-identical for the same decisions" needs. Moving the stack/decision
search to the device is a follow-up that must re-run this gate.

## Files

```text
src/cuda/m2/
  m2_gpu.h              public integration API + stats
  m2_pipeline.cpp       frame loop, partition/MV/mode decisions, OBU assembly
  m2_backend.h          backend contract (pixel/transform work only)
  m2_backend_cuda.cu    CUDA backend + three M2-owned kernels
  m2_backend_cpu.cpp    host-emulation backend (no GPU/nvcc needed)
  m2_device.cuh         shared device arithmetic (variance, SSE, residual, recon)
  m2_selftest.cpp       inter residual TX/quant/inverse parity vs CPU
  m2_gate.cpp           standalone CPU-vs-GPU gate + dav2d decode
  build_and_test.sh     emulation gate (this machine, no GPU)
  build_gpu_gate.sh     real CUDA gate (nvcc + device)
```

`m2_device.cuh` is compiled by both backends, so the M2-owned arithmetic cannot
drift. The existing CUDA components have their own CPU-vs-GPU bit-exactness
tests; this gate verifies the orchestration and the byte stream.

`src/cuda/rdoq` is intentionally not on this path: the M2 baseline quantises
with `txv_forward_transform` (the CPU M2 reference does the same), so adding
RDOQ would change the coefficients and break byte parity. It stays available
for a later rate-aware M2 that also changes the CPU reference.

## Integration API

`m2_gpu.h` is a drop-in replacement for the CPU entry point:

```cpp
#include "cuda/m2/m2_gpu.h"

av2enc::m2::M2SequenceResult result;
av2enc::cuda::m2::M2GpuStats stats;
std::string error;
const bool ok = av2enc::cuda::m2::encode_m2_sequence_gpu(
    frames,          // std::span<const av2enc::m2::M2InputFrame>
    options,         // av2enc::m2::M2Options, identical semantics
    result,          // av2enc::m2::M2SequenceResult, identical contents
    error,           // std::string&
    &stats);         // optional, may be nullptr
```

Everything else (input validation, `base_q_idx = 90`, frame headers, temporal
units) matches `av2enc::m2::encode_m2_sequence()`. `m2_backend_name()` returns
`"cuda"` or `"cpu-emulation"` for honest labeling.

`M2GpuStats` reports kernel launches by stage (`variance`, `me`, `mc`, `sse`,
`intra`, `tx`, `residual`, `recon`), host/device transfer bytes, sync count,
ME job and inter-candidate counts, and the per-frame device interval. The CUDA
backend measures the interval with one `cudaEvent` pair per frame; the
emulation backend reports the wall time of the emulated backend calls.

### Device memory

The standalone backend reserves scratch with `cudaMalloc` once per frame size at
`begin_frame` and reuses it. A product integration should carve the same
buffers from `src/cuda/arena.*`; there is no `cudaMalloc` in the block loop.
There is exactly one frame-boundary sync in `end_frame` plus the unavoidable
syncs when a decision needs a device value back on the host (see below).

## Decision parity

The pipeline in `m2_pipeline.cpp` is a line-by-line mirror of
`src/encoder/m2/m2_frame.cpp`:

- partition costs are discovered with the CPU traversal, computed by the device
  kernel, then fed back into `partition::decide_frame()` so the tree is the CPU
  tree;
- `inter::mc::build_mv_stack` / `resolve_mode` derive NEARMV/NEWMV exactly as
  the baseline (including the conservative NEARMV DRL cap for blocks <= 32
  described in the CPU README);
- ME is batched per frame but each job is independent, so results match the
  CPU per-leaf calls;
- GLOBALMV/NEARMV/NEWMV candidates are all scored in one MC+SSE batch and
  considered in the CPU order (strict `<`, so earlier candidates win ties);
- the intra fallback is evaluated for every leaf, exactly like the CPU;
- EOB scan, level ordering and `BlockSyntax` fields are produced on the host
  with the CPU syntax writer.

Optional CPU M2 tools are mirrored when the linked `M2Options` provides them:

- `enable_inter_residual` / `force_inter_residual` — inter leaves can code
  Y/U/V residual. The decision copies the CPU heuristic (SSE + `lambda=0.5*66²`
  × `estimate_tx_bits`); the MC → residual → downsample → forward/quant →
  inverse → recon chain runs on the device.
- `drl_reorder` — passed to `MvStackParams::reorder` and to the M2 config.

`m2_inter_residual_selftest()` runs that chain on the linked backend and
compares every level and reconstructed sample of all three planes against the
CPU transform reference. It is wired into both build scripts.

FP caveat: partition variance is the only floating-point decision input. The
CUDA build uses `--fmad=false` and explicit `__dadd_rn`/`__dmul_rn`; the
expression and loop order match the CPU. A 1-ULP difference would only matter
if a variance landed within ~1e-8 of an integer before the `int64` cast, which
the gate would catch as a partition/bitstream difference.

## Build and run

Emulation (no GPU, no nvcc — CPU components stand in for the device):

```sh
./src/cuda/m2/build_and_test.sh
./src/cuda/m2/build_and_test.sh --frames 8 --full-tree
./src/cuda/m2/build_and_test.sh --inter-residual
./src/cuda/m2/build_and_test.sh --force-inter-residual
./src/cuda/m2/build_and_test.sh --drl-reorder 2
./src/cuda/m2/build_and_test.sh --no-hd
```

CUDA (nvcc + device; default `sm_86`):

```sh
./src/cuda/m2/build_gpu_gate.sh
ARCH=89 ./src/cuda/m2/build_gpu_gate.sh
./src/cuda/m2/build_gpu_gate.sh --frames 8 --full-tree
```

Both scripts build `m2_gate.cpp` and run, for each phase:

1. CPU M2 encode and GPU M2 encode;
2. per-unit byte comparison (`units`, `frame_bytes`, `tile_bytes`,
   `inter_blocks`, `intra_blocks`);
3. per-sample Y/U/V reconstruction comparison;
4. `dav2d` decode of the GPU stream and a decoded-vs-recon comparison;
5. the launch/transfer/timing report.

Phases: 2-frame and 8-frame 64x64 synthetic, and 2-frame and 8-frame 720p from
`testdata/shine_1280x720_60f.y4m`. 720p defaults to `--single-block` (mirrors
the CPU gate); `--full-tree` runs the variance-driven tree. `DAV2D=` overrides
decoder discovery.

If the working tree's `src/encoder/m2/` is mid-refactor, point
`M2_CPU_REF=/path/to/checkout` (containing `src/encoder/m2/`) at a clean
reference; the gate builds against that snapshot without touching the tree.

## Gate results

Measured on the development host (Apple M4, **CPU-emulation backend**, macOS).
These are proxy numbers: they validate byte/recon parity and report the pipeline
shape, not GPU speed. The CUDA build was not runnable here (no CUDA device); the
dependency, `dav2d` and parity checks are the same, and the CUDA backend reuses
components already gated on the RTX 3060.

Default (720p single-block, 8 frames):

```text
== 64x64 short clip: 8 frame(s) ==
  GPU backend=cpu-emulation launches=85 [var=8 me=7 mc=12 sse=11 intra=12 tx=24 recon=11]
  PASS: bitstream byte-identical, recon 0 diffs
  PASS: dav2d decoded 8 frame(s), recon exact
== 720p short clip: 8 frame(s) ==
  GPU backend=cpu-emulation launches=11055 [var=8 me=7 mc=3351 sse=1680
      intra=1920 tx=3840 recon=249] me_jobs=1680 cand=7614 device≈2.19 s
  PASS: bitstream byte-identical, recon 0 diffs
  PASS: dav2d decoded 8 frame(s), recon exact
M2 GPU GATE PASS
```

Per-frame launch shape (720p single-block, 8 frames): ≈1,382 launches/frame.
Full-tree 720p 8 frames: 28,359 launches total ≈3,545/frame. Most launches are
per-leaf intra candidates (the CPU evaluates the DC fallback on every leaf), MC
candidate batches and one MC+download for the chosen inter block. The
host-in-the-loop design synchronises at each decision download; batching the
decision search on the device is the obvious next step and does not change the
bitstream. 64x64 8 frames: 85 launches, ≈13 ms proxy.

Feature gates (all `PASS`, byte-identical, recon 0 diffs, `dav2d` exact):

| Run | Result |
|---|---|
| `--frames 8 --full-tree` | 720p 8f, 28,359 launches |
| `--force-inter-residual --frames 2` | 720p frame 1: 2,230 → 12,570 bytes, still exact |
| `--inter-residual --frames 2` | heuristic keeps skip on this content, still exact |
| `--drl-reorder 1/2 --frames 2` | exact |
| `--force-global`, `--no-newmv`, `--force-intra`, `--no-intra-fallback` | exact |
| `--selftest 128` | inter residual levels + recon parity |

## Scope / limitations

- Baseline decision tools only: GLOBALMV, NEARMV, NEWMV, DC-intra fallback.
  B frames, multiple references, compound, skip-mode, warp/OBMC/inter-intra,
  BAWP, TIP, OPFL, refine-MV, temporal MVs, ref-MV bank, loop filters are not
  implemented here.
- Inter leaves are `skip_flag = 1` unless the optional inter-residual tool is
  enabled; that matches the CPU reference.
- The decision search (partition/stack/candidate compare/EOB) stays on the
  host; only pixels/transforms/reconstruction are device-resident.
- Every decision that needs a score synchronises the stream. This is a
  correctness-first milestone, not a throughput claim.
- No CUDA device was available on the development host, so the CUDA numbers are
  not reported. Run `build_gpu_gate.sh` on the `sm_86` box to fill them in; the
  emulation gate is the byte/recon reference.
