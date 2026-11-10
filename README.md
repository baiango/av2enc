# av2enc — from-scratch GPU AV2 encoder

> **Vibe-coded:** This project is built with **DeepSeek V4.1 Flash** in
> **OpenCode**, with AI subagents writing most of the code and a human directing
> the architecture and review. Treat it accordingly.

## Hard constraint

No encoder code from AVM, SVT-AV1/AV2, clave, or any other encoder.
Implement from the **AV2 v1.0.0 specification**.

`avmdec` / `dav2d` may only be used as **black-box validators** (decode our
output, compare pixels). Do not read or copy their encoder-side code.

## Target

- Primary target: **8 GB Ampere (`sm_86`)** — RTX 3060 8 GB, RTX 3050 8 GB,
  RTX 3070 8 GB. Default profile `compat8g` with a 7,500 MB VRAM cap.
- Development environment (fixed): **RTX 3060 12 GB running `compat8g`**, i.e.
  8 GB emulation. No other GPU rentals or profiles are current work.
- RTX 5070 (`sm_120`) is a build/smoke target only. No Blackwell-only code.
- Product speed target: **≥1 encoded fps at 1080p**. Surplus compute goes into
  quality (RDOQ, partitions, modes, search, references), never into more fps.
- Build: **Linux + CUDA 12.8 + C++20 + CMake/Ninja**. CPU is a reference/test
  path only, not the product.
- Fat binary: `sm_86;sm_120`. No Blackwell-only accelerators (TMA, clusters,
  tensor-core/FP4 paths) are required.
- VRAM: hard default ceiling **7,500 MB**, so the 8 GB RTX 3060 variant is
  supported. The 12 GB box is tested with the same cap.
- Extra VRAM scales **precompute** (`compat8g` default; `compat4g`, `compat6g`,
  `auto12g`, `auto16g`, `auto24g` opt-in), never compression-tool shortcuts.
- Encoder architecture: GPU-resident. CPU only launches work, handles OBU
  headers and file I/O; pixel work, mode decision, filters, and tile entropy
  stay on the GPU.

## Remote box

```
ssh -p <port> <user>@<host>
/workspace/av2enc
```

`/workspace` is **not** a persistent volume. The local repo at
`$HOME/work/av2enc` is the source of truth; sync to the box before builds.
Disk is 16GB: keep test media tiny (<100MB total), prefer 64x64/256x256 frames.

## Platforms

- **Linux x86_64 + CUDA** — the only supported encoder target (Vast.ai RTX 3060,
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
The AVM reference baseline is in `bench/avm_baseline/README.md`. Compare at
matched bitrate, and always confirm `dav2d` decoded the stream first.

`tools/score_speed.py` reports speed vs CVVDP. `tools/score_tradeoff.py` enforces
the three-way budget: max CVVDP subject to bitrate <= R_max and speed >= S_min.
The planned 720p60 gate is CVVDP >= 9.0 at <= 500 kbps, with speed measured
against AVM on the same machine.

## Profiling

The encoder has a built-in profiler that works without GPU counter permissions:

```sh
AV2ENC_PROFILE=/tmp/run.jsonl AV2ENC_VRAM_LIMIT_MB=7500 ./build-cuda/av2enc ...
```

It logs per-frame CPU and CUDA stage times, JSONL plus a summary, and VRAM
peak/capacity. `AV2ENC_PROFILE_NVTX=1` adds NVTX ranges for Nsight Systems.
See `docs/profiling.md`.
