# Agent instructions

> **Docs go stale in hours.** This repo moves fast; when a document disagrees
> with the code or tests, the **code and tests win**. Read the implementation
> before trusting a status line or table.

Read these before writing code:

1. `README.md`
2. `docs/architecture.md`
3. `docs/cuda-interface.md`
4. `docs/negative-results.md` — **do not repeat listed failed approaches**

Bug fixing: read `refactor/bugfix-protocol.md` first. Claim a bug with
`tools/agents/claim.sh`, use `tools/agents/new_fix.sh` for the worktree, and
leave the shared roots (`src/main.cpp`, `CMakeLists.txt`, `still_*`, `video_*`,
`baseline.json`) to an integrator.

## Non-negotiables

- Encoder target: Linux + CUDA, 8 GB Ampere `sm_86` primary. macOS is not an
  encoder build target. Windows is unsupported.
- VRAM arena is sized from the device: `min(total, free) - 512 MiB`, optionally
  bounded by `--vram-cap <MiB>`. There is no fixed 7,500 MB preset. Do not tune
  for one specific larger card.
- No encoder code from AVM, SVT-AV1/AV2, clave, or av2-turbo. Use the AV2 v1.0.0
  spec. `dav2d`/`avmdec` are black-box validators only.
- Every stream must pass `tools/validate_dav2d.sh` before it is called working.
- No `cudaMalloc` in hot paths; use `src/cuda/arena.*`.
- No Blackwell-only instructions (TMA, clusters, `wgmma`/`tcgen05`, FP4).
- CPU-only mode is a correctness oracle, not a performance target. Do not
  benchmark or optimize it for speed; the product is GPU-resident on Vast.ai.
- Do not edit another agent's files. Check `docs/cuda-interface.md` ownership.
- Commit source/docs, never build dirs, test media, or `tools/bin`.

## Current reference bitstreams

- 64x64: MD5 `7efc3162bc0a4785685eba291a8096fd`
- 640x360: MD5 `56f11d4a63ff07f2356ef5064b84fe0e`

The CPU path is the reference. The CUDA path must match it or explain exactly
why it cannot.
