# Agent waves: M1 → M4

Sixteen workstreams, run as two waves of eight. This keeps file ownership
disjoint and prevents eight agents from editing the same bitstream path.

## Rules for every agent

- Read `AGENTS.md`, `docs/architecture.md`, `docs/cuda-interface.md`,
  `docs/negative-results.md`, and `docs/tool-roadmap.md`.
- Spec-only. No AVM/SVT/clave/av2-turbo encoder code.
- **Component only:** no root `CMakeLists.txt`, `src/main.cpp`,
  `src/encoder/basic_encoder.cpp`, `src/entropy/av2/`, or `src/bitstream/` edits.
- CPU reference first, then CUDA. Every CUDA kernel gets a CPU-vs-GPU
  bit-exact test.
- Standalone build/test script; no claiming end-to-end integration.
- Every final stream must pass `tools/validate_dav2d.sh`.
- Use the 7,500 MB cap and `sm_86`; no Blackwell-only instructions.

## Wave 1 — M1 still quality

| # | Workstream | Owns | Depends on |
|---:|---|---|---|
| 1 | Recursive partition tree (CPU) | `src/encoder/partition/` | spec; M0 geometry |
| 2 | Full intra predictors (CPU) | `src/encoder/intra_modes/` | spec; reconstruction buffers |
| 3 | Full intra predictors (CUDA) | `src/cuda/intra_modes/` | #2 reference |
| 4 | Transform variants (CPU+CUDA) | `src/encoder/tx_variants/`, `src/cuda/tx_variants/` | spec; DCT reference |
| 5 | CfL (CPU+CUDA) | `src/encoder/cfl/` | luma/chroma recon |
| 6 | Deblock filter (CPU+CUDA) | `src/encoder/filters/deblock/` | partition/transform edges |
| 7 | RDO/RDOQ framework | `src/encoder/rdo/` | transform/intra cost APIs |
| 8 | Wavefront/CUDA-graph scheduler | `src/cuda/scheduler/` | M0 pipeline API |

## Wave 2 — filters, inter, control

| # | Workstream | Owns | Depends on |
|---:|---|---|---|
| 9 | CDEF | `src/encoder/filters/cdef/` | reconstruction |
| 10 | CCSO + Wiener/PC-Wiener + GDF + film grain | `src/encoder/filters/restoration/` | reconstruction |
| 11 | Inter ME: SAD/SATD, search, subpel | `src/encoder/inter/me/` | reference frames |
| 12 | Inter MC + skip/merge/NEWMV/NEARMV | `src/encoder/inter/mc/` | #11 |
| 13 | B-frames, compound, warp, OBMC, inter-intra | `src/encoder/inter/advanced/` | #12, RDO |
| 14 | Rate control, QP slices, CQP/CRF/VBR | `src/encoder/rc/` | entropy QP contexts |
| 15 | 10-bit / 4:2:2 / 4:4:4 | `src/encoder/format/` | headers, entropy, kernels |
| 16 | Tiles/SVC/multi-GPU + deterministic scheduling | `src/encoder/tiles/`, `src/multi_gpu/` | #8, tile syntax |

## Integration

Wave 1 components are wired into the bitstream by an integration pass after
their tests pass. Do not let component agents edit the M0 core. Wave 2 starts
after M1 is integrated, measured, and still passes `dav2d`.

The M1 integration agent owns `src/encoder/m1/` and may touch `src/main.cpp`,
the root `CMakeLists.txt`, and `tests/CMakeLists.txt` only for wiring. Everything
else stays component-only.

## Quality gates

- Per-tool: minimal stream using only that tool + M0 baseline must decode.
- Integrated: 64x64, 640x360, 1080p stills must decode.
- Only then compare bitrate/PSNR/CVVDP against M0.
- No BD-rate claim until the decoder gate passes.
