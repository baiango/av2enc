# CUDA M2 P-frame (reserved)

This directory is reserved for the device-side M2 P-frame pipeline. The M2
correctness milestone is gated on the CPU reference in `src/encoder/m2/`;
nothing here is wired into the product build yet.

Intended split, following `docs/cuda-interface.md`:

- `src/encoder/m2/` owns the bitstream decisions, frame headers, reference-slot
  bookkeeping and the CPU oracle.
- `src/cuda/m2/` will own the GPU-resident frame/reference pool, the batched
  inter search and mode decision, the per-block residual path, and the
  device-side reconstruction/reference update.
- Reuse the existing GPU components: `src/cuda/inter/me/`, `src/cuda/inter/mc/`,
  `src/cuda/intra_modes/`, `src/cuda/tx_variants/`, `src/cuda/transform.*`.
- Entropy remains CPU in the near term (`src/entropy/av2/`), as documented in
  `docs/architecture.md`; the M2 tile is one tile and is serial within the
  range coder.

The CPU path must remain the byte-exact reference. Any GPU implementation must
match `src/encoder/m2/` for the same decisions and pass
`src/encoder/m2/build_and_test.sh` against `dav2d` before it is called working.

Not implemented in this milestone:

- B frames, multiple references, compound prediction.
- Warp/affine, OBMC, inter-intra, BAWP, TIP, OPFL, refine-MV, temporal MV
  projection, reference MV bank.
- Inter residual coding (P-frame inter leaves are `skip_flag = 1`).
- Multi-tile entropy and 10-bit / non-4:2:0 formats.
