# M1 GPU reconstruction + deblock-metadata stage

Workstream 5 of `docs/m1-gpu.md`.  Owns only `src/cuda/m1/recon/`.

This stage turns per-block predictions and inverse-transform residuals into the
padded reconstruction planes, keeps the block/edge state prediction needs, and
invokes the existing `src/cuda/deblock` component once per frame **after** all
reconstruction.  Deblocking is never applied during prediction (normative order
is reconstruct -> deblock -> CDEF/CCSO/LR).

Bit-exact with the CPU M1 path: for identical inputs the padded planes, the
prediction availability state and the `DeblockMi` grids are identical, so the
cropped planes equal `M1Result.recon_y/u/v`.

## Files

```text
src/cuda/m1/recon/
  recon.h            public PODs + API
  recon.cu           kernels: frame/state init, recon writes, availability,
                     block state commit + deblock metadata, deblock wrapper
  recon_test.cu      CPU-vs-GPU parity (64x64, 640x360, with/without deblock),
                     tile pass-through, synthetic padded-edge tests
  build_and_test.sh  standalone build/run (no root CMake changes)
  README.md          this file
```

## API

All pointers are device pointers; the caller owns the arena buffers and the
stream.  Nothing here allocates or synchronises.

```cpp
namespace av2enc::cuda::m1 {

struct ReconPlane { uint8_t* data; int stride; int height; };

struct ReconFrame {
  ReconPlane plane[3];                 // padded planes, 64-aligned like CPU
  int mi_cols, mi_rows;                // luma MI grid (§5.18.4.4 formula)
  deblock::DeblockMi* deblock_mi[3];   // mi_cols*mi_rows each
  uint8_t base_q_idx;                  // M1: 90
  const uint16_t* tile_col_starts; int num_tile_cols;   // optional, device
  const uint16_t* tile_row_starts; int num_tile_rows;
};

struct ReconPredState {
  int mi_cols, mi_rows;
  uint8_t* decoded[3];  int decoded_cols[3], decoded_rows[3];
  int* intra_joint;        // mi_rows*mi_cols
  uint8_t* uv_smooth;      // (mi_cols/2)*(mi_rows/2)
  int sb_mi_row, sb_mi_col, sb_cols[3], sb_rows[3];  // set by recon_begin_sb
};

cudaError_t recon_init_frame(const ReconFrame&, cudaStream_t = nullptr);
cudaError_t recon_init_state(const ReconPredState&, cudaStream_t = nullptr);
cudaError_t recon_begin_sb(ReconPredState&, int sb_mi_row, int sb_mi_col,
                           cudaStream_t = nullptr);

struct ReconWriteJob {
  const uint8_t* pred;      // w*h uint8
  const int16_t* residual;  // w*h int16 (coded/rect grid, row stride w)
  uint8_t* plane; int stride, height;   // padded plane extent
  int x0, y0, w, h;
};
cudaError_t recon_write_blocks(const ReconWriteJob* d_jobs, int count,
                               cudaStream_t = nullptr);
cudaError_t recon_write_block(const ReconWriteJob&, cudaStream_t = nullptr);

struct ReconBlockInfo { /* mi_row/col, w4/h4, chroma_w4/h4, joint, uv_smooth,
                           qindex, tx_size_luma/chroma, skip/region/lossless */ };
cudaError_t recon_commit_blocks(const ReconBlockInfo* d_infos, int count,
                                const ReconPredState&, const ReconFrame&,
                                cudaStream_t = nullptr);
cudaError_t recon_commit_block(const ReconBlockInfo&, const ReconPredState&,
                               const ReconFrame&, cudaStream_t = nullptr);

struct ReconAvailRequest { int plane, x4, y4, w4, h4; };
struct ReconAvail { int have_left, have_above, num4_above_right,
                    num4_below_left; };
cudaError_t recon_block_availability(const ReconPredState&,
                                     const ReconAvailRequest* d_requests,
                                     ReconAvail* d_out, int count,
                                     cudaStream_t = nullptr);

cudaError_t recon_deblock_frame(const ReconFrame&,
                                const deblock::DeblockOptions&,
                                cudaStream_t = nullptr);
}
```

### Reconstruction writes

```text
recon[py, px] = clip1(pred[y, x] + residual[y, x])
write only when 0 <= x0+x < plane.stride and 0 <= y0+y < plane.height
```

That is exactly the CPU `Encoder::write_recon`: it clamps to the *padded* plane
extent, so an edge transform block writes its full parsed extent (the decoder
does the same); prediction later clamps reads to `MiCols*4` / `MiRows*4`.

`recon_write_blocks` is one launch for the whole job array (one CUDA block per
job, threads stride over `w*h`, mixed sizes allowed).  Jobs must cover disjoint
regions, which partition leaves always do.  `recon_write_block` takes the
descriptor by value for one-off use.

### Prediction state

`recon_begin_sb` resets the three `DecodedMap` grids and records the SB origin
and SB-local plane extents.  `recon_commit_blocks` applies, per block:

* decoded flags (luma at `mi_col`, chroma at `mi_col/2`);
* IntraJointModes over the luma MI extent;
* the UV smooth flag over the chroma extent;
* the three `DeblockMi` grids (block/tx/sub-PU bases, clamped `w4/h4`,
  `tx_size`, `qindex`, skip/intra/lossless), matching `fill_deblock_mi`.

`recon_block_availability` returns the CPU predictor's
`have_left`/`have_above` and `count_top_right` / `count_bottom_left` values
from the current state.  Query it *before* committing the current block.

Batched commits are safe for the MI cells (leaves are disjoint); keep the
syntax order of the blocks, since later availability depends on earlier
commits.

### Deblock

`recon_deblock_frame` fills a `cuda::DeblockDeviceFrame` from the `ReconFrame`
and calls `cuda::deblock_frame` once.  The deblock component itself issues the
normative two passes (vertical then horizontal); the caller makes one
invocation per frame.  Tile starts are forwarded unchanged; M1 is a single
tile, so `num_tile_cols = num_tile_rows = 0`.  Frame edges come from
`mi_cols`/`mi_rows`, exactly like the CPU reference.

`DeblockOptions` for M1: `apply_deblocking_filter[0..3] = true`,
`frame_is_intra = true`, `enable_sdp = false`, 8-bit.

## Tests

`build_and_test.sh` compiles the CPU reference (partition, intra_modes,
tx_variants, CfL, deblock, M1 encoder + syntax) with the host compiler, compiles
`recon.cu`, `recon_test.cu` and `src/cuda/deblock/deblock.cu` with nvcc, and
runs the parity test.

The test builds a small CPU *shadow* of the default M1 decision path from the
public components (DC-only luma, DC/CfL chroma, variance partition, DCT_DCT,
CfL, deblock metadata).  It first gates the shadow against the official
`m1::encode_m1_still_picture` result, so a change in the encoder defaults fails
loudly instead of silently testing different blocks.  It does not include
encoder-private internals, so it survives refactors of `m1_still.cpp`.

Gates:

1. shadow CPU recon == `M1Result.recon_*` (visible crop), with and without
   deblock, 64x64 and 640x360;
2. GPU padded planes == shadow CPU planes (full padded planes) and their
   visible crop == `M1Result.recon_*`;
3. GPU deblocked planes == CPU deblocked planes (full padded) and the visible
   crop == `M1Result.recon_*` (deblock enabled);
4. GPU availability == shadow `count_top_right`/`count_bottom_left` at every
   leaf and plane (all three planes);
5. GPU decoded flags / IntraJointModes / UV smooth grid == shadow state;
6. GPU `DeblockMi` grids == CPU grids on every field;
7. tile-start pass-through (luma tile edge at MI column 8) CPU vs GPU;
8. synthetic padded-edge jobs: saturation, padded right/bottom clamping,
   batched == serial.

Measured on the RTX 3060 (12 GB, CUDA 12.8, `sm_86`):

```text
64x64    deblock=1  leaves=11   checks=149  failures=0
64x64    deblock=0  leaves=11   checks=250  failures=0
640x360  deblock=1  leaves=218  checks=1457 failures=0
640x360  deblock=0  leaves=218  checks=2652 failures=0
checks: 2652, failures: 0
compute-sanitizer --tool memcheck: 0 errors
nvcc -Wall -Wextra -Wconversion -Wshadow: 0 warnings
```

### Timing (640x360, 218 leaves, CUDA events, 50 iterations)

```text
recon writes (one batched launch): 0.014 - 0.016 ms
deblock (one wrapper call; 2 internal passes): 0.044 - 0.092 ms
```

The box is shared; run-to-run spread is a few percent.

## Limitations / notes

- M1 is 8-bit 4:2:0, one tile, one keyframe; the stage does not model tiles
  beyond forwarding `MiColStarts`/`MiRowStarts` to the deblock component.
- `recon_write_blocks` assumes disjoint write regions.  Blocks that overlap
  (which the partition tree never produces) would race.
- The deblock wrapper is one call per frame; the deblock component internally
  launches the two normative passes.  Do not call it per block, and do not call
  it before the frame is fully reconstructed.
- Deblock is deliberately not applied to the recon planes used by prediction.
  The test compares the unfiltered planes before deblock and the filtered planes
  afterwards.
- No allocation, no global state, no hidden synchronisation.  Buffers come from
  the orchestrator's arena.
