# CUDA M0 intra: DC prediction, residual, reconstruction

Component-owned files (intra agent):

```text
src/cuda/intra.h              API + arithmetic contract
src/cuda/intra.cu             kernels (no global state, no hidden sync)
src/cuda/intra_test.cu        CPU-reference bit-exactness + timing test
src/cuda/build_intra_test.sh  standalone build/run script
src/cuda/README-intra.md      this file
```

Not wired into `src/main.cpp` or the root `CMakeLists.txt`; integration is a
separate step. Nothing outside `src/cuda/` is modified.

## Arithmetic contract (bit-exact with the CPU M0 path)

Mirrors `encode_residual_plane()` in `src/encoder/m0_still.cpp` (read-only),
including the CPU's `approx_divide` behavior for non-power-of-two block areas:

```text
pred = 128
if (have_left && have_above)
  pred = clip1(round2((sum_left + sum_above) * 512,
                      9 + floor_log2(width + height)))
else if (have_left)
  pred = round2(sum_left, floor_log2(height))
else if (have_above)
  pred = round2(sum_above, floor_log2(width))

residual[i] = source(clamped) - pred      // int16, range [-255, 255]
recon[i]    = clip1(pred + residual[i])   // uint8
```

- `sum_left` reads `recon[x0 - 1][clamp(y0 + i, 0, plane_max_y)]`;
  `sum_above` reads `recon[clamp(x0 + j, 0, plane_max_x)][y0 - 1]`.
  The clamp is to the visible plane bounds, exactly like the CPU. Padded
  reconstruction-plane samples are never read.
- Source samples are likewise clamped to `source_max_x` / `source_max_y`, so a
  block that extends past the visible frame reproduces the CPU's padded-block
  residual.
- `add_residual_clip_to_plane` drops writes outside
  `[0, plane_max_x] x [0, plane_max_y]`. The CPU writes into the 64-aligned
  padded plane instead, but those padded samples never feed prediction, so the
  visible output is identical.
- Integer only. No FP, no `-fmad` sensitivity.
- `have_left` / `have_above` are caller-supplied context flags (in the CPU they
  come from `is_inside(mi_row, mi_col - 1)` / `is_inside(mi_row - 1, mi_col)`,
  i.e. frame-level, not tile-level).

## API

All entry points are pure functions: device pointers + `cudaStream_t`, no
global state, no allocations, no device synchronisation. Invalid arguments
return `cudaErrorInvalidValue`; launch failures come back from
`cudaGetLastError()`.

```cpp
namespace av2enc::cuda {

struct DcBlock {
  const uint8_t* recon_plane; int recon_stride;   // prediction samples
  int x0, y0;                                     // block origin, samples
  int block_w, block_h;                           // per-axis <= 64
  int plane_max_x, plane_max_y;                   // visible bounds, inclusive
  int have_left, have_above;                      // context flags
  const uint8_t* source_plane; int source_stride; // residual source
  int source_max_x, source_max_y;                 // visible source bounds
};

cudaError_t dc_pred_value(const DcBlock&, int* d_pred, cudaStream_t = nullptr);
cudaError_t dc_predict(const DcBlock&, uint8_t* pred_block, cudaStream_t = nullptr);
cudaError_t dc_residual(const DcBlock&, const uint8_t* pred_block_or_null,
                        int16_t* residual_block, cudaStream_t = nullptr);
cudaError_t dc_predict_residual(const DcBlock&, uint8_t* pred_block,
                                int16_t* residual_block, cudaStream_t = nullptr);
cudaError_t dc_predict_residual_batch(const DcBlockJob* d_jobs, int job_count,
                                      cudaStream_t = nullptr);
cudaError_t add_residual_clip(const uint8_t* pred_block,
                              const int16_t* residual_block,
                              uint8_t* recon_block, int block_w, int block_h,
                              cudaStream_t = nullptr);
cudaError_t add_residual_clip_to_plane(
    const uint8_t* pred_block, const int16_t* residual_block, uint8_t* plane,
    int stride, int x0, int y0, int block_w, int block_h,
    int plane_max_x, int plane_max_y, cudaStream_t = nullptr);
}
```

Notes:

- `pred_block` is per-sample `uint8` (the DC value replicated). Use
  `dc_pred_value` when only the scalar predictor is needed.
- `DcBlockJob` bundles a `DcBlock` with output pointers; the array must live in
  device memory. Jobs are not revalidated per job and must be mutually
  independent (frozen input planes, disjoint outputs). One CUDA block per job,
  256 threads.
- `dc_predict_residual` fuses the first two stages in one launch; the normal
  per-block encode path is `dc_predict_residual` -> transform agent ->
  `add_residual_clip_to_plane`. There is no dependency on the transform
  component: `add_residual_clip*` accepts any `int16` inverse-transform
  residual and saturates to `[0, 255]`.
- Supported/validated sizes: luma 32x32, 32x16, 32x8, 64x64, 64x32, 32x64;
  chroma 16x16, 16x8, 16x4, 32x32, 32x16, 16x32; the kernels are generic for
  any `1 <= block_w, block_h <= 64`. 64x64 luma / 32x32 chroma are included
  because 640x360 with the M0 `k32x32` tree emits a 64x64 PARTITION_NONE for
  the partial bottom superblock.
- The CPU site stores the predictor scalar and duplicates a 64x64 residual from
  32x32 when `TX_64X64` is used. That residual duplication belongs to the
  transform agent; this module predicts over the full 64x64 block.

## Tests

`intra_test.cu` uses a line-for-line CPU mirror of the CPU M0 loop (not the
transform), with the same `clip3` / `clip1` / `round2` / `floor_log2` /
`approx_divide`.

Coverage:

1. Seven synthetic 640x360 luma and 320x180 chroma plane patterns (gradient,
   black, white, checker, noise, stripes, ramp); source pattern differs from
   the reconstruction pattern.
2. Explicit block classes: top-left / top / left / interior / top-right /
   bottom-right; 32x16 and 32x8 edge blocks; 64x64 blocks that extend past the
   640x360 bottom edge (the real M0 edge-SB case); blocks extending past the
   right/bottom bounds; all four `(have_left, have_above)` combinations at
   interior positions (including "no neighbours" -> 128).
3. Random fuzzing: 400 luma + 400 chroma blocks per pattern, dimensions drawn
   from 15 shapes, positions anywhere in the plane, random neighbour flags.
4. Padded stride: visible 100x70 reconstruction plane stored with stride 128
   and `0xEE` sentinel padding; padding must never be read or written.
5. Full-plane raster with an evolving reconstruction (the dependency pattern
   the encoder actually has): per block, DC prediction from the current plane,
   CPU-vs-GPU per-block pred/residual comparison, then an arbitrary synthetic
   inverse-transform residual is added and the plane evolves for the next
   block. Runs over 640x360 luma 32x32 (+32x8 bottom row) and 320x180 chroma
   16x16 (+16x4 bottom row) for all seven patterns.
6. `add_residual_clip` stress with full-range random `int16` residuals and
   random predictors (saturation), plus the strided plane variant with a block
   crossing the plane edge (sentinel checks).
7. Batched path: 64 independent jobs (one CUDA block each, mixed block sizes,
   random positions and neighbour flags) compared per job against the CPU
   reference.
8. Invalid-argument checks (`nullptr`, zero extent, `have_left` with `x0 == 0`,
   etc.).

Each stage reports sample count, mismatch count, and max absolute difference.
The test exits non-zero on any mismatch.

### Measured results (RTX 3060 12 GB, sm_86, CUDA 12.8)

```text
stage                   samples   mismatches  max|diff|
prediction             12158656            0          0
pred_value                 6417            0          0
residual               17216448            0          0
recon_block             5057792            0          0
recon_plane           883876608            0          0
clip_stress               42768            0          0
```

`compute-sanitizer --tool memcheck`: **0 errors**. (`racecheck` was attempted
but is impractically slow with this many small launches and was aborted; the
shared-memory reduction is guarded by `__syncthreads()` and memcheck is clean.)

## Build / run

On the 3060 box:

```sh
ssh -p <port> <user>@<host>
cd /workspace/av2enc/src/cuda
./build_intra_test.sh                 # sm_86 build + run
./build_intra_test.sh --no-run        # build only
ARCH=sm_120 BUILD_DIR=build-intra-sm120 ./build_intra_test.sh --no-run
```

The script uses `nvcc -std=c++20 -O3 -lineinfo -arch=sm_86` and
`-Wall -Wextra -Wconversion -Wshadow` for host code. `sm_120` compile-only was
verified with CUDA 12.8. Kernel resources: 26-40 registers, 32-36 bytes of
static shared memory, no spills.

## Timing (measured)

Run-to-run spread on the shared Vast.ai box is significant (idle-GPU vs loaded),
so ranges over four runs are shown. `us/launch` is wall time per block for one
kernel launch on one stream, i.e. mostly launch latency.

| Path (32x32 luma) | Time |
|---|---:|
| `dc_predict` | 2.5 - 4.0 us/launch |
| `dc_residual` | 2.9 - 4.6 us/launch |
| `dc_predict_residual` | 2.95 - 4.1 us/launch |
| `add_residual_clip` | 1.9 - 2.8 us/launch |
| `dc_predict_residual_batch`, 8192 blocks | 0.145 - 0.230 ms (36 - 56 Mblocks/s, 36 - 58 Gsamples/s) |
| 640x360 raster, 720 blocks, 2 launches/block | 3.6 - 5.2 ms/frame |

The batch figure (dependency-free blocks, one CUDA block per job) is
memory-bound at roughly 36-58 Gsamples/s; the per-block figures are entirely
launch-bound. Extrapolating the 640x360 launch sequence to 1080p (6120 blocks)
gives roughly 31-44 ms/frame for the predict+residual+clip launches alone,
versus ~0.1-0.2 ms of actual work at batch rate.

## Integration notes / blockers

- **Launch count is the blocker, not arithmetic.** The M0 path allows one block
  per launch for bring-up, and it works, but 2 launches per block at ~2-4 us
  each dominates. A later wavefront pass should batch
  independent blocks (e.g. anti-diagonal superblock waves, one launch per
  wave) or use CUDA graphs. `dc_predict_residual_batch` already provides the
  batched arithmetic once dependencies are handled by the caller.
- Dependencies are strict: prediction reads reconstructed left/above samples.
  The batch entry point must only be used for blocks whose prediction sources
  are frozen.
- The caller must pass the padded plane stride and visible `maxX` / `maxY`.
  Writes outside the visible bounds are dropped, so the module can operate
  directly on a padded CPU-equivalent reconstruction plane.
- `have_left` / `have_above` follow the CPU frame-level rule. For a block at
  `x0 == 0` / `y0 == 0` they must be 0. The API rejects `have_left` with
  `x0 == 0` and `have_above` with `y0 == 0`.
- M0 has no intra edge filters; none are applied. Directional/Paeth/smooth
  modes and edge filtering are later milestones.
- No hidden `cudaDeviceSynchronize`, no `cudaMalloc`, no static/global device
  state. One sync per frame remains the caller's responsibility.
