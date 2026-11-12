# Inter motion compensation — CUDA kernels (AV2 §7.13.3.18)

GPU counterpart of `src/encoder/inter/mc/` (component 12, M2 P-frame baseline).
Single-reference translational MC, 8-bit 4:2:0, switchable subpel filters, and
batched prediction jobs. No bitstream, no ME, no BD-rate claim.

The kernels call the same `mc_core.h` arithmetic as the CPU reference, so CPU
and GPU output is bit-exact. **Verified on the RTX 3060 box: 1,069 checks,
0 failures.**

## Files

```text
src/cuda/inter/mc/
  mc.h              device frame/job structs and public API
  mc.cu             per-block and batched kernels + host validation
  mc_test.cu        CPU-vs-CUDA tests (1,069 checks)
  build_and_test.sh standalone nvcc build/run
  README.md         this file
```

## Design

`mc_predict_sample()` (shared with the CPU) evaluates the normative two-pass
filter per output sample:

1. eight horizontal 8-tap sums, each rounded with `Round2(s, 3)`;
2. one vertical 8-tap sum, `Round2(s, 11)`, then `Clip1`.

The CPU reference materialises the spec's `intermediate` array once per block.
The GPU recomputes the eight horizontal sums per output sample instead. Both
perform the same rounded operations in the same order, so the results are
identical; the GPU needs no shared memory or intermediate buffer. Cost is
~72 integer MACs per sample (plus clipped loads), which is memory-bound on the
3060.

- One thread per output sample; 32x8 threads per CUDA block.
- `mc_predict_plane`: one CUDA block per prediction block (small/edge use).
- `mc_predict_batch`: one CUDA block per `McBlockJob`; threads iterate over the
  job's width/height. This is the encoder path: a launch per prediction block
  is launch-bound at video scale.
- No `cudaMalloc`, no hidden synchronisation, no global mutable state. Every
  buffer is caller-owned.
- Standard CUDA only: no TMA, clusters, `wgmma`/`tcgen05`, FP4, or other
  Blackwell-only features. Builds for `sm_86` and `sm_120`.

## API

```cpp
namespace av2enc::cuda {

struct McDeviceFrame {
  const uint8_t* plane[3];  // device pointers, 8-bit 4:2:0
  int stride[3];
  int mi_cols, mi_rows;     // luma MI grid; planes are MI-aligned
};

cudaError_t mc_predict_plane(const McDeviceFrame&, int plane,
                             const inter::mc::BlockRect&, inter::mc::Mv,
                             inter::mc::InterpFilter, uint8_t* dst,
                             int dst_stride, cudaStream_t = nullptr);

cudaError_t mc_predict_block_420(const McDeviceFrame&,
                                 const inter::mc::BlockRect& luma_rect,
                                 inter::mc::Mv, inter::mc::InterpFilter,
                                 uint8_t* dst[3], int dst_stride[3],
                                 cudaStream_t = nullptr);

struct McBlockJob {
  uint8_t* dst; int dst_stride;
  int x, y, w, h;           // plane samples
  int plane;                // 0/1/2
  inter::mc::Mv mv;         // 1/8 luma sample
  inter::mc::InterpFilter filter;
};

cudaError_t mc_predict_batch(const McDeviceFrame&, const McBlockJob* device_jobs,
                             int count, cudaStream_t = nullptr);
}
```

`McBlockJob` is trivially copyable; upload an array of jobs once per frame and
issue one launch. Chroma jobs carry the already-halved rectangle (the same
convention as `mc_predict_block_420`). The batch path does not validate jobs on
the host because they live in device memory; the caller is responsible for
valid geometry, strides and planes, exactly as for any device-side descriptor.

The host-side per-block entry points return `cudaErrorInvalidValue` for null
pointers, out-of-plane rectangles, odd 4:2:0 geometry, blocks above 64x64, and
short strides. Otherwise they return `cudaGetLastError()`; no sync is performed.

## Tests

```sh
./build_and_test.sh                 # sm_86, build + run
ARCH=sm_120 ./build_and_test.sh
./build_and_test.sh --no-run
```

The script compiles `src/encoder/inter/mc/mc.cpp` with the host compiler and
links it with the CUDA test; nothing outside these two directories is touched.

1,069 checks, 0 failures on RTX 3060 (sm_86, 28 SMs). Coverage:

1. Device `Subpel_Filters` table: all 96 phases sum to 128.
2. 11 block shapes (4x4 … 64x64, rectangular) x 4 filters x 9 subpel MVs, CPU
   `predict_plane` vs GPU, including 4x4/2x2 four-tap downgrade cases.
3. Integer-MV copy for three displacements.
4. Partial right/bottom-edge blocks with large out-of-picture MVs; destination
   guard bands verified intact after every launch.
5. 4:2:0 chroma per plane and through `mc_predict_block_420`.
6. Mixed batch: six jobs spanning all three planes, 4x4 through 32x16, four
   filters and fractional MVs; each compared to CPU with guard checks.
7. Real frame `tests/data/basic_640x360_1f.y4m`: 240 luma 32x32 blocks, one batch
   launch, all match both the CPU reference and a known `(+2,-4)` integer shift.
8. Invalid-argument rejection: out-of-plane, 128x128, null dst, odd
   4:2:0, invalid filter, and the full 17-bit MV range.

## Measured timing (RTX 3060 12 GB, `compat8g` host, 640x360)

One batch launch predicting 240 luma 32x32 blocks (whole frame, one MV):

| Path | Time | Relative |
|---|---:|---:|
| `mc_predict_batch` (1 launch) | **0.052 ms** | 1.0x |
| `mc_predict_plane` (240 launches) | 0.829 ms | 15.9x |

The batch path is ~4.5 G luma samples/s at 640x360. This is kernel timing
including launch, not an encode-speed or QBS claim; ME and RDO dominate a real
P-frame. Numbers vary with host load on the shared Vast.ai Xeon (observed
0.052–0.077 ms for the batch).

For 1080p with 32x32 blocks, expect ~2,040 luma jobs + chroma per frame; one
batch launch keeps the GPU busy while the per-block path would spend
~7–8 ms just on launches.

## Integration notes

1. **Upload once per frame.** Frame/reference planes stay device-resident in the
   arena; the CPU only uploads input and downloads finished payloads.
   `McDeviceFrame` is a view, not an allocation.
2. **Batch per frame.** Collect `(dst, rect, mv, filter)` for every prediction
   job of the frame and call `mc_predict_batch` once; do not launch per block.
3. **Reference/prediction aliasing.** The kernels read reference planes and
   write separate destination planes. Do not use the current reconstruction
   buffer as both source and destination for the same block.
4. **No sync.** The functions are asynchronous on the given stream; the frame
   pipeline's single sync is the integration owner's responsibility.
5. **Not wired in.** No root CMake, pipeline, entropy, or bitstream edits. The
   integration pass owns wiring and the `dav2d` gate.

## Blockers / open items

- **No bitstream/decode gate.** This component emits no stream; M2 remains `U1`
  until an integrated key+P stream decodes with `dav2d`.
- **No ME.** MVs come from component 11 through the CPU-side mode decision; the
  GPU does not search.
- **Scaled references unsupported.** Same limitation as the CPU reference.
- **No fused residual/transform.** MC writes a prediction plane; the transform
  stage reads it. Fusing prediction into the residual kernel is a later
  optimization, not needed for correctness.
- **Stack subset is host-side.** MV prediction runs on the CPU, not the GPU. At
  one block per decision this is acceptable for M2; a fully GPU-resident
  pipeline needs a device MV-stack kernel later.
