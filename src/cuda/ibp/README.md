# AV2 IBP — CUDA

GPU implementation of AV2 intra bi-prediction, bit-exact against the CPU
reference in `src/encoder/ibp/`.

Normative source: AV2 v1.0.0 §7.13.2.7/.8/.9/.12/.14/.17/.18, §5.4.5.

## Files

```text
src/cuda/ibp/
  ibp.h                        public API, host/device decision helper
  ibp_predict_device.cuh       IBP orchestration on top of the shared intra
                               device body (edge build / directional sample)
  ibp.cu                       kernels + host wrappers
  ibp_test.cu                  GPU vs CPU parity test (needs nvcc/GPU)
  ibp_device_emulation_test.cpp  host-emulation parity (no nvcc/GPU)
  build_cuda_test.sh           nvcc build/run script
  build_emulation_test.sh      no-nvcc build/run script
  README.md                    this file
```

Nothing outside this directory is modified. The component reuses the
read-only `src/cuda/intra_modes/` code: `intra_predict_device.cuh` supplies the
§7.13.2.1 edge construction, §7.13.2.8 single-directional predictor and DC
value, and `intra_modes.h` supplies `IntraBlock`, mode enums, WAIP and the
normative tables. Only the IBP-specific orchestration (edge-filter parameter
split, two-angle blend, IBP DC post-filter) is new. The IBP tables in
`src/encoder/ibp/ibp_tables.h` are the single source for both CPU and device.

## API

```cpp
namespace av2enc::cuda::ibp {

struct IbpBlock {
  IntraBlock intra;  // recon plane, x/y/w/h, maxX/maxY, mode, angle_delta,
                     // filter_type_above/left, enable_edge_filter, plane
  int enable_ibp;    // sequence-level §5.4.5 flag
  int is_inter;      // inter-intra: skip WAIP
  int use_dip;       // DC applicability predicate only
  int uv_is_cfl;     // chroma DC CfL exemption
};

struct IbpDecision {
  int valid, apply_ibp_edge_rules, use_ibp, p_angle, second_angle;
};
IbpDecision ibp_decide(const IbpBlock&);  // host/device

cudaError_t ibp_predict(const IbpBlock&, uint8_t* pred,
                        cudaStream_t = nullptr);

struct IbpJob { IbpBlock block; uint8_t* pred; };
cudaError_t ibp_predict_batch(const IbpJob* d_jobs, int count,
                              cudaStream_t = nullptr);
}
```

`ibp_predict()` owns the two paths IBP can change:

* **DC**: DC prediction plus the §7.13.2.12 post-filter when
  `enable_ibp && !use_dip && !(w==4&&h==4) && (plane==0 || !uv_is_cfl)`.
* **Directional**: IBP-aware edge filtering, then one §7.13.2.8 prediction, or
  two plus the §7.13.2.9 blend when `useIBP` is true.

SMOOTH / SMOOTH_V / SMOOTH_H / PAETH are unchanged by IBP; the host wrapper
delegates them to `intra_modes::intra_predict()`. Batch jobs must be DC or
directional.

Every function is pure: device pointers + stream in, `cudaError_t` out. No
global state, no allocation in the kernel, no hidden device synchronisation.
One CUDA block predicts one transform block; edge filtering is performed by
thread 0 (above) and thread 1 (left) in parallel, followed by a barrier.

## Parity

### Host emulation (runs here, no GPU/nvcc)

`./build_emulation_test.sh` compiles the exact `ibp_predict_device.cuh` with
CUDA keywords stubbed and one emulated thread, then compares every byte against
`ibp::predict()`:

```text
ibp_device_emulation_test: PASS (273958171 checks)
emulation prediction cases: 593408
emulation real blocks 64x64: 48 cases
emulation real blocks 640x360: 2880 cases
```

Coverage: every DC/directional mode, angle delta −3..3, 19 shapes (4×4 through
64×64, including 1:2, 1:4, 1:8 and 1:16), luma/chroma, `enable_ibp` 0/1,
edge-filter 0/1, `is_inter` 0/1, smooth-neighbour filter types, interior and
frame-edge positions. Real blocks come from `tests/data/*.y4m`.

### Real GPU

```sh
cd src/cuda/ibp
./build_cuda_test.sh                 # sm_86
ARCH=sm_120 ./build_cuda_test.sh     # secondary target
```

The GPU test runs the same prediction matrix through `ibp_predict()`, adds a
96-job `ibp_predict_batch()` comparison, real y4m blocks and invalid-argument
checks.

**Status:** this workspace is macOS and has no `nvcc` or CUDA device, so the
`sm_86` run was not executed here. The device body is covered bit-exactly by
the emulation test; the `.cu` kernel is a thin launch wrapper around that same
body.

## Integration

IBP has no per-block syntax and no new CDF. To use this kernel in the encoder:

1. Set `enable_ibp = 1` in the sequence header
   (`src/encoder/m1/m1_headers.cpp:120`, `src/encoder/m2/m2_headers.cpp:147`,
   `src/encoder/m2/post/tiled_frame.cpp:157`;
   `src/bitstream/m0_headers.cpp:224` only if M0 gains the full intra path).
2. Use `ibp_predict()` for luma mode search and reconstruction **and** for the
   chroma paths, with the block's `enable_ibp` set. The decoder derives
   `applyIbp` for every non-4×4 directional block when the sequence bit is 1,
   so a mixed search that keeps the old predictor would desynchronise.
3. Keep `IntraBlock.filter_type_above/left`, `x0/y0/maxX/maxY`,
   `num4_above_right`/`num4_below_left` populated exactly as for the existing
   intra predictor.

CUDA-side call sites to extend:

* `src/cuda/m1/search/luma_search.cu` (mode search; device intra body),
* `src/cuda/m1/chroma/m1_chroma.cu` (chroma edge filtering / DC),
* `src/cuda/m1/recon/` (reconstruction),
* `src/cuda/m1/m1_gpu.cu` (orchestration).

No files outside `src/cuda/ibp/` and `src/encoder/ibp/` were changed.
