# Morphological prediction — CUDA

CUDA side of `src/encoder/inter/morph/`, the AV2 v1.0.0 morphological
prediction process. §7.13.3.26 has no body of its own: it invokes the
§7.13.3.25 block adaptive weighted prediction process with
`plane = 0, morphPred = 1`. The CPU reference (`morph_common.h`) is the
contract, and this component does not reinterpret it.

The process per block:

1. derive the full-pel reference origin from the block's MI position and the
   decoded 1/8-pel vector (`to_fullmv`, bias 3 plus 1 for non-negative);
2. check the §7.13.3.25 conformance box (`refX >= 1`, `refY >= 1`,
   `refX+bw <= planeWidth`, `refY+bh <= planeHeight`);
3. fit `alpha`/`beta` (Q8) from the reconstructed above/left template, using
   the 12 -> 8 extent rule, the availability branches, and the Div_Lut
   `resolve_division` approximation;
4. apply `clip1((orig * alpha + beta) >> 8, 8)` to **every** sample of the
   full `w*h` rectangle, including the part outside the visible plane.

The reference has no dense per-sample motion field. The optional
`MorphJob::field_out` materialises the block-constant full-pel displacement
`(dx, dy)` per sample for callers that need one; it never affects prediction.

## Files

| File | Purpose |
|---|---|
| `morph_device.cuh` | The single arithmetic transcription, annotated `__host__ __device__`. Compiles into the kernel and into the host emulation test. |
| `morph.h` | Public API: `MorphJob`, `MorphWorkspace`, `required_device_bytes`, launchers. |
| `morph.cu` | Kernels. One CUDA block per job; thread 0 fits, a barrier, all threads apply. |
| `morph_workspace.cpp` | Host-side arena allocation, upload, staged scratch path. No allocation from the launchers. |
| `morph_test_cases.h` | Deterministic case list shared by both tests. |
| `morph_test.cu` | Real CPU-vs-GPU parity test (needs nvcc + device). |
| `morph_emulation_test.cpp` | Host parity test against the CPU reference (no nvcc). |
| `morph_emulation.cpp` | Host emulation of the entry points, used only by the emulation build. |

## API

```cpp
namespace av2enc::cuda::morph {

struct MorphJob {
  MorphContext ctx;      // mi_col/mi_row/mi_cols/mi_rows, avail_u/avail_l, bit_depth
  MorphRect rect;        // x, y, w, h
  int mv_row, mv_col;    // 1/8-pel, as decoded
  std::uint8_t* frame;   // device luma plane with the IBC prediction in place
  int stride;
  MorphFit* fit_out;     // optional device output
  int* status_out;       // optional device output: 1 accepted, 0 rejected
  std::int16_t* field_out;  // optional 2*w*h int16 (dx, dy) per sample
};

cudaError_t morph_predict(const MorphJob& job, cudaStream_t = nullptr);
cudaError_t morph_predict_batch(const MorphJob* d_jobs, int count,
                                cudaStream_t = nullptr);
cudaError_t morph_predict_batch_persist(const MorphJob* d_jobs,
                                        MorphFit* d_fits, int* d_statuses,
                                        int count, cudaStream_t = nullptr);

std::size_t required_device_bytes(const MorphArenaConfig&);
std::size_t required_scratch_bytes(int job_count);

class MorphWorkspace {
  Status init(Arena&, const MorphArenaConfig&, std::string* error = nullptr);
  MorphJob* jobs(); MorphFit* fits(); int* statuses(); std::int16_t* field();
  std::size_t required_bytes() const;
};

Status morph_upload_jobs(MorphWorkspace&, const MorphJob* host_jobs, int count,
                         cudaStream_t = nullptr);
Status morph_run_batch(const MorphWorkspace&, int count,
                       cudaStream_t = nullptr);

Status morph_stage_and_run(Arena&, const MorphJob* host_jobs, int count,
                           MorphStagedResult* result, cudaStream_t = nullptr,
                           std::size_t* scratch_bytes_out = nullptr);
}
```

`morph_predict`/`morph_predict_batch` do not allocate and do not synchronize.
A rejected job leaves the plane, fit output and field untouched and reports
`status_out = 0`, matching `morph_predict()` returning `false`.

### Arena policy

All device memory comes from the caller's single persistent
`av2enc::cuda::Arena`. This component never calls `cudaMalloc`, `cudaFree`,
`cudaMallocHost`, or any per-launch allocation.

- `required_device_bytes(config)` is the exact persistent size:
  `max_jobs` descriptors + `max_jobs` fits + `max_jobs` statuses +
  `2 * max_field_samples` `int16` field entries, each 256-byte aligned.
- `MorphWorkspace::init()` allocates those arrays once. Two launches through
  the workspace consume zero additional arena bytes.
- A short arena returns `Status::kArenaExhausted`, leaves the bump pointer
  untouched, and reports the required size through `required_bytes()`.
- The production path is workspace upload + `morph_run_batch`; the staged
  path is for callers without a workspace. `morph_stage_and_run()` takes
  `arena.mark()`, allocates staging descriptors/fits/statuses, launches,
  synchronizes the stream so the scratch is no longer in flight, copies
  results back and rewinds. It reports its exact scratch size in
  `scratch_bytes_out`.

For the test configuration `{max_jobs = 16, max_field_samples = 4096}`:

```
required_device_bytes = 19200 bytes
required_scratch_bytes(16) = 2816 bytes
```

## 8-bit and 10-bit

The CPU reference is luma 8-bit: `morph_reference_geometry()` rejects
`bit_depth != 8` before reading the plane, and `morph_apply_fit()` clips to
8 bits. The CUDA path matches that decision exactly. The shared case list
includes 24 10-bit cases; both sides reject them and leave the plane, fit and
field untouched. Bit-exact prediction is therefore asserted for 8-bit, and
bit-exact normative rejection for 10-bit.

## Build and test

Local host pre-flight (no nvcc; this Mac):

```sh
./build_and_test.sh
```

This compiles `morph_device.cuh` as host C++ with CUDA keywords stubbed,
links the emulated entry points plus the real `arena.cpp`/`morph_workspace.cpp`,
and compares 2292 cases against `morph_common.h`.

Real gate (Linux + CUDA, sm_86):

```sh
./build_gpu_test.sh                 # build + run
./build_gpu_test.sh --no-run
ARCH=sm_120 ./build_gpu_test.sh
```

Remote command from the development Mac:

```sh
AV2ENC_HOST=<user>@<host> AV2ENC_PORT=<port> ./tools/sync_remote.sh
ssh -p <port> <user>@<host> \
  'cd /workspace/av2enc && ./src/cuda/morph/build_gpu_test.sh'
```

Neither script touches the root CMake build.

## Coverage and expected counts

The shared case list has **2292 cases**: all 22 IBC-signallable block sizes
(4x4 through 64x32, including 4x32/32x4), 64x64 (process-valid, not
IBC-signallable), all four `(AvailU, AvailL)` combinations, interior and
right/bottom-overhanging rectangles, the `bw == 12`/`bh == 12` template rule,
zero templates, zero/integer/sub-pel/negative/±32k motion vectors, every
`to_fullmv` residue in both signs, ±65528/±65536 out-of-box rejections,
all conformance-box failures, and 0/255 sample extremes.

Local emulation (`./build_and_test.sh`):

```
morph_emulation_test: cases=2292 accepted=1889 rejected=403 bit_depth10=24
                      checks=11593 failures=0
```

Expected on the sm_86 box (`./build_gpu_test.sh`):

```
morph_cuda_test: cases=2292 accepted=1889 rejected=403 bit_depth10=24
                 checks=36940 failures=0 required_device_bytes=19200
                 scratch=2816 ...
```

The GPU run also covers the raw batch entry point, the persistent workspace,
two `mark()`/`rewind()` staged launches with no net arena growth, and
one-byte-short arenas for both the workspace and the staged path.

## Rules followed

- Integer-only arithmetic; no floating point.
- One arithmetic transcription shared by device and emulation.
- No `cudaMalloc`/`cudaFree`/`cudaMallocHost`; every device buffer comes from
  `src/cuda/arena.*`.
- One launch per batch, one `__syncthreads()` per job, no hidden
  `cudaDeviceSynchronize()` in the component (the staged helper synchronizes
  its stream only because it must rewind arena scratch).
- No Blackwell-only instructions, no TMA/clusters/`wgmma`/`tcgen05`, no FP4.
- Component-local build scripts; root `CMakeLists.txt`, `src/main.cpp` and
  other agents' files untouched.
