# CUDA M0 interface contract

Primary target: 8 GB Ampere (`sm_86`), CUDA 12.8. Secondary build: `sm_120`.
No Blackwell-only instructions. No root `CMakeLists.txt` edits by component
agents; put CUDA build fragments under `src/cuda/`.

## Goal

Port the working CPU M0 still-picture path to CUDA without changing the
bitstream. End-to-end CUDA output must decode with `dav2d` and match the CPU
reference MD5s:

- 64x64: `7efc3162bc0a4785685eba291a8096fd`
- 640x360: `56f11d4a63ff07f2356ef5064b84fe0e`

If exact MD5 matching proves impossible due to FP contraction, report the
difference and a PSNR comparison; do not silently change the test.

## Ownership

| Path | Agent |
|---|---|
| `src/cuda/context.*`, `arena.*`, `error.*`, `CMakeLists.txt`, `build_test.sh` | runtime agent |
| `src/cuda/transform.*`, `transform_test.cu` | transform agent |
| `src/cuda/intra.*`, `intra_test.cu` | intra agent |
| `src/cuda/entropy.*`, `entropy_test.cu` | entropy agent |
| `src/cuda/basic_pipeline.*` | integration agent |
| `bench/cuda_dct_perf/` | DCT performance agent |
| `bench/avm_1080p/` | 1080p baseline agent |

Do not edit `src/encoder/basic_encoder.cpp`, `src/entropy/av2/`, `src/bitstream/`,
`src/main.cpp`, or the root `CMakeLists.txt`.

## Data model

- 8-bit 4:2:0 planar, row-major tightly packed.
- Luma block sizes: 32x32, 64x32, 32x64, 64x64, plus edge 32x16/32x8.
- Chroma: half width/height of the corresponding luma block.
- Residual: `int16_t`, coefficient: `int32_t`, reconstruction: `uint8_t`.
- All buffers device-resident; host copies only input frame and final payload.

## Runtime API

Provide a small `av2enc::cuda` layer:

- device selection, stream creation, error checking
- persistent arena sized from the device (`min(total, free) - 512 MiB`, optional
  `--vram-cap`) with peak tracking
- `upload` / `download` helpers (pinned staging preferred)
- no hidden `cudaDeviceSynchronize()` in kernels; one sync per frame at most
- no `cudaMalloc` after init

## Single device arena

Every device buffer comes from one `av2enc::cuda::Arena` (`src/cuda/arena.h`).
Only `arena.cpp` calls `cudaMalloc`/`cudaFree`; `context.cpp` pinned host
staging (`cudaMallocHost`) is the only other permitted allocation. Modules,
backends and launchers must not allocate, and no launcher may allocate per
frame or per block.

- **One arena.** Pass an `Arena&` (or an already-allocated device buffer) into
  a module entry point. Do not create a second arena for a sub-stage of the
  same pipeline.
- **`required_device_bytes(config)`.** Each module reports the capacity its
  persistent buffers need, including the 256-byte alignment padding
  (`ibc_required_device_bytes()` for IBC). Size the arena once from the device
  budget and this value.
- **`allocate_checked`.** Persistent buffers are bump-allocated once with
  `allocate_checked()`. An undersized arena returns the module's distinct
  `kArenaTooSmall` status carrying the required byte count; do not retry in a
  block loop.
- **Mark / rewind.** Transient scratch takes `mark()` before allocation and
  `rewind(mark)` before returning. Bump allocation never frees by itself.
- **Growth is a frame-boundary operation.** Growing means the caller releases
  and re-initialises the arena (or rewinds past an older mark). That invalidates
  every pointer handed out before it. Never grow with outstanding kernels or
  inside a block loop.
- **Enforcement.** `tools/check_arena.sh` fails on `cudaMalloc`, `cudaFree`,
  `cudaMallocHost` (and the async variants) outside `arena.cpp`,
  `context.cpp`, and test/build-stub trees. It treats `cudaFree(nullptr)` as a
  non-allocation and recognizes the independent multi-GPU pinned-staging
  provider; the script header documents those two narrow distinctions.

## Transform API

Match the CPU arithmetic order in `src/encoder/basic_encoder.cpp`:

- forward 2D DCT (`double`, same loop order)
- quantization with the same `quant_factor` and `std::lround` behavior
- dequantization + inverse 2D DCT
- edge 64x64 -> adjusted 32x32 residual duplication
- compile with `-fmad=false` or explicit `__fadd_rn`/`__fmul_rn` to prevent
  contraction differences

Tests: randomized blocks, CPU vs GPU bit-exact, 32x32/16x16 and edge sizes.

## Intra API

- DC prediction from reconstructed left/above samples
- edge clamping to visible plane bounds (`maxX`/`maxY`)
- residual = source - prediction
- reconstruct = clip(prediction + inverse transform residual)
- no filters (M0)
- one block per kernel launch is acceptable for the first CUDA path; wavefront
  optimization comes later

Tests: synthetic planes, edge blocks, CPU vs GPU bit-exact.

## Validation

Run on the RTX 3060 box:

```sh
ssh -p <port> <user>@<host>
```

Use `nvcc -arch=sm_86` for component tests; build the fat binary only for the
final integration. Nsight Systems 2025.1.3 is available. `ncu` counters are
blocked.
