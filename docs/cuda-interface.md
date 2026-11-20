# CUDA M0 interface contract

Targets: RTX 3060 12 GB dev and RTX 3080 Ti production (`sm_86`), RTX 5090 (`sm_120`), **CUDA 13.0** (verified 2026-11-13 on
driver 580.178.04); CUDA 12.8 still builds. Production: RTX 3080 Ti (`sm_86`);
maximum-speed build: RTX 5090 (`sm_120`).
No Blackwell-only instructions. No root `CMakeLists.txt` edits by component
agents; put CUDA build fragments under `src/cuda/`.

## Goal

Port the working CPU M0 still-picture path to CUDA without changing the
bitstream. End-to-end CUDA output must decode with `dav2d` and match the CPU
reference MD5s:

- 64x64: `c9294fa48f70fc82a26c64908ddc0b12`
- 640x360: `0d90c958b8dd5d12c6ab2eef2a2c19bf`

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
- **Enforcement.** `tools/check_arena.sh` fails on every `cudaMalloc*`/`cudaFree*`
  spelling (`cudaMallocManaged`, `cudaMallocPitch`, `cudaMalloc3D`,
  `cudaMallocArray`, `cudaMallocAsync`, ... included), `cudaHostAlloc` and
  `cudaHostRegister` outside `arena.cpp`, `context.cpp`, and test/build-stub
  trees; a call whose `(` sits on the next line is reported too. It treats
  `cudaFree(nullptr)` as a non-allocation, recognizes the independent multi-GPU
  pinned-staging provider, and reports (without failing on) the allocation
  sites inside `bench/` standalone harnesses. The script header documents
  those distinctions.

- **Still-path clamp (documented exception).** `encode_still_gpu()`
  (`src/cuda/pipeline/still/still_gpu.cu`) computes the same device budget
  (`min(total, free) - 512 MiB`) and honours an explicit `vram_cap_mb`, but then
  caps the reservation at `max(256 MiB, 1.5 * required_device_bytes + 16 MiB)`.
  The still working set is a few MiB, so the arena is deliberately kept small
  while other processes share the GPU; this is not a card-specific preset.
  `Arena::capacity_cap_bytes()` (the global set by `apply_vram_cap`) is not
  consulted on this path; the CLI passes `--vram-cap` explicitly instead.

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

Every component suite is discoverable and runnable in one pass:

```sh
tools/cuda_module_suites.sh --list        # what exists
tools/cuda_module_suites.sh -j 5          # pass/fail table + logs
```

`docs/cpu-cuda-matrix.md` records the last full merge run (2026-11-13:
**96/96** suites on `sm_86`, including both M2 video suites).

## Verification on RTX 3060 (2026-11-17)

Host: RTX 3060 12 GB, driver 580.173.02, CUDA 13.0, `sm_86`, 32 cores.
Clean CUDA build: **168 s** (`-j16`). Everything below is CPU vs GPU on the
same host, `cmp` with no tolerance.

| Check | Result |
|---|---|
| Still 64x64 (`--still` vs `--still-gpu`) | IVF `2b31e378…`, identical |
| Video 240p QP106, 8 frames (`--video` vs `--video-gpu`) | IVF `c48cf494…`, identical |
| Calibration 240p QP189, 60 frames | **`a1d0dba1…`, 25,009 B**, identical; `--video-recon` dumps identical |
| 720p QP90, 24 frames (synthetic source) | identical; GPU **1.79x** faster (6.8 s -> 3.8 s) |
| `build_runtime_test`, `build_entropy_test`, `build_intra_test`, `build_transform_test` | OK |
| Still `build_pipeline_test.sh` | OK |
| CPU cross-architecture (Linux x86_64 vs macOS arm64) | identical IVF |

`--fast-decode` is CPU-only by design (the CUDA backend keeps the historical
decisions so its streams stay CPU-reference exact) and `--cpu-threads` is a CPU
knob; both are rejected by name with `--video-gpu`.

Two defects were found by running this check, not by reading the code:

1. **`AV2ENC_VIDEO_GPU_CALLABLE` was translation-unit local.** After the CLI
   split it was defined only in `main.cpp`, while `cli/validate.cpp` tested it;
   undefined identifiers are 0 in `#if`, so the "backend not linked" rejection
   was compiled in unconditionally and a working CUDA build refused
   `--video-gpu`. Fixed by `src/cli/gpu_callable.h`, included by the CLI TU.
2. **The two product GPU gates failed instead of skipping on Linux.** They
   accepted any `-x` decoder, so they tried to execute the macOS
   `tools/bin/dav2d-macos` and reported "dav2d rejected the stream". They now
   probe the binary and exit 77 when it cannot run.

Linux GPU host verification (second 3060, 2026-11-17): the decoder gap is
closed by building it from source.

```sh
tools/build_dav2d_linux.sh     # static build, installs tools/bin/dav2d
```

On this host it took 75 s and produced sha256 `061b79a8…`, so the gates'
discovery finds a native decoder and their decode half runs. Results:

| Check | Result |
|---|---|
| `gate_video_gpu.sh` | **PASS**, 747 s — every wired mode (temporal layers, B frames, loop filters) plus the rejection matrix |
| `gate_gpu_throughput.sh` | **PASS**, 338 s — zero drift, identical decoded Y4M, 720p QP90 at 1.73 fps on the GPU |

Running them found a real GPU defect that byte-comparison alone would have
missed earlier: the temporal-layer encoder coded only `ref_display[0]` while the
frame header advertised `num_refs`, and its plan omitted `ref_mode = kTwo` and
the `order_hint_bits = 5` rule for a 16-picture GOP. `tlayers4` diverged
(CPU `bf9526ab` vs GPU `957c3a51`). Fixed by mirroring the CPU plan and
reference list; `tlayers4` is now byte-identical (`b06a1128`).
