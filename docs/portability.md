# Portability — RTX 3060/3080 Ti (sm_86) and RTX 5090 (sm_120)

## Focus freeze

| Role | Hardware | Architecture | Status |
|---|---|---|---|
| **Primary target** | **RTX 3060 8 GB (Ampere)** | **sm_86** | **8 GB hardware pending**; dev-box emulation only |
| **Dev box** | RTX 3060 12 GB | sm_86 | Validated under the 7500 MiB cap |
| Maximum-speed target | RTX 5090 | sm_120 | production throughput host |
| Out of scope | 4/6/16/24 GB cards, Turing | — | Not a target; profile notes are estimates only |

The encoder is built from one source tree. **sm_86 is the baseline feature
set**; sm_120 is a second fatbin slice for build/smoke coverage, not a
performance target. The primary runtime budget is the 8 GB card with a
**7500 MiB VRAM cap**; the 12 GB dev box runs the same cap.

The RTX 3060 12 GB is the development host; the 3080 Ti and 5090 are production/max-speed targets. The old 8 GB claim is
**12 GB emulation, actual 8 GB hardware pending**.

## Build

`bench/hello_cuda/CMakeLists.txt` sets the default before `project()` so
CMake's CUDA detection cannot seed its own `sm_52` default:

```cmake
set(CMAKE_CUDA_ARCHITECTURES "86;120" CACHE STRING "CUDA architectures")
```

An explicit `-D` still wins:

```sh
cd /workspace/av2enc/bench/hello_cuda

# Default fatbin: sm_86 (primary) + sm_120 (build/smoke)
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cuobjdump -lelf build/hello_cuda_bench

# 3060-only fast iteration
cmake -G Ninja -S . -B build86 -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=86

# RTX 5090 (sm_120) build
cmake -G Ninja -S . -B build120 -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=120
```

Verified output of the default build:

```text
$ cuobjdump -lelf build/hello_cuda_bench
ELF file    1: main.sm_86.cubin
ELF file    2: main.sm_120.cubin
ELF file    3: kernels.sm_86.cubin
ELF file    4: kernels.sm_120.cubin

$ cuobjdump -lptx build/hello_cuda_bench
PTX file    1: main.sm_86.ptx
PTX file    2: main.sm_120.ptx
PTX file    3: kernels.sm_86.ptx
PTX file    4: kernels.sm_120.ptx
```

Both SASS slices are present; PTX is also emitted because CMake's default code
list is `[compute_XX,sm_XX]`. Validate SASS execution, not PTX, with PTX JIT
disabled:

```sh
$ CUDA_DISABLE_PTX_JIT=1 ./build/hello_cuda_bench --tiny
device: NVIDIA GeForce RTX 3060 (sm_86), 11920 MiB, 28 SMs, clock 1.78 GHz
shared memory: 48 KiB/block default, 99 KiB/block opt-in, 100 KiB/SM
loaded kernel image: sm_86
...
ALL CHECKS PASSED
```

`loaded kernel image` comes from `cudaFuncGetAttributes().binaryVersion`, i.e.
the architecture of the image the driver selected. With PTX JIT disabled, a
successful run proves the `sm_86` cubin executed.

## Allowed baseline features

These are permitted in the common path and compile for sm_86 and sm_120:

- Standard CUDA cores: FP32 and INT32 arithmetic.
- Shared memory (static and dynamic).
- Warp shuffles: `__shfl_sync`, `__shfl_down_sync`, `__shfl_up_sync`,
  `__shfl_xor_sync`.
- `__dp4a` (sm_61+).
- `cp.async` (sm_80+; baseline because sm_86 is the floor).
- CUDA graphs, CUDA events, streams.
- `atomicAdd`, `__ldg`, vector loads/stores, constant memory, `__syncthreads`.
- Runtime device queries and the CUDA occupancy API.

## Forbidden dependencies

Do not use any of these in required code, and do not add an unconditional
`#if __CUDA_ARCH__ >= 900` path:

- Tensor cores: WMMA, `mma`, `ldmatrix`, FP4/FP8 tensor operations.
- TMA and `cp.async.bulk`.
- Thread block clusters and distributed shared memory.
- `wgmma` (Hopper), `tcgen05` (Blackwell), `setmaxnreg` (Hopper+).
- Hopper/Blackwell-only PTX instructions or intrinsics.
- Libraries/APIs requiring sm_90/sm_100/sm_120 at build or run time.
- Any Blackwell-only accelerator as a hard requirement.

Architecture-specific optimizations may be added later only if guarded and a
correct sm_86 fallback exists. Until then, the common kernels use the feature
set above.

## Runtime device query — never hardcode 28 SMs

```cpp
cudaDeviceProp prop{};
cudaGetDeviceProperties(&prop, 0);
const int sms = prop.multiProcessorCount;          // 28 on RTX 3060
const int cc  = prop.major * 10 + prop.minor;      // 86 on RTX 3060

int optin_kib = 0;
cudaDeviceGetAttribute(&optin_kib,
                       cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);

cudaFuncAttributes fa{};
cudaFuncGetAttributes(&fa, kernel);
const int loaded_arch = fa.binaryVersion;          // 86 or 120
```

Use `prop.multiProcessorCount` for grid sizing and
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` for launch configuration.
`28` may appear in documentation as the measured RTX 3060 value; it must never
appear as a launch or allocation constant. Query `sharedMemPerBlock`,
`sharedMemPerMultiprocessor`, and the opt-in attribute instead of assuming
48/100 KiB. The VRAM limit is runtime configuration (`VRAM_LIMIT_MB=7500`),
not a compile-time assumption.

`hello_cuda` prints:

```text
device: NVIDIA GeForce RTX 3060 (sm_86), 11920 MiB, 28 SMs, clock 1.78 GHz
shared memory: 48 KiB/block default, 99 KiB/block opt-in, 100 KiB/SM
loaded kernel image: sm_86
```

## RTX 5090 — maximum-speed target

**The RTX 5090 is the maximum-speed target.** The `sm_120` SASS is built and
inspected with `cuobjdump`; execution is pending a 5090 host. Do not report a
5090 as validated until this smoke run passes.

Smoke procedure when a 5090 host is available:

```sh
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv
# Expected: NVIDIA GeForce RTX 5090, 12.0, ~32768 MiB, driver >= 570

cd /workspace/av2enc/bench/hello_cuda
rm -rf build
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cuobjdump -lelf build/hello_cuda_bench | grep -E 'sm_(86|120)'
CUDA_DISABLE_PTX_JIT=1 ./build/hello_cuda_bench --tiny
# Expected: compute capability 12.0, loaded kernel image: sm_120,
#           ALL CHECKS PASSED
```

This is a build/smoke check only. Performance numbers from a 5090 are not yet
measured. `nsys` tracing is expected to work; `ncu` counters depend on host
permissions and may still fail with `ERR_NVGPUCTRPERM`.

## Retired 8 GB target

8 GB cards, the RTX 4070 Super and the RTX 5070 are no longer targets. The
`compat8g` 7500 MiB cap is historical and is not part of the current target
matrix. The arena is sized from the device on the 3060 12 GB, 3080 Ti and 5090.

## What was actually tested

| Item | Status |
|---|---|
| RTX 3060 12 GB, full suite, sm_86 SASS (PTX JIT disabled) | **PASS** |
| 7500 MiB cap / `compat8g` on the 12 GB dev box | **PASS** (emulation) |
| 8 GB cards / 4070 Super / 5070 | **Retired targets** |
| `sm_120` SASS present for the RTX 5090 | **PASS (compiled/inspected only)** |
| RTX 5090 execution | **PENDING HOST** |
| 4/6/16/24 GB cards | **Out of scope; estimates only** |
| Turing sm_75 | **Out of scope; unsupported** |
| Nsight Systems tracing on the 3060 | **PASS** (2025.1.3.140) |
| Nsight Compute counters on the 3060 | Blocked (`ERR_NVGPUCTRPERM`) |

## Toolchain note

`sm_120` requires CUDA 12.8 or newer. CUDA 12.8.93 emits both sm_86 and sm_120
and runs on this driver (570.181). Do not upgrade the driver or CUDA to add a
target; a newer toolkit is only needed if a new architecture is deliberately
added, and that is a matrix decision, not an incidental upgrade.
