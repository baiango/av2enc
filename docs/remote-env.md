# Remote CUDA environment and baseline

Status of the Vast.ai RTX 3060 box and the first `bench/hello_cuda` baseline.
Measured 2026-11-11 on the instance below. Local repo
`$HOME/work/av2enc` remains the source of truth; nothing here is
persistent (see [Persistence](#disk-and-persistence)).

## Instance

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3060, 11,920 MiB usable, compute capability **8.6** |
| SMs / max SM clock | 28 / 2,100 MHz (observed boost 1,777 → 1,927 MHz under load) |
| Memory clock / bus | 7,501 MHz, 192-bit (≈360 GB/s theoretical) |
| Power / thermals | 110 W cap; 19.7 W idle, 110 W peak observed, 50 °C peak |
| Driver | 570.181 |
| CUDA | 12.8 (runtime and toolkit) |
| nvcc | 12.8.93 (`V12.8.93`) |
| Host OS | Ubuntu 24.04.5 LTS (unprivileged Docker container) |
| Kernel | 5.4.0-216-generic (host kernel, shared/read-only) |
| CPU / RAM | 56 vCPU / 125 GiB, 8 GiB swap |
| Disk | 16 GiB overlay at `/workspace`; 354 MiB used after a clean build |
| PCIe | Gen3 ×16 max (`pcie.link.gen.max=3`), idle link shows Gen1; measured host link ≈12–13 GB/s |
| CMake / Ninja / GCC | 3.28.3 / 1.11.1 / 13.3.0 |

Not installed: `nsys`. Present: `ncu` 2025.1.1.0 and `compute-sanitizer`
2025.1.1.0 in `/usr/local/cuda/bin`.

## Commands

### Sync (from the Mac)

`rsync` cannot create missing parent directories on the receiver. Create them
once, then sync:

```sh
ssh -p <port> <user>@<host> 'mkdir -p /workspace/av2enc/bench/hello_cuda'
rsync -az --exclude build -e "ssh -p <port> -o BatchMode=yes" \
  $HOME/work/av2enc/bench/hello_cuda/ \
  <user>@<host>:/workspace/av2enc/bench/hello_cuda/
```

### Build and run

```sh
ssh -p <port> <user>@<host>
cd /workspace/av2enc/bench/hello_cuda
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/hello_cuda_bench          # correctness + median-of-10 benchmarks
./build/hello_cuda_bench --tiny   # correctness only, for profilers
```

Build products: 6.0 MiB. Clean configure + build takes <20 s on this box.

### Profilers / sanitizer

```sh
ncu --target-processes all --set basic --launch-count 1 \
    --kernel-name regex:dct_naive ./build/hello_cuda_bench --tiny

compute-sanitizer --tool memcheck --error-exitcode 9 \
    ./build/hello_cuda_bench --tiny
```

`ncu` currently fails with `ERR_NVGPUCTRPERM`; `compute-sanitizer` works and
reports 0 errors. Details in [Profiling](#profiling).

### Environment checks

```sh
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,pcie.link.gen.max,pcie.link.width.max,\
clocks.max.sm,clocks.mem,power.draw,temperature.gpu --format=csv
nvcc --version
cmake --version && ninja --version && gcc --version
nproc && free -h && df -h /workspace
vast-capabilities | jq '.instance.workspace_is_volume'   # false
cuobjdump --list-elf build/hello_cuda_bench              # main/kernels .sm_86.cubin
```

## Benchmark suite

`bench/hello_cuda` is standalone and is **not** wired into the root
`CMakeLists.txt` or `src/`.

| File | Purpose |
|---|---|
| `CMakeLists.txt` | C++20, CUDA 12.8, `CMAKE_CUDA_ARCHITECTURES=86`, `-lineinfo`, Release |
| `src/common.cuh` | error macro, fixed-point constants, launcher declarations |
| `src/kernels.cu` | vector add, naive integer DCT (8/16/32), 16×16 SAD, vector copy |
| `src/main.cu` | CPU references, exactness checks, event timing, bandwidth tests |

Workloads:

- **vector_add** — 4,194,304 floats, 256-thread blocks.
- **DCT** — naive direct 2D integer DCT, one CUDA block per transform block,
  one thread per output coefficient, `O(N^4)` work. Basis is a Q6 fixed-point
  cosine table (`round(cos·2^6)`), accumulated in `int32`. **This is a
  throughput baseline, not the AV2 transform** and not usable for encoding.
- **SAD 16×16** — one warp per block, 8 warps per CUDA block, lane-strided byte
  loads, `__shfl_down_sync` reduction.
- **Bandwidth** — 64 MiB buffers: copy-engine `cudaMemcpyAsync` D2D, pinned and
  pageable H2D, pinned D2H, plus a 16-byte-vectorized SM copy kernel. Each
  kernel/copy has a scalar CPU reference and an exactness check (round trip for
  copies).
- Every benchmark warms up once, then reports the **median of 10 timed runs**.
  Each timed run itself loops many iterations (e.g. DCT32 = 5 iterations over
  16,384 blocks), so event resolution is not a factor.

### Build gotcha (fixed)

CMake's CUDA compiler detection initialises `CMAKE_CUDA_ARCHITECTURES` to
`52` in the cache. A plain `set(... CACHE ...)` loses to that existing entry,
which produced `--generate-code=arch=compute_52,code=[compute_52,sm_52]` and
left the GPU to PTX-JIT compute_52 code. `CMakeLists.txt` now hard-pins with
`set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING ... FORCE)`. Verified with:

```sh
grep -o -- '--generate-code=[^ ]*' build/build.ninja | sort -u
# --generate-code=arch=compute_86,code=[compute_86,sm_86]
cuobjdump --list-elf build/hello_cuda_bench
# main.sm_86.cubin, kernels.sm_86.cubin
```

## Correctness

All checks compare against scalar CPU code with exact integer/float equality.
Exit status is non-zero on any mismatch.

| Check | Coverage | Result |
|---|---|---|
| vector_add | 4,194,304 elements | PASS, 0 mismatches |
| dct8x8 | 256 blocks: zero, constant 128, impulse, checkerboard, random | PASS, 0 mismatches |
| dct16x16 | 256 blocks, same patterns | PASS, 0 mismatches |
| dct32x32 | 256 blocks, same patterns | PASS, 0 mismatches |
| sad16x16 | 65,536 random blocks | PASS, 0 mismatches |
| memcpy_bw | pinned H2D, D2D, pageable H2D, SM copy round trips | PASS |
| compute-sanitizer memcheck | full `--tiny` run | **0 errors** |

Final clean-build run: `ALL CHECKS PASSED`, exit code 0.

## Timings (median of 10 runs)

Canonical run after `rm -rf build`; 64 MiB copy buffers.

| Benchmark | Timed work | Median | Min | Derived |
|---|---|---:|---:|---|
| vector_add | 4.19 M floats × 100 iters | 14.796 ms | 14.795 ms | **340.2 GB/s** |
| dct8x8 | 262,144 blocks × 20 | 20.445 ms | 18.002 ms | 3.9 ns/block, 1,050 Gterm/s |
| dct16x16 | 65,536 blocks × 10 | 70.334 ms | 70.253 ms | 107.3 ns/block, 611 Gterm/s |
| dct32x32 | 16,384 blocks × 5 | 138.698 ms | 137.322 ms | 1,693 ns/block, 619 Gterm/s |
| sad16x16 | 262,144 blocks × 50 | 19.859 ms | 19.669 ms | 1.52 ns/block, 169.0 Gpix/s |
| memcpy_d2d | 64 MiB × 30 | 11.778 ms | 11.775 ms | 170.9 GB/s copy |
| copy_kernel_d2d | 64 MiB × 30 | 11.776 ms | 11.774 ms | 171.0 GB/s copy |
| memcpy_h2d_pinned | 64 MiB × 10 | 56.093 ms | 55.779 ms | **11.96 GB/s** |
| memcpy_h2d_pageable | 64 MiB × 10 | 86.302 ms | 85.343 ms | 7.78 GB/s (unstable) |
| memcpy_d2h_pinned | 64 MiB × 10 | 51.429 ms | 51.000 ms | 13.05 GB/s |

Notes:

- **DCT "Gterm/s"** counts input terms: `blocks × iters × N^4`. Each term is
  two integer multiplies, so this is not a MAC/s figure. The naive direct
  transform is expected to be far slower than a separable/butterfly transform;
  use these numbers only as the starting point.
- **D2D copy bandwidth counts bytes copied, not DRAM traffic.** A copy reads and
  writes, so physical traffic is ≈341.9 GB/s — the same ceiling as
  `vector_add` (340.2 GB/s) and the SM copy kernel (171.0 GB/s copy, ≈342 GB/s
  traffic). D2D copies are HBM-bandwidth-bound, not copy-engine-bound.
- Run-to-run medians over 5 process invocations were stable for compute and
  pinned copies: vector_add 14.796–14.800 ms, DCT32 137.6–138.7 ms, SAD
  19.84–19.86 ms, pinned H2D 56.1–56.6 ms. **Pageable H2D is the exception:**
  process medians ranged 85–143 ms (4.7–7.9 GB/s). Use pinned memory.
- Measured host link is PCIe Gen3 ×16, not Gen4: pinned H2D ≈12.0 GB/s,
  D2H ≈13.1 GB/s. Batch frame transfers and keep them pinned.
- GPU reached 1,927 MHz SM / 7,501 MHz memory at 110 W during the run; the
  numbers above are with default clocks, no locking.

## Profiling

| Tool | Status | Notes |
|---|---|---|
| `ncu` 2025.1.1.0 | **Present, blocked** | Connects, then `ERR_NVGPUCTRPERM`: no permission for GPU performance counters on device 0. The instance is an unprivileged container; this is not fixable from inside and was not fought. |
| `nsys` | **Not installed** | `apt-cache policy nsight-systems` offers 2022.4.2 (Ubuntu noble multiverse), which predates this CUDA 12.8 stack. Not installed; no newer NVIDIA-repo package was attempted. |
| `compute-sanitizer` 2025.1.1.0 | **Working** | `memcheck` on `--tiny`: 0 errors. |
| `perf` / eBPF | Unavailable | Kernel profilers are blocked by the container (Vast base-image policy). |

Consequence: detailed per-kernel metrics (occupancy, memory throughput,
stall reasons) are unavailable on this box. Use the event-timed suite plus
`compute-sanitizer`; if Nsight Compute counters become necessary, move to a
VM/bare-metal instance or one where the NVIDIA counter permission is granted.

## Disk and persistence

- `/workspace` is **not** a host volume:
  `vast-capabilities | jq '.instance.workspace_is_volume'` → `false`.
  `stop`/`start` preserves the container filesystem, but **recycle or destroy
  wipes everything**. Never leave unsynced work on the box.
- 16 GiB overlay. Clean state after this work: 354 MiB used, 16 GiB available;
  `bench/hello_cuda/build` is 6.0 MiB. Keep test media tiny (<100 MiB total,
  64×64/256×256 frames as agreed in the README).
- Build artifacts stay remote only; `rsync --exclude build` is intentional.
- Observed files are owned by uid 501 (preserved from macOS) but are writable by
  root; this is harmless.

## Blockers and caveats

1. **`ncu` counters are blocked** (`ERR_NVGPUCTRPERM`). Hard blocker for
   hardware-counter profiling until the instance type changes. Report only;
   do not attempt driver/permission changes.
2. **No `nsys`.** Timing is currently CUDA-event based. The apt candidate is
   from 2022 and was not installed.
3. **Pageable H2D is slow and erratic** (4.7–7.9 GB/s vs 12.0 GB/s pinned).
   All frame I/O must use pinned buffers, one batch per frame.
4. **PCIe Gen3 ×16**, so host↔device traffic is the scarce resource
   (≈12–13 GB/s each way). Avoid per-block transfers; keep reference frames and
   workspaces resident on the device.
5. The naive DCT is **not** a spec transform; do not use its timings to size
   the real DCT kernel without implementing the separable/butterfly version.
6. Driver and CUDA were **not** modified. Do not install NVIDIA drivers or
   upgrade CUDA on this box; the driver is host-injected.

## Bottom line for M0

- All kernels are bit-exact against their scalar references; the toolchain and
  sm_86 code generation are verified.
- The GPU has ample headroom for 64×64/256×256 stills: even the deliberately
  naive DCT32 costs ≈1.7 µs per 32×32 block, and SAD runs at ≈169 Gpix/s.
- Memory ceilings: ≈340 GB/s on-device, ≈12 GB/s pinned H2D, ≈13 GB/s pinned
  D2H. Design the frame pipeline around pinned, batched transfers and
  persistent device workspaces.
