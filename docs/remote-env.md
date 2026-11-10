# Remote CUDA environment and baseline

Status of the Vast.ai RTX 3060 box and the `bench/` baseline. Measured
2026-11-11.

**Focus:** the primary target is **8 GB Ampere (sm_86)** with a **7500 MiB**
VRAM cap. This 12 GB box is the development machine, run under the same 7500 MiB
cap. The 8 GB claim is **12 GB emulation, actual 8 GB hardware pending**.
RTX 5070 is build/smoke only. 4/6/16/24 GB and Turing are out of scope.

## Instance

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3060, 11,920 MiB reported by CUDA, sm_86 |
| SMs / max SM clock | 28 / 2,100 MHz (observed boost 1,777 → 1,927 MHz under load) |
| Memory clock / bus | 7,501 MHz, 192-bit (≈360 GB/s; query estimate 360.0 GB/s) |
| Power / thermals | 110 W cap; 19.7 W idle, 110 W peak observed, 50 °C peak |
| Driver | 570.181 |
| CUDA / nvcc | 12.8 runtime and toolkit / 12.8.93 |
| Host OS | Ubuntu 24.04.5 LTS (unprivileged Docker container) |
| Kernel | 5.4.0-216-generic (host kernel, shared/read-only) |
| CPU / RAM | 56 vCPU / 125 GiB, 8 GiB swap |
| Disk | 16 GiB overlay at `/workspace`; **~1.5 GiB used, 15 GiB free** |
| PCIe | Gen3 ×16 max; measured host link ≈12–13 GB/s |
| CMake / Ninja / GCC | 3.28.3 / 1.11.1 / 13.3.0 |
| Nsight Systems | **2025.1.3.140 installed and verified** |
| Nsight Compute | 2025.1.1.0 installed; counters blocked |
| compute-sanitizer | 2025.1.1.0, working |

## Commands

### Sync (from the Mac)

```sh
ssh -p <port> <user>@<host> 'mkdir -p /workspace/av2enc/bench/hello_cuda'
rsync -az --exclude build -e "ssh -p <port> -o BatchMode=yes" \
  $HOME/work/av2enc/bench/hello_cuda/ \
  <user>@<host>:/workspace/av2enc/bench/hello_cuda/
```

Repeat for `bench/vram_budget/` and `bench/vram_config/` with the same parent
directory already created.

### Build and run the main benchmark

```sh
ssh -p <port> <user>@<host>
cd /workspace/av2enc/bench/hello_cuda
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
CUDA_DISABLE_PTX_JIT=1 ./build/hello_cuda_bench
```

Default architectures are `86;120`. The 3060 selects `sm_86`; the sm_120 slice
is present for the 5070 build/smoke target. Override with
`-DCMAKE_CUDA_ARCHITECTURES=86` for 3060-only iteration.

### VRAM / profile checks

```sh
cd /workspace/av2enc/bench/vram_budget
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
VRAM_LIMIT_MB=7500 ./build/vram_budget

cd /workspace/av2enc/bench/vram_config
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
./build/vram_config --width 3840 --height 2160 --bit-depth 10 --cap-mb 7500
```

### Profilers / sanitizer

```sh
# Nsight Systems: verified working (CUDA trace, kernel summary)
nsys profile --trace=cuda --sample=none --force-overwrite=true \
    -o /tmp/av2enc_nsys ./build/hello_cuda_bench --tiny
nsys stats --report cuda_gpu_kern_sum /tmp/av2enc_nsys.nsys-rep

# Nsight Compute: connects, then fails with ERR_NVGPUCTRPERM
ncu --target-processes all --set basic --launch-count 1 \
    --kernel-name regex:dct_naive ./build/hello_cuda_bench --tiny

# Sanitizer: 0 errors
compute-sanitizer --tool memcheck --error-exitcode 9 \
    ./build/hello_cuda_bench --tiny
```

`nsys` produced a 71 KiB report and a valid `cuda_gpu_kern_sum` table listing
`dct_naive_kernel<32>`, `copy_kernel`, `vector_add_kernel`, `sad16_kernel`, etc.
No GPU counter permission is needed for this trace.

## Benchmark suite

| Bench | Purpose |
|---|---|
| `bench/hello_cuda` | vector add, naive integer 8/16/32 DCT, 16×16 SAD, D2D/H2D/D2H copies, exactness checks, median-of-10 timings |
| `bench/vram_budget` | 7500 MiB arena allocation/touch, cap rejection, clean over-allocation failure |
| `bench/vram_config` | frame/persistent/scratch/cache estimator and `compat8g` selection |

`hello_cuda` is standalone and deliberately not wired into the root CMake
project. The naive DCT is a throughput baseline, not the AV2 transform.

## Build / architecture verification

```sh
$ cuobjdump -lelf build/hello_cuda_bench
ELF file    1: main.sm_86.cubin
ELF file    2: main.sm_120.cubin
ELF file    3: kernels.sm_86.cubin
ELF file    4: kernels.sm_120.cubin

$ CUDA_DISABLE_PTX_JIT=1 ./build/hello_cuda_bench --tiny
device: NVIDIA GeForce RTX 3060 (sm_86), 11920 MiB, 28 SMs, clock 1.78 GHz
shared memory: 48 KiB/block default, 99 KiB/block opt-in, 100 KiB/SM
loaded kernel image: sm_86
...
ALL CHECKS PASSED
```

With PTX JIT disabled, a successful run proves the `sm_86` SASS executed. The
`sm_120` slice is compiled and inspected only; no RTX 5070 host is available.

## Correctness

| Check | Coverage | Result |
|---|---|---|
| vector_add | 4,194,304 elements | PASS, 0 mismatches |
| dct8/16/32 | 256 blocks each: zero, constant, impulse, checkerboard, random | PASS, 0 mismatches |
| sad16x16 | 65,536 blocks | PASS, 0 mismatches |
| memcpy_bw | pinned H2D, D2D, pageable H2D, SM copy round trips | PASS |
| `compute-sanitizer` memcheck | full `--tiny` run | 0 errors |
| `vram_budget` at 7500 MiB | arena/touch/cap/over-allocation/recovery | PASS |
| `vram_config` at cap 7500 | `compat8g` selection | OK |

## Timings (median of 10 runs, fatbin build)

The 3060 executes the sm_86 image; PTX JIT was disabled for this run. 64 MiB
copy buffers.

| Benchmark | Timed work | Median | Min | Derived |
|---|---|---:|---:|---|
| vector_add | 4.19 M floats × 100 iters | 14.800 ms | 14.798 ms | 340.1 GB/s |
| dct8x8 | 262,144 blocks × 20 | 20.348 ms | 17.558 ms | 3.9 ns/block, 1,055 Gterm/s |
| dct16x16 | 65,536 blocks × 10 | 70.268 ms | 70.077 ms | 107.2 ns/block, 611 Gterm/s |
| dct32x32 | 16,384 blocks × 5 | 138.367 ms | 137.032 ms | 1,689 ns/block, 621 Gterm/s |
| sad16x16 | 262,144 blocks × 50 | 19.861 ms | 19.646 ms | 1.52 ns/block, 169.0 Gpix/s |
| memcpy_d2d | 64 MiB × 30 | 11.780 ms | 11.778 ms | 170.9 GB/s copy |
| copy_kernel_d2d | 64 MiB × 30 | 11.783 ms | 11.778 ms | 170.9 GB/s copy |
| memcpy_h2d_pinned | 64 MiB × 10 | 56.254 ms | 55.723 ms | 11.93 GB/s |
| memcpy_h2d_pageable | 64 MiB × 10 | 106.263 ms | 104.950 ms | 6.32 GB/s (varies 4.7–7.9 across runs) |
| memcpy_d2h_pinned | 64 MiB × 10 | 51.406 ms | 51.257 ms | 13.05 GB/s |

Notes:

- DCT "Gterm/s" counts input terms (`blocks × iters × N^4`); each term costs two
  integer multiplies. It is not a MAC/s figure.
- D2D copy bandwidth counts bytes copied; physical traffic is 2× (~342 GB/s),
  matching the vector-add triad. D2D is HBM-bound, not copy-engine-bound.
- Pageable H2D is unstable; use pinned memory.
- PCIe Gen3 ×16 limits H2D to ~12 GB/s and D2H to ~13 GB/s.

## VRAM: `compat8g` at 7500 MiB (measured)

`vram_budget` on the 12 GB dev box:

| Metric | Value |
|---|---:|
| Total VRAM | 11,920 MiB |
| Free before | 11,795 MiB |
| Baseline used | 125 MiB |
| Arena (`limit - 256 MiB`) | 7,244 MiB |
| Free after alloc/touch | 4,551 MiB |
| Peak used | **7,369 MiB** |
| Spare under cap | 130 MiB |
| Free after release | 11,795 MiB (0 MiB delta) |
| 512 MiB over-cap request | rejected |
| 4,807 MiB `cudaMalloc` beyond free | clean `cudaErrorMemoryAllocation` |

`vram_config` with `--cap-mb 7500` selects `compat8g`:

- 4K10 estimate: 980.86 MiB total (persistent 239.28, scratch 205.84,
  candidate 23.73, headroom 512).
- Result: `RESULT: OK - profile 'compat8g' with cap 7500 MiB`.

**8 GB claim:** 12 GB emulation, actual 8 GB hardware pending. The same peak
would leave ~823 MiB on an 8 GB card; that margin has not been tested on real
8 GB hardware. See `docs/vram.md`.

## Profiling

| Tool | Status | Notes |
|---|---|---|
| `nsys` 2025.1.3.140 | **Working** | CUDA trace + kernel summary verified on the 3060. No counter permission needed. |
| `ncu` 2025.1.1.0 | **Present, blocked** | `ERR_NVGPUCTRPERM`: no GPU performance-counter permission in this unprivileged container. Not fought. |
| `compute-sanitizer` 2025.1.1.0 | **Working** | `memcheck` on `--tiny`: 0 errors. |
| `perf` / eBPF | Unavailable | Kernel profilers blocked by the container. |

## Disk and persistence

- `/workspace` is **not** a host volume:
  `vast-capabilities | jq '.instance.workspace_is_volume'` → `false`.
  `stop`/`start` preserves the container; recycle/destroy wipes it.
- 16 GiB overlay, ~1.5 GiB used, 15 GiB free. Build dirs are ~6 MiB each.
- Keep test media tiny (<100 MiB total; 64×64/256×256 frames).
- `rsync --exclude build` is intentional.

## Blockers and caveats

1. **`ncu` counters blocked** (`ERR_NVGPUCTRPERM`). Use `nsys` tracing and the
   event-timed suite; detailed occupancy/stall metrics are unavailable.
2. **Actual 8 GB hardware pending.** All 8 GB statements are 12 GB emulation
   under the 7500 MiB cap.
3. **RTX 5070 runtime pending host.** sm_120 SASS is present but never executed.
4. **Pageable H2D is slow and erratic**; use pinned buffers and batch per frame.
5. **PCIe Gen3 ×16**; avoid per-block host transfers.
6. Driver and CUDA were **not** modified. Do not install drivers or upgrade
   CUDA on this box.
7. The naive DCT is not a spec transform; do not size the real DCT kernel from
   its timings.

## Bottom line

- The primary configuration is `compat8g` with a **7500 MiB** cap; on this
  12 GB dev box it peaks at **7369 MiB**, leaves **130 MiB** of cap headroom,
  and releases all VRAM cleanly.
- All `hello_cuda` checks are bit-exact, and the 3060 executes `sm_86` with PTX
  JIT disabled.
- Actual 8 GB and RTX 5070 validation remain pending hardware. See
  `docs/portability.md` and `docs/vram.md`.
