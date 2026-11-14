# Architecture — av2enc

## Language / toolchain (locked)

- **CUDA/Linux is the product target.** macOS is not an encoder target; it is
  only an AVM/CVVDP scoring host.
- Host: **C++20**
- GPU: **CUDA C++**, CUDA 12.8, `sm_86`
- Build: CMake >= 3.24 + Ninja
- CPU scalar code is a correctness/test reference, not a supported encoder
  deliverable.
- Warnings: `-Wall -Wextra -Wpedantic -Wconversion -Wshadow`
- No exceptions in hot paths; use error codes / `std::optional`
- Prefer `std::span`, `std::byte`, `std::unique_ptr`, RAII
- No third-party codec code. Standard library + CUDA + CUB only.

## Process split

The encoder is **GPU-resident**. The CPU is a launch/control processor, not a
second encoder. On the Vast.ai box the CPU is a 2016 Xeon E5-2690 v4 running at
38–43% scaling; any per-block CPU work will starve the 3060.

```
input (NVDEC or pinned upload)
  -> GPU frame pool / reference pool
  -> tile scheduler
       -> intra/inter mode search        (GPU)
       -> RDO / mode decision            (GPU)
       -> transform + quant              (GPU)
       -> reconstruction                 (GPU)
       -> loop filters                   (GPU)
       -> tile symbol generation         (GPU)
       -> per-tile range coder           (GPU, one workgroup per tile)
  -> tile payloads -> OBU/IVF assembly   (CPU, per frame)
  -> final bitstream write               (CPU)
```

CPU owns only:
- input demux / CUDA graph launch
- per-frame rate-control policy (not per-block)
- OBU headers and final file write
- validation harness

GPU owns:
- frame and reference pools, all pixel buffers
- motion estimation, intra search, SAD/SATD
- RDO and mode decision
- forward/inverse transform, quantization, reconstruction
- deblock/CDEF/CCSO/Wiener/GDF/restoration
- tile symbol generation and, where tile-parallel, the range coder itself

No per-block host round trips. No `cudaMemcpy` in the block loop. The only
regular host transfer is the finished tile payloads / bitstream.

## Entropy strategy

AV2's range coder is serial **within a tile**, but tiles are independent. So:

- **Basic:** one tile, CPU entropy coder as a correctness reference, run on the
  Mac/dev host. This is allowed only while bringing up the bitstream.
- **Target:** multiple tiles (2×2, 4×2, …), one CUDA workgroup per tile, each
  workgroup running the normative AV2 §8.2 coder. This keeps the entropy stage
  on the 3060 instead of the E5.
- Tile boundaries cost compression. Measure the BD-rate penalty per tile layout;
  use the smallest tile count that keeps the GPU occupied.
- `dav2d` f9dafbc was verified to decode a 2×2-tile 720p stream, so multi-tile
  is inside the decoder gate. Test each new tile layout before relying on it.
- If a layout fails `dav2d`, fall back to one tile and a single overlapped CPU
  entropy thread; do not silently emit streams the gate rejects.

Implementation status:

- CPU normative §8.2 coder: `src/entropy/av2/`, 116k checks, ASan/UBSan clean.
- CUDA tile coder: `src/cuda/entropy.*`, byte-identical to CPU, 986k checks,
  zero-width/adaptive CDFs included.
- **Measured:** one 2048-symbol GPU tile is ~4.57 ms vs ~100–135 µs on the CPU,
  so the GPU coder is **34–46× slower per tile** at Basic tile counts. 4×2 = 8 and
  8×2 = 16 blocks are below one warp per SM.
- Therefore **Basic keeps CPU entropy**. GPU entropy is only for 256+ tiles,
  batched independent frames, or overlapped next-frame work — not a default.
- The 4 KiB shared staging window is smaller than the full ~46.7 KiB CDF set;
  larger sets fall back to global memory. Reset per-tile CDFs before each
  independent launch; the kernel mutates caller CDF copies.

## CPU-only path is not a performance target

The CPU reference is a correctness oracle. It is deliberately simple: no SIMD,
no CPU threading, no cache tuning. Do not benchmark it as the product and do not
optimize it for speed. The deployment target is a Vast.ai Linux container with a
weak, shared Xeon and a GPU; CPU work is limited to launch/control, OBU headers,
rate-control policy, and (for now) entropy. Pixel work, transforms, prediction,
reconstruction, and filters belong on the 3060.

AVM on the Mac is a quality/rate anchor, not a speed prediction for our encoder.

## Fast decode mode (deferred)

After quality and encode speed are settled, add an optional fast-decode
profile. It trades compression efficiency for decoder work:

- no film grain
- no warps/OBMC/compound prediction
- no B-frames / deep references
- limit partitions to 32x32/64x64, no deep RDO splits
- optionally disable CDEF/GDF/restoration (measure the quality cost)
- tile layout chosen for decoder parallelism, not just encoder occupancy

Measure `decode_fps` / `decode_realtime_x` with `dav2d` on the target hardware.
The gate is: fast-decode mode must still pass `dav2d` and meet a decode-speed
target at the chosen QBS budget. Do not silently disable tools to claim it.

This is not implemented yet. `tools/quality.py` records decode speed now so the
baseline exists.

## GPU targets

Development and feature baseline: **RTX 3060 12 GB (`sm_86`)**. Production:
**RTX 3080 Ti 12 GB (`sm_86`)**. Maximum speed: **RTX 5090 32 GB (`sm_120`)**.
The arena is sized from the device (`min(total, free) - 512 MiB`); there is no
fixed VRAM preset. RTX 4070 Super, RTX 5070 and 8 GB cards are no longer
targets.

There are no fixed "compat" presets. Extra VRAM only changes how much precompute
fits; it never changes the encoded tool set.

Allowed baseline (works on both):

- standard CUDA cores, shared memory, warp shuffles
- `__dp4a` integer dot product
- `cp.async` asynchronous copies
- CUDA graphs, events, streams
- blocks up to 1024 threads; <= 48 KB static/shared memory baseline

Do **not** require:

- tensor cores, FP4/FP6/FP8
- TMA / `cp.async.bulk`
- thread-block clusters / distributed shared memory
- `wgmma`, `tcgen05`, `setmaxnreg`, `griddepcontrol`
- Hopper/Blackwell-only PTX or architecture-specific fast paths

Build fat binaries for both target architectures:

```sh
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DAV2ENC_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="86;120"
```

The runtime selects native SASS; no JIT. Verify with:

```sh
cuobjdump -lelf build-cuda/av2enc
```

Query SM count, shared memory, and compute capability at runtime. Never hardcode
28 SMs or 3060-specific launch geometry. The 3080 Ti and 5090 have more SMs
and bandwidth; the same kernels should scale, with launch-config tuning only if
profiling shows it.

## VRAM budget: 8 GB target

**RTX 3060 12 GB is the development baseline; the 3080 Ti is production.** There is no fixed VRAM ceiling. The
runtime reserves the largest block the device can serve, minus a 512 MiB
headroom, and an optional `--vram-cap <MiB>` bounds it. On an 8 GB card this
lands below the physical limit automatically.

Rules:

- Allocate one persistent arena at first GPU use; no `cudaMalloc` after warm-up.
- Split into persistent regions (frame/reference pools, CDF/state, graphs) and a
  reusable scratch region (RDO candidates, tile buffers).
- Before allocating, estimate: frames × bytes/frame, reference count, tile
  buffers, lookahead, and scratch. Keep the reservation inside the device-derived
  capacity.
- If the requested config does not fit, reduce lookahead/reference count/tile
  batch or fail with a clear message. Do not silently OOM.
- `--vram-cap <MiB>` is an optional upper bound; `0` (default) takes the
  device-derived maximum. `cudaMalloc` is the only hard limit.
- Log `peak_vram_mb`, `vram_capacity_mb`, and `vram_limit_mb` in the profiler
  summary and quality report.
- The 8 GB card has a 128-bit bus (~240 GB/s vs ~360 GB/s). Memory-bound kernels
  (ME/SAD) will be slower; correctness and the sizing rule are the same.

Rough frame sizes (8-bit 4:2:0):

| Resolution | Per frame | 16 refs | Notes |
|---|---:|---:|---|
| 1080p | 3.1 MB | ~50 MB | easy |
| 4K | 12.4 MB | ~200 MB | fits; scratch dominates |
| 8K | 49.8 MB | ~800 MB | tile/stream; not an initial target |

10-bit doubles the frame sizes. 4K10 with a deep reference pool still fits in
7.5 GB; the risk is per-block RDO buffers, so those live in the capped scratch
arena and are reused.

### Precompute scaling

More VRAM may buy more precompute, but only in a deterministic, logged profile.
The default gate remains the **8 GB-compatible profile**.

Historical planning estimates (the encoder no longer selects a preset; these
remain only as precompute sizing examples):

| Profile | VRAM cap | Lookahead | Refs | Tiles | RDO cache | Search | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| `compat4g` | 3,500 MB | 1 | 2 | 2×2 | 1× | ±32 | 1080p recommended |
| `compat6g` | 5,500 MB | 2 | 3 | 2×2 / 4×2 | 1× | ±48 | 1080p, light 1440p |
| `compat8g` (default) | 7,500 MB | 4 | 4 | 4×2 | 1× | ±64 | 8 GB 3060 |
| `auto12g` | 11,000 MB | 12 | 8 | 8×2 | 2× | ±96 | 12 GB 3060 |
| `auto16g` | 15,000 MB | 16 | 12 | 8×2 / 16×2 | 3× | ±112 | 16 GB cards |
| `auto24g` | 20,000 MB | 24 | 16 | 8×2 / 16×2 | 4× | ±128 | 24 GB cards |

`auto` picks by capacity (4/6/8/12/16/24 GB) and memory bandwidth. A 4 GB
GTX 1650 and a 4 GB RTX 3050 are not the same target: the former is Turing
(`sm_75`), the latter Ampere (`sm_86`). The current fatbin is `sm_86;sm_120`.
Supporting Turing requires a separate `sm_75` build and a synchronous-load
fallback for code that assumes `cp.async`; that is not enabled yet.


Priority order when spending extra VRAM:

1. reference pool (quality)
2. lookahead / temporal decisions (quality)
3. RDO candidate cache (quality and speed)
4. tile batching (GPU occupancy)
5. larger search ranges (quality, compute-heavy)

Rules:

- `--vram-cap <MiB>` bounds the device-derived reservation; there is no
  preset switch.
- Estimate `frame_bytes × (refs + lookahead + 2) + scratch + candidate cache`
  before allocating. Keep at least 512 MB headroom.
- 8 GB cards have less bandwidth; extra precompute may not pay off. The 5090’s
  higher bandwidth makes aggressive precompute more attractive.
- Log `precompute_profile`, `precompute_level`, `vram_limit_mb`, and
  `peak_vram_mb`. Compare QBS only at the same profile.
- Precompute may change encoder decisions, but the output must still pass
  `dav2d`; it must never reduce quality just to use memory.

## Project objectives (2026-11-14)

- **CPU mode:** maximize **CVVDP/JOD at 480×240 (240p, 2:1) @ 30 fps and 100 kbps**.
  Quality is the KPI; CPU speed is not. NASA baseline: CVVDP 4.802 JOD at
  100.0 kbps (AVM CBR: 8.452 JOD at 99.5 kbps; SVT-AV1 4.1.0: 9.756 JOD at 100.2 kbps).
- **GPU calibration:** encode the CPU point (240p30, 100 kbps, QP 189) on GPU
  and require minimal drift: CVVDP ±0.05 JOD, bitrate ±2%, PSNR-Y ±0.1 dB.
  Current drift is zero (byte-identical stream, 4df16901...).
- **GPU throughput:** the GPU uses the **same decisions** calibrated at
  240p30/100 kbps and maximizes **encode fps at 1440×720**. The 720p run is a
  cross-encoder throughput benchmark, not a separate decision/quality point.
  The RTX 3080 Ti is the production host and the RTX 5090 the maximum-speed
  host.
- The old 1080p ≥1 fps floor and the 8 GB/4070 Super/5070 targets are retired.

If a frame finishes faster than 1 s, the surplus time is spent on quality, in
this order:

1. RDOQ / trellis quantization
2. deeper partition search
3. more intra/inter modes and candidates
4. larger motion search range and subpel refinement
5. more references/lookahead (only if the 7.5 GB cap allows)
6. multi-pass / temporal analysis

If a frame takes longer than 1 s, reduce in reverse order. Protect the core
tools: partitions, full intra, transforms, deblock, CfL. Never disable those to
hit the speed target.

Controller:

- `--target-fps 1.0 --target-resolution 1080p`
- sliding window over the last N frames, target 0.9–1.1 s/frame with hysteresis
- adjust a quality-level index, not individual kernels ad hoc
- log `quality_level`, `encode_fps`, and `frame_time_ms` in the profiler summary
- first frames may run at a conservative level until the estimate stabilizes

The QBS gate becomes **QBS@1fps**:

```text
max CVVDP  subject to  encode_fps >= 1.0 at 1080p
                       bitrate    <= R_max
```

Compare quality and bitrate at the fixed 1 fps target. Extra frames per second
are not a win; extra CVVDP at the same bitrate is.

## Multi-GPU: 2× 3060 and beyond

Each GPU owns its own device-sized arena (no fixed preset). VRAM is **not
pooled**; a 2× 3060 machine has two independent arenas, not one 24 GB process.

Two modes:

### Tile-parallel (default)

- Split the frame's independent tiles across GPUs: 4×2, 8×2, 16×2 depending on
  resolution and GPU count.
- Each GPU encodes a contiguous tile range with its own CDF state.
- The coordinator merges tile payloads into one frame OBU.
- Reference frames are duplicated on each GPU. At 1080p this is ~3 MB per frame;
  at 4K ~12 MB. Consumer 3060s have no NVLink and GeForce P2P is restricted, so
  use host-mediated copies or duplicate refs; PCIe traffic is negligible at 1 fps.
- For determinism: reset CDFs per tile, disable cross-tile prediction, and use a
  fixed tile order. Loop filters across tile boundaries must be explicitly
  handled or disabled; measure the bitrate cost.
- 8×2 = 16 tiles, so 2 GPUs get 8 each. 4 GPUs need 16×2 or GOP mode.

### GOP-parallel

- Split the sequence into closed GOPs; each GPU encodes independent GOPs.
- Near-linear scaling and no per-frame synchronization.
- Every segment starts with a keyframe, so expect a bitrate penalty at segment
  boundaries. Use longer GOPs to amortize it.

CLI:

```text
--gpus 0,1
--multi-gpu tile|gop
--scale-mode quality|throughput
```

Default is `tile` + `quality`: the 1080p 1 fps floor stays fixed, and extra GPUs
buy more RDO/search/tiles until quality saturates. `throughput` mode raises fps
instead.

Scaling target: 2 GPUs in tile mode should reach **1.8–1.95×** the single-GPU
work at the same quality; report per-GPU utilization and load imbalance. Real
multi-GPU validation is pending hardware. The profiler must aggregate per-GPU
stage times and peak VRAM.

**Multi-GPU development target:** N× RTX 3060 **12 GB** cards, each with its own
device-sized arena. The target-validation requirement of an 8192 MiB enumerated
card remains separate and applies to final single-GPU performance/QBS claims.

## Quality gate

GPU residency must not reduce compression. Speed is only scored after the
stream passes `dav2d` and meets the quality/bitrate budget.

The initial 720p60 QBS budget:

- `CVVDP >= 9.0 JOD`
- `bitrate <= 500 kbps` initially, then `<= 400 kbps`
- `speed >= AVM on the same machine` (rerun AVM on the 3060 box; the M4 number
  is not comparable)

AVM's measured QBS is **9.3623 JOD at 368.6 kbps** under the 400 kbps ceiling.
Our encoder must match or beat that before speed tuning matters. The encoder
must add the Still tools before performance tuning:

1. recursive partitioning (square + rectangular)
2. full intra prediction (directional/smooth/paeth + edge filtering)
3. variable transforms (DCT/ADST/IDTX, TX partitioning)
4. deblocking, then CfL

If a tool is slow on GPU, optimize it. Do not disable it to win CSS.

Measured tile overhead (1 frame, 720p, QP 160, AVM `cd7d8bc3db`):

| Layout | Tiles | IVF bytes | Overhead |
|---|---:|---:|---:|
| 1×1 | 1 | 3,140 | — |
| 2×1 | 2 | 3,194 | +1.7% |
| 2×2 | 4 | 3,224 | +2.7% |
| 4×2 | 8 | 3,288 | +4.7% |
| 8×2 | 16 | 3,301 | +5.1% |

All five decoded with `dav2d`. The byte overhead is not the full BD-rate cost;
measure real video clips before choosing a layout. 4×2 or 8×2 gives the GPU
enough entropy workgroups while keeping the penalty small.

## First milestone (Basic)

Still-picture encoder, 8-bit 4:2:0, one tile:

- fixed 64x64 superblocks
- fixed 32x32 luma coding blocks (chroma 16x16)
- DC intra only
- DCT 32x32 only
- scalar quantization with fixed QP
- no loop filters, film grain, SVC, palette, CfL
- output: IVF containing one frame

Basic status: **achieved 2026-11-11** with the CPU reference path. A 64×64
DC-intra stream (1,138 bytes) decodes with `dav2d` f9dafbc; MD5
`7efc3162bc0a4785685eba291a8096fd`, PSNR 42.13 dB. The CUDA kernels are not
wired into the encode path yet. Adaptive CDFs are disabled for this frame.

The original gate was `avmdec`; `dav2d` is the hard decoder gate now.
Bit-exactness of our reconstruction vs the decoder is the next correctness step.

## Module layout

```
src/
  main.cpp                 CLI and per-frame orchestration only
  io/y4m.{h,cpp}           Y4M reader (8-bit 4:2:0 first)
  io/ivf.{h,cpp}           IVF muxer
  bitstream/bitwriter.*    MSB-first bit writer
  bitstream/obu.*          OBU header + payload framing
  entropy/av2/             normative AV2 §8.2 coder (CPU reference)
  entropy/range_coder.*    generic adaptive range coder (scaffold/tests only)
  cpu/transform.*          scalar reference DCT/quant
  cpu/intra.*              scalar reference DC prediction
  cuda/pipeline.cu         frame pool, tile scheduler, CUDA graph
  cuda/transform.cu        GPU transform/quant kernels
  cuda/intra.cu            GPU intra search kernels
  cuda/entropy.cu          per-tile range coder (one workgroup per tile)
tests/
  y4m_test.cpp
  ivf_test.cpp
  bitwriter_test.cpp
  range_coder_test.cpp
  av2_entropy_test.cpp
```

## CUDA rules

1. Every kernel has a scalar CPU reference; tests assert equality on randomized
   blocks before the kernel is used in the encoder.
2. No `cudaMalloc` / `cudaFree` in per-frame or per-block paths. Preallocate a
   persistent workspace.
3. Use pinned host memory for frame I/O; batch transfers once per frame.
4. One CUDA block per superblock for search kernels; warp reductions for
   SAD/SATD; shared memory for reference windows.
5. Use `cudaGraph` for the per-frame sequence once the kernel set stabilizes.
6. Compile with `-lineinfo`; profile with Nsight Systems/Compute.
7. `CMAKE_CUDA_ARCHITECTURES=86`.
8. Keep frame and reference pools in device memory for the whole sequence. The
   CPU sees only finished tile payloads.
9. No synchronization inside block/tile loops. One graph launch per frame,
   events only for correctness/debug builds.
10. Use NVDEC + CUDA interop for input decode when the source is H.264/HEVC/
    VP9/AV1, so decoded frames never touch the E5.
11. Measure GPU busy %, CPU seconds per frame, and host-device bytes per frame.
    Target: CPU < 5% of wall time and one host transfer per frame.

## Validation

- `avmdec` and `dav2d` are black boxes. Do not read their encoder-side source.
- Unit tests: IVF structure, Y4M parse, bit writer, range-coder round-trip.
- Integration: encode one 64x64 frame, decode with `avmdec`, compare PSNR.
- Keep test media tiny; the remote disk is 16GB and non-persistent.
