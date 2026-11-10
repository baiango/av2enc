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

- **M0:** one tile, CPU entropy coder as a correctness reference, run on the
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

## GPU portability: 3060 primary, 5070 secondary

The **RTX 3060 (sm_86, 12 GB, 28 SMs)** is the baseline. The **RTX 5070
(sm_120, Blackwell)** is a secondary target. The encoder must not depend on
Blackwell-only accelerators.

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

Build fat binaries for both architectures:

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
28 SMs or 3060-specific launch geometry. The 5070 has more SMs and bandwidth; the
same kernels should scale, with launch-config tuning only if profiling shows it.

## VRAM budget: 8 GB target

The 8 GB RTX 3060 is a supported variant. Default hard ceiling: **7,500 MB**
(leaves ~500 MB for the driver/context on an 8 GB card). The 12 GB box must run
with the same cap so compatibility is tested, not assumed.

Rules:

- Allocate one persistent arena at init; no `cudaMalloc` after warm-up.
- Split into persistent regions (frame/reference pools, CDF/state, graphs) and a
  reusable scratch region (RDO candidates, tile buffers).
- Before allocating, estimate: frames × bytes/frame, reference count, tile
  buffers, lookahead, and scratch. Choose a config that fits under the cap.
- If the requested config does not fit, reduce lookahead/reference count/tile
  batch or fail with a clear message. Do not silently OOM.
- `--vram-limit-mb` default `7500`; `0` means auto (90% of device memory, for
  12 GB experiments only).
- Log `peak_vram_mb`, `vram_capacity_mb`, and `vram_limit_mb` in the profiler
  summary and quality report.
- The 8 GB card has a 128-bit bus (~240 GB/s vs ~360 GB/s). Memory-bound kernels
  (ME/SAD) will be slower; correctness and the cap are the same.

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

Initial profiles (starting points, tune with profiling):

| Profile | VRAM cap | Lookahead | Refs | Tiles | RDO cache | Search |
|---|---:|---:|---:|---:|---:|---:|
| `compat8g` (default) | 7,500 MB | 4 | 4 | 4×2 | 1× | ±64 |
| `auto12g` | 11,000 MB | 12 | 8 | 8×2 | 2× | ±96 |
| `auto24g` | 20,000 MB | 24 | 16 | 8×2 / 16×2 | 4× | ±128 |

Priority order when spending extra VRAM:

1. reference pool (quality)
2. lookahead / temporal decisions (quality)
3. RDO candidate cache (quality and speed)
4. tile batching (GPU occupancy)
5. larger search ranges (quality, compute-heavy)

Rules:

- `--vram-limit-mb` and `--precompute <compat8g|auto12g|auto24g|auto>` select the
  profile. `auto` picks from total VRAM and memory bandwidth.
- Estimate `frame_bytes × (refs + lookahead + 2) + scratch + candidate cache`
  before allocating. Keep at least 512 MB headroom.
- 8 GB cards have less bandwidth; extra precompute may not pay off. The 5070’s
  higher bandwidth makes aggressive precompute more attractive.
- Log `precompute_profile`, `precompute_level`, `vram_limit_mb`, and
  `peak_vram_mb`. Compare QBS only at the same profile.
- Precompute may change encoder decisions, but the output must still pass
  `dav2d`; it must never reduce quality just to use memory.

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
must add the M1 tools before performance tuning:

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

## First milestone (M0)

Still-picture encoder, 8-bit 4:2:0, one tile:

- fixed 64x64 superblocks
- fixed 32x32 luma coding blocks (chroma 16x16)
- DC intra only
- DCT 32x32 only
- scalar quantization with fixed QP
- no loop filters, film grain, SVC, palette, CfL
- output: IVF containing one frame

M0 is done when `avmdec` decodes the IVF and the decoded YUV is close to the
source. Bit-exactness of our reconstruction vs the decoder is the next gate.

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
