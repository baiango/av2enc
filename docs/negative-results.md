# Negative results and do-not-repeat

Read this before proposing an implementation. Repeating a listed approach is a
waste of time unless you have new measured evidence.

Status legend:

- **CONFIRMED** — measured or directly observed
- **DEFERRED** — deliberately out of scope for now
- **OPEN** — not yet measured; do not cite as a result

## Bitstream / encoder

| Status | Do not do | Why | Do instead |
|---|---|---|---|
| CONFIRMED | Use `src/entropy/range_coder.*` for the AV2 stream | It is a generic arithmetic coder, not the normative §8.2 state machine; it cannot produce AV2-decodable bytes | Use `src/entropy/av2/` |
| CONFIRMED | Pad 640x360 to 640x384 | Product owner rejected padding; exact 640x360 is now supported | Encode exact dimensions; use edge partition handling |
| CONFIRMED | Use `PARTITION_VERT` at a bottom-edge superblock | `dav2d` returns `Error decoding frame: Invalid argument`; VERT is not allowed when `has_rows` is false | Use `PARTITION_NONE` at the 64x64 SB for the 640x360 bottom edge; 32x16/32x8 implied paths for narrower edges |
| CONFIRMED | Hand-type CDF, scan, DCT, or quant tables | Error-prone and already generated | Use `spec/tables/` and `src/encoder/generate_*.py` |
| CONFIRMED | Enable adaptive CDFs by default | It changes pinned MD5s | Keep `disable_cdf_update=1`; opt in with `--adaptive-cdf` |
| CONFIRMED | Copy AVM/SVT/clave/av2-turbo encoder code | Project constraint is spec-only | Implement from the spec; use decoders as black boxes |
| DEFERRED | Inter frames, loop filters, RDO, rate control | M0 is still-picture only | Finish the CUDA M0 path first |

## CUDA

| Status | Do not do | Why | Do instead |
|---|---|---|---|
| CONFIRMED | Depend on Nsight Compute counters | `ncu` fails with `ERR_NVGPUCTRPERM` in the unprivileged Vast.ai container | Use Nsight Systems 2025.1.3, NVTX, and the built-in profiler |
| CONFIRMED | Use Blackwell-only features | 3060 `sm_86` is primary | Standard CUDA, `__dp4a`, `cp.async`, CUDA graphs; no TMA/clusters/`wgmma`/`tcgen05`/FP4 |
| CONFIRMED | `cudaMalloc` in the block/frame loop | Fragmentation, sync, and no hard cap | Use the 7,500 MB arena (`src/cuda/arena.*`) |
| CONFIRMED | Assume 8/16 tile blocks saturate the GPU | 4x2=8, 8x2=16 blocks on a 28-SM 3060; entropy is serial per tile | Overlap entropy with next-frame compute; buffer output; batch where legal |
| CONFIRMED | Per-bit global writes from the entropy coder | Latency-bound | Buffer tile output in shared/global memory and flush bytes/words |
| CONFIRMED | Use `__fadd_rn` / `__fmul_rn` for `double` DCT math | They are **float** intrinsics and silently down-convert doubles; produced ~1.27M coefficient mismatches | Use `__dadd_rn` / `__dmul_rn` plus `-fmad=false`; compile CPU with `-ffp-contract=off` |
| CONFIRMED | Launch one kernel per transform/helper | ~30 µs dispatch per launch; a 1-block 4-launch pipeline is 271 µs, latency-bound | Batch `nblocks` and fuse stages |
| CONFIRMED | Use one block/launch for intra/recon at video scale | 640x360 raster is 3.6–5.2 ms/frame; 1080p extrapolates to 31–44 ms/frame, almost all launch overhead | Batch `dc_predict_residual_batch` where predictions are frozen, or use a wavefront/CUDA-graph scheduler |
| CONFIRMED | Keep a duplicate fallback transform/intra implementation in the pipeline | Diverges from the bit-exact components and creates two codepaths | Use `transform.*` and `intra.*`; only the 64x64 predictor adapter is allowed |
| CONFIRMED | Use one kernel per M0 stage | A 32x32 block costs ~13 launches; 640x360 uses 2,740 launches and is launch-bound at ~34 ms/frame | Wavefront/CUDA graph batching is the next optimization |
| CONFIRMED | Claim GPU entropy in the CUDA M0 path | Entropy is still `av2::Encoder` on the CPU; `src/cuda/entropy.*` is not wired | Keep CPU entropy until tile-parallel GPU entropy is integrated and gated |
| CONFIRMED | Use float/fixed-point DCT as the unguarded normative path | Float flips 3 coefficients in one 1080p luma frame; fixed flips 2/frame; both need correction/fallback | Use double or the bit-exact hybrid; `bench/cuda_dct_perf/README.md` |
| CONFIRMED | Serialize one DCT block per launch | 65.5 µs/block; 6,120 launches is ~0.4 s/frame before entropy | Batch `nblocks`; the transform API already supports it |
| CONFIRMED | Judge DCT speed by coefficient accuracy alone | The benchmark uses a block-mean predictor, not M0 reconstruction | End-to-end MD5 and `dav2d` remain the gate |
| OPEN | Assume fixed-point Q20/Q60 is safe without correction | Q20 needs the delta correction (37 → 3,084 failures/2.1M otherwise); Q60 needs ceil for 16x16 | Keep double/hybrid until a fixed path is proven end-to-end |
| CONFIRMED | Use `dc_predict_residual_batch` on dependent blocks | Batch jobs assume frozen prediction inputs; later blocks depend on earlier reconstruction | Use the raster path until a wavefront scheduler exists |
| CONFIRMED | Pass tightly packed strides or omit `maxX`/`maxY` | Edge blocks read/write outside the visible plane | Pass the padded plane stride plus visible bounds; see `src/cuda/intra.h` |
| CONFIRMED | Use 64x32 / 32x64 transform modes | CPU reference calls `dct_kernel(64)` while looping to 64, an out-of-bounds read; only diagnostic modes use them | Fix the CPU reference first; default `k32x32` and 64x64 edge path are covered |
| CONFIRMED | Recompute `scale_*`/`quant_factor` with a different expression | Breaks bit-exactness | Use `make_tx_params()` from `src/cuda/transform.h` |
| DEFERRED | Tune 4/6/16/24 GB profiles or Turing `sm_75` | Current scope is 8 GB Ampere | Use `compat8g`; leave other profiles as estimates |
| OPEN | Assume double-precision DCT is fast enough | `bench/cuda_dct_perf` is measuring FP64/FP32/int | Wait for its numbers before choosing precision |

## Benchmark / hardware

| Status | Do not do | Why | Do instead |
|---|---|---|---|
| CONFIRMED | Compare CSS speed across the Mac M4 and the 3060/E5 box | Hardware-specific; the script refuses by default | Rerun AVM on the same machine; `score_speed.py` / `score_tradeoff.py` enforce this |
| CONFIRMED | Rent a 3060 Ti 8 GB as a 3060 speed proxy | Ti has 4864 cores / 448 GB/s vs 3060's 3584 / 240-360 GB/s | Use the 12 GB 3060 with `compat8g` for dev; real 8 GB only for validation |
| CONFIRMED | Treat the 12 GB 3060 with a 7.5 GB cap as 8 GB validation | It is an emulation/proxy | Run `tools/check_target_host.py`; target is 8192 MiB enumerated, cc 8.6 |
| CONFIRMED | Use `ffmpeg -r 60` to fix `dav2d` Y4M `F32:1` | It duplicated 60 frames to 113 and broke alignment | `tools/quality.py` rewrites only the header |
| CONFIRMED | Compare raw ffmpeg PSNR without normalizing frame rate | Wrong result: 35.98 dB instead of 49.62 dB | Use `tools/quality.py` |
| CONFIRMED | Use a 10-frame AVM bitrate as steady-state | Keyframe-dominated | Use the 2-second clip; see `bench/avm_1080p/` |
| CONFIRMED | Assume area-based tile assignment proves 2-GPU scaling | Content skew/throttling: a 10% straggler drops the modeled 1.917× to ~1.77× | Use previous-frame cost-aware LPT or finer tiles; measure per-tile time |
| OPEN | Claim measured multi-GPU speedup | `bench/multi_gpu_plan` is an analytic model; no 2-GPU host | Label as simulation; validate on real hardware |
| OPEN | Use 16x2 tiles without decoding them | Only 1x1 through 8x2 have been verified with `dav2d` | Test 16x2 with the pinned decoder first |

## Input / NVDEC

| Status | Do not do | Why | Do instead |
|---|---|---|---|
| CONFIRMED | Trust `ffmpeg -hwaccel cuda` exit 0 | On this box it silently falls back to software decode; GPU stayed at 0% | Check `pix_fmt == AV_PIX_FMT_CUDA` after `avcodec_open2`, or run `-c:v vp9_cuvid` and treat failure as failure |
| CONFIRMED | Block M0 on NVDEC in this container | `cuvidGetDecoderCaps` returns `CUDA_ERROR_NO_DEVICE` (100) on driver 570.181; NVIDIA container-toolkit multi-GPU regression | Use `Y4mInputSource`/software decode + pinned upload for now |
| CONFIRMED | Alias `/dev/nvidia*` or fight the container from inside | Device cgroup is read-only; aliases still return 100 | Fix at instance launch: expose all host GPUs or the host's last GPU, select with `CUDA_VISIBLE_DEVICES` |
| CONFIRMED | Install/upgrade NVIDIA drivers in the container | Host driver is injected; this breaks compute | Report the infrastructure blocker; do not touch drivers |
| OPEN | Quote NVDEC fps from this box | NVDEC never opened a decoder here | 3060 hardware is viable; throughput is unmeasured until the container is fixed |

Software decode baseline on the E5 box (shared, ranges vary): 4K VP9 ~113 fps,
1080p ~283 fps. Pinned 4K NV12 upload is ~1.05 ms/frame at 11.8 GB/s. The
synthetic FFmpeg CUDA-hwframe handoff works: device pointers,
`host_accessible=no`, 0 host frame bytes, checksum match.

## Build / tooling

| Status | Do not do | Why | Do instead |
|---|---|---|---|
| CONFIRMED | Benchmark or optimize the CPU-only encoder as the product | The deployment target is a weak Vast.ai Xeon plus a 3060; CPU mode is a correctness oracle with no SIMD/threading tuning | Measure on the GPU path; treat CPU-only as reference |
| CONFIRMED | Judge CUDA design by CPU-only speed | CPU-only is deliberately simple and may be far slower | Use GPU stage timings and end-to-end MD5/dav2d gates |
| CONFIRMED | Build/debug the encoder on macOS | Not a supported encoder target | Linux + CUDA; Mac only for AVM/CVVDP |
| CONFIRMED | Support Windows | No Windows machine, POSIX tooling | Good luck; not a target |
| CONFIRMED | Use `rsync` to the Vast.ai box | Not installed | `tools/sync_remote.sh` tar pipe |
| CONFIRMED | Assume `/workspace` persists | It is not a volume | Back up with `tools/backup_to_mac.sh` |
| CONFIRMED | Rely on `perf`/`ncu` for deep profiling | Container restrictions | `nsys` + built-in profiler; `gprof` for host-only |

## Open items — do not claim these

- Multi-GPU scaling is only simulated; no 2-GPU host.
- NVDEC input is still a feasibility study.
- CUDA M0 pipeline is not yet bit-exact; CPU path is the reference.
- Fast-decode profile is deferred until quality/encode speed are settled.
- Actual 8 GB 3060 hardware validation is pending.
