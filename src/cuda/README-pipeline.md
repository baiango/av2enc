# CUDA M0 still-picture pipeline

End-to-end CUDA port of the M0 still-picture path. The M0 partition tree and
the §5.20 entropy syntax stay on the CPU; the GPU performs prediction, residual,
forward DCT, quantisation, dequantisation, inverse DCT and reconstruction
through the shared component layer.

## Status

**PASS** on the RTX 3060 development box (sm_86, CUDA 12.8, driver 570.181):

| Case | Frame payload | CPU/GPU frame MD5 | `dav2d` decoded MD5 | PSNR (Y / avg) |
|---|---|---:|---|---|
| 64x64 | 1,344 B | `c83ace80774c4349d1d8e02549d23537` | `c9294fa48f70fc82a26c64908ddc0b12` | 41.051 / 42.238 dB |
| 640x360 | 38,351 B | `c479f7fa89800698b6cec3cfba47051f` | `0d90c958b8dd5d12c6ab2eef2a2c19bf` | 38.177 / 39.797 dB |

The GPU frame payloads are **byte-identical** to
`av2enc::basic::encode_still_picture()`, and the reconstructed planes are
**bit-exact** (GPU-vs-CPU PSNR is infinite on every plane). The payload/PSNR rows were re-recorded on the CPU reference for the
public-domain fixtures (`tests/data/basic_64x64_1f.y4m`,
`tests/data/basic_640x360_1f.y4m`); the historical tile-only payload
accounting from the original GPU run is not reproduced here. The decoded MD5s
come from the pinned `dav2d` f9dafbc build through
`tools/validate_dav2d.sh` with `EXPECTED_MD5` set.

Steady-state timing (one warm-up encode first; RTX 3060, single frame):

| Case | CPU reference | GPU pipeline | Speedup | Kernel launches |
|---|---:|---:|---:|---:|
| 64x64 | 5.24 ms | 1.34 ms | 3.90x | 52 |
| 640x360 | 42.03 ms | 37.67 ms | 1.12x | 2,740 |

Measured when the dev GPU was idle. The box is shared: under concurrent GPU
load the same runs measured 2.1-4.2 ms (64x64) and ~64 ms (640x360). Correctness
is deterministic; timing is indicative. The 640x360 path is still launch-bound
(2,740 component launches), so the speedup is small.

The first call in a process additionally pays one-time `Context`/`Arena`/
`StagingBuffer` setup and CUDA module load (~300 ms on this box); the table is
per-frame after that.

## Files

| File | Role |
|---|---|
| `basic_pipeline.h` | Public encode entry point, POD block/job descriptors, timing structs |
| `basic_pipeline.cpp` | M0 partition planner, host §5.20 syntax writer, OBU assembly, PSNR-side cropping |
| `basic_pipeline.cu` | Runtime/transform/intra orchestration, pinned staging, CUDA events, 64x64 recon adapter |
| `basic_pipeline_test.cpp` | CPU-vs-GPU harness: MD5, PSNR, timing, IVF output |
| `build_pipeline_test.sh` | Build + run + `dav2d` MD5 gate |

## Architecture

```
StillPicture
   │
   ├─ PartitionPlanner (host)  ── mirrors basic_encoder.cpp §5.20.3 tree
   │     ops: clear / do_split / rect_type / block, in coding order
   │     blocks: mi position, luma/chroma sizes, tx_sz_ctx, coeff offsets
   │
   ├─ run_gpu_job (basic_pipeline.cu)
   │     Context + Arena + pinned StagingBuffer        (runtime component)
   │     for each real block, raster order:
   │        dc_pred_value / dc_predict_residual        (intra component)
   │        forward_transform / inverse_transform      (transform component)
   │        add_residual_clip_to_plane                  (intra component)
   │        U/V transform stages batched with nblocks=2 (same TxParams)
   │     64x64 luma: downsample_residual_64x64 + recon adapter
   │     one cudaStreamSynchronize per frame
   │
   ├─ HostEntropyWriter (host)
   │     av2::Encoder + the same CDFs/contexts/scan order as basic_encoder.cpp
   │     coefficients and predictions came from the GPU
   │
   └─ av2enc::basic::assemble_basic_encoder_picture()  (existing OBU writer)
```

Important properties:

- **No arithmetic duplication.** DCT/quantisation/dequantisation/inverse DCT,
  DC prediction, edge clamping and source clipping are delegated to
  `src/cuda/transform.*` and `src/cuda/intra.*`. The only pipeline-local kernel
  is `recon_64x64_adjusted_kernel`, a 20-line adapter that applies the
  component's adjusted 32x32 inverse residual with the normative 2x2
  duplication. It exists because `transform::reconstruct_64x64()` takes the
  predictor as a host scalar, while this pipeline keeps the predictor
  device-resident.
- **Exact M0 syntax.** The host writer ports the serial §5.20 syntax from
  `basic_encoder.cpp` (partition contexts, `txb_skip`, EOB, coefficient base/BR,
  `read_quant` escapes, DC sign) and keeps the same CDF tables and order. The
  entropy component (`src/cuda/entropy.*`) is a device tape executor; entropy
  remains on the CPU here, as planned for the first CUDA path.
- **Memory.** One persistent `Arena` reservation (`compat8g`, 7,500 MB hard
  cap), per-frame bump allocations, no `cudaMalloc` in the hot path. Source,
  coefficients and recon use one pinned upload staging block and one pinned
  download staging block. `cudaMalloc` appears only inside runtime setup.
- **Synchronisation.** Component kernels are enqueued on the runtime stream;
  exactly one `Context::synchronize()` per frame. CUDA events measure upload,
  kernel and download time without extra syncs.
- **64x32 / 32x64 blocks.** Only the zero-residual diagnostic modes
  (`k64x64VertNone`, `k64x64HorzNone`) can emit those coding-block sizes; they
  never reach the GPU because `real_coeffs` is false. The M0 target tree
  (`k32x32`) emits 32x32 and 64x64 `PARTITION_NONE` blocks, both covered. A
  guard reports a clear error if a future tree ever emits an unsupported
  transform size.

## Build and run

On the CUDA host (from the repository root):

```sh
src/cuda/build_pipeline_test.sh
```

The script compiles the host orchestrator/test with `g++`, the kernels and
shared components with `nvcc --fmad=false -arch=sm_86`, runs both Y4M cases,
and then validates each GPU IVF with the pinned `tools/bin/dav2d`:

```text
md5: c9294fa48f70fc82a26c64908ddc0b12 (ok)
md5: 0d90c958b8dd5d12c6ab2eef2a2c19bf (ok)
PASS: CUDA M0 pipeline matches the CPU reference and decoded MD5 gates
```

Useful environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `CUDA_ARCH` | `sm_86` | nvcc architecture |
| `BUILD_DIR` | `$TMPDIR/av2enc-m0-pipeline-build` | build/output location (outside the repo) |
| `CXX` / `NVCC` | `g++` / `nvcc` | host/CUDA compilers |
| `DAV2D` | `tools/bin/dav2d` | decoder override |

Manual harness use (writes `gpu_<name>.ivf` and `cpu_<name>.ivf`):

```sh
basic_pipeline_test --y4m tests/data/basic_640x360_1f.y4m --out-dir /tmp/out --name 640x360
basic_pipeline_test --width 128 --height 96 --out-dir /tmp/out --name synth
basic_pipeline_test --y4m tests/data/basic_64x64_1f.y4m --partition 64x64-single
```

The harness prints:

- CPU and GPU frame-payload MD5s and a byte-for-byte comparison,
- reconstructed-plane exactness / first difference,
- GPU-vs-CPU and GPU-vs-source PSNR for Y/U/V and the weighted average,
- CPU, upload, kernel, download, entropy and total timing,
- the IVF file MD5s.

Exit code is non-zero unless the frame payloads and reconstructed planes are
bit-identical.

## Validation evidence

CPU-reference capture on the public-domain fixtures (the GPU run was
byte-identical at every point; the pre-swap GPU capture was dropped because its
byte counts described the retired promo fixture):

```
== 64x64 ==
  CPU frame: 1344 bytes, MD5 c83ace80774c4349d1d8e02549d23537
  bitstream: matches tests/data/basic_64x64_1f.y4m reference
  recon:     bit-exact
  GPU vs source      Y  41.051 dB  U  45.841 dB  V  47.384 dB  avg  42.238 dB
  md5: c9294fa48f70fc82a26c64908ddc0b12 (ok)

== 640x360 ==
  CPU frame: 38351 bytes, MD5 c479f7fa89800698b6cec3cfba47051f
  bitstream: matches tests/data/basic_640x360_1f.y4m reference
  recon:     bit-exact
  GPU vs source      Y  38.177 dB  U  49.644 dB  V  50.396 dB  avg  39.797 dB
  md5: 0d90c958b8dd5d12c6ab2eef2a2c19bf (ok)
```

Additional bit-exact checks run during bring-up: `k64x64Single` on 64x64 and
640x360, `k64x64VertNone` on 64x64, `k32x32HorzFirst` on 640x360, synthetic
patterns at 48x48, 66x34, 100x50, 128x96, and 100x50 with `k32x32HorzFirst`.
All matched the CPU frame bytes and reconstructed planes exactly.

## Known limitations / next steps

1. **Launch-bound component path.** The shared components expose one kernel per
   stage, so a 32x32 coding block costs 13 launches (luma prediction/residual,
   transform, reconstruction; U/V prediction/residual, batched transform with
   `nblocks=2`, two reconstructions). The 640x360 frame is 2,740 launches and
   only ~1.1x faster than the CPU reference. The planned fix is a wavefront
   scheduler that batches transforms/reconstruction across independent blocks
   (and eventually a CUDA graph); the M0 bitstream does not change.
2. **Entropy on CPU.** The host syntax writer is a port of `basic_encoder.cpp`'s
   serial writer. Moving it to the device tape coder in `src/cuda/entropy.*`
   is the next entropy step; it must preserve CDF order exactly.
3. **Cold start.** The first encode in a process includes context/arena setup
   (~300 ms). The reported speedups are steady-state.
4. **M0 scope only:** 8-bit 4:2:0, DC intra, `TX_MODE_LARGEST`, no filters /
   palette / CfL / film grain, `base_q_idx = 90`.
5. `ncu` counters are blocked on the dev box; timing uses CUDA events. `nsys`
   is available if a kernel-level breakdown is needed.
