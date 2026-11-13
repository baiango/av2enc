# CUDA M2 P-frame baseline

> **Status 2026-11-13 (CUDA 13.0, sm_86): builds and runs, gate FAILS at frame
> 1.** `build_and_test.sh` and `build_gpu_gate.sh` both reach the stream
> comparison and report `unit_mismatches=1 recon_mismatches=0 max_diff=0
> first=f1 off=6` (one header byte, 0x5A vs 0x56, tile payload identical).
> The mirror predates the CPU writer changes that landed after `a0b4ba1`
> (`BUG-CDF-001`, `BUG-INTER-002`, and the compound/bframes/multiref/tiles/MFH
> integration waves in `src/pipeline/video/video_frame.cpp`).  Do not treat
> this gate as green until the mirror is re-synced.

GPU path for the M2 P-frame encoder. It is a decision-identical mirror of
`src/pipeline/video/`: same partition tree, same MV-stack derivation, same
candidate ordering/tie-breaking, same transform and same §8.2 range coder. The
output is **byte-identical** to the CPU M2 encoder for the same input and
options, and `dav2d` decodes both to the same reconstruction.

The CPU directory is not modified and remains the reference. This tree owns
only `src/cuda/pipeline/video/`.

## What runs where

| Stage | Owner |
|---|---|
| Partition block-variance cost | `video_variance_kernel` (this tree) |
| Integer + half/quarter subpel ME | `src/cuda/inter/me` (`me_search_batch`) |
| Single-ref §7.13.3 MC | `src/cuda/inter/mc` (`mc_predict_batch`) |
| Luma SSE against source | `video_sse_kernel` (this tree) |
| DC intra prediction + residual | `src/cuda/intra_modes` (`intra_predict_batch`) |
| Residual downsample / forward / quant | `src/cuda/tx_variants` (`txv_*_multi`) |
| Inverse transform | `src/cuda/tx_variants` (`txv_inverse_transform_multi`) |
| Reconstruction `Clip1(pred + residual)` | `video_recon_kernel` (this tree) |
| Reconstruction-plane writes / reference upload | `video_backend_cuda.cu` |
| Partition recursion, MV stack, candidate comparison, EOB scan | host (CPU decision helpers) |
| OBU/IVF assembly and §8.2 entropy | host (`src/pipeline/video/`) |

The split is deliberate. Pixel work, ME/MC, transform and reconstruction are
device-resident; the decisions are the CPU decisions by construction, which is
what "byte-identical for the same decisions" needs. Moving the stack/decision
search to the device is a follow-up that must re-run this gate.

## Files

```text
src/cuda/pipeline/video/
  video_gpu.h              public integration API + stats
  video_sequence_internal.h internal plain/wrapper entries
  video_pipeline.cpp       frame loop, partition/MV/mode decisions, OBU assembly
  video_frame_modes.cpp    long-term/OLK, leading, SEF, SWITCH/RAS wrappers
  video_backend.h          backend contract (pixel/transform work only)
  video_backend_cuda.cu    CUDA backend + four M2-owned kernels
  video_backend_cpu.cpp    host-emulation backend (no GPU/nvcc needed)
  video_device.cuh         shared device arithmetic (variance, SSE, residual, recon)
  video_selftest.cpp       inter residual TX/quant/inverse parity vs CPU
  video_gate.cpp           standalone CPU-vs-GPU gate + dav2d decode
  build_and_test.sh     emulation gate (this machine, no GPU)
  build_gpu_gate.sh     real CUDA gate (nvcc + device)
```

`video_device.cuh` is compiled by both backends, so the M2-owned arithmetic cannot
drift. The existing CUDA components have their own CPU-vs-GPU bit-exactness
tests; this gate verifies the orchestration and the byte stream.

`src/cuda/rdoq` is intentionally not on this path: the M2 baseline quantises
with `txv_forward_transform` (the CPU M2 reference does the same), so adding
RDOQ would change the coefficients and break byte parity. It stays available
for a later rate-aware M2 that also changes the CPU reference.

## Integration API

`video_gpu.h` is a drop-in replacement for the CPU entry point:

```cpp
#include "cuda/m2/video_gpu.h"

av2enc::video::VideoSequenceResult result;
av2enc::cuda::video::VideoGpuStats stats;
std::string error;
const bool ok = av2enc::cuda::video::encode_video_sequence_gpu(
    frames,          // std::span<const av2enc::video::VideoInputFrame>
    options,         // av2enc::video::VideoOptions, identical semantics
    result,          // av2enc::video::VideoSequenceResult, identical contents
    error,           // std::string&
    &stats);         // optional, may be nullptr
```

Everything else (input validation, `base_q_idx = 90`, frame headers, temporal
units) matches `av2enc::video::encode_video_sequence()`. `video_backend_name()` returns
`"cuda"` or `"cpu-emulation"` for honest labeling.

`VideoGpuStats` reports kernel launches by stage (`variance`, `me`, `mc`, `sse`,
`intra`, `tx`, `residual`, `recon`), host/device transfer bytes, sync count,
ME job and inter-candidate counts, the total device interval, and
`frame_ms` — one entry per frame. The CUDA backend fills `frame_ms` from one
`cudaEvent` pair per frame; the emulation backend reports the wall time of the
emulated backend calls.

### Build-system wiring

The M2 pipeline is now a product target: `av2enc_video_gpu` in the root
`CMakeLists.txt` compiles `video_pipeline.cpp` and `video_backend_cuda.cu` and
links the shared CUDA components (`inter/me`, `inter/mc`, `intra/modes`,
`transform/variants`) **once**.  When the M1 target is also present, any shared
component it already provides is skipped so the executable does not get duplicate
object symbols.  `--video-gpu` is accepted only in that build; a CPU-only or
target-less build rejects the flag with a usage error and exit code 2.

The CMake target needs these objects if you build the gate scripts by hand:

```cmake
add_library(av2enc_video_gpu OBJECT
  src/cuda/pipeline/video/video_pipeline.cpp
  src/cuda/pipeline/video/video_backend_cuda.cu
  src/cuda/pipeline/video/video_selftest.cpp)target_link_libraries(av2enc_video_gpu PUBLIC av2enc_core CUDA::cudart)
target_include_directories(av2enc_video_gpu PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
# video_backend_cuda.cu and the shared component .cu files: --fmad=false
```

`video_pipeline.cpp` is plain host C++; only `video_backend_cuda.cu` needs nvcc.
`video_selftest.cpp` is optional (gate/test builds). Do not link
`video_backend_cpu.cpp` in the same binary: it defines the same
`detail::make_backend()` factory.

### Product CLI (`--video-gpu`)

`--video-gpu` implies `--video` and selects `encode_video_sequence_gpu()`.  The
CPU `--video` path remains the default and the reference.  There is **no silent
CPU fallback**: the CUDA backend's `VideoGpuStats::frame_bytes_from_cpu` stays
`false` and the CLI prints
`av2enc: --video-gpu backend=cuda frame_bytes_from_cpu=0`.

Wired and gated combinations:

| Mode | CPU flag | GPU status |
|---|---|---|
| Keyframe + flat P chain | `--video` | byte-identical |
| Partition tree / `--video-force-single` / `--video-max-depth` | same | byte-identical |
| ME + GLOBALMV/NEARMV/NEWMV | default | byte-identical |
| 1/8, 1/4, 1/2-pel MV precision | `--video-mv-precision` | byte-identical |
| 1/8-pel ME refinement | `--video-subpel-eighth` | byte-identical |
| Inter residual (heuristic or forced in the gate) | `--video-inter-residual` | byte-identical |
| DRL/merge stack | default | spec-exact host stack, byte-identical |
| Multi-reference (2..3 refs, per-block ref index) | `--video-refs N` | byte-identical, ref-indexed MC/ME |
| Long-term / closed-GOP LTR | `--video-long-term` | byte-identical (plain GPU tiles + CPU header assembly) |
| Open random access (OLK) | `--video-olk` | byte-identical (4-frame profile) |
| Leading pictures | `--video-leading N` | byte-identical |
| Show-existing-frame | `--video-sef` | byte-identical |
| SWITCH / RAS | `--video-switch` / `--video-ras` | byte-identical (2/3-frame profiles) |
| B frames (one closed IPBB GOP) | `--b-frames N` | byte-identical (plan coding order) |
| Hierarchical-P temporal layers | `--video-tlayers N` | byte-identical (4/8-frame GOP) |
| Compound / CWP AVERAGE | `--video-compound` (refs >= 2) | byte-identical (GLOBAL_GLOBALMV average); device `compound_full::skip_batch` |
| Multi-frame header | `--video-mfh` | byte-identical (§5.7 MFH on the M2 stream) |
| Timing + content interpretation | `--video-timing` | byte-identical (decoder-compatible §5.4.12 form) |
| Average-CDF / adaptation | `--avg-cdf [--adaptive-cdf]` | byte-identical (§8.2.4 bank + §7.5 frame end) |

Rejected by name (usage exit 2, no CPU fallback):
DRL reordering (`--video-drl-reorder`, also rejected on the CPU path),
`--rc crf/vbr` (the device transform quantiser is pinned to qlookup(90)),
multi-tile, cross-frame CDF carry (`--cdf-continuity`), `--svc-layers`.
These need a different pixel stage or header/CDF model; the backend is
deliberately an honest mirror of the modes it implements.


### Device memory

The backend owns (or accepts) one `av2enc::cuda::Arena`. Persistent frame
planes and per-block scratch are allocated once per frame geometry at
`begin_frame`; per-batch job arrays use `mark`/`rewind` inside each call. There
is no `cudaMalloc` and no per-frame free. `required_device_bytes()` reports the
capacity for a frame configuration, and an undersized arena returns
`ArenaStatus::kArenaTooSmall` with the size to grow to. Growing means
re-initialising the arena at a frame boundary: all pointers handed out before
that point become invalid. There is exactly one frame-boundary sync in
`end_frame` plus the unavoidable syncs when a decision needs a device value
back on the host (see below).

## Decision parity

The pipeline in `video_pipeline.cpp` is a line-by-line mirror of
`src/pipeline/video/video_frame.cpp`:

- partition costs are discovered with the CPU traversal, computed by the device
  kernel, then fed back into `partition::decide_frame()` so the tree is the CPU
  tree;
- `inter::mc::build_mv_stack` / `resolve_mode` derive NEARMV/NEWMV exactly as
  the baseline (including the conservative NEARMV DRL cap for blocks <= 32
  described in the CPU README);
- ME is batched per frame but each job is independent, so results match the
  CPU per-leaf calls;
- GLOBALMV/NEARMV/NEWMV candidates are all scored in one MC+SSE batch and
  considered in the CPU order (strict `<`, so earlier candidates win ties);
- the intra fallback is evaluated for every leaf, exactly like the CPU;
- EOB scan, level ordering and `BlockSyntax` fields are produced on the host
  with the CPU syntax writer.

Optional CPU M2 tools are mirrored when the linked `VideoOptions` provides them:

- `enable_inter_residual` / `force_inter_residual` — inter leaves can code
  Y/U/V residual. The decision copies the CPU heuristic (SSE + `lambda=0.5*66²`
  × `estimate_tx_bits`); the MC → residual → downsample → forward/quant →
  inverse → recon chain runs on the device.
- `drl_reorder` — passed to `MvStackParams::reorder` and to the M2 config.

`video_inter_residual_selftest()` runs that chain on the linked backend and
compares every level and reconstructed sample of all three planes against the
CPU transform reference. It is wired into both build scripts.

FP caveat: partition variance is the only floating-point decision input. The
CUDA build uses `--fmad=false` and explicit `__dadd_rn`/`__dmul_rn`; the
expression and loop order match the CPU. A 1-ULP difference would only matter
if a variance landed within ~1e-8 of an integer before the `int64` cast, which
the gate would catch as a partition/bitstream difference.

## Build and run

Emulation (no GPU, no nvcc — CPU components stand in for the device):

```sh
./src/cuda/pipeline/video/build_and_test.sh
./src/cuda/pipeline/video/build_and_test.sh --frames 8 --full-tree
./src/cuda/pipeline/video/build_and_test.sh --inter-residual
./src/cuda/pipeline/video/build_and_test.sh --force-inter-residual
./src/cuda/pipeline/video/build_and_test.sh --drl-reorder 2
./src/cuda/pipeline/video/build_and_test.sh --no-hd
```

CUDA (nvcc + device; default `sm_86`):

```sh
./src/cuda/pipeline/video/build_gpu_gate.sh
ARCH=89 ./src/cuda/pipeline/video/build_gpu_gate.sh
./src/cuda/pipeline/video/build_gpu_gate.sh --frames 8 --full-tree
SANITIZE=1 ./src/cuda/pipeline/video/build_gpu_gate.sh   # memcheck selftest + smoke
```

Both scripts build `video_gate.cpp` and run, for each phase:

1. CPU M2 encode and GPU M2 encode;
2. per-unit byte comparison (`units`, `frame_bytes`, `tile_bytes`,
   `inter_blocks`, `intra_blocks`);
3. per-sample Y/U/V reconstruction comparison;
4. `dav2d` decode of the GPU stream and a decoded-vs-recon comparison;
5. the launch/transfer/timing report.

Phases: 2-frame and 8-frame 64x64 synthetic, and 2-frame and 8-frame 720p from
`testdata/shine_1280x720_60f.y4m`. 720p defaults to `--single-block` (mirrors
the CPU gate); `--full-tree` runs the variance-driven tree. `DAV2D=` overrides
decoder discovery.

If the working tree's `src/pipeline/video/` is mid-refactor, point
`VIDEO_CPU_REF=/path/to/checkout` (containing `src/pipeline/video/`) at a clean
reference; the gate builds against that snapshot without touching the tree.
The scripts link `src/codec/inter/merge/merge.cpp` when it exists (the current
CPU gate needs it); if the CPU reference grows another helper module, mirror
the addition from `src/pipeline/video/build_and_test.sh` into both CORE lists.

## Gate results

### Product CPU-vs-GPU gate (`--video-gpu`)

`gate_video_gpu.sh` generates a deterministic 640x360 clip, encodes it with
both paths for every wired mode, compares the temporal units, the
`--video-recon` dumps and the `dav2d --strict 1` decode, then asserts that each
unported combination is rejected with exit code 2.  It is registered as CTest
`gate_video_gpu` only when `AV2ENC_PRODUCT_GATES=ON` and the
`av2enc_video_gpu` target exists.

RTX 3060 (12 GB), CUDA 13, `sm_86`, 4-frame 640x360 synthetic clip
(inter-sensitive cases use the smooth-pan and A/B/mid clips generated by the
gate):

```text
PASS video-gpu baseline       f90c098499e293a2b019872e63c5f4bf
PASS video-gpu quarter        735f787872eb84f583e9eea1c30d5da6
PASS video-gpu half           b4ec6e1cb27fc98dae95ca936621fca5
PASS video-gpu subpel-eighth  f90c098499e293a2b019872e63c5f4bf
PASS video-gpu force-single   24fe02a91f8460fa3d4925c5267c7b70
PASS video-gpu depth2         7e19c9f944542fb88b1c86bfccadd994
PASS video-gpu min-block8     f1825028f70713d37d4066b52d87b752
PASS video-gpu me-range8      a8ea5aa8541f39e7a8cfda9e1cc6faa2
PASS video-gpu inter-residual f90c098499e293a2b019872e63c5f4bf
PASS video-gpu inter-baseline 457440dc8422bcf1821b473f1a5044ad
PASS video-gpu refs2          7ce934bfca66bde47e5928be79a40445
PASS video-gpu refs3          e3bb47cdcba82d8c8be0e94dcbe859c9
PASS video-gpu refs2-quarter  2fd26460b399f85548024825ae38782f
PASS video-gpu refs2-half     5994ea23ec5b03d4163a56a8d17beca9
PASS video-gpu mfh            a895466ceebe60fb2c4a48b51202b810
PASS video-gpu timing         27f4e8084123561c7c20ab85a3bf0bfb
PASS video-gpu avg-cdf        4f3e451f82f09fced2cb6f772b2e720f
PASS video-gpu adaptive-cdf   4ee704d2f3554c0c5e3f7c62484a2357
PASS video-gpu long-term      ec86d616e394c81665e75043f7b2b722
PASS video-gpu olk            11ae750a86cdbd0a6e0758a03eb0990c
PASS video-gpu leading        030e6d998fbabd3db4dda08a2f0c0411
PASS video-gpu sef            44d63b9864aaa9d806477f99c9bf8db3
PASS video-gpu switch         a50a0f3cb25b21fb8d1b23ab6c46b19f
PASS video-gpu ras            4ef352d711c72b47d20c813d21087c86
PASS video-gpu bframes2       71d970d517850cc01e466e1c912a9120
PASS video-gpu bframes3       0c46731c4a700e5b617b0f1f7523debe
PASS video-gpu tlayers4       957c3a51d801c1ac3dd48871df7564fa
PASS video-gpu tlayers8       f42ba113e70767773de0d6fb6db92fa8
PASS video-gpu tlayers8-drop1 30ccf21fcd2faca78c2b2104b18e28eb
PASS video-gpu compound       c80da95dc2b00e459aa04414746d5f04
PASS video-gpu reject rc-crf / tiles / cdf-continuity / svc
gate_video_gpu: PASS
```

Additional real runs on the same box: 640x360 8 frames default
`12d2df474b9d6314af81428c39651108` and 1280x720 6 frames default are
byte-identical to the CPU reference; `dav2d --strict 1` decoded output and the
encoder recon dump are identical; `build_gpu_gate.sh --input <720p> --frames 6
--full-tree` and `--force-inter-residual` both report byte-identical units,
recon 0 diffs and a clean decode.  The CUDA inter-residual selftest reports
`mc=128 tx=256 resid=384 recon=128` PASS.

### RTX 3060 12 GB, CUDA 12.8, `sm_86` (real event times)

Default run (`build_gpu_gate.sh`, 720p single-block, `testdata/shine_1280x720_60f.y4m`):

```text
m2 inter-residual selftest: PASS (128 trials, backend=cuda, ...)
== 64x64 two-frame ==
  GPU backend=cuda launches=15 ... device=23.99ms
  frame ms (cuda-event): 15.9429 8.0519
  PASS: bitstream byte-identical, recon 0 diffs
  PASS: dav2d decoded 2 frame(s), recon exact
== 64x64 short clip: 8 frames ==
  GPU backend=cuda launches=97 ... device=52.39ms
  frame ms (cuda-event): 4.67184 7.27645 6.56291 6.50026 6.35373 6.20726 7.90659 6.90896
  PASS: bitstream byte-identical, recon 0 diffs
  PASS: dav2d decoded 8 frame(s), recon exact
== 720p two-frame ==
  GPU backend=cuda launches=2883 ... device=195.89ms
  frame ms (cuda-event): 65.0762 130.814
  PASS: bitstream byte-identical, recon 0 diffs
  PASS: dav2d decoded 2 frame(s), recon exact
== 720p short clip: 8 frames ==
  GPU backend=cuda launches=12975 ... device=911.675ms
  frame ms (cuda-event): 58.6221 125.816 126.553 120.272 118.13 121.041 120.953 120.287
  PASS: bitstream byte-identical, recon 0 diffs
  PASS: dav2d decoded 8 frame(s), recon exact
M2 GPU GATE PASS
```

The same run on the 2016 Xeon CPU reference took 13.46 s for the 720p
8-frame clip (≈1.68 s/frame), so the GPU path is ≈14.5× faster on this
content. Frame 0 includes context/module warm-up.

Additional real runs, byte-identical / recon 0 diffs / `dav2d` exact:

| Run | GPU launches | event time per frame | total |
|---|---:|---|---:|
| 720p 8f `--full-tree` | 33,289 | 119,198,195,190,191,195,193,193 ms | 1,473 ms |
| 720p 8f `--force-inter-residual` | 26,415 | 60,185,174,174,174,172,174,173 ms | 1,285 ms |
| `compute-sanitizer --tool memcheck` (selftest 128 + 64x64 smoke) | — | — | **0 errors** |

### CPU-emulation proxy (Apple M4, macOS)

The same gate also runs with no GPU/nvcc; these numbers validate byte/recon
parity and the pipeline shape, not device speed:

| Phase | launches | proxy backend | CPU reference |
|---|---:|---:|---:|
| 64x64 2 frames | 13 | 2–3 ms | 3–5 ms |
| 64x64 8 frames | 85 | 20–24 ms | 18–25 ms |
| 720p 2 frames, single-block | 2,403 | 430–490 ms | 350–450 ms |
| 720p 8 frames, single-block | 11,055 | 2.2–2.9 s | 2.25 s |
| 720p 2 frames, full tree | 6,183 | 395–450 ms | 420 ms |
| 720p 8 frames, full tree | 28,359 | 2.26 s | 2.25 s |

Per-frame launch shape (720p single-block, 8 frames): ≈1,622 launches/frame
(12,975/8). Full-tree 720p 8 frames: 33,289 launches ≈4,161/frame. Most launches
are per-leaf intra candidates (the CPU evaluates the DC fallback on every leaf),
MC candidate batches and the chosen inter block's MC + those per-block copies.
The host-in-the-loop design synchronises at each decision download; batching the
decision search on the device is the obvious next step and does not change the
bitstream.

Feature gates (emulation host, all `PASS`, byte-identical, recon 0 diffs,
`dav2d` exact): `--frames 8 --full-tree`, `--force-inter-residual`,
`--inter-residual`, `--drl-reorder 1/2`, `--force-global`, `--no-newmv`,
`--force-intra`, `--no-intra-fallback`, `--selftest 128`.

### TX scratch allocation invariant

`d_recon_out_` is a `std::uint8_t*`; its element count is the number of
**samples** (`3 * 64 * 64`), not the number of planes. The first GPU run showed
this as `tx download: invalid argument` because the buffer was reserved with a
count of 3 bytes while the download copied 3×4096. All TX/recon scratch is now
reserved in `ensure_intra_buffers()` before the first call, and
`tx_and_recon_420()` checks every capacity before any copy. `download()` returns
`cudaError_t` and each copy site names itself in the error path.

## Scope / limitations

- Baseline decision tools only: GLOBALMV, NEARMV, NEWMV, DC-intra fallback.
  The host decision path uses the spec-exact `src/codec/inter/merge` stack, so
  the shared MC stack's documented deviations cannot leak into the bitstream.
- Frame-type wrappers (long-term/OLK, leading, SEF, SWITCH/RAS) reuse the
  plain GPU sub-encodes; only the header/OBU assembly is shared with the CPU
  reference modules.  B frames and hierarchical-P temporal layers reuse the
  same device frame encoder with plan-owned coding order and explicit
  per-frame reference lists.  Compound is the gated GLOBAL_GLOBALMV /
  CWP_EQUAL AVERAGE subset, computed by the device compound skip kernel.
  Skip-mode beyond that subset, warp/OBMC/inter-intra, BAWP, TIP, OPFL,
  refine-MV, temporal MVs, the ref-MV bank and loop filters are not
  implemented on this path; the CLI rejects every one by name.
- DRL reordering is wired to `merge::DrlReorder` but disabled; the CPU product
  path rejects it for the same decoder-gap reason.
- Inter leaves are `skip_flag = 1` unless the inter-residual tool is enabled;
  that matches the CPU reference.
- The decision search (partition/stack/candidate compare/EOB) stays on the
  host; only pixels/transforms/reconstruction are device-resident.
- Every decision that needs a score synchronises the stream. This is a
  correctness-first milestone, not a throughput claim.
- The CUDA backend was validated on an RTX 3060 (sm_86): selftest, full gate,
  full-tree, forced inter-residual, 360p and 720p product parity all
  byte-identical with recon 0 diffs.  The emulation gate remains the no-GPU
  reference.
