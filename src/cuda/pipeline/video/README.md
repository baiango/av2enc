# CUDA M2 P-frame baseline

> **Status 2026-11-13 (CUDA 13.0, sm_86): gate PASSES.** Both
> `build_and_test.sh` (host emulation) and `build_gpu_gate.sh` (sm_86) are green,
> and the product `gate_video_gpu.sh` matrix passes at 4 and 8 frames including
> the opt-in `--video-deblock` cases.  The header mirror was re-synced by the
> integration waves after `a0b4ba1` (`BUG-CDF-001`, `BUG-INTER-002`, compound/
> bframes/multiref/tiles/MFH); the stale "gate FAILS at frame 1" note is gone.

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
| Full luma intra candidate search + TX plan (`--video-intra-full`) | `src/cuda/pipeline/video/intra` (`Workspace::evaluate`, `download_best`) |
| Residual downsample / forward / quant | `src/cuda/tx_variants` (`txv_*_multi`) |
| Inverse transform | `src/cuda/tx_variants` (`txv_inverse_transform_multi`) |
| Reconstruction `Clip1(pred + residual)` | `video_recon_kernel` (this tree) |
| Deblocking before reference/emit | `src/cuda/codec/filters/deblock` (`cuda::deblock_frame`) |
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
| §7.17 deblocking before reference/emit | `--video-deblock` | byte-identical, device `cuda::deblock_frame` |
| Full 61-candidate luma intra search | `--video-intra-full` | byte-identical || DRL/merge stack | default | spec-exact host stack, byte-identical |
| Multi-reference (2..7 refs, per-block ref index) | `--video-refs N` | byte-identical, ref-indexed ME/MC over per-slot device reference planes |
| `NumRefFrames` pool 4..8 | `--video-num-ref-frames N` | byte-identical (rotating write slot, `f(N)` refresh mask) |
| §7.8 implicit reference lists | `--video-implicit-ref-map` | rejected by name on the GPU path (implemented on the CPU path) |
| §5.20.6/§5.20.8 luma TX size/type/partition search | `--video-tx-search` | byte-identical, device `videotx::video_tx_select_batch` |
| DRL/merge stack | default | spec-exact host stack, byte-identical |
| Multi-reference (2..3 refs, per-block ref index) | `--video-refs N` | byte-identical, ref-indexed MC/ME || Long-term / closed-GOP LTR | `--video-long-term` | byte-identical (plain GPU tiles + CPU header assembly) |
| Open random access (OLK) | `--video-olk` | byte-identical (4-frame profile) |
| Leading pictures | `--video-leading N` | byte-identical |
| Show-existing-frame | `--video-sef` | byte-identical |
| SWITCH / RAS | `--video-switch` / `--video-ras` | byte-identical (2/3-frame profiles) |
| B frames (one closed IPBB GOP) | `--b-frames N` | byte-identical (plan coding order) |
| Hierarchical-P temporal layers | `--video-tlayers N` | byte-identical (4/8-frame GOP) |
| Compound / CWP AVERAGE | `--video-compound` (refs >= 2) | byte-identical (GLOBAL_GLOBALMV average); device `compound_full::skip_batch` |
| §7.13.3.27 wedge compound | `--video-compound-wedge` | byte-identical; `SkipJob.type = WEDGE`, device luma-mask build + chroma subsample |
| §7.13.3.28 diffwtd compound | `--video-compound-diffwtd` | byte-identical; `SkipJob.type = DIFFWTD` |
| §7.13.3.29–.30 inter-intra | `--video-compound-interintra` | rejected by name (`motion_mode_allowed()` needs `!isCompound`) |
| Multi-frame header | `--video-mfh` | byte-identical (§5.7 MFH on the M2 stream) |
| Timing + content interpretation | `--video-timing` | byte-identical (decoder-compatible §5.4.12 form) |
| Average-CDF / adaptation | `--avg-cdf [--adaptive-cdf]` | byte-identical (§8.2.4 bank + §7.5 frame end) |
| Per-frame rate control | `--rc crf` / `--rc vbr` | byte-identical; the device quantizer follows each frame's qlookup(base_q_idx) |
| Multi-tile (§5.19) | `--tiles CxR` | byte-identical; tile-local coded geometry/recon + per-tile MV grids, CPU tile-group writer |

`--video-tx-search` (opt-in, default off) is the M2 consumer of the verified
device module under `src/cuda/pipeline/video/tx/`.  For every coded luma
residual it evaluates the §5.20.6.3 partition candidates, the legal §5.20.8.3
TX-type set of every leaf (DCT/ADST/FDST/IDTX; 1-D classes stay off until the
writer codes `TX_CLASS_HORIZ/VERT`), and returns the RD-best plan.  The syntax
writer emits `tx_mode_select = 1`, `tx_do_partition` / `tx_partition_type` and
the per-leaf `TX_TYPE`; levels are the module's §5.20.7.30 scan-order output.
The CPU reference is `src/pipeline/video/video_tx_search.{h,cpp}`, the same
candidate sets, arithmetic, rate model and tie-breaks, so the two paths stay
byte-identical.

- Partition search is **inter-only**: §5.20.7.24 predicts per transform block
  and feeds each part's reconstruction into the next part's prediction, while
  the module's contract is a fixed block predictor.  That is equivalent for MC
  with one MV but not for intra, so intra leaves use `TX_PARTITION_NONE` and
  search the TX type only.  A full intra partition search needs per-part
  prediction/recon feedback in the module.
- Chroma stays `TX_MODE_LARGEST` with `enable_chroma_dctonly = 1`.
- The search cost (`sse + lambda * rate`) only selects the plan; the
  skip/residual and intra/inter decisions keep the historical per-leaf
  `estimate_tx_bits` heuristic.  `--video-tx-lambda` (default 256) tunes the
  plan selection.
- Rejected by name: `--video-deblock` (the §7.17 grid derives its transform
  edges from one transform per leaf and does not model §5.20.6.3 parts),
  `--avg-cdf`/`--adaptive-cdf` (the §8.2.4 bank has no TX-partition CDFs),
  `--tiles > 1x1`, `--svc-layers`, and the long-term/OLK/leading/SEF/SWITCH/RAS
  frame-type wrappers.

Rejected by name (usage exit 2, no CPU fallback):
DRL reordering (`--video-drl-reorder`, also rejected on the CPU path),
cross-frame CDF carry (`--cdf-continuity`), `--svc-layers`, and the §7.8
implicit reference lists (`--video-implicit-ref-map`; the GPU path emits
§5.18.2 explicit maps, while the CPU path implements the implicit form — see
`src/pipeline/video/README.md`).  `--video-deblock` is wired on the
default/B/temporal/RC paths,
but is rejected with multi-tile (the pinned decoder filters interior tile
edges) and with the frame-type wrappers whose header writers do not signal
`deblocking_filter_params()`.
These need a different pixel stage or header/CDF model; the backend is
deliberately an honest mirror of the modes it implements.

### Multi-tile geometry

`--tiles CxR` runs one tile-local encode per tile and frame.  `begin_frame`
receives the whole-frame MI size (reference planes and arena sizing) plus the
tile origin and tile MI size; the device reconstruction planes are indexed with
the tile stride and cleared per tile, so intra availability stops at the tile
edge exactly like the decoder.  Motion estimation, MC and residual source reads
use absolute frame coordinates while their reconstruction destination is
converted back to tile-local (x - tile_x, y - tile_y).  The partition tree,
block syntax and MV grid are tile-local.  Tile payloads are in raster order and
are framed by the CPU `post::assemble_video_tile_group_from_payloads()`, so
`tile_info()`, `TileSizeBytes` and the interleaved size fields have one writer.
Compound prediction is not reachable in tile mode (`--tiles` requires
`--video-refs 1`); the backend rejects it explicitly if it is ever called.


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

### Full intra search (`--video-intra-full`)

The default stream keeps the DC-only fallback. With `--video-intra-full` the
luma leaf runs the still product's 61-candidate SSE search (DC, V, H, the eight
directional modes, SMOOTH/SMOOTH_V/SMOOTH_H, PAETH, then the six non-zero angle
deltas per directional mode = 61; chroma stays DC_PRED). Both paths share
`src/pipeline/video/video_intra_search.h`:

- candidates come from `intra::build_mode_candidates(..., true)` and the
  tie-break is the still product's strict `<` in that order;
- the cost is SSE of the prediction against the (source-clamped) source;
- `--video-intra-full` is the only CPU reference change; the GPU path delegates
  the predictors and the TX plan to `src/cuda/pipeline/video/intra/`, so no
  predictor math is duplicated.

The orchestrator owns the availability inputs because they depend on the coded
stream: `§5.20.9.1` `haveLeft`/`haveAbove` plus the `§7.13.2.1` above-right /
below-left counts are derived from a `§6.19.2.3 BlockDecoded` replay
(`video_intra_search.h::BlockDecoded`), reset at every superblock root exactly
like `clear_block_decoded_flags()`. Below-left can be available inside a
superblock when the left neighbour is taller, so the earlier "raster order means
0" shortcut was wrong for directional modes. The `§7.13.2.15/.16` smooth
neighbour flags come from an `§5.20.5.5 IntraJointMode` grid that mirrors the
syntax writer's state.

Device shape per intra leaf: one 63-job candidate batch (luma 61 + U/V DC, no
TX), `download_best`, then one small batch for the winning mode group + DC
chroma with `run_tx` (the shared `still_tx` plan), then the shared inverse
transform and `video_recon_kernel`. The device TX quantiser is pinned to
`qlookup(90)` like the rest of the M2 GPU path (default `--qp 90`).

Measured effect (RTX 3060, 120-frame 640x360 session clip, QP 90, x264-free
A/B): 1,254,794 B / 41.001 dB PSNR-Y default vs 1,316,505 B / 40.985 dB with
the flag. More intra leaves are selected (19,407 vs 18,271) because the
intra/inter decision is still SSE-only; the full search is a quality/parity
capability, not a rate win. Keep it opt-in until a rate-aware decision lands.


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
./src/cuda/pipeline/video/build_and_test.sh --intra-full
./src/cuda/pipeline/video/build_and_test.sh --drl-reorder 2
./src/cuda/pipeline/video/build_and_test.sh --no-hd
```

CUDA (nvcc + device; default `sm_86`):

```sh
./src/cuda/pipeline/video/build_gpu_gate.sh
ARCH=89 ./src/cuda/pipeline/video/build_gpu_gate.sh
./src/cuda/pipeline/video/build_gpu_gate.sh --frames 8 --full-tree
./src/cuda/pipeline/video/build_gpu_gate.sh --intra-full --frames 8
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
`av2enc_video_gpu` target exists.  `gate_video_gpu_intra` (`--frames 8
--intra-only`) runs just the full intra search case; the same two cases also
run inside `gate_video_gpu` (4 frames) and `gate_video_gpu_long` (8 frames).

RTX 3060 (12 GB), CUDA 13, `sm_86`, 4-frame 640x360 synthetic clip
(inter-sensitive cases use the smooth-pan and A/B/mid clips generated by the
gate):

`--video-tx-search` cases (RTX 3060, 4-frame 640x360 gate clip):

```text
PASS video-gpu tx-search             557c5dc0d1ec19dce367ce16d3209e3c
PASS video-gpu tx-search-inter-residual 6a344afe787f1c4a3e2d3c2fb1b3781f
PASS video-gpu tx-search-force-single   13b2f996f469504a02d607a66015c1af
PASS video-gpu tx-search-depth2         7a19ec884b6a009418f2180b6615462c
PASS video-gpu tx-search-inter          6972054a72f21cbb67b578507be4a86b
PASS video-gpu tx-search-refs2          c5db01c1a2b0105c38487342e22b91f4
PASS video-gpu tx-search-quarter        89f81242cf2e09f7a64e9b407ffe2d75
PASS video-gpu tx-search-bframes2       31a2ac8c19f6f52058362e5a27d28fe1
PASS video-gpu tx-search-tlayers4       3a70391f691070257b8b70edc9a11fc2
```

```text
PASS video-gpu baseline       f90c098499e293a2b019872e63c5f4bf
PASS video-gpu quarter        735f787872eb84f583e9eea1c30d5da6
PASS video-gpu half           b4ec6e1cb27fc98dae95ca936621fca5
PASS video-gpu subpel-eighth  f90c098499e293a2b019872e63c5f4bf
PASS video-gpu force-single   24fe02a91f8460fa3d4925c5267c7b70
PASS video-gpu depth2         7e19c9f944542fb88b1c86bfccadd994
PASS video-gpu min-block8     f1825028f70713d37d4066b52d87b752
PASS video-gpu me-range8      a8ea5aa8541f39e7a8cfda9e1cc6faa2
PASS video-gpu inter-residual ccd321df563aa65d041fe1cf8fb9a30c
PASS video-gpu inter-residual f90c098499e293a2b019872e63c5f4bf
PASS video-gpu intra-full     fcea0cb6065ebddb9cc5eae1e5973c33
PASS video-gpu intra-full-inter 2b1a5474d481f55193ee352b84052b3bPASS video-gpu inter-baseline 457440dc8422bcf1821b473f1a5044ad
PASS video-gpu refs2          7ce934bfca66bde47e5928be79a40445
PASS video-gpu refs3          e3bb47cdcba82d8c8be0e94dcbe859c9
PASS video-gpu refs2-quarter  2fd26460b399f85548024825ae38782f
PASS video-gpu refs2-half     5994ea23ec5b03d4163a56a8d17beca9
PASS video-gpu refs4          bb2393dafff71d53c10b22442062fe9e
PASS video-gpu refs7          d89e651b6d8cafa0754fe7986a5dac9f
PASS video-gpu nrf4-refs4     8cfc8aab5ecc063dec27828d49c12a98
PASS video-gpu nrf5-refs5     f6e448734d4def209e600a5013261f5c
PASS video-gpu nrf6-refs6     a32bdd9b068bf3f9bb62c5ca5c2fe366
PASS video-gpu nrf7-refs7     5a25d563dc13666bf768655a7ffee2f7
PASS video-gpu nrf8-refs1     d996af3944eac53ad617be615382a284
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
PASS video-gpu reject ref-lists / refs-bounds
gate_video_gpu: PASS
```

> Consistency-wave matrix (2026-11-13, 8-frame 640x360 gate clip, RTX 3060):

```text
PASS video-gpu gme-refs2-8        e3f5d3471182d93c74bce23ff1fbde5e
PASS video-gpu gme-refs3-8        4f92dea888b91f35dacefc141605937f
PASS video-gpu bframes2-cdef      ec1b59d4dd5316be206f16363cb1bfa3
PASS video-gpu bframes2-cdef-ccso d42474f1cd4a161bef797a3a47cd633e
PASS video-gpu tlayers4-cdef      ae06cd3f771f5790946506372c4a414d
PASS video-gpu tlayers8-cdef-ccso-gdf        658eb69b22e778a435e635e751911bd3
PASS video-gpu tiles2x1-cdef      9042ebc00c3fa5493ba586dd29fa3276
PASS video-gpu tiles2x2-cdef      9d88640c614000aa43bb301dc4d25510
PASS video-gpu tiles2x2-ccso      ce23af433e06af8ec1a63ff3a64743f5
PASS video-gpu tiles2x2-gdf       cd85de7bb5861419c0b7b0badcb0ce85
PASS video-gpu tiles2x2-cdef-ccso-gdf    e8d1db981132a830719157b681e69795
PASS video-gpu tiles2x2-cdef-ccso-gdf-8  36b94be7f0bd5161269e8dfff26a3b7d
PASS video-gpu tx-search-deblock  29285c5cdc0cbc49fa3bbfd4ed9fc537
PASS video-gpu tx-search-avg-cdf  d130109b2c1530cc40f5f53933803928
PASS video-gpu tx-search-tiles2x2 f84647c19de4c94a5c6837a186e86c09
PASS video-gpu tx-search-tiles2x2-avg-8 1d0dcdd9e8166479352f3390e135dc26
```

Loop restoration keeps its named `--tiles > 1x1` rejection and the frame-type
wrappers keep theirs (see `docs/negative-results.md`).

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
  `--video-intra-full` adds the still product's 61-candidate luma intra search
  (device module, opt-in).  The host decision path uses the spec-exact
  `src/codec/inter/merge` stack, so the shared MC stack's documented deviations
  cannot leak into the bitstream.
- Frame-type wrappers (long-term/OLK, leading, SEF, SWITCH/RAS) reuse the
  plain GPU sub-encodes; only the header/OBU assembly is shared with the CPU
  reference modules.  B frames and hierarchical-P temporal layers reuse the
  same device frame encoder with plan-owned coding order and explicit
  per-frame reference lists.  Compound is the gated GLOBAL_GLOBALMV /
  CWP_EQUAL AVERAGE subset, computed by the device compound skip kernel.
  Skip-mode beyond that subset, warp/OBMC/inter-intra, TIP, OPFL, refine-MV,
  temporal MVs, the ref-MV bank and loop filters are not implemented on this
  path; the CLI rejects every one by name.  BAWP is wired for the fitted luma
  form (`--video-bawp`): the orchestration applies the shared
  `video_bawp.h` fit to the host prediction and mirrors the adapted block into
  the device recon, and `gate_video_bawp` requires byte-identical CPU/GPU
  streams at 360p, 4 and 8 frames.
- DRL reordering is wired to `merge::DrlReorder` but disabled; the CPU product
  path rejects it for the same decoder-gap reason.
- Inter leaves are `skip_flag = 1` unless the inter-residual tool is enabled;
  that matches the CPU reference.
- The decision search (partition/stack/candidate compare/EOB) stays on the
  host; only pixels/transforms/reconstruction are device-resident.
- Every decision that needs a score synchronises the stream.  The per-leaf
  round trips are the throughput ceiling; see "720p throughput" below.
- The CUDA backend was validated on an RTX 3060 (sm_86): selftest, full gate,
  full-tree, forced inter-residual, 360p and 720p product parity all
  byte-identical with recon 0 diffs.  The emulation gate remains the no-GPU
  reference.

## 720p throughput

The GPU path is the product path, so the 1440x720 encode is also the
cross-encoder throughput benchmark.  Two points are measured separately
(`gate_gpu_throughput.sh`):

| Point | Purpose | Gate |
|---|---|---|
| 480x240@30 QP150 (240p, 2:1) | decision-policy calibration | CVVDP <= 0.05 JOD, bitrate <= 2 %, PSNR-Y <= 0.1 dB drift vs the CPU reference |
| 1440x720 QP90, 60 frames | throughput | fps (best of N) plus the same drift row |

The calibration clip is the benchmark clip scaled to 480x240 (240p, 2:1), so
both rows describe the same NASA content.  The GPU pipeline is a
decision-identical mirror, so every drift row is 0 and the throughput
optimizations below only change *how* the same decisions are executed.

### Measured stage costs (RTX 3060, 720p, 60 frames, QP 90)

Per-frame host wall time of each backend/orchestration stage (instrumented
build, `AV2ENC_GPU_PROF=1`; the harness is not committed):

| Stage | before | after |
|---|---:|---:|
| `inter_sse_batch` (candidate MC + SSE) | 44.7 ms | 22.4 ms |
| `intra_candidate` (DC pred + TX/quant/inverse + recon) | 54.1 ms | 51.2 ms |
| `me_batch` (one launch per frame) | 30.2 ms | 30.3 ms |
| `mc_420` (winning prediction) | 14.8 ms | 9.8 ms |
| `write_recon_420` | 4.0 ms | 2.3 ms |
| `variance_costs` | 2.2 ms | 2.2 ms |
| partition tree (host, `decide_partition`) | 19.2 ms | 8.1 ms |
| entropy + syntax write (host) | 15.5 ms | 15.7 ms |

Whole-frame fps and transfer counters for the same clip:

| Metric | before | after |
|---|---:|---:|
| encode wall time (60 frames) | 12.02-12.21 s | 9.19-9.38 s |
| fps | 4.95 | 6.53 |
| kernel launches | 254,747 | 254,747 |
| stream syncs | 184,947 | 92,254 |
| host->device bytes | 238 MB | 238 MB |
| device->host bytes | 974 MB | 360 MB |

What is left is dominated by two things outside the orchestrator: the
`forward_job_kernel` double-precision DCT (~24 ms/frame; one block per plane,
so only three SMs are busy per leaf) and `me_search_kernel` (~30 ms/frame in
one batched launch).  Further gains need those components to expose more
intra-leaf parallelism, or a graph-captured per-leaf launch chain.

### Optimizations that landed

1. **Warp-parallel candidate SSE.**  `video_sse_kernel` scored one candidate
   per thread, i.e. a 64x64 block summed 4096 samples serially (~50 us per
   launch, the largest single kernel).  One warp per candidate with a shuffle
   reduction is exact (integer sum) and costs ~2 us.
2. **Pinned staging + one sync per leaf batch.**  Every per-leaf copy used
   pageable host memory, where a D2H has to wait for the copy, so each level /
   recon download drained the stream and the backend synced once per copy.  The
   pinned staging ring (`av2enc::cuda::StagingBuffer` + a bump allocator that
   is recycled only after a sync) makes uploads true async enqueues, batches the
   3-6 copies of `mc_420` / the TX chain into one sync, and removes the sync
   from `write_recon_420`/`write_recon_luma` (the bytes stay valid in pinned
   memory until the next sync).  Syncs 184,947 -> 92,254.
3. **Size-accurate TX result downloads.**  `tx_and_recon_planes` downloaded a
   fixed 3 x 1024 levels + 3 x 4096 recon samples per leaf (24 KiB) regardless
   of block size, i.e. 15 MB/frame for an average ~40x40 leaf.  Per-plane
   coded sizes only: 974 MB -> 360 MB downloaded.
4. **Partition node-set cache.**  `decide_partition` walked the CPU partition
   tree twice per frame (discovery + real).  The node set is a pure function of
   the coded geometry and the partition options, so it is discovered once per
   sequence; the real pass also uses hash maps instead of `std::map`.
   `decide_partition` 19.2 -> 8.1 ms/frame.

### Benchmarks against other encoders

Same box, same 60-frame 720p clip (RTX 3060 for `av2enc`, 28-core CPU for the
software encoders).  Bitrate/quality are not matched here; the row exists to
put the absolute fps number in context.

| Encoder | setting | fps | bitrate |
|---|---|---:|---:|
| `av2enc --video --video-gpu` (before) | QP 90 | 4.95 | 6,635 kbps |
| `av2enc --video --video-gpu` (after) | QP 90 | 6.53 | 6,635 kbps |
| x264 | medium, CRF 18 | 99 | 739 kbps |
| x264 | medium, CRF 23 | 96 | 369 kbps |
