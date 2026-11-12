# M1 GPU orchestrator (`src/cuda/pipeline/still/` root)

Owner: workstream 1 in `docs/m1-gpu.md`. Root files only:
`still_gpu.h` (contract), `still_gpu.cu` (orchestrator), `still_gpu_test.cc`
(data-flow test), `build_still_gpu_test.sh`. Component directories
(`plan/`, `search/`, `tx/`, `recon/`, `chroma/`, `cfl/`, `sched/`) are owned by
the other workstreams and are never edited here.

Status: **`--still-gpu` produces frame bytes from GPU stages and is byte-identical
to CPU `--still`.** Verified at 64×64, 640×360 and 1920×1080 across default,
full-intra, no-CfL, no-deblock, depth 2/0 and adaptive-CDF.

## API

Namespace is frozen: `av2enc::cuda::still_gpu`.

```cpp
bool encode_still_gpu(const av2enc::still::StillPicture& picture,
                   const GpuOptions& options,
                   GpuResult& result, std::string& error);
```

Shared PODs (`still_gpu.h`):

| POD | Contents |
|---|---|
| `FramePlan` | width/height, padded strides, MI/SB grid, leaf count, coefficient counts, block list in syntax order |
| `BlockPlan` | MI origin, luma/chroma/TX sizes, plane/coefficient/packed-prediction/packed-residual offsets, availability bits, depth |
| `BlockDecision` | luma mode + angle delta, UV mode, `is_cfl`, Q8 alpha U/V, TX types, EOB per plane, SSE |
| `GpuFrame` | host source/recon/coeff spans, decisions, arena device planes, stream (hook ABI) |
| `GpuOptions` | M1 option set + device/VRAM/callback/verify controls |
| `GpuResult` | frame bytes, recon, blocks/decisions, per-plane coefficient levels, stage status, stats |

`GpuResult::frame_bytes_from_cpu` is **false** when the GPU assembled the
frame. `GpuStats::gpu_frame_verified` is true when the GPU frame and all recon
planes matched the CPU reference. `linked_stage_count()` is **7** for the full
pipeline (plan, search, chroma/CfL, TX, recon, deblock, scheduler availability).

## Pipeline

Per frame, one stream, one sync:

```
plan::build_gpu_frame_plan (+ preorder stream::TreeNode tree)
  -> sched_bridge::build_still_plan (availability counts, host-only)
  -> per block, in syntax order:
       recon_block_availability replay (filter_type flags from IntraJointModes)
       luma_search(1 block)                    mode/angle/SSE (CPU tie-break)
       intra_modes::intra_predict_batch        chosen-mode predictor
       still_tx_batch (luma) + recon_write_blocks
       chroma_decide (DC/CfL, writes U/V predictor)
       still_tx_batch (U, V) + recon_write_blocks
       recon_commit_blocks                     IntraJointModes + deblock metadata
  -> recon_deblock_frame (when enabled)
  -> one download batch: recon Y/U/V, levels Y/U/V, EOB, decisions
  -> stream::assemble_gpu_frame (CPU §8.2 writer, GPU decisions/coefficients)
```

Descriptors are built on the host and uploaded once per frame (source planes,
`LumaSearchBlock`, `IntraJob`, 3×`StillTxJob`, 3×`ReconWriteJob`,
`ReconBlockInfo`). Only mode decisions flow device→device between stages via
three one-thread glue kernels (search filter flags, chosen prediction, commit
joint). Nothing allocates or synchronizes inside the block loop.

Deliberate simplification: blocks run in syntax order (per-block launches), not
in scheduler wavefront levels. This is byte-exact and keeps dependencies on one
stream; batching by `sched::run_level`/`LevelGraph` is the remaining speed step.

## Stage status

| Stage | State |
|---|---|
| plan | GPU path, real tree (`plan_gpu_bridge` tree overload) |
| search | GPU per block (`luma_search`, 61 candidates gated by options/env) |
| chroma + CfL | GPU (`chroma_decide`; the winning DC/CfL prediction is written in-kernel) |
| TX | GPU (`still_tx_batch`, DCT_DCT, 3 jobs per block) |
| recon | GPU (`recon_write_blocks` + `recon_commit_blocks`) |
| deblock | GPU (`recon_deblock_frame`) when `enable_deblock` |
| scheduler | used host-side for availability replay; level batching not yet used |
| entropy | GPU inputs (decisions/levels) → CPU §8.2 writer (`assemble_gpu_frame`) |

The separate `still_cfl_predict_pairs` pass is not needed: `chroma_decide` already
applies the derived-alpha CfL prediction for the winning candidate.

### Option semantics

`GpuOptions` carries every M3 tool flag and its decision knobs, even when the
stage is not ported.  `gpu_options_to_m1()` (in `still_gpu.h`) is the single
mapping used by the internal CPU parity encode, so a parsed flag can never be
missing from the comparison.  `gpu_m3_options_supported()` accepts the wired
in-loop filters and rejects each unported tool (`--still-wiener`,
`--still-chroma-lr`, `--still-pc-wiener`, `--still-gdf`, `--still-rdoq`,
`--still-tx-type`, `--still-rdo-intra`, `--still-uv-full`) by name with a
"not ported" message; the M3 gate reports those as `SKIP-REJECTED`.  Additional
CPU-only flags (`--still-qm`, `--still-ist`, `--still-dc-ac-delta-q`,
`--still-ibp/mrl/dip/mhccp/cfl-*`, `--still-sb128/256`,
`--still-ext-partitions`, segmentation/delta-Q) are rejected before the GPU
stage.  No flag is silently ignored.

### Wired in-loop filters

The CPU reference still plans the decisions (CDEF strengths/index grid, CCSO
offset tables and CcsoBlks, film-grain model); the GPU applies the normative
pixel kernels and the syntax bridge replays the planned symbols.

| Flag | GPU stage | Verified |
|---|---|---|
| `--still-cdef` | §7.18 `still_cdef_filter` after deblock | 64×64 / 256×256 / 640×360, strengths 1/2/4/8, ±deblock |
| `--still-ccso` | §7.19 `ccso_frame` after CDEF | same sizes, bands 0/3/6 |
| `--still-ccso-edge` | §7.19.1 edge-classified form | same sizes, per-unit CcsoBlks |
| `--still-grain` | metadata only (decoder-side synthesis) | 64×64 / 640×360 |

Every combination above produces frame bytes identical to the CPU reference
(`--still` with the same flags) and decodes with `dav2d --strict 1`.  Without
CDEF the CDEF frame is seeded from the deblocked reconstruction so CCSO's
`CurrFrame`/`CdefFrame` pair matches the CPU.  Loop restoration (Wiener NS,
chroma LR, PC-Wiener, GDF) stays rejected: the component suite still reports
`loop_restoration_frame launch (invalid argument)`, and PC-Wiener/GDF depend on
restoration tables that are not yet re-verified in the integrated tree.

`src/cuda/pipeline/still/still_gpu_options_test.cpp` (built by `build_options_test.sh`, host
only) pins the full mapping and the rejection set.

### Fallback

If a component is unavailable, a TX size/chroma geometry is unsupported, or a
debug override (`AV2ENC_STILL_FORCE_Y*`) is set, the frame falls back to
`av2enc::still::encode_still_picture`, marks stages `cpu-reference`, and leaves
`frame_bytes_from_cpu = true`. The CLI prints a note in that case. Verified
with `AV2ENC_STILL_FORCE_Y=6:0` at 1080p: `--still-gpu` equals `--still`
(MD5 `eddd9a4ce26b5dde4f9ed72f66b05fdb`).

## Verification

### CPU-vs-GPU matrix (CLI, same options both sides)

| Input | default | full-intra | no-cfl | no-deblock | depth 2 | depth 0 | adaptive |
|---|---|---|---|---|---|---|---|
| 64×64 | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 640×360 | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 1920×1080 | PASS | PASS | PASS | PASS | PASS | PASS | PASS |

Every cell compares the full IVF MD5 of `--still` and `--still --still-gpu`. Example
64×64 default: `0ef9825062a25c4d950ee924113c6f89`; 640×360 default:
`e61de628b0812b716b0efa7dcafcdd06`; 1080p default:
`59fc458d00ab5085cdbb926184009fa6`. The 1080p input is a Lanczos upscale of
the 640×360 fixture (`ffmpeg -vf scale=1920:1080`), matching the docs' proxy.

### `tools/still_gpu/verify_gpu.sh`

PASS at 64×64 (160 kernels), 640×360 (3,027 kernels) and 1920×1080 (15,729
kernels), with Nsight Systems attestation: IVF frame payloads identical, both
streams decode with `dav2d`, decoded Y/U/V planes bit-exact. `--recon` passes
through the decoded-frame fallback ("recon parity: fallback"); the CLI has no
GPU recon dump.

### `tools/still_verify/run_matrix.sh`

43/45 cases bit-exact. The two differences are **CPU-side forced-mode**
recon-vs-decoder cases at 1080p, not GPU:

```
1920x1080  F6   Y 16 samples, max |d| 211, first (171,81)   DIFF(Y)
1920x1080  F11  V  2 samples, max |d| 222, first (387,229)  DIFF(V)
```

`F6/F11` are `AV2ENC_STILL_FORCE_Y=6:0/11:0` with the CPU encoder. The GPU path
intentionally rejects those overrides and uses the CPU reference, so the GPU
bytes still match. This is the pre-existing forced-directional CPU/decoder
discrepancy at the 1080p proxy; it is not caused by the orchestrator.

### Standalone data-flow test

`still_gpu_test.cc` runs both default and forced-DC modes on the fixtures plus
synthetic edge geometry, and asserts: GPU-sourced bytes, one sync, 3 TX jobs
per block, GPU stage status, coefficient planes present, plan/EOB invariants,
byte-identical frame and all recon planes, and arena reuse on the second call.

## Hardening matrix (2026-11-11)

105-case integrated run (CPU `--still` vs `--still --still-gpu`, same options; full IVF
file bytes, MD5, `dav2d` decode, per-sample Y/U/V compare, GPU repeated in a
second process):

- **72 PASS**: bytes identical, Y/U/V differences `0/0`, deterministic MD5,
  GPU bytes (no `frame_bytes_from_cpu` note). Covers 64x64 and 640x360 with all
  13 options; native 1080p `testdata/shine_1920x1080_10f.y4m` with all options
  except depth 8; natural 3840x2160 with default/no-deblock/depth-0/adaptive;
  ten random even natural sizes with default/full-intra/no-CfL.
- **17 DECODE-FAIL**: GPU bytes equal CPU bytes, but the pinned `dav2d` rejects
  the CPU-sourced stream. `136x88` is strict-incompliant (decodes with
  `--strict 0`); `1280x720` fails even with `--strict 0`, and the **M0 CPU
  encoder fails the same way** while the AVM 720p baseline decodes. Isolated to
  the shared CPU bitstream/header/plan, not this orchestrator.
- **16 REJECT**: both CPU and GPU reject identically: 8x8/8x16/16x8
  (spec-forced partition conflict), depth 8 on 640x360/1080p/136x88/720p
  (`EOB multisize 0 is not supported by M1 transform sizes`), and depth
  0/1/2/3/`--still-single` on 136x88/720p (spec-forced partition conflict).
- One transient decode anomaly: an early 1080p default run reported 2 V samples
  differing; re-decoding the identical IVF is bit-exact and the clean run with
  `dav2d --threads 1` had 0 diffs. The same transient appeared once in
  `run_matrix.sh` (1080p full, 2 Y samples); re-running `verify.py` on the same
  IVF+dump reports `BIT-EXACT`. The pinned decoder occasionally produces a
  couple of different samples under concurrent load; use `--threads 1`.

Additional gates:

- `compute-sanitizer --tool memcheck`: 0 errors at 64x64, 640x360 and 64x64
  full-intra. `--tool racecheck`: 0 hazards at 64x64.
- Determinism: every PASS case repeated in a second process with the same MD5;
  the unit test performs two encodes per case in one process (arena reuse),
  including a 16 MiB `--vram-cap` and a 64x64 -> 640x360 growth/reinit.
- Error paths: odd 63x65 rejected by the CLI; no input -> usage/exit 2; CPU-only
  build -> clear `--still-gpu requires a CUDA build`; `--vram-cap` below the working
  set -> `--vram-cap N MiB is below the required working set (M MiB)`; no CUDA
  device -> CPU fallback with a note and identical bytes; unsupported tiny
  geometries -> clear partition error (CPU and GPU).

Integration fixes from this pass:

1. `validate_options` rejects odd dimensions before plan/GPU work.
2. An explicit `--vram-cap` below the required working set is now a hard error
   instead of a silent CPU fallback; an explicit cap also forces the arena to
   shrink/reinit if a previous reservation exceeded it.
3. `still_gpu_test --vram-cap N` exercises capped arenas and the growth/retry path
   in one process.

No bitstream-integration bug was found: every encodable case has identical CPU
and GPU IVF bytes. The decode failures and depth-8 rejections are in the shared
CPU plan/syntax/header path (outside `src/cuda/pipeline/still/` root files) and are
reported, not worked around.

## Runtime ownership

- One `Context` + persistent `Arena` per process. Capacity is device-derived
  (`min(total, free) - 512 MiB`, optional `vram_cap_mb`, clamped to
  `max(256 MiB, 1.5 × working set)`); grow-only at a frame boundary.
- Pinned staging: one upload group, one download group, one
  `Context::synchronize()` per frame.
- All scratch (packed pred/residual, search scratch, descriptors, 3× levels,
  EOB, recon state, deblock metadata) is arena-resident.

## Build and test

```sh
BUILD_DIR=build-m1-orch/m1-gpu-test src/cuda/pipeline/still/build_still_gpu_test.sh
BUILD_DIR=build-m1-orch/m1-gpu-test \
  src/cuda/pipeline/still/build_still_gpu_test.sh -- --synthetic 1920x1080
```

The standalone script compiles the CPU reference, plan bridge, stream bridge,
scheduler bridge, `intra_modes`/`deblock`/`cfl`/`tx`/`recon`/`search`/`chroma`
CUDA components, and the orchestrator with `nvcc --fmad=false -arch=sm_86`
(the tree has both CPU and CUDA `intra_modes`/`deblock` TUs; object names are
indexed to avoid collisions). The root CMake `av2enc_still_gpu` OBJECT target
links the same set (`tools/sync_remote.sh` excludes `build-*`).

## Remaining work

1. Batch the block loop by scheduler dependency level (`sched::run_level` /
   `LevelGraph`) instead of one launch per block. Correctness is already gated.
2. Add a GPU recon dump (`AV2ENC_STILL_GPU_DUMP`) so `verify_gpu.sh --recon` can
   compare encoder-side recon instead of the decoded-frame fallback.
3. Update the two forced-directional CPU cases in `run_matrix.sh` (F6/F11 at
   1080p) if the CPU forced-mode path is expected to be exact at the proxy.
4. Wire video frames / multiple stills; this path is one still per call.
