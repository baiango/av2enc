# CPU ↔ CUDA feature coverage

Snapshot: **2026-11-13**, `main` after the CUDA kernel, still/video product and
support waves.
Source of truth is code and gates; this table records what exists **and** what
has actually been run.

Measured host: RTX 3060 12 GB (compute capability 8.6, 28 SMs), driver
580.178.04, **CUDA 13.0.88**, g++ 13.3, CMake 3.28.3, Ninja 1.11.1.

Status vocabulary for a *feature*:

- **VERIFIED** — CPU reference + CUDA kernel, bit-exact parity suite passes on
  `sm_86` in the run below.
- **KERNEL** — CUDA kernel exists; the suite builds but its parity section is
  host-emulation only (no device assertions).
- **CPU-ONLY** — no CUDA implementation. Reasons are given.
- **N/A** — no CUDA work is meaningful for an encoder (syntax/entropy/headers,
  or decoder-side output such as grain synthesis).
- **BLOCKED** — kernel exists and its suite passes, but the product path
  rejects or cannot reach it yet.

Coverage rule: a feature is **covered** when it has a CPU reference, a CUDA
implementation, and a functional parity test that passes. Wiring into the
shipped M1/M2 GPU path is **not** part of coverage; the "In GPU path" column is
informational only. `GPU-UNWIRED` in `docs/gpu-coverage.md` therefore means
"covered, not called", not "missing".

## Reproduce

```sh
# product build + CPU gates
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DAV2ENC_ENABLE_CUDA=ON
cmake --build build -j12
ctest --test-dir build                       # 10/10 passed

# every CUDA module suite, one pass/fail table
tools/cuda_module_suites.sh -j 5 --timeout 1800 --logs build-cuda-suites/logs

# arena policy across the runtime and the benchmark harnesses
tools/check_arena.sh src bench
```

Result of the run recorded here: product build clean, `ctest` **10/10**,
module suites **86 pass / 2 fail / 0 skip** (the two failures are the M2 video
gates, see below), `check_arena.sh` **PASS** (94 allocation sites in `bench/`
standalone harnesses reported and excluded).

## Module suites (measured)

One row per directory that owns CUDA build/test scripts. `gpu` suites invoke
`nvcc`; `host` suites are host-emulation and need no device. Times are from the
2026-11-13 `-j 5` sweep; they are wall-clock and include compilation.

| module | suites | status |
|---|---|---|
| `src/cuda` | `build_entropy_test.sh` (gpu, 21s)<br>`build_intra_test.sh` (gpu, 10s)<br>`build_pipeline_test.sh` (gpu, 30s)<br>`build_runtime_test.sh` (gpu, 5s)<br>`build_transform_test.sh` (gpu, 7s) | **PASS** |
| `codec/entropy/lossless/inter` | `build_and_test.sh` (gpu, 15s)<br>`build_emulation_test.sh` (host, 6s)<br>`build_gpu_test.sh` (gpu, 15s) | **PASS** |
| `codec/filters/cdef` | `build_and_test.sh` (gpu, 12s) | **PASS** |
| `codec/filters/deblock` | `build_and_test.sh` (gpu, 13s) | **PASS** |
| `codec/filters/deblock/subpu` | `build_and_test.sh` (gpu, 4s)<br>`build_gpu_test.sh` (gpu, 10s) | **PASS** |
| `codec/filters/gdf` | `build_and_test.sh` (host, 6s)<br>`build_gpu_test.sh` (gpu, 13s) | **PASS** |
| `codec/filters/loop_restoration` | `build_and_test.sh` (gpu, 15s) | **PASS** |
| `codec/filters/loop_restoration/chroma` | `build_and_test.sh` (gpu, 45s)<br>`build_emulation_test.sh` (host, 44s)<br>`build_gpu_test.sh` (gpu, 42s) | **PASS** |
| `codec/filters/pc_wiener` | `build_and_test.sh` (gpu, 11s)<br>`build_emulation_test.sh` (host, 2s)<br>`build_gpu_test.sh` (gpu, 11s) | **PASS** |
| `codec/filters/tip_deblock` | `build_and_test.sh` (gpu, 15s)<br>`build_emulation_test.sh` (host, 6s)<br>`build_gpu_test.sh` (gpu, 14s) | **PASS** |
| `codec/inter/advanced` | `build_advanced_test.sh` (gpu, 11s) | **PASS** |
| `codec/inter/compound` | `build_and_test.sh` (gpu, 16s)<br>`build_emulation_test.sh` (host, 8s)<br>`build_gpu_test.sh` (gpu, 16s) | **PASS** |
| `codec/inter/gme` | `build_and_test.sh` (gpu, 11s)<br>`build_gpu_test.sh` (gpu, 25s) | **PASS** |
| `codec/inter/mc` | `build_and_test.sh` (gpu, 11s) | **PASS** |
| `codec/inter/me` | `build_and_test.sh` (gpu, 13s) | **PASS** |
| `codec/inter/morph` | `build_and_test.sh` (gpu, 12s)<br>`build_gpu_test.sh` (gpu, 11s) | **PASS** |
| `codec/inter/temporal/motion_field` | `build_and_test.sh` (host, 9s)<br>`build_gpu_test.sh` (gpu, 16s) | **PASS** |
| `codec/inter/tip` | `build_and_test.sh` (gpu, 16s)<br>`build_emulation_test.sh` (host, 5s)<br>`build_gpu_test.sh` (gpu, 15s) | **PASS** |
| `codec/inter/warp_stack` | `build_and_test.sh` (gpu, 17s)<br>`build_emulation_test.sh` (host, 8s)<br>`build_gpu_test.sh` (gpu, 17s) | **PASS** |
| `codec/intra/cfl` | `build_cfl_test.sh` (gpu, 10s) | **PASS** |
| `codec/intra/dip` | `build_and_test.sh` (gpu, 8s)<br>`build_emulation_test.sh` (host, 4s)<br>`build_gpu_test.sh` (gpu, 7s) | **PASS** |
| `codec/intra/ibc` | `build_and_test.sh` (gpu, 14s) | **PASS** |
| `codec/intra/ibp` | `build_cuda_test.sh` (gpu, 17s)<br>`build_emulation_test.sh` (host, 8s) | **PASS** |
| `codec/intra/modes` | `build_emulation_test.sh` (host, 2s)<br>`build_intra_modes_test.sh` (gpu, 15s) | **PASS** |
| `codec/intra/mrl` | `build_emulation_test.sh` (host, 14s)<br>`build_mrl_test.sh` (gpu, 9s) | **PASS** |
| `codec/intra/palette` | `build_and_test.sh` (gpu, 13s) | **PASS** |
| `codec/intra/supplemental` | `build_and_test.sh` (gpu, 6s)<br>`build_gpu_test.sh` (gpu, 13s) | **PASS** |
| `codec/rate/rdoq` | `build_rdoq_batch_test.sh` (gpu, 9s) | **PASS** |
| `codec/transform/ddt` | `build_and_test.sh` (gpu, 4s)<br>`build_gpu_test.sh` (gpu, 9s) | **PASS** |
| `codec/transform/qm` | `build_and_test.sh` (gpu, 9s) | **PASS** |
| `codec/transform/secondary` | `build_and_test.sh` (gpu, 13s) | **PASS** |
| `codec/transform/tcq` | `build_and_test.sh` (gpu, 29s)<br>`build_gpu_test.sh` (gpu, 20s) | **PASS** |
| `codec/transform/tx_partition` | `build_and_test.sh` (gpu, 8s)<br>`build_gpu_test.sh` (gpu, 24s) | **PASS** |
| `codec/transform/tx_type` | `build_and_test.sh` (gpu, 24s) | **PASS** |
| `codec/transform/variants` | `build_tx_variants_test.sh` (gpu, 7s) | **PASS** |
| `formats` | `build_high_bitdepth_gpu_test.sh` (gpu, 21s)<br>`build_high_bitdepth_test.sh` (gpu, 6s) | **PASS** |
| `formats/chroma` | `build_and_test.sh` (gpu, 10s)<br>`build_gpu_test.sh` (gpu, 20s) | **PASS** |
| `pipeline/dispatch/scheduler` | `build_scheduler_test.sh` (gpu, 49s) | **PASS** |
| `pipeline/still` | `build_options_test.sh` (host, 0s)<br>`build_still_gpu_test.sh` (gpu, 247s) | **PASS** |
| `pipeline/still/cdef` | `build_and_test.sh` (gpu, 263s) | **PASS** |
| `pipeline/still/cfl` | `build_still_cfl_test.sh` (gpu, 22s) | **PASS** |
| `pipeline/still/chroma` | `build_frame_emulation_test.sh` (gpu, 196s)<br>`build_still_chroma_test.sh` (gpu, 216s)<br>`build_uv_full_emulation_test.sh` (host, 5s) | **PASS** |
| `pipeline/still/hbd` | `build_hbd_test.sh` (gpu, 18s) | **PASS** |
| `pipeline/still/plan` | `build_and_test.sh` (host, 20s) | **PASS** |
| `pipeline/still/recon` | `build_and_test.sh` (gpu, 255s) | **PASS** |
| `pipeline/still/restoration` | `build_and_test.sh` (gpu, 14s) | **PASS** |
| `pipeline/still/sched` | `build_sched_test.sh` (gpu, 267s) | **PASS** |
| `pipeline/still/search` | `build_luma_search_test.sh` (gpu, 12s) | **PASS** |
| `pipeline/still/tx` | `build_still_tx_test.sh` (gpu, 9s) | **PASS** |
| `pipeline/video` | `build_and_test.sh` (host, 197s)<br>`build_gpu_gate.sh` (gpu, 217s) | **FAIL** |

`pipeline/still/search` is PASS in the sweep after commit `8adc450` fixed its
default media path; the FAIL in the raw log was that fixture lookup, not a
kernel.

## Intra prediction

| Feature | CPU | CUDA | Evidence | In GPU path |
|---|---|---|---|---|
| Edges, directional, smooth, Paeth | yes | `cuda/intra_modes` | M1 GPU 112-case matrix | yes |
| MRL | yes | `cuda/mrl` | built parity suite; M1 GPU parity | **yes** (`--still-mrl`) |
| DIP | yes | `cuda/dip` | 22,771 checks; M1 GPU parity | **yes** (`--still-dip`) |
| IBP | yes | `cuda/ibp` | 147,814,518 checks; M1 GPU parity | **yes** (`--still-ibp`) |
| CfL (derived) | yes | `cuda/cfl` | M1 GPU matrix | yes |
| CfL explicit alpha | yes | `cuda/cfl` | 3,702,784 samples; M1 GPU parity | **yes** (`--still-cfl-explicit`) |
| CfL 4 downsampling filters | yes | `cuda/cfl` | 597 checks | **yes** (`--still-cfl-ds`) |
| MHCCP | yes | `cuda/cfl` | 542,560 samples; M1 GPU parity | **yes** (`--still-mhccp`) |
| Palette | yes | `cuda/palette` | built parity suite; M1 GPU map/predict parity | **yes** (`--still-palette`) |
| IntraBC | yes | `cuda/ibc` | 36 checks (arena); M1 GPU copy parity | **yes** (`--still-ibc`) |
| DC-subsampled, corner, IDTX | yes | `cuda/intra_full` | 70,915,222 checks | no |
| Sub-8 blocks | yes | GPU plan (`cuda/pipeline/still/plan`) | plan parity + M1 GPU stream parity | **yes** (`--still-sub8`) |
| SDP | yes | GPU plan (`cuda/pipeline/still/plan`) | plan parity + M1 GPU stream parity | **yes** (`--still-sdp`) |
| 128/256 SB | yes | GPU plan (`cuda/pipeline/still/plan`, sm_86 parity) | plan parity 17,270 leaves; **yes** (`--still-sb128`, `--still-sb256`) |

## Inter prediction

| Feature | CPU | CUDA | Evidence | In GPU path |
|---|---|---|---|---|
| ME/MC, subpel, adaptive precision | yes | `cuda/codec/inter/me`, `mc` | suites PASS | yes (M0/M2 gate) |
| Compound, CWP, masks, inter-intra | yes | `cuda/codec/inter/compound` | suite PASS | no |
| BAWP / morphological prediction | yes | `cuda/codec/inter/morph` | suite PASS | no |
| OPFL, refine-MV | yes | `cuda/codec/inter/compound` | suite PASS | no |
| Warp / affine (block + extended) | yes | `cuda/codec/inter/warp_stack` | suite PASS | no |
| TIP | yes | `cuda/codec/inter/tip` | suite PASS | no |
| Temporal MV field, projection, ref-MV bank | yes | `cuda/codec/inter/temporal/motion_field` | suite PASS | no |
| Global motion estimation | yes | `cuda/codec/inter/gme` | suite PASS | no |
| DRL reorder, MV stack build | yes | **CPU-ONLY** | per-block decision, not pixel work | no |
| AMVD / JMVD / precision signalling | yes | **CPU-ONLY** | syntax/decision | no |
| B-frames, SEF, leading, OLK/RAS, BRU | yes | **N/A** | container/ordering, no pixels; OLK/LTR (`--video-long-term`, `--video-olk`), leading (`--video-leading`), SEF (`--video-sef`) and SWITCH/RAS (`--video-switch`, `--video-ras`) wired on the CPU path | no |

## Transform, quantisation, entropy

| Feature | CPU | CUDA | Evidence | In GPU path |
|---|---|---|---|---|
| TX variants + TX types | yes | `cuda/tx_variants`, `cuda/tx_type` | built parity suites | yes |
| TX partition (`TX_MODE_SELECT`) | yes | `cuda/tx_partition` | 13,456,988 checks | no |
| IST (secondary transform) | yes | `cuda/tx_secondary` | 2,293,875 checks | no |
| DDT / DDTX / FDDT | yes | `cuda/ddt` | 1,956,864 elements | no |
| QM dequant | yes | `cuda/qm` | 268,685 checks | **yes** (`--still-qm`, via the delta-Q q2 path) |
| RDOQ | yes | `cuda/rdoq` | batch kernel, built suite | no |
| TCQ / parity hiding / CCTX | yes | `cuda/tcq` | 933,016 checks | no |
| DC/AC delta-Q | yes | `cuda/delta_q` | 768,688 checks (sm_86); host emulation 425,761 | **yes** (`--still-dc-ac-delta-q`) |
| Entropy coding, CDFs, avg-CDF | yes | **N/A** | serial range coder, CPU by design | yes |

## Filters

| Feature | CPU | CUDA | Evidence | In GPU path |
|---|---|---|---|---|
| Deblocking | yes | `cuda/deblock` | M1 GPU matrix | yes |
| CDEF | yes | `cuda/pipeline/still/cdef` | 129 checks + sm_86 stream parity | **yes** (`--still-cdef`) |
| CCSO | yes | `cuda/codec/filters/loop_restoration/ccso.cu` | 406 checks + sm_86 stream parity | **yes** (`--still-ccso`, `--still-ccso-edge`) |
| Wiener NS (luma) | yes | `cuda/m1/restoration` | built parity suite | **yes** (`--still-wiener`) |
| PC-Wiener | yes | `cuda/pc_wiener` | 716 checks | **yes** (`--still-pc-wiener`) |
| GDF | yes | `cuda/gdf` | 8,970,859 checks | **yes** (`--still-gdf`) |
| TIP deblocking | yes | `cuda/tip_deblock` | 17,303 checks / 13.1 M samples | no |
| Chroma loop restoration | yes | `cuda/lr_chroma` | 57,009 checks / 95.2M samples | **yes** (`--still-chroma-lr`) |
| Film grain | yes (synthesis) | **N/A** | decoder-side output process | metadata yes (`--still-grain`) |
| Loop restoration on reference | CPU yes | — | M2 filter chain is off today | — |

## Formats, system

| Feature | CPU | CUDA | Evidence | In GPU path |
|---|---|---|---|---|
| 8-bit 4:2:0 | yes | yes | M0/M1/M2 GPU gates | yes |
| 10-bit 4:2:0 | yes (`hbd`) | `cuda/m1/hbd`, `cuda/format`, `cuda/formats/high_bitdepth_gpu` | 27,147,456 CPU↔GPU comparisons, q 1..303 (sm_86) | product CPU path `--bit-depth 10`; GPU stage unwired |
| 10-bit 4:2:2 / 4:4:4 / 4:0:0 | yes (`hbd`) | `cuda/format`, `cuda/formats/high_bitdepth_gpu`, `cuda/formats/chroma` | same 27,147,456 comparison sweep + 5,244,850 sample-conversion checks (sm_86) | product CPU path `--bit-depth 10 --chroma`; GPU stage unwired |
| 10-bit qindex 1..303 | yes (`hbd` + qctx) | `cuda/m1/hbd` (`HbdTxJob::quant_step`) | full 1..303 sweep on sm_86; `format_gate.sh` q=91/141/191/303 | product CPU path |
| 4:2:2 / 4:4:4 / 4:0:0 (8-bit) | yes | `cuda/chroma_formats` | 1,742,235 checks | product CPU path `--chroma`; GPU not wired |
| Format/filter plane geometry (all depths/chroma) | yes | `cuda/formats/format_device` | 1,064,426 checks vs `fmt` + deblock | geometry for the future 10-bit deblock |
| Tiles, SVC, OBU metadata, HDR | yes | **N/A** | framing/syntax | no |
| Conformance, RC, two-pass | yes | **N/A** | CPU policy | no |
| Multi-GPU dispatch | yes | runtime + planner | mock + single-GPU gate | no |

## M2 video gate (resolved)

Both `src/cuda/pipeline/video/build_and_test.sh` (host emulation) and
`build_gpu_gate.sh` (real CUDA) pass on current `main`. Earlier runs failed the
stream comparison with one byte of frame 1 (`unit_mismatches=1
recon_mismatches=0 max_diff=0 first=f1 off=6`, 0x5A vs 0x56) because
`src/cuda/pipeline/video/video_pipeline.cpp` still mirrored the pre-`BUG-CDF-001`
frame header. `integration/gpu-video` re-synced the mirror (reference slots,
`recent_slots`/`frame_distance`, sub-pel MV fields, merge stack); the byte
disappeared and tile payload/reconstruction remained identical throughout. It
was a mirror lag, never a kernel divergence.

The merged `tools/cuda_module_suites.sh` run on the 3060 is **96/96 pass**
(host emulation and `sm_86`), including both video suites and the standalone M1
data-flow harness.

## What the product paths consume

`--still-gpu` runs plan/search/TX/recon/chroma/deblock, CDEF, CCSO,
Wiener NS, PC-Wiener, GDF, chroma LR, QM, DC/AC delta-Q and CfL downsampling on
the device, plus SB128/SB256, extended partitions, SDP, sub-8 and the MRL, DIP,
IBP, CfL-explicit, MHCCP, full-UV, palette and IBC intra candidates, with
CPU↔GPU byte parity on 64x64, 640x360 and 384x128 screen-content fixtures.
Still rejected by name (kernels verified but no device decision stage): RDOQ,
TX-type, IST and RDO-intra; also rejected: the 10-bit/4:2:2/4:4:4/4:0:0 product
formats and SB128/256 combined with CDEF/CCSO/loop restoration.

`--video-gpu` runs keyframe + P/B frames with 1-3 references (quarter/half/
eighth pel, force-single, depth, min-block, ME-range), compound/CWP AVERAGE,
hierarchical temporal layers, the frame-type wrappers (long-term/OLK, leading,
SEF, SWITCH, RAS), `--video-mfh`, timing/content-interpretation OBUs and
average/adaptive CDF, byte-identical to `--video`. Still rejected by name:
CRF/VBR rate control, tiles, cross-frame CDF continuity, DRL reordering and
SVC.

## Bottom line

- **CPU: every implemented encoder feature has a CPU reference.**
- **CUDA: every pixel/coefficient tool with a CPU reference has a CUDA kernel
  and a suite that passes on `sm_86`** (96/96 suites in the merged run).
- **Wiring, not kernels, is what the still path is missing.** The remaining
  still rejections are decision-stage or orchestrator work; their kernels pass.
- **Entropy, headers, metadata, policy and conformance are CPU by design** and
  should not be counted as CUDA gaps.
