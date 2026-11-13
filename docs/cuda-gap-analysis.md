# CUDA integration gap analysis

Snapshot: **2026-11-13**, `main` at `fcf83e0`, plus the three in-flight video
branches listed in §6.

Scope: every encoder-relevant tool in the AV2 Bitstream & Decoding Process
Specification v1.0.0 (§5.20 syntax, §6.4/§6.17 headers, §7.9–§7.22 processes,
§8 entropy, §9 tables) mapped to its CPU reference, its CUDA kernel and its
product wiring (`--still-gpu`, `--video-gpu`).

Spec identity: 1169 pages, commit `38f28e7`, SHA-256
`e9916f091e4e83446aad6b4601641c5b292e569c144c4163b26a4497573b533f`
(also recorded in `docs/bitrate-features.md`).

## Reading the tables

- **CPU ref** — the reference implementation every CUDA path must match
  bit-exactly.
- **CUDA kernel** — the device implementation. Unless marked otherwise it has
  a passing CPU↔GPU parity suite on `sm_86`; `tools/cuda_module_suites.sh`
  currently reports 96/96 suites passing.
- **Product state** — what `--still-gpu` / `--video-gpu` actually consumes
  today. "rejected" means the flag exits 2 by name; there is no silent CPU
  fallback.

## 0. Summary

Pixel-kernel coverage is essentially complete: every major prediction,
transform, quantization and filter process in §7.13–§7.20 has a device kernel
with a parity suite. The missing work is:

| Class | Count | Meaning |
|---|---:|---|
| Video product integration | 15 tools | kernel + parity exist, `--video-gpu` does not use them |
| Still product integration | 11 tools | kernel + parity exist, `--still-gpu` rejects them |
| No CUDA kernel | 5 areas | decision/search or framing work, not pixel kernels |
| CPU by design | 6 areas | entropy writer, headers, reference lists, output |

Overlaps (10-bit formats, tiles, delta-Q, `dc_sign`) appear once per product
path where the consequence differs.

## 1. `--video-gpu` — spec tools with kernels but no product path

Product wiring lives in `src/cuda/pipeline/video/video_pipeline.cpp`;
`src/pipeline/video/video_frame.cpp` is the CPU reference it mirrors.

| Spec | Tool | CPU ref | CUDA kernel | Product state |
|---|---|---|---|---|
| §7.13.2.2–.18 | directional / smooth / edge-filtered intra | `src/codec/intra/modes/`, `src/codec/intra/supplemental/` | `src/cuda/codec/intra/modes/intra_modes.cu`, `.../supplemental/intra_full.cu`, `src/cuda/pipeline/video/intra/` | DC by default; `--video-intra-full` runs the still product's 61-candidate luma search (device module, opt-in, byte-identical CPU/GPU) |
| §5.20.6, §7.15 | TX size / type / partition search | `src/codec/transform/variants/`, `tx_type/`, `tx_partition/` | `src/cuda/codec/transform/{variants,tx_type,tx_partition}/`, `src/cuda/pipeline/video/tx/` | wired: `--video-tx-search` (opt-in, byte-identical CPU/GPU) |
| §7.17 | deblocking | `src/codec/filters/deblock/` | `src/cuda/codec/filters/deblock/` | wired: `--video-deblock` (§5.18.7.9 header + filtered reference) |
| §7.18 | CDEF | `src/codec/filters/cdef/` | `src/cuda/codec/filters/cdef/cdef.cu` | wired: `--video-cdef` |
| §7.19 | CCSO | `src/codec/filters/loop_restoration/` | `src/cuda/codec/filters/loop_restoration/ccso.cu` | wired: `--video-ccso` |
| §7.20 | Wiener NS / chroma LR / GDF | `src/codec/filters/{loop_restoration,gdf}/` | `src/cuda/codec/filters/{loop_restoration,gdf}/` | wired: `--video-wiener`, `--video-chroma-lr`, `--video-gdf` |
| §7.20.4 | PC-Wiener | `src/codec/filters/pc_wiener/` | `src/cuda/codec/filters/pc_wiener/` | **not wired** on video (needs its own `lr_params()`/`read_lr()` writer path) |
| §7.13.3.19–.24 | warp / affine / warp estimation | `src/codec/inter/{warp_stack,advanced}/` | `src/cuda/codec/inter/{warp_stack,advanced}/` | wired: `--video-gme/--video-warp` GLOBALMV warp |
| §7.13.3.23 | global motion estimation | `src/codec/inter/gme/` | `src/cuda/codec/inter/gme/gme.cu` | wired: `--video-gme` on the single-reference P chain (`--video-refs > 1` rejected until the per-slot model is re-synced) |
| §7.10, §7.13.3.2 | TIP | `src/codec/inter/tip/` | `src/cuda/codec/inter/tip/tip.cu` | never called |
| §7.13.3.3–.12 | OPFL / refine-MV | `src/codec/inter/compound/` | `src/cuda/codec/inter/compound/compound_full.cu` | never called |
| §7.13.3.25 | BAWP | `src/codec/inter/{advanced,morph}/` | `src/cuda/codec/inter/advanced/advanced.cu` | never called |
| §7.13.3.26 | morphological prediction | `src/codec/inter/morph/` | `src/cuda/codec/inter/morph/morph.cu` | never called |
| §7.13.3.27–.30 | wedge / diffwtd / intra-variant / mask blend | `src/codec/inter/advanced/` | `src/cuda/codec/inter/advanced/` | only equal-weight AVERAGE compound |
| §7.13.3.15 | save sub-PU | `src/codec/inter/` | `src/cuda/codec/filters/deblock/subpu/subpu.cu` | not wired |
| §7.22 | motion-field storage / temporal MV | `src/codec/inter/temporal/` | `src/cuda/codec/inter/temporal/motion_field/` | not wired |
| §7.13.5 | CfL chroma | `src/codec/intra/cfl/` | `src/cuda/codec/intra/cfl/cfl.cu` | header enables CfL; the encoder never emits `is_cfl = 1` |
| §7.13.6–.8 | MHCCP | `src/codec/intra/cfl/mhccp/` | `src/codec/intra/cfl/mhccp/mhccp_gpu.cu` | header signals `enable_mhccp = 0`; still path searches it |
| §5.18.2, §7.8 | NumRefFrames 4–8, reference lists | `src/pipeline/video/` | `src/cuda/codec/inter/mv_stack/` | NumRefFrames 4–8 + `NumTotalRefs` 1–7 wired (per-block `single_ref`); §7.8 implicit past/future lists rejected by name (`--video-implicit-ref-map`), explicit map only |
| §5.20.7.27, §8.3 | `dc_sign` | `src/codec/intra/supplemental/` | `src/cuda/codec/intra/supplemental/intra_full.cu` | kernel exists, not wired |
| §5.13, §5.20.7 | QM / delta-Q / segmentation signalling | `src/codec/transform/qm/`, `src/codec/entropy/deltaq/` | `src/cuda/codec/transform/{qm,delta_q}/` | header writes `using_qmatrix = 0`, `segmentation_enabled = 0`, `delta_q_present = 0` |
| §5.19 | tiles | `src/pipeline/tiles/` | — | wired for `--video-gpu` (tile-local coded geometry/recon, per-tile MV grids, CPU tile-group writer) |
| §5.4.12 | CRF/VBR rate control | `src/codec/rate/rc/` | — | wired for `--video-gpu` (per-frame `base_q_idx` drives the device qlookup step) |
| §5.18.2 | cross-frame CDF continuity | `src/codec/entropy/` | — | rejected by name (CPU path ungated by the pinned decoder) |
| §7.3.6, §6.8–.10 | SVC / extended layers / LCR | `src/pipeline/svc/` | — | rejected by name |
| §6.4.1, §7.15 | 10-bit and 4:2:2 / 4:4:4 / 4:0:0 | `src/formats/` | `src/cuda/formats/` | `--video` rejects non-8-bit/420 entirely |

### Wired in `--video-gpu`

keyframe + P/B frames, 1–7 references over a 4–8 slot `NumRefFrames` pool,
quarter/half/eighth pel, force-single,
depth/min-block/ME-range, compound AVERAGE (equal CWP), B frames, temporal
layers + drop, long-term/OLK/leading/SEF/SWITCH/RAS, MFH, timing +
content-interpretation OBU, average/adaptive CDF, the rate-aware
inter-residual decision (`fcf83e0`), §5.18.9 global motion + §7.13.3.19/.20
GLOBALMV warp (`--video-gme` / `--video-warp`; single-reference), §5.20.6/
§5.20.8 TX size/type/partition search (`--video-tx-search`), §7.13.2 full intra
search (`--video-intra-full`), §7.17 deblocking, §7.18 CDEF, §7.19 CCSO,
§7.20 Wiener/chroma LR/GDF, §5.4.12 CRF/VBR and §5.19 multi-tile.

### Remaining advanced-inter prerequisites

The other §7.13.3 tools (TIP with the §7.22 motion field, BAWP/morph,
OPFL/refine-MV, wedge/diffwtd/intra-variant compound) share one blocker: the
product tile writer only carries the minimal CDF/context set in
`src/pipeline/video/video_cdf_tables.h`, while each tool codes new arithmetic
symbols (`tip_mode`, `tip_pred_mode`, `use_bawp`, `comp_group_idx`,
`compound_type`, wedge bits, `use_refinemv`, ...).  The component CDFs exist
(`src/codec/inter/advanced/video/advanced_cdf_tables.h`,
`src/codec/inter/multiref/gate/compound_cdf_tables.h`; the TIP CDFs are in
`spec/tables/default_cdf.json` but are not generated into the video tables),
but the block-symbol writers and decision loops are not wired into
`video_syntax.cpp`/`video_frame.cpp`.  Enabling `enable_ref_frame_mvs` also
inserts temporal merge candidates that the encoder stack does not currently
feed, so TIP needs that stack parity first.  These are integration gaps, not
decoder rejections: no stream carrying the symbols has been built yet.
inter-residual decision (`fcf83e0`), and the luma TX size/type/partition search
(`--video-tx-search`, device `videotx` module).
## 2. `--still-gpu` — spec tools with kernels but no product path
Product wiring lives in `src/cuda/pipeline/still/still_gpu.cu`; the rejection
list is `reject_cpu_only` in `src/main.cpp` and `gpu_m3_options_supported()` in
`src/cuda/pipeline/still/still_gpu.h`.

| Spec | Tool | CPU ref | CUDA kernel | Product state |
|---|---|---|---|---|
| §7.15.3 | IST secondary transform | `src/codec/transform/secondary/` | `src/cuda/codec/transform/secondary/tx_secondary_gpu.cu` | wired (`--still-ist`) |
| §5.20.7.27, §9.6 | TX type search | `src/codec/transform/tx_type/` | `src/cuda/codec/transform/tx_type/tx_type.cu` | wired (`--still-tx-type`, `--still-idtx`) |
| §5.20.6.1–.3 | TX partition | `src/codec/transform/tx_partition/` | `src/cuda/codec/transform/tx_partition/tx_partition_gpu.cu` | rejected |
| §7.14.4 | RDOQ | `src/codec/rate/rdoq/` | `src/cuda/codec/rate/rdoq/rdoq_batch.cu` | rejected |
| §7.15.4 | IDTX | `src/codec/transform/variants/` | `src/cuda/codec/transform/variants/tx_variants.cu` | wired (`--still-idtx`) |
| §7.14.6 | DDT / DDTX / FDDT | `src/codec/transform/ddt/` | `src/cuda/codec/transform/ddt/ddt_gpu.cu` | rejected |
| §9.3 | TCQ / parity hiding / CCTX | `src/codec/transform/tcq/` | `src/cuda/codec/transform/tcq/tcq.cu` | CCTX wired (`--still-cctx`); TCQ / parity hiding still rejected |
| §7.13.4 | palette | `src/codec/intra/palette/` | `src/cuda/codec/intra/palette/palette.cu` | wired (GPU map/predict; CPU-planned colour set) |
| §5.20.5.4 | IntraBC | `src/codec/intra/ibc/` | `src/cuda/codec/intra/ibc/ibc.cu` | wired (GPU copy; CPU-planned block vector) |
| §7.13.5 | full-UV chroma search | `src/codec/intra/cfl/` | `src/cuda/pipeline/still/chroma/still_chroma.cu` | wired |
| §5.20.7.27, §8.3 | `dc_sign` | `src/codec/intra/supplemental/` | `src/cuda/codec/intra/supplemental/intra_full.cu` | kernel exists, not wired |
| §6.12, §9.4 | user-defined QM OBU | `src/codec/transform/qm/` | `src/cuda/codec/transform/qm/qm_gpu.cu` | rejected (pinned decoder rejects QM OBUs) |
| §5.20.7 | segmentation / neutral delta-Q / per-segment Q | `src/codec/entropy/deltaq/still/` | `src/cuda/pipeline/still/still_gpu.cu` device segment map + `src/cuda/codec/transform/delta_q/deltaq_gpu.cu` | wired: neutral table/delta-Q and the per-SB `SEG_LVL_ALT_Q` device map |
| §6.13, §7.21.7 | adaptive film grain | `src/codec/filters/tables/film_grain.cpp` (encoder signals metadata; synthesis is a decoder-side output process) | — | `--still-grain-adaptive` rejected; static signalling wired |
| §6.14 | content interpretation | `src/pipeline/still/` | — (metadata OBU) | rejected |
| §5.19 | multi-tile | `src/pipeline/tiles/` | — | rejected |
| §5.20.6.1 | lossless FSC / DPCM | `src/codec/entropy/lossless/` | `src/cuda/codec/entropy/lossless/inter/lossless_inter.cu` (inter) | `--lossless` is CPU-only |
| §6.4.1, §7.15 | 10-bit / 4:2:2 / 4:4:4 / 4:0:0 | `src/formats/`, `src/pipeline/still/high_bitdepth/` | `src/cuda/formats/`, `src/cuda/pipeline/still/hbd/` | wired: 10-bit 420/422/444/400 and 8-bit 422/444/400 (still; video still rejects formats) |
| §5.18.1, §5.18.4 | SB128/256 with CDEF/CCSO/LR | `src/codec/filters/` | filter kernels derive unit grids from `plan.sb_size` | wired (CCSO 6/7, LR sb_shift 4/5, PC-Wiener/chroma-LR 128 units at SB128) |
| §5.20.6 | RDO intra decision | `src/codec/rate/rdo/intra_decision.cpp` | none | rejected |

### Wired in `--still-gpu`

plan/search/recon/chroma/deblock, CDEF (on-skip/tx), CCSO band/edge, Wiener NS,
PC-Wiener, GDF, chroma LR, QM, DC/AC delta-Q, neutral segmentation / delta-Q,
device per-SB `SEG_LVL_ALT_Q` segmentation, CfL downsampling, CfL explicit,
MHCCP, MRL, DIP, IBP, SB128/SB256 (including CDEF/CCSO/LR unit grids),
extended partitions, SDP, sub-8, full-UV chroma search, palette, intra block
copy, film-grain signalling, and the product optical formats: 10-bit
4:2:0/4:2:2/4:4:4/4:0:0 (`high_bitdepth_gpu` block stage + `hbd` writer) and
8-bit 4:2:2/4:4:4/4:0:0 (`ChromaGpuExecutor` seam).

## 3. Spec areas with no CUDA kernel

| Spec | Area | Why it is not a pixel kernel |
|---|---|---|
| §7.13.2 + §7.15 | RDO intra decision | encoder search over candidate modes; CPU `src/codec/rate/rdo/intra_decision.cpp` |
| §5.20.7 | segmentation map / per-segment Q | per-superblock state, not a block transform |
| §5.4.12 | CRF/VBR + HRD / two-pass | frame-level rate control |
| §5.19 | tile loop / tile-group assembly | framing and per-tile geometry |
| §7.3.6, §6.8–.10 | SVC / LCR / atlas / operating point | multi-layer framing |

## 4. Correctly CPU-only by design

| Spec | Area | Reason |
|---|---|---|
| §8 | range coder, coefficient CDFs | serial arithmetic coding; a byte-exact CUDA coder exists (`src/cuda/entropy.cu`) but the product keeps the host writer |
| §6.2–§6.19 | OBU and header syntax | tens of bytes per frame |
| §7.7–§7.8 | reference-list construction | frame-serial bookkeeping |
| §7.23 | reference frame update | frame-serial bookkeeping |
| §7.21 | output process, film-grain synthesis | decoder-side output |
| §9 | tables | constants |

## 5. Merged implementation wave

All branches from the first completion wave have merged and passed their
product gates on the 3060:

| Branch | Landed |
|---|---|
| `integration/video-deblock` | §7.17 `--video-deblock` (header, MI grid, filtered reference, CPU+CUDA) |
| `integration/video-intra` + `-wire` | §7.13.2 61-candidate luma search (`--video-intra-full`) |
| `integration/video-tx` + `-wire` | §5.20.6/§5.20.8 TX size/type/partition search (`--video-tx-search`) |
| `integration/video-loopfilters` | §7.18 CDEF, §7.19 CCSO, §7.20 Wiener/chroma LR/GDF |
| `integration/video-refs` | §5.4.6 NumRefFrames 4–8, NumTotalRefs 1–7 |
| `integration/video-rc-tiles` | §5.4.12 CRF/VBR, §5.19 multi-tile |
| `integration/video-advanced-inter` | §5.18.9 global motion + §7.13.3.19/.20 GLOBALMV warp |
| `integration/still-tx-tools` | ±IST, TX type, IDTX, CCTX |
| `integration/still-intra-tools` | palette, IntraBC, full-UV chroma |
| `integration/still-formats` | 10-bit and 8-bit chroma product paths |
| `integration/still-seg-filters` | segmentation/delta-Q/per-segment Q, SB128/256 filter grids |

Remaining advanced-inter work (TIP + motion field, BAWP/morph, OPFL/refine-MV,
wedge/diffwtd/intra-variant compound) still needs the missing video CDFs and the
§7.22 merge-stack feed documented in `docs/negative-results.md`.

## 6. Suggested order for the remaining work

1. Video: attach the missing advanced-inter CDFs and §7.22 motion field, then wire TIP, BAWP/morph, OPFL/refine-MV and the masked compound variants.
2. Video: PC-Wiener (`lr_params` dispatch) and §5.13/§5.20.7 QM/delta-Q/segmentation signalling.
3. Video: §5.18.2 cross-frame CDF continuity once the pinned decoder is gated; SVC layers.
4. Still: TX-partition, RDOQ and TCQ device rate-model stages; `dc_sign`; RDO intra.
5. Still: per-segment Q combined with DC/AC delta-Q; user-QM OBU (decoder-gated); 10-bit deblock arithmetic.
6. Remaining metadata-only tools (adaptive film grain, content interpretation) and tiles/SVC framing.

## 7. Evidence

```sh
# kernel + parity matrix on the RTX 3060 (96/96 at the snapshot)
tools/cuda_module_suites.sh -j 5 --timeout 1800

# product surfaces
src/main.cpp                      # reject_cpu_only(), --video-gpu reject_gpu()
src/cuda/pipeline/still/still_gpu.h   # gpu_m3_options_supported()
src/pipeline/video/video_headers.cpp  # deblock/QM/segmentation header bits
```

Quality context for the video gaps (120-frame 360p heavy clip):

| Configuration | Size | PSNR-Y |
|---|---:|---:|
| delivered default (skip-only inter) | 12.15 MB | 39.17 dB |
| residual + RD fix (`fcf83e0`) | 3.56 MB | 39.86 dB |
| + 3 references + adaptive CDF | 2.44 MB | 39.70 dB |
| x264 CRF 16 | 1.16 MB | 39.71 dB |
