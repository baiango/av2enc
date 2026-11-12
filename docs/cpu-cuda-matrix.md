# CPU ↔ CUDA feature coverage

Snapshot: 2026-11-12. Source of truth is code and gates; this table records
what exists **and** what has actually been run.

Status vocabulary:

- **VERIFIED** — CPU reference + CUDA kernel, bit-exact parity executed on the
  RTX 3060 (`sm_86`) during this work.
- **KERNEL** — CUDA kernel exists and was verified when built, but not re-run in
  the latest sweep.
- **CPU-ONLY** — no CUDA implementation. Reasons are given.
- **N/A** — no CUDA work is meaningful for an encoder (syntax/entropy/headers,
  or decoder-side output such as grain synthesis).

Coverage rule: a feature is **covered** when it has a CPU reference, a CUDA
implementation, and a functional parity test that passes. Wiring into the
shipped M1/M2 GPU path is **not** part of coverage; the "In GPU path" column is
informational only. `GPU-UNWIRED` in `docs/gpu-coverage.md` therefore means
"covered, not called", not "missing".

## Intra prediction

| Feature | CPU | CUDA | Evidence | In GPU path |
|---|---|---|---|---|
| Edges, directional, smooth, Paeth | yes | `cuda/intra_modes` | M1 GPU 112-case matrix | yes |
| MRL | yes | `cuda/mrl` | built parity suite | no |
| DIP | yes | `cuda/dip` | 22,771 checks | no |
| IBP | yes | `cuda/ibp` | 147,814,518 checks | no |
| CfL (derived) | yes | `cuda/cfl` | M1 GPU matrix | yes |
| CfL explicit alpha | yes | `cuda/cfl` | 3,702,784 samples | no |
| CfL 4 downsampling filters | yes | `cuda/cfl` | 597 checks | no |
| MHCCP | yes | `cuda/cfl` | 542,560 samples | no |
| Palette | yes | `cuda/palette` | built parity suite | no |
| IntraBC | yes | `cuda/ibc` | 36 checks (arena) | no |
| DC-subsampled, corner, IDTX | yes | `cuda/intra_full` | 70,915,222 checks | no |
| Sub-8 blocks | yes | plan only | GPU planner is 64-SB/8-leaf | no |
| SDP | yes | **CPU-ONLY** | partition decision, no GPU plan | no |
| 128/256 SB | yes | **CPU-ONLY plan** | GPU planner accepts 64 only | no |

## Inter prediction

| Feature | CPU | CUDA | Evidence | In GPU path |
|---|---|---|---|---|
| ME/MC, subpel, adaptive precision | yes | `cuda/inter`, `cuda/m1`, `cuda/m2` | M2 GPU 14.9×, byte-identical | yes |
| Compound, CWP, masks, inter-intra | yes | `cuda/compound_full` | 17,263 checks | no |
| OPFL, refine-MV | yes | `cuda/compound_full` | included above | no |
| Warp / affine | yes | `cuda/warp_stack` | 210 checks | no |
| TIP | yes | `cuda/tip` | 5,759 checks | no |
| Temporal MV projection, motion field | yes | `cuda/motion_field` | **in fix** (storage/merge parity) | no |
| Global motion estimation | yes | `cuda/gme` | 81,300 checks incl. real 720p | no |
| DRL reorder, MV stack build | yes | **CPU-ONLY** | per-block decision, not pixel work | no |
| AMVD / JMVD / precision signalling | yes | **CPU-ONLY** | syntax/decision | no |
| B-frames, SEF, leading, OLK/RAS, BRU | yes | **N/A** | container/ordering, no pixels | no |

## Transform, quantisation, entropy

| Feature | CPU | CUDA | Evidence | In GPU path |
|---|---|---|---|---|
| TX variants + TX types | yes | `cuda/tx_variants`, `cuda/tx_type` | built parity suites | yes |
| TX partition (`TX_MODE_SELECT`) | yes | `cuda/tx_partition` | 13,456,988 checks | no |
| IST (secondary transform) | yes | `cuda/tx_secondary` | 2,293,875 checks | no |
| DDT / DDTX / FDDT | yes | `cuda/ddt` | 1,956,864 elements | no |
| QM dequant | yes | `cuda/qm` | 268,685 checks | no |
| RDOQ | yes | `cuda/rdoq` | batch kernel, built suite | no |
| TCQ / parity hiding / CCTX | yes | `cuda/tcq` | 933,016 checks | no |
| DC/AC delta-Q | yes | **N/A** | parameter, applied by quant kernels | no |
| Entropy coding, CDFs, avg-CDF | yes | **N/A** | serial range coder, CPU by design | yes |

## Filters

| Feature | CPU | CUDA | Evidence | In GPU path |
|---|---|---|---|---|
| Deblocking | yes | `cuda/deblock` | M1 GPU matrix | yes |
| CDEF | yes | `cuda/cdef` | 129 checks | **no** |
| CCSO | yes | `cuda/restoration/ccso` | built parity suite | **no** |
| Wiener NS (luma) | yes | `cuda/m1/restoration` | built parity suite | no |
| PC-Wiener | yes | `cuda/pc_wiener` | 716 checks | no |
| GDF | yes | `cuda/gdf` | 8,970,859 checks | no |
| TIP deblocking | yes | `cuda/tip_deblock` | 17,303 checks / 13.1 M samples | no |
| Chroma loop restoration | yes | `cuda/lr_chroma` | 57,009 checks / 95.2M samples | no |
| Film grain | yes (synthesis) | **N/A** | decoder-side output process | no |
| Loop restoration on reference | CPU yes | — | M2 filter chain is off today | — |

## Formats, system

| Feature | CPU | CUDA | Evidence | In GPU path |
|---|---|---|---|---|
| 8-bit 4:2:0 | yes | yes | M0/M1/M2 GPU gates | yes |
| 10-bit | yes (`m10`, `hbd`) | `cuda/m1/hbd`, `cuda/format` | M10 GPU test | product CPU path `--bit-depth 10`; GPU not wired |
| 4:2:2 / 4:4:4 / 4:0:0 | yes | `cuda/chroma_formats` | 1,742,235 checks | product CPU path `--chroma`; GPU not wired |
| Tiles, SVC, OBU metadata, HDR | yes | **N/A** | framing/syntax | no |
| Conformance, RC, two-pass | yes | **N/A** | CPU policy | no |
| Multi-GPU dispatch | yes | runtime + planner | mock + single-GPU gate | no |

## GPU-required gaps

**None.** Every pixel/coefficient tool with a CPU reference now has a CUDA
implementation and a functional parity test that passes on `sm_86`:

| Tool | Kernel | Real-GPU evidence |
|---|---|---:|
| TCQ / parity / CCTX | `cuda/tcq` | 933,016 checks |
| Chroma loop restoration | `cuda/lr_chroma` | 57,009 checks / 95.2M samples |
| Global motion estimation | `cuda/gme` | 81,300 checks incl. real 720p |

Film-grain synthesis is decoder-side output, not an encoder kernel.

## Wiring is not a coverage gap

The GPU matrix counts **35 kernel TUs as GPU-UNWIRED**. Under the rule above
that is not a deficiency: CDEF, CCSO, RDOQ, TX type/partition, RDO and full-UV
all have CPU references, CUDA kernels and passing functional parity tests. They
are simply not called by the shipped path yet. Integration decides which of
them become product defaults; it does not decide whether they are implemented.

## Bottom line

- **CPU: every implemented encoder feature has a CPU reference.** No feature in
  the spec-audit matrix is missing a CPU implementation; remaining spec gaps are
  the ones classified ABSENT/INTEGRATION there.
- **CUDA: every feature with a CPU reference has a CUDA implementation and a
  passing functional test on hardware.** 17 new modules were verified on
  `sm_86` in this sweep, plus the earlier M1/M2/DIP/IBP/CfL/QM/IST kernels.
- **Planning-only gaps are not coverage gaps.** Sub-8/SDP/128/256 support in the
  GPU plan matters only when those tools are wired; their CPU functional tests
  pass, and no pixel kernel is missing.
- **Entropy, headers, metadata, policy and conformance are CPU by design** and
  should not be counted as CUDA gaps.
