# AV2 compression tools — catalog and implementation priority

Status: planning document for `av2enc`. M0 is deliberately DC-intra-only and will
be far from competitive; this document catalogs the AV2 tools that close the gap.
The dependency-ordered implementation plan is in
[`tool-roadmap.md`](tool-roadmap.md).

Normative reference: **AV2 Bitstream & Decoding Process Specification v1.0.0**
(2026-05-28, commit `38f28e7`, 1169 pages, SHA-256
`e9916f091e4e83446aad6b4601641c5b292e569c144c4163b26a4497573b533f`).
Section numbers below are from that document. The spec defines syntax and decoding,
not encoder algorithms and not BD-rate.

## 0. Read this first

### 0.1 Every feature has a decoder gate

The validation decoder is **dav2d, pinned at `f9dafbc`** (nightly 2026-11-10).
Every encoder feature is considered done only after a stream that exercises it
decodes with that binary:

```sh
tools/validate_dav2d.sh out.ivf
```

The gate order is always:

1. **Decoder gate** — the command above exits 0, the decoder reports the expected
   frame count and resolution, and no decode errors are emitted.
2. **Reconstruction gate** — the decoder's Y4M output matches the encoder's internal
   reconstruction (0 LSB preferred; more than 1 LSB is a bug, because in-loop
   prediction will drift).
3. **Quality/BD-rate gate** — only now measure PSNR/SSIM/BD-rate on the *decoded*
   output, never on the encoder's internal reconstruction.

A feature that fails the decoder gate is **not done**, no matter how good its
encoder-side statistics look. It stays disabled by default. Record the failure as
`K` (known incomplete at `f9dafbc`) with the smallest reproducer.

### 0.2 M0 and early-stage restriction

M0 and all pre-M1 bring-up streams must stay inside the baseline subset the project
requires dav2d to decode:

> 8-bit 4:2:0, one tile, key frame, DC intra, 32x32 DCT, no filters, no film grain,
> no SVC, no palette, no CfL.

Do not "fix" an M0 stream by adding an M1 tool. M0 must pass exactly as scoped.
Only after M0 passes do M1 tools get added, one gated feature at a time.

### 0.3 Gain figures are estimates, not normative

The spec does not state BD-rate gains. Every percentage in this document is an
**engineering estimate** for 1080p natural content at ordinary QPs, unless a cell
explicitly says otherwise. Ranges are wide on purpose. They are not measurements
from this repository and must not be quoted as AV2 guarantees. Actual gains depend
on content, QP, encoder decisions, and how well the tool is integrated.

### 0.4 Column definitions

**Side**

| Code | Meaning |
|---|---|
| `E` | Encoder-only decision/processing. No new bitstream syntax. Output still must decode. |
| `B` | Bitstream syntax plus normative decoder process. Encoder must both decide and mirror the decoder exactly. |
| `D` | Decoder-side process/synthesis. Encoder estimates/signals parameters; if the process is in the prediction loop, the encoder still needs a mirror. |
| `N/A` | No bitstream syntax, or tool absent from AV2. |

**Complexity** (solo developer, scalar reference first, then GPU)

| Code | Rough meaning |
|---|---|
| `S` | Days. Mostly syntax, table lookup, or a small kernel. |
| `M` | 1–3 weeks. New syntax plus a moderate kernel/search, or a decoder process with several edge cases. |
| `L` | 3–8 weeks. Multiple interacting modules, large tables, or a search/RDO loop. |
| `XL` | 8+ weeks. New subsystem, motion-field/pyramid machinery, or a large decoder-process surface. |

**Stage / needed?**

| Code | Meaning |
|---|---|
| `M0` | Already in the M0 baseline. |
| `M1` | Needed for the smallest respectable still-image encoder. |
| `M2` | Needed for the baseline P-frame/inter encoder. |
| `M3` | Needed for a minimally competitive encoder (filters + RDO). |
| `M4` | Advanced: required to approach the frontier, not required for a credible M3 encoder. |
| `Content` | Content-specific (screen content, fades, grain, etc.). |
| `No` | Not needed, not useful, or not in AV2. |

**dav2d gate**

| Code | Meaning |
|---|---|
| `G0` | M0 baseline subset. Must pass before M0 exits. Not yet proven for our stream. |
| `U1` | Unverified at `f9dafbc`; lower risk (well-established syntax family, simpler process). `U1` is **not** a support claim. |
| `U2` | Unverified at `f9dafbc`; higher risk (new AV2 tool, large tables, or many decoder interactions). Isolate and gate before building dependencies. |
| `K` | Known incomplete/failing at `f9dafbc`. None recorded yet; populate when a minimal stream fails. |
| `N/A` | No bitstream syntax, or the tool is absent from AV2. |

> No tool in this document is marked "supported." At the time of writing no
> isolated stream has been produced in this repository for any tool, including M0.
> M0 is `G0`; every other entry is `U1` or `U2`. A row becomes `PASS(f9dafbc)` only
> after the exact command above succeeds and the result is recorded.

---

## 1. Partitioning and block structure

Partitioning is the highest-leverage family in the codec: it changes the granularity
of prediction, transform, and mode signaling at the same time. M0 fixes a 64x64
superblock and a 32x32 coding block, which is the single largest quality/rate
handicap after DC-only intra.

**AV2 v1.0.0 correction:** `PARTITION_SPLIT` is defined only for
128x128→64x64 and 256x256→128x128. It is illegal at 64x64 and below; use
`PARTITION_HORZ`/`VERT` (the M0 rect tree) to reach 32x32 quadrants. Do not
implement SPLIT at 64x64.

| Tool | Spec § | Side | Complexity / depends on | What it buys (est.) | Needed? | dav2d gate |
|---|---|---|---|---|---|---|
| Recursive square partition tree (`PARTITION_NONE` / `PARTITION_SPLIT`) | §5.20.3.1, §5.20.3.2, §6.19.3.1, §6.19.3.2 | B | M; needs block bookkeeping and neighbor availability | 10–20% BD-rate vs fixed 32x32; the base for every later tool | M1 | U1 — unverified |
| Rectangular splits (`PARTITION_HORZ` / `PARTITION_VERT`, 2:1) | §5.20.3.1, §5.20.3.2, §6.19.3.1 | B | M; depends on square tree | 2–5% incremental on edges, text, and mixed content | M1 | U1 — unverified |
| Extended partitions: 3-way (`HORZ_3`/`VERT_3`) and uneven 4-way (`HORZ_4A/4B`, `VERT_4A/4B`) | §5.4.3 (`enable_ext_partitions`, `enable_uneven_4way_partitions`), §5.20.3.2, §6.4.3 | B | L; depends on rectangular tree; more partition contexts and edge cases | 1–4% incremental | M4 | U2 — CPU `--video-ext-partitions` gated (`src/pipeline/video/gate_video_partitions.sh`) |
| Semi-decoupled partitioning (SDP; luma/chroma use different trees) | §5.4.3 (`enable_sdp`, `enable_extended_sdp`), §5.20.3.1, §6.4.3 | B | L; depends on partition tree and chroma reconstruction | 1–4%, mostly 4:2:0 mixed/detail content | M4 | U2 — unverified |
| Block-size range 4x4 through 64x64 (and larger blocks under 128/256 superblocks) | §5.20.3, §5.20.4.1, §6.19.3 | B | M; same tree, more size cases | Enables the partition gains above; small blocks help detail, large blocks help flat areas. No separate BD-rate claim. | M1 (8x8–64 first; 4x4 after its own gate) | U1 — unverified |
| 128x128 superblocks (`use_128x128_superblock`) | §5.4.3, §5.18.7.6, §6.4.3 | B | M; larger context arrays, more neighbors, memory | 0.5–2%; fewer partition symbols and better large-block coding on flat content | M4 | U2 — CPU `--video-sb128` gated (`gate_video_partitions.sh`) |
| 256x256 superblocks (`use_256x256_superblock`, inter only; intra is capped at 128x128) | §5.4.3, §5.18.7.6, §6.4.3 | B | L; larger trees, motion-field and filter interactions | 0.3–1.5%; mostly flat/high-resolution content | M4 (optional) | U2 — CPU `--video-sb256` gated (inter roots; keys clamp to 128) |

**Notes.**

- `PARTITION_SPLIT` is the recursive workhorse. Rectangular splits are where most
  of the edge/text gain comes from; extended 3/4-way splits are diminishing returns
  until the rest of the encoder is mature.
- SDP is not a quality tool by itself; it prevents chroma from being forced into a
  luma decision that is bad for chroma. Gate it separately because the tree state
  (`TreeType`, `ChromaMiRow/Col`, chroma offsets) is a common source of decoder
  mismatch.
- 128x128 superblocks are a better first step than 256x256. 256x256 only applies to
  inter frames and increases the blast radius of motion/filter bugs.
- Do not hand-type partition tables. Generate `Partition_Subsize`,
  `H_Partition_Midsize`, and block geometry from the spec tables with a script.

---

## 2. Intra prediction

M0 uses DC only. The full intra tool set is the difference between "valid stream"
and "still image people would ship." The base enum has 13 modes (5 non-directional
plus 8 directional), and angle deltas expand the directional set to 56 variants
(§3 constants; §5.20.5.5).

| Tool | Spec § | Side | Complexity / depends on | What it buys (est.) | Needed? | dav2d gate |
|---|---|---|---|---|---|---|
| Basic intra: DC, V, H | §7.13.2.10, §7.13.2.11, §5.20.5.5 | B | S; M0 has DC only | V/H alone are a large step over DC on edges; folded into the directional gain below | M1 | G0 for DC; U1 for V/H |
| Directional intra, 8 base directions expanded to 56 variants by angle delta (±3 × 3°) | §5.20.5.5, §6.19.5.5, §7.13.2.7, §7.13.2.8 | B | M; needs edge sample preparation and wide-angle mapping | 10–20% vs DC-only on natural stills; the core intra gain | M1 | U1 — unverified |
| Smooth intra (`SMOOTH`, `SMOOTH_V`, `SMOOTH_H`) | §7.13.2.13, §5.20.5.5 | B | S; same edge samples | 1–3% incremental on gradients/soft content | M1 | U1 — unverified |
| Paeth intra | §7.13.2.2, §5.20.5.5 | B | S; same edge samples | 0.5–2% incremental | M1 | U1 — unverified |
| Intra edge filtering (corner filter, above/left filter type, strength selection, edge filter) | §7.13.2.14–§7.13.2.18 | B | S–M; depends on directional prediction | 0.5–2%; mostly visual smoothing along edges | M1 | U1 — unverified |
| Multiple reference line selection (MRL), including secondary-line prediction (`mrl_index`, `mrl_sec_index`) | §5.20.5.5, §6.19.5.5, §7.13.2.7 | B | M; depends on directional prediction; extra line buffers | 0.5–2%; can help strong edges/texture | M4 | U2 — unverified (secondary-line variant is new) |
| Chroma from luma (CfL), explicit and derived alpha | §5.4.5, §5.20.5.6, §5.20.7.32, §7.13.5 | B | M; needs reconstructed luma and chroma DC prediction | 2–6% on 4:2:0; one of the best chroma tools | M1 | U1 — unverified |
| MHCCP (multi-hypothesis cross-component prediction; `cfl_index = CFL_MULTI`) | §5.20.5.6, §5.20.7.32, §7.13.6, §7.13.7, §7.13.8 | B | L; depends on CfL and a 3x3 least-squares solve | 1–3% incremental over CfL on chroma | M4 | U2 — unverified |
| DIP (data-driven intra prediction) | §5.4.5, §5.20.7.9, §6.19.7.8, §7.13.2.3–§7.13.2.6 | B | M; 11-feature 8x8 linear predictor and resampling | 0.5–2%; content-dependent | M4 | U2 — unverified |
| IBP (intra bi-prediction, blending two directional predictions) | §5.4.5, §7.13.2.9, §7.13.2.12 | B | M; depends on directional prediction | 0.5–2% | M4 | U2 — unverified |
| Palette mode (2–8 colors, identity rows, palette cache) | §5.4.7, §5.20.8.1, §5.20.8.4, §6.19.8.1, §7.13.4 | B | M; needs screen-content config and color-map coding | 5–20% on screen content; ~0 on natural video | Content | U2 — unverified |
| Intra block copy (IntraBC), local/global IBC buffers | §5.4.7, §5.18.3.4, §5.20.5.4, §6.17.3.3, §6.19.5.4 | B | M; needs block matching/hash and IBC buffer management | 5–15% on screen content; ~0 on natural video | Content | U2 — unverified |
| DPCM intra (lossless/DPCM luma and chroma) | §5.20.5.5, §5.20.5.6, §7.15.4 | B | S; mostly lossless/screen-content path | Small; only relevant for lossless or synthetic content | Optional | U2 — unverified |

**Notes.**

- The intra mode syntax is not "one of 13 modes." Read §5.20.5.5 carefully:
  `y_mode_set`/`y_mode_index`/`y_mode_offset` and `y_second_mode` select from mode
  sets, and directional modes carry `AngleDeltaY`/`AngleDeltaUV` in ±3 steps of 3°.
  Getting this wrong is a silent quality loss and a decoder mismatch.
- CfL and MHCCP share `cfl_index`; MHCCP is not a separate UV mode. Implement CfL
  first, then extend the same path.
- Palette and IntraBC require screen-content configuration (§5.4.7). Do not enable
  them for natural-content presets just because the syntax is present.
- MRL's secondary-line mode (`mrl_sec_index`) is new relative to older codecs;
  gate it separately from plain MRL.

---

## 3. Transform, quantization, and coefficient coding

M0 has one transform (32x32 DCT) and scalar quantization. AV2 has transform sizes
from 4x4 to 64x64 (including rectangular and 4:1/1:4), several transform types,
transform partitioning, secondary transforms, cross-component transform, and
coefficient-coding tools.

| Tool | Spec § | Side | Complexity / depends on | What it buys (est.) | Needed? | dav2d gate |
|---|---|---|---|---|---|---|
| Variable transform size, 4x4–64x64 including rectangular (2:1, 4:1, 1:4) | §5.20.6.1, §6.19.6.1, §7.15 | B | M; needs transform kernels and block geometry | 2–5% vs fixed 32x32 | M1 | U1 — unverified |
| Transform partitioning (`TX_PARTITION_NONE/SPLIT/HORZ/VERT/HORZ4/VERT4/HORZ5/VERT5`) | §5.20.6.3, §6.19.6.3 | B | M; depends on TX size | 1–4%; lets residual structure choose its own granularity | M1 (split only), M4 (4/5-way) | U2 — unverified (5-way new) |
| Transform type selection: DCT, ADST, FLIPADST, identity (IDTX), long-side variants | §5.20.8.2, §6.19.8.2, §7.15.2 | B | M; depends on TX size and intra direction | 1–4%; directional residuals benefit from ADST | M1 | U1 — unverified |
| Secondary transform (IST: intra/inter secondary transforms, 4x4 and 8x8) | §5.20.8.2 (`sec_tx_type`), §7.15.3, §9.7 | B | L; large kernel tables and per-mode mapping | 1–3% | M4 | U2 — unverified |
| CCTX (cross-component transform, Cb/Cr rotation, 6 angles) | §5.4.8, §5.20.7.27, §6.19.7.23, §7.14.6 | B | S–M; depends on chroma dequant | 0.5–2% on chroma | M4 | U2 — unverified |
| DDT / DDTX (data-dependent transform kernels for inter ADST/FDST) | §5.4.8 (`enable_inter_ddt`), §7.15.2.1 | B | S once transform infra exists; large kernel tables | 0.3–1.5% | M4 | U2 — unverified |
| Quantization matrices (QM OBU, `setup_qm_params`, 15 matrix levels) | §5.13, §6.12, §5.4.11, §6.4.11, §5.18.6.2, §6.17.6.2, §9.4 | B | S; dequant scaling plus matrix generation | 0.3–1.5%, content- and QP-dependent | M3 | U2 — unverified |
| TCQ (trellis coded quantization, parity-state dependent levels) | §5.4.8 (`enable_tcq`), §5.20.7.27 | B | M; encoder trellis search plus decoder parity state | 1–3% | M4 | U2 — unverified |
| Parity hiding (DC level parity hidden in coefficient sum) | §5.4.8 (`enable_parity_hiding`), §5.20.7.27 | B | S; coefficient coding only | 0.3–1% | M3 | U2 — unverified |
| FSC (forward skip coding; IDTX-only coefficient coding) | §5.4.8 (`enable_fsc`), §5.20.5.5, §5.20.7.27 | B | M; separate coefficient path | 0.5–2% on screen/lossless content; ~0 on natural video | M4 | U2 — unverified |

**Notes.**

- Transform sizes and types are not independent: §5.20.8.3 maps TX size and intra
  mode to a transform set. Generate the mapping tables with a script.
- `TX_PARTITION_HORZ5/VERT5` are uneven 5-way splits. They are not needed for M1;
  implement `SPLIT` first, then `HORZ/VERT`, then 4/5-way.
- QM is cheap to add but easy to get wrong: the QM OBU is separate from the frame
  header, and `setup_qm_params` selects matrix indices. Gate it with a stream that
  uses a non-default matrix.
- TCQ and parity hiding interact. If TCQ is on, parity hiding is constrained
  (§5.4.8); do not enable both blindly.
- FSC is not a natural-video tool. It is for IDTX/screen-content/lossless paths.

---

## 4. Inter prediction

Inter is where a video encoder earns its keep. M0 has no inter at all; M2 introduces
the baseline P-frame path. The advanced tools are the difference between "works" and
"competitive," but almost none of them are needed before the baseline works.

AV2 stores up to 16 reference frames and allows up to 7 active references per frame
(§3 constants; §5.18.2). "B-frames" are not a separate syntax type; they are inter
frames that reference future frames, selected by the reference lists in §7.7/§7.8.

| Tool | Spec § | Side | Complexity / depends on | What it buys (est.) | Needed? | dav2d gate |
|---|---|---|---|---|---|---|
| Skip / merge mode (`skip_flag`, skip mode params, skip syntax) | §5.18.8.2, §5.20.5.9, §5.20.5.10, §6.19.5.9, §6.19.5.10 | B | S–M; depends on inter mode decision | 5–15% of inter rate; cheap coding of static blocks | M2 | U1 — unverified |
| Single-reference inter with NEWMV / NEARMV (and NEAR_NEARMV, NEAR_NEWMV, NEW_NEARMV) | §5.20.7.6, §5.20.7.13, §6.19.7.5, §7.13.3.18 | B | L; needs motion search, MV prediction, interpolation, residual path | 30–50% vs intra-only on typical video; the baseline inter gain | M2 | U1 — unverified |
| MV prediction: find MV stack, DRL index, temporal/ref-MV candidates | §5.20.7.8, §6.19.7.7, §7.12.2 | B | M; depends on neighbor state and reference MVs | 2–5% of inter rate; also reduces motion bits | M2 | U1 — unverified |
| Subpel interpolation filters (regular/switchable, `interp_filter`) | §5.18.5.1, §6.17.5.1, §7.13.3.14, §7.13.3.18 | B | S–M; depends on reference buffers | 1–3%; necessary for fractional motion | M2 | U1 — unverified |
| Multiple reference frames (up to 7 active, 16 slots) | §5.18.2, §7.7, §7.8 | B | M; needs reference management | 2–6% | M2 | U1 — unverified |
| Compound prediction: average, CWP, wedge, diffwtd | §5.20.7.11, §5.20.7.16, §5.20.7.18, §7.13.3.13, §7.13.3.27–§7.13.3.30 | B | L; depends on two-reference MC and masks | 3–8%; compound average is the cheap first step | M4 | U2 — unverified |
| B-like frames / hierarchical reference structures | §5.18.2, §7.7, §7.8 | B | M for syntax; L for scheduler/RC | 8–15% vs P-only when paired with hierarchical GOP | M4 | U1 — unverified |
| Global motion (identity, rotation-zoom, affine; translation components) | §5.18.9, §6.17.9, §7.12.2.1 | B | M–L; needs global motion estimation | 2–6% on pans/zooms; ~0 on static content | M4 | U2 — unverified |
| Local warp / affine (`LOCALWARP`, `DELTAWARP`, `WARPMV`, `WARP_NEWMV`) | §5.4.6, §5.20.7.7, §7.13.3.19–§7.13.3.24 | B | L–XL; depends on inter MC and MV prediction | 2–8% on camera motion/rotation; content-dependent | M4 | U2 — unverified |
| Interintra (inter + intra blend, 4 intra modes, wedge option) | §5.20.7.15, §6.19.7.14, §7.13.3.29, §7.13.3.30 | B | M; depends on inter and intra prediction | 0.5–2% | M4 | U2 — unverified |
| BAWP (block adaptive weighted prediction) | §5.4.6 (`enable_bawp`), §5.20.7.6, §7.13.3.25 | B | M; encoder estimates scale/offset | measured ≈−4% rate at iso-CVVDP on a fade (nightly 240p, `--video-bawp-carrier`); 0.5–2% otherwise | M4 | **verified** (CPU carrier; CUDA skip-only) |
| TIP (temporally interpolated prediction; frame/block mode) | §5.4.6 (`enable_tip`), §7.10, §7.13.3.2 | B | XL; motion-field construction, hole filling, TIP deblock | 2–6%; alt-ref/pyramid content | M4 | U2 — unverified |
| OPFL (optical-flow refinement; `enable_opfl_refine`) | §5.4.6, §7.13.3.8–§7.13.3.12 | B | L; encoder must mirror decoder refinement | 0.5–2% | M4 | U2 — unverified |
| Ref MV bank / temporal MV prediction (`enable_refmvbank`) | §5.4.6, §7.9, §7.12.2.21 | B | L; motion-field storage and projection | 1–3% | M4 | U2 — unverified |
| Adaptive MVD / flexible MV precision / sign derivation | §5.4.6 (`enable_adaptive_mvd`, `enable_flex_mvres`, `enable_mvd_sign_derive`), §5.20.7.6 | B | S–M; depends on MV coding | 0.3–1.5% | M4 | U2 — unverified |
| BRU (backwards reference update) | §5.4.6 (`enable_bru`), §5.20.5.2, §6.19.5.2 | B | L; complex reference/region update | 0–2%; niche multi-pass use | M4 (optional) | U2 — unverified |
| OBMC (overlapped block motion compensation) | **Not present in AV2 v1.0.0**; no syntax or process found in §5–§7 | N/A | N/A | 0; do not implement | No | N/A |

**Notes.**

- **OBMC is not an AV2 tool.** There is no OBMC syntax or decoding process in the
  v1.0.0 spec. The closest tools are interintra, wedge/masked compound, and local
  warp. Do not budget time for OBMC.
- Skip/merge and NEWMV/NEARMV are the minimum inter set. Compound, B-frames, warp,
  and TIP are force multipliers, not prerequisites.
- `NEWMV` needs a real motion search. A cheap diamond/hex search plus a few
  predictors is enough for M2; do not build a hierarchical motion field before the
  baseline P-frame path decodes.
- Global motion and local warp are different tools. Global motion is frame-level
  and cheap for the decoder; local warp is block-level and much more complex.
  Gate them separately.
- TIP depends on the motion-field machinery in §7.9/§7.10 and on TIP deblocking
  (§7.16). Treat it as an XL subsystem, not a mode flag.
- BAWP and OPFL are decoder-side refinements that the encoder must mirror. Budget
  for the mirror, not just the syntax.

---

## 5. In-loop filters and film grain

Filter order is normative: deblock → CDEF → CCSO → loop restoration (Wiener /
PC-Wiener / GDF) → output; film grain is applied at output only. Filters are
in-loop: their output becomes the reference for later frames, so the encoder must
run them exactly.

| Tool | Spec § | Side | Complexity / depends on | What it buys (est.) | Needed? | dav2d gate |
|---|---|---|---|---|---|---|
| Deblocking filter | §5.18.5.2, §6.17.5.2, §7.17 | B | M; depends on partition/transform edges | 3–8% BD-rate plus a large visual improvement; the first filter to add | M1 | U1 — unverified |
| CDEF (constrained directional enhancement filter) | §5.18.7.10, §6.17.7.6, §5.20.10.1, §7.18 | B | M; 64x64 params, 8x8 blocks, direction search | 2–5% plus ringing reduction | M3 | U2 — unverified |
| CCSO (cross-component sample offset) | §5.18.7.12, §6.17.7.8, §5.20.10.2, §7.19 | B | M; band/edge classifiers and offsets | 0.5–2% | M3 | U2 — unverified |
| Loop restoration framework (units, stripes, per-unit types) | §5.18.7.11, §6.17.7.7, §5.20.10.4–§5.20.10.6, §7.20 | B | M; 64-luma stripes, unit bookkeeping | Prerequisite for Wiener/PC-Wiener/GDF; no separate gain | M3 | U2 — unverified |
| Non-separable Wiener filter (Wiener NS) | §7.20.3, §9.8 | B | L; per-unit filter estimation and large coefficient tables | 1–4% | M3 | U2 — unverified |
| Pixel-classified Wiener (PC-Wiener) | §7.20.4, §9.8 | B | L; pixel classification features and multiple filters | 0.5–2% incremental over Wiener NS | M4 | U2 — unverified |
| GDF (guided detail filter) | §5.18.7.9, §6.17.7.5, §5.20.10.3, §7.20.5 | B | L; guided filtering, per-block enable | 0.5–2%; detail retention | M4 | U2 — unverified |
| Film grain synthesis | §5.14, §6.13, §5.18.10, §6.17.10, §7.21.7 | D | M encoder analysis; decoder synthesis is normative | Perceptual only; allows lower coded bitrate on grainy sources. PSNR/BD-rate may get worse while perceived quality improves. | M4 (content) | U2 — unverified |

**Notes.**

- Deblock is the one filter that belongs in M1. Without it, a still encoder is not
  respectable at ordinary bitrates, and inter prediction inherits blocking.
- CDEF, CCSO, and loop restoration are decoder-side processes, but the encoder
  must run the exact same code to keep reconstruction aligned. Do not implement
  them only in the decoder-validation path.
- Filter strength/parameter selection is an encoder decision; start with frame-level
  defaults and add per-block/per-unit RDO later.
- Film grain is not a PSNR tool. Measure it with a perceptual metric and a visual
  check, and keep a separate "grain on/off" preset. The dav2d CLI has a
  `--filmgrain` switch, but that does not prove bitstream-level support for our
  streams; gate it anyway.

---

## 6. Rate-distortion and encoder control

These are encoder-only: no syntax is added, but they determine whether the bitstream
tools above actually pay off. They still require decoder validation of the resulting
streams, because a bad RD decision can produce a valid but poor stream.

| Tool | Spec § | Side | Complexity / depends on | What it buys (est.) | Needed? | dav2d gate |
|---|---|---|---|---|---|---|
| Lagrangian RDO for partition/intra/inter/TX decisions | None (encoder-only); QP syntax in §5.18.6.1 | E | M–L; needs candidate generation and a distortion metric | 5–15% vs greedy SATD decisions | M3 | N/A — no syntax; output still gated |
| RDOQ (rate-distortion optimized quantization) | None (encoder-only); coefficient syntax in §5.20.7.27 | E | M; depends on quantizer and entropy costs | 1–4% | M3 | N/A — no syntax; output still gated |
| Delta-Q: frame-level `delta_q_present`/`delta_q_res` plus per-block quantizer index delta | §5.18.6.3, §6.17.6.3, §5.18.7.8, §5.20.5.11 | B | S–M; depends on QP control | 1–3% | M3 | U2 — unverified |
| Segmentation (up to 8 or 16 segments; ALT_Q, SKIP, GLOBALMV features) | §5.18.7.1, §6.17.7.1, §5.20.5.7, §5.20.5.8, §6.19.5.7, §6.19.5.8 | B | M; depends on delta-Q and frame analysis | 1–5%; ROI and content-adaptive QP | M4 | U2 — unverified |
| Rate control (CBR/VBR, VBV/HRD, QP adaptation) | None (encoder-only); frame QP in §5.18.6.1; decoder model in Annex E | E | M; depends on RDO and buffer model | No direct BD-rate gain; quality consistency and bitrate accuracy | M4 | N/A — no syntax; output still gated |
| Lookahead / two-pass analysis | None (encoder-only) | E | M; depends on frame scheduler | 1–4% from better QP/GOP/reference decisions | M4 | N/A — no syntax; output still gated |
| Perceptual RD (SSIM/butteraugli-style metric) | None (encoder-only) | E | S–M; depends on distortion metric | Metric-dependent; can improve perceived quality at same PSNR | Optional | N/A — no syntax; output still gated |

**Notes.**

- RDO is the multiplier on every other tool. A partition tree with greedy SATD
  decisions leaves a large fraction of its potential on the table.
- Start RDO with a simple Lagrangian cost (`D + λR`) and SSD/SATD; add RDOQ after
  the entropy costs are stable. Do not build a trellis before the baseline decoder
  gate passes.
- Segmentation is not a general quality tool. Use it for ROI, static regions, and
  content-adaptive QP; gate it with a stream that actually uses multiple segments.
- Rate control is not a compression feature. It is a product feature; do not claim
  BD-rate gains from it.

---

## 7. Parallelism and scalability

| Tool | Spec § | Side | Complexity / depends on | What it buys (est.) | Needed? | dav2d gate |
|---|---|---|---|---|---|---|
| Tiles (up to 64 columns × 64 rows; independent entropy/prediction boundaries) | §5.18.7.2, §6.17.7.2, §5.18.7.3, §6.17.7.3 | B | M; depends on tile-aware availability and entropy reset | Parallelism, not compression; costs roughly 0.1–0.5% BD-rate at tile boundaries | M4 (for speed) | U1 — unverified |
| Tile groups (multiple tile-group OBUs per frame) | §5.19, §6.18, §5.20.1, §6.19.1 | B | M; depends on tiles | Parallel encode/transmit and low-latency slicing; ~0 compression gain | M4 (for speed) | U1 — unverified |
| Temporal layers (`obu_tlayer_id`, dependency maps) | §5.2.2, §6.2.2, §5.4.1 | B | M–L; depends on GOP scheduler | No direct gain; enables temporal scalability and hierarchical B (the gain is counted under B-frames) | M4 | U2 — unverified |
| Extended/embedded layers (SVC-like scalable/multistream) | §5.8, §6.8, §7.3.6, Annex F, Annex G | B | XL; layer management, extraction, composition | Packaging/scalability, not a proven single-layer BD-rate gain. Cross-xlayer prediction is forbidden (§6.8.2); only the embedded (mlayer) ladder has cross-layer prediction. Do not budget compression gain for it. | Optional | **U1 — CPU path gated** (`gate_cli_svc.sh`: temporal 8/8, embedded 12/12 pictures exact; avmdec only for embedded) |
| Multi-stream / atlas composition | §5.9, §6.9, Annex D, Annex G | B | XL; atlas/region mapping | Delivery/composition only; no single-stream gain | Optional | U2 — unverified |

**Notes.**

- Tiles are a speed tool. Do not add them to M1; they complicate neighbor
  availability and filter boundaries for no quality gain.
- Tile groups are not a compression tool. They are useful for GPU/CPU pipelining
  and low-latency delivery.
- AV2's layer machinery is primarily for scalable/multistream delivery. There is
  no normative "free" inter-layer prediction gain to assume; treat SVC as
  packaging until a specific layered use case proves otherwise. Cross-xlayer
  prediction cannot be signalled at all: §6.8.2 requires
  `lcr_dependent_xlayers_flag = 0`. The only decodable cross-layer tool is the
  embedded (mlayer) refinement ladder (`--svc-embedded`), which is gated by
  `src/pipeline/svc/gate_cli_svc.sh` with `avmdec --all-layers`.

---

## 8. Highest-leverage tools, ranked

1. **Flexible partitioning** (square + rectangular; 4x4–64x64). It changes the
   granularity of everything else. Est. 10–20% directly, and it is a multiplier on
   intra, transform, and inter gains.
2. **Full intra prediction** (directional modes + angle deltas + smooth + Paeth +
   edge filtering). Est. 10–20% on stills versus DC-only. Without this, M1 is not a
   still-image encoder.
3. **Inter baseline** (single reference, NEWMV/NEARMV, skip/merge, MV prediction,
   subpel interpolation, multiple refs). Est. 30–50% on video versus intra-only.
   This is the single biggest step for video.
4. **Transform adaptation** (variable TX size, DCT/ADST/FLIPADST/IDTX, TX split).
   Est. 3–8%; cheap once the transform infrastructure exists, and it compounds with
   partitioning.
5. **In-loop deblocking**. Est. 3–8% plus a large visual gain; required for
   reference quality and for inter prediction not to inherit blocking.

Force multiplier, not a bitstream tool: **Lagrangian RDO + RDOQ** (est. 5–15%
on top of greedy decisions). Honorable mentions: **CfL** (2–6% on 4:2:0 chroma)
and **B-frames/hierarchical GOP** (8–15% on video).

---

## 9. Decoder-risk shortlist

These are the tools most likely to expose a dav2d gap or an encoder/decoder mismatch.
Gate them early, with a minimal stream that uses only the tool plus the M0 baseline,
before building dependent features.

- **New AV2 intra:** DIP, MHCCP, MRL secondary-line, IBP.
- **New AV2 transform/quant:** IST, CCTX, DDT, TCQ, parity hiding, FSC, QM.
- **New AV2 inter:** BAWP, TIP, OPFL, local warp, ref MV bank, BRU, compound
  wedge/diffwtd/CWP.
- **New AV2 filters:** CDEF, CCSO, non-separable Wiener, GDF, PC-Wiener, film grain.
- **Structural:** SDP, extended 3/4-way partitions, 128/256 superblocks, tile
  groups, layered/SVC streams.
- **Content tools:** palette and IntraBC (gate only if a screen-content preset is
  actually shipped).

If a tool fails, do not assume a later decoder will fix it. Record the minimal
stream, the dav2d commit, and the exact error. Keep the feature off by default and
continue with tools that pass.

---

## 10. What not to do

- **Do not implement OBMC.** It is not in AV2 v1.0.0.
- **Do not add 256x256 superblocks early.** 128x128 is the better first step; 256
  is inter-only and increases the bug surface.
- **Do not use SVC as a compression argument.** AV2 layers are delivery/scalability
  machinery; there is no assumed single-layer BD-rate gain.
- **Do not use film grain as a PSNR win.** It is a perceptual tool; measure it
  separately.
- **Do not add palette/IntraBC to natural-content presets.** They are screen-content
  tools and add decoder risk for near-zero natural-video gain.
- **Do not report a BD-rate number before the decoder gate passes.** A number from
  the encoder's internal reconstruction is not a codec result.
- **Do not treat `U1` as "supported."** It means "unverified, lower risk." Only a
  passing `tools/validate_dav2d.sh out.ivf` run changes that.
