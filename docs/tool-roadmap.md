# Tool roadmap — M0 to M4

Status: dependency-ordered implementation plan for `av2enc`. The tool catalog and
per-tool gain estimates are in [`bitrate-features.md`](bitrate-features.md).

Normative reference: **AV2 Bitstream & Decoding Process Specification v1.0.0**
(2026-05-28, commit `38f28e7`). Spec section numbers are cited where a stage adds
syntax.

## 0. Decoder gate policy

The validation decoder is **dav2d, pinned at `f9dafbc`** (nightly 2026-11-10).
Every encoder feature must pass the decoder gate before it is considered done.

The exact command is:

```sh
tools/validate_dav2d.sh out.ivf
```

Gate order, for every stage and every tool:

1. **Decoder gate** — the command above exits 0, reports the expected frame count
   and resolution, and emits no decode errors. A feature with no passing stream is
   not done.
2. **Reconstruction gate** — the decoder's Y4M output matches the encoder's internal
   reconstruction (0 LSB preferred; more than 1 LSB is a bug). Compare with film
   grain disabled where applicable.
3. **Quality/BD-rate gate** — only now measure PSNR/SSIM/BD-rate on the **decoded**
   output. Never report a BD-rate number from the encoder's internal reconstruction.

Do not reorder these. A quality number produced before the decoder gate is not a
codec result.

### 0.1 Decoder status codes

| Code | Meaning |
|---|---|
| `G0` | M0 baseline subset. Must pass before M0 exits. Not yet proven for our stream. |
| `U1` | Unverified at `f9dafbc`; lower risk. **Not** a support claim. |
| `U2` | Unverified at `f9dafbc`; higher risk (new AV2 tool, large tables, many interactions). Isolate and gate before depending on it. |
| `K` | Known incomplete/failing at `f9dafbc`. None recorded yet; populate when a minimal stream fails. |
| `N/A` | No bitstream syntax, or the tool is absent. Output is still gated. |

No tool in this roadmap is marked "supported." Every non-M0 tool is `U1` or `U2`
until a minimal stream passes `tools/validate_dav2d.sh out.ivf` and the result is
recorded. A failing tool stays disabled by default; record the smallest reproducer,
the dav2d commit, and the exact error.

### 0.2 M0 / early-stage restriction

M0 and all pre-M1 bring-up streams must stay inside the subset the project requires
dav2d to decode:

> 8-bit 4:2:0, one tile, key frame, DC intra, 32x32 DCT, no filters, no film grain,
> no SVC, no palette, no CfL.

Do not add an M1 tool to make M0 pass. M0 is gated exactly as scoped. After M0
passes, tools are added one at a time, each with its own gate.

---

## 1. Stage overview

| Stage | Scope | Est. gain vs previous stage | Est. cumulative vs M0 | Cost (person-weeks) | dav2d gate (before quality) |
|---|---|---|---|---|---|
| **M0** | Valid still-picture bitstream: DC intra, 32x32 DCT, fixed QP, one tile | — (anchor) | 0% | 4–6 | `G0`: decode gate first, then recon, then PSNR sanity |
| **M1** | Smallest respectable still encoder: partition tree, full intra, variable transforms, CfL, deblock | 35–55% | 35–55% | 8–12 | `G0` + M1 per-tool `U1` gates; decode before any BD-rate |
| **M2** | P-frame/inter baseline: skip, NEWMV/NEARMV, MV prediction, subpel, multiple refs | 35–55% on video | 60–75% | 6–10 | M2 per-tool `U1` gates; decode before any BD-rate |
| **M3** | Remaining filters and RDO: CDEF, CCSO, Wiener LR, QM, delta-Q, RDO/RDOQ | 10–20% | 65–80% | 8–12 | M3 per-tool `U2` gates; decode before any BD-rate |
| **M4** | Advanced inter, advanced intra/transform/quant, finishing filters, segmentation/RC, tiles/SVC | 15–30% | 70–85% | 18–28 | M4a/b/c per-tool gates; decode before any BD-rate |

The percentages are engineering estimates for typical 1080p natural content, not
spec claims. They are not strictly multiplicative; cumulative numbers are rounded
and content-dependent. Still-image gains and video gains are mixed in M1/M3/M4;
measure them on separate corpora.

Unless a stage says otherwise, every stage stays at **8-bit 4:2:0, one tile**.
Tiles and higher bit depths are not part of the M0–M3 critical path; tiles appear
only as an optional M4c speed feature.

Total rough cost: **44–68 person-weeks**, or roughly 10–16 months for one
experienced codec developer, excluding heavy GPU optimization and test-infrastructure
work. Mark all cost figures as estimates.

---

## 2. Dependency order

The critical path is bitstream/reconstruction first; everything else depends on it.

```
M0  bitstream + entropy + OBU/IVF + reconstruction
 |
 +--> M1 partition tree
 |      +--> M1 intra prediction
 |      +--> M1 variable transform / TX split
 |      +--> M1 CfL (needs chroma reconstruction)
 |      +--> M1 deblock (needs partition + transform edges)
 |
 +--> M2 inter baseline (needs M1 reconstruction and reference buffers)
 |      +--> M2 skip, NEWMV/NEARMV, MV prediction
 |      +--> M2 subpel interpolation, multiple references
 |
 +--> M3 RDO / RDOQ (needs M1 + M2 candidate generation)
 |      +--> M3 CDEF -> CCSO -> Wiener LR (each needs exact recon)
 |      +--> M3 QM, delta-Q
 |
 +--> M4a compound / B-frames / global motion (needs M2 + M3 RDO)
 +--> M4b advanced intra/transform/quant (needs M1 + M3; mostly independent of inter)
 +--> M4c exotic inter + finishing filters + RC + tiles/SVC (needs M4a + M3)
```

Practical ordering rules:

- Do not start inter until M1's reconstruction path is exact. Inter inherits every
  intra/transform/deblock mismatch.
- Do not start RDO until M1 and M2 can generate candidate modes reliably. RDO on
  top of broken candidates just makes better choices among broken options.
- Do not start advanced inter before the P-frame baseline decodes. A failed
  compound/warp/TIP stream is much harder to debug on top of an unstable MVP.
- Decoder-risk spikes (the `U2` shortlist) belong at the **start** of the stage
  that needs them, not the end. If a tool fails, the stage plan changes early.

---

## 3. M0 — current milestone (reference anchor)

**Goal.** A valid AV2 still-picture IVF that dav2d decodes: 8-bit 4:2:0, one tile,
key frame, DC intra, 32x32 DCT, fixed QP. 64x64 superblock, 32x32 luma coding
block (16x16 chroma).

**Tools in scope.**

| Tool | Spec § | dav2d gate |
|---|---|---|
| OBU framing, sequence header, frame header, tile group | §5.2, §5.4, §5.18, §5.19, §5.20.1 | G0 |
| 8-bit 4:2:0, one tile, key frame | §5.4, §5.18.7.2, §6.17.7.2 | G0 |
| DC intra prediction | §7.13.2.10, §7.13.2.11 | G0 |
| 32x32 DCT, scalar quantization, fixed QP | §5.20.6.1, §5.20.7.27, §7.14, §7.15 | G0 |
| Symbol decoder / CDF init / coefficient coding | §8.2, §8.3, §9.3 | G0 |
| IVF muxing | Project I/O, not spec | G0 |

**Explicitly out of scope.** Any filter (deblock, CDEF, CCSO, LR, GDF), film grain,
SVC/layers, palette, IntraBC, CfL, MRL, DIP, MHCCP, inter, tiles beyond one, RDO,
RDOQ, rate control. Do not enable them in an M0 stream.

**Estimated gain.** Zero by definition; M0 is the anchor for all later BD-rate
measurements.

**Cost.** 4–6 person-weeks remaining (scaffold + entropy + first valid stream +
gate harness). Estimate.

**Dependencies.** Scaffold I/O, bit writer, OBU/IVF writer, range coder, CDF tables,
transform/quant reference, reconstruction loop. No inter, no filters.

**dav2d gate (must pass before M0 is done).**

1. **Decoder gate.** Encode 64x64 and 256x256 stills and run:
   ```sh
   tools/validate_dav2d.sh out.ivf
   ```
   Require exit 0, one decoded frame at the expected resolution, and no errors.
2. **Reconstruction gate.** Dump the encoder reconstruction as Y4M and compare with
   dav2d's decoded Y4M. Require exact match (0 LSB; more than 1 LSB is a bug).
3. **Quality sanity.** At fixed QP, decoded PSNR should be sane for a 32x32
   DCT/DC-only encoder (for example, high-20s to 30s dB at a mid QP; the exact value
   is content-dependent). This is not a competitive number; it only catches
   catastrophic syntax/transform errors.
4. **Regression pin.** Record `EXPECTED_MD5` for the M0 streams and rerun with:
   ```sh
   EXPECTED_MD5=<hash> tools/validate_dav2d.sh out.ivf
   ```

**Exit criteria.** Both resolutions decode; recon matches; MD5 pinned; M0 contains
no tool outside the scoped G0 subset. Only then start M1.

---

## 4. M1 — smallest respectable still encoder

**Goal.** A still-image encoder that is credible at ordinary bitrates, not just
valid. M1 is still key-frame-only, 8-bit 4:2:0, one tile.

**Tools added.**

| Tool | Spec § | dav2d gate |
|---|---|---|
| Recursive square partition tree, `PARTITION_NONE`/`SPLIT` | §5.20.3.1, §5.20.3.2, §6.19.3.1 | U1 |
| Rectangular splits, `PARTITION_HORZ`/`VERT` | §5.20.3.1, §5.20.3.2 | U1 |
| Block sizes 8x8–64x64 (add 4x4 after its own gate) | §5.20.3, §5.20.4.1 | U1 |
| Full intra mode set: DC/V/H, directional + angle deltas, smooth, Paeth | §5.20.5.5, §7.13.2.2, §7.13.2.7, §7.13.2.8, §7.13.2.10, §7.13.2.11, §7.13.2.13 | U1 |
| Intra edge filtering | §7.13.2.14–§7.13.2.18 | U1 |
| Variable TX size 4x4–64x64, including rectangular | §5.20.6.1, §6.19.6.1, §7.15 | U1 |
| TX partition `NONE`/`SPLIT`/`HORZ`/`VERT` (4/5-way later) | §5.20.6.3, §6.19.6.3 | U1 |
| Transform types DCT/ADST/FLIPADST/IDTX | §5.20.8.2, §6.19.8.2, §7.15.2 | U1 |
| CfL, explicit and derived alpha | §5.4.5, §5.20.7.32, §7.13.5 | U1 |
| Deblocking filter (the one M1 filter) | §5.18.5.2, §7.17 | U1 |
| Fixed QP per frame, greedy/SATD mode decision | §5.18.6.1 | N/A (no new syntax) |

**Explicitly out of scope.** Extended 3/4-way partitions, SDP, 128x128/256x256
superblocks, MRL, DIP, MHCCP, IBP, palette, IntraBC, QM, IST, CCTX, DDT, TCQ,
parity hiding, FSC, CDEF, CCSO, loop restoration, GDF, film grain, inter, B-frames,
RDO/RDOQ, rate control, tiles, SVC.

**Estimated gain.** 35–55% BD-rate versus M0 at equal PSNR on natural stills. This
is a wide estimate: M0 is intentionally far below a real intra encoder. Expect the
largest gains from partitioning and directional intra; transform adaptation and CfL
add the rest. Low-rate stills will still look blocky until M3's advanced filters;
M1's deblock is the minimum, not the finish.

**Cost.** 8–12 person-weeks. Estimate. GPU-heavy: intra mode search, transform/
quantization, deblock. CPU-heavy: partition/context bookkeeping, entropy.

**Dependencies.** M0's entropy/recon must be exact. Partition tree before intra and
transform; CfL after chroma reconstruction; deblock after partition/transform so
edges exist.

**dav2d gate (must pass before any BD-rate number).**

1. **Per-tool decoder gate.** For each M1 row above, produce a minimal stream that
   uses only that tool plus the M0 baseline. Run:
   ```sh
   tools/validate_dav2d.sh out.ivf
   ```
   Require exit 0, expected resolution/frame count, no errors. Keep the minimal
   stream as a regression.
2. **Integrated decoder gate.** Encode the M1 corpus (64x64, 256x256, 480x270,
   720p stills) with the full M1 feature set. Run the same command on every output.
   Any failure blocks M1 exit; disable the offending tool and record `K`.
3. **Reconstruction gate.** Compare decoded Y4M to encoder reconstruction on every
   M1 stream. Exact match required.
4. **Quality gate.** Only now measure BD-rate on decoded output versus M0 at QP
   {22, 32, 42} on the still corpus. Require a conservative **≥25%** BD-rate
   improvement to call M1 a success; investigate if it is below that. (Estimate;
   the real number depends on content and mode decision.)
5. **Regression pin.** Pin MD5s for one stream per partition shape, intra mode
   family, TX size family, and CfL on/off.

**Exit criteria.** All M1 tools have a passing minimal stream or are explicitly
disabled; integrated corpus decodes; recon matches; ≥25% BD-rate vs M0; no `U2`
tool is required for M1.

---

## 5. M2 — P-frame / inter baseline

**Goal.** A baseline inter encoder: key frame plus P frames, single-reference
prediction, skip/merge, NEWMV/NEARMV, MV prediction, subpixel interpolation, and
multiple references. Still 8-bit 4:2:0, one tile.

**Tools added.**

| Tool | Spec § | dav2d gate |
|---|---|---|
| Inter frame header, reference slots, refresh flags, order hints | §5.18.2, §6.17.2 | U1 |
| Reference lists, past/future selection | §7.7, §7.8 | U1 |
| Skip / merge mode | §5.18.8.2, §5.20.5.9, §5.20.5.10, §6.19.5.9, §6.19.5.10 | U1 |
| Single-reference inter, NEWMV/NEARMV/NEAR_NEARMV | §5.20.7.6, §5.20.7.13, §6.19.7.5, §7.13.3.18 | U1 |
| MV prediction stack / DRL index | §5.20.7.8, §7.12.2 | U1 |
| Subpel interpolation, switchable `interp_filter` | §5.18.5.1, §6.17.5.1, §7.13.3.14 | U1 |
| Multiple references (start with 2–3 active, up to 7; 16 slots) | §5.18.2, §7.7 | U1 |
| Basic frame scheduler: key + P, fixed QP | Project-level; QP syntax §5.18.6.1 | N/A |

**Explicitly out of scope.** Compound prediction, B-like/hierarchical frames,
global motion, local warp, interintra, BAWP, TIP, OPFL, ref MV bank, adaptive MVD,
BRU, switch/SEF/bridge frames, tiles, rate control, film grain, SVC.

**Estimated gain.** 35–55% BD-rate versus M1 on typical video at equal PSNR. Still
images are unchanged. The exact number depends heavily on motion content and how
good the motion search is; a weak search can leave most of this on the table.

**Cost.** 6–10 person-weeks. Estimate. GPU-heavy: motion search, interpolation,
residual/transform. CPU-heavy: reference management, MVP, bitstream.

**Dependencies.** M1 reconstruction and deblock must be exact. Reference buffers
and frame scheduling are new. Do not add compound/B until the single-ref path is
stable.

**dav2d gate (must pass before any BD-rate number).**

1. **Per-tool decoder gate.** Minimal key+P streams for skip, NEWMV, NEARMV, each
   interpolation filter family, and 2-/3-reference cases. Run:
   ```sh
   tools/validate_dav2d.sh out.ivf
   ```
   Require exit 0 and the expected number of output frames.
2. **Integrated decoder gate.** Encode 60-frame 480x270 and 720p sequences,
   key+P, with the M2 feature set. Every frame must decode; no errors.
3. **Reconstruction gate.** Compare decoded frames to encoder reconstruction;
   exact match. Pay special attention to frame 2+ (reference drift).
4. **Quality gate.** Measure BD-rate on decoded output versus M1 at QP {22, 32, 42}
   over the video corpus. Require a conservative **≥30%** improvement on video to
   call M2 a success. Keep still-image measurements separate (M2 should not regress
   them).
5. **Regression pin.** MD5 for one key+P stream per reference count and per
   interpolation filter mode.

**Exit criteria.** All M2 tools pass individually; integrated video corpus decodes;
recon matches; ≥30% video BD-rate vs M1; no B/compound/warp dependency yet.

---

## 6. M3 — remaining filters and RDO

**Goal.** Close the low-rate quality gap and make encoder decisions rate-aware.
This is the stage that turns "works" into "minimally competitive."

**Tools added.**

| Tool | Spec § | dav2d gate |
|---|---|---|
| Lagrangian RDO across partition/intra/inter/TX decisions | Encoder-only; QP syntax §5.18.6.1 | N/A (no syntax; output gated) |
| RDOQ | Encoder-only; coefficient syntax §5.20.7.27 | N/A (no syntax; output gated) |
| CDEF | §5.18.7.10, §6.17.7.6, §5.20.10.1, §7.18 | U2 |
| CCSO | §5.18.7.12, §6.17.7.8, §5.20.10.2, §7.19 | U2 |
| Loop restoration framework + non-separable Wiener | §5.18.7.11, §6.17.7.7, §5.20.10.4–§5.20.10.6, §7.20.3 | U2 |
| Quantization matrices (QM OBU + setup) | §5.13, §6.12, §5.18.6.2, §9.4 | U2 |
| Delta-Q: frame-level and per-block quantizer delta | §5.18.6.3, §5.18.7.8, §5.20.5.11 | U2 |
| Deblock parameter/RDO refinement (M1 filter already present) | §5.18.5.2, §7.17 | U1 |

**Explicitly out of scope.** PC-Wiener, GDF, film grain, segmentation, rate
control, multi-pass lookahead, advanced inter (compound/B/global motion/warp/TIP/
BAWP/OPFL), advanced intra (MRL/DIP/MHCCP/IBP), palette/IntraBC, TCQ/parity/FSC,
tiles, SVC.

**Estimated gain.** 10–20% BD-rate versus M2, with the largest visual gain at low
bitrates. CDEF/CCSO/Wiener are the main quality tools; RDO/RDOQ determine how much
of the M1/M2 toolset actually gets used.

**Cost.** 8–12 person-weeks. Estimate. GPU-heavy: filter kernels and filter
parameter search. CPU-heavy: RDO candidate costing, entropy-cost estimation.

**Dependencies.** M1/M2 candidate generation must be stable. Filters must run in
the exact normative order (deblock → CDEF → CCSO → LR) and update references.
RDO needs reliable rate estimates from the entropy coder.

**dav2d gate (must pass before any BD-rate number).**

1. **Per-tool decoder gate.** Add one filter at a time; for each, produce a stream
   that exercises it on at least one frame and run:
   ```sh
   tools/validate_dav2d.sh out.ivf
   ```
   Require exit 0. Do not add the next filter until the previous one passes.
2. **Filter-order gate.** Encode a stream with all M3 filters enabled; decode and
   compare recon. Any mismatch usually means filter order, stripe/unit boundaries,
   or reference update is wrong.
3. **Quality gate.** Only after decode/recon pass, measure BD-rate versus M2 at QP
   {22, 32, 42} on stills and video. Require a conservative **≥10%** improvement;
   run a filter on/off ablation so a broken filter is not hidden by RDO.
4. **Regression pin.** One stream per filter combination (deblock only, +CDEF,
   +CCSO, +Wiener) and one QM stream.

**Exit criteria.** All filters decode and recon-match; RDO/RDOQ improve BD-rate
without breaking streams; ≥10% vs M2; no tool is left enabled without its own
passing stream.

---

## 7. M4 — advanced inter, finishing tools, rate control

**Goal.** Approach the frontier. M4 is intentionally split into three internal
passes so a solo developer can land value incrementally and stop if time runs out.

### M4a — video compression core

| Tool | Spec § | dav2d gate |
|---|---|---|
| Compound prediction: average, CWP, wedge, diffwtd | §5.20.7.11, §5.20.7.16, §5.20.7.18, §7.13.3.13, §7.13.3.27–§7.13.3.30 | U2 |
| B-like frames / hierarchical reference structures | §5.18.2, §7.7, §7.8 | U1 |
| Multiple references beyond the M2 baseline | §5.18.2, §7.7 | U1 |
| Global motion: identity, rotation-zoom, affine | §5.18.9, §6.17.9, §7.12.2.1 | U2 |
| Local warp / affine (`LOCALWARP`, `DELTAWARP`, `WARPMV`, `WARP_NEWMV`) | §5.4.6, §5.20.7.7, §7.13.3.19–§7.13.3.24 | U2 |

**Est. gain:** 8–15% versus M3 on video. **Cost:** 6–8 person-weeks.

**dav2d gate (before any quality number).** Each tool gets a minimal stream and
must pass the decoder gate first:
```sh
tools/validate_dav2d.sh out.ivf
```
Then check recon, then measure BD-rate. B-frame streams must decode all frames and
recon-match; global/local motion streams need content with actual pan/zoom/rotation.

### M4b — advanced intra, transform, quantization

| Tool | Spec § | dav2d gate |
|---|---|---|
| MRL (including secondary-line prediction) | §5.20.5.5, §7.13.2.7 | U2 |
| DIP | §5.4.5, §5.20.7.9, §7.13.2.3–§7.13.2.6 | U2 |
| MHCCP | §5.20.5.6, §5.20.7.32, §7.13.6–§7.13.8 | U2 |
| IBP | §5.4.5, §7.13.2.9, §7.13.2.12 | U2 |
| Palette | §5.4.7, §5.20.8.1, §7.13.4 | U2 |
| IntraBC | §5.4.7, §5.18.3.4, §5.20.5.4, §6.17.3.3 | U2 |
| IST (secondary transform) | §5.20.8.2, §7.15.3, §9.7 | U2 |
| CCTX | §5.4.8, §5.20.7.27, §7.14.6 | U2 |
| DDT/DDTX | §5.4.8, §7.15.2.1 | U2 |
| TCQ + parity hiding | §5.4.8, §5.20.7.27 | U2 |
| FSC | §5.4.8, §5.20.5.5, §5.20.7.27 | U2 |
| Extended 3/4-way partitions, SDP | §5.4.3, §5.20.3.1, §5.20.3.2 | U2 |
| 128x128 superblocks (and optional 256x256 inter) | §5.4.3, §5.18.7.6 | U2 |

**Est. gain:** 4–8% mixed, content-dependent. **Cost:** 5–8 person-weeks.

**dav2d gate (before any quality number).** Minimal stream per tool, then:
```sh
tools/validate_dav2d.sh out.ivf
```
Only after decode and recon pass, measure the gain. Palette/IntraBC only on a
screen-content corpus; TCQ/parity tested together and separately; IST/CCTX/DDT
tested with non-default transform sets.

### M4c — finishing filters, rate control, parallelism

| Tool | Spec § | dav2d gate |
|---|---|---|
| Interintra | §5.20.7.15, §7.13.3.29, §7.13.3.30 | U2 |
| BAWP | §5.4.6, §5.20.7.6, §7.13.3.25 | U2 |
| OPFL | §5.4.6, §7.13.3.8–§7.13.3.12 | U2 |
| TIP (+ TIP deblock) | §5.4.6, §7.10, §7.13.3.2, §7.16 | U2 |
| Ref MV bank | §5.4.6, §7.9, §7.12.2.21 | U2 |
| Adaptive MVD / flexible MV precision | §5.4.6, §5.20.7.6 | U2 |
| BRU (optional) | §5.4.6, §5.20.5.2 | U2 |
| PC-Wiener | §7.20.4, §9.8 | U2 |
| GDF | §5.18.7.9, §5.20.10.3, §7.20.5 | U2 |
| Film grain synthesis + encoder model estimation | §5.14, §6.13, §5.18.10, §7.21.7 | U2 |
| Segmentation (ALT_Q, SKIP, GLOBALMV) | §5.18.7.1, §5.20.5.7, §5.20.5.8 | U2 |
| Rate control (CBR/VBR, VBV/HRD) | Encoder-only; Annex E for decoder model | N/A (output gated) |
| Lookahead / two-pass | Encoder-only | N/A (output gated) |
| Tiles | §5.18.7.2, §5.18.7.3 | U1 |
| Tile groups | §5.19, §5.20.1 | U1 |
| Temporal layers | §5.2.2, §5.4.1 | U2 |
| SVC-like extended/embedded layers | §5.8, §6.8, §7.3.6, Annex F/G | U2 |

**Est. gain:** 5–10% versus M4a+M4b, mostly from rate control/segmentation and
content-specific tools. **Cost:** 7–12 person-weeks.

**dav2d gate (before any quality number).** Minimal stream per tool, then:
```sh
tools/validate_dav2d.sh out.ivf
```
Only after decode and recon pass, measure the gain. Rate-control streams must hit
target bitrate within ±5% while decoding; film grain is measured perceptually, not
by PSNR.

### M4 overall

**Estimated gain.** 15–30% versus M3; cumulative 70–85% versus M0. The top of the
range requires B-frames, compound, global motion, and a competent RDO/RC stack.
If only M4a lands, expect roughly its 8–15% range rather than the full M4 number.

**Cost.** 18–28 person-weeks. Estimate.

**Dependencies.** M4a needs M2+M3; M4b needs M1+M3 and is mostly independent of
inter; M4c needs M4a+M3. Do not start M4c's TIP/OPFL/ref-MV-bank before M4a's
compound/B path is stable.

**dav2d gate (must pass before any BD-rate number).**

1. **Per-tool decoder gate.** Every row in M4a/b/c gets a minimal stream; run:
   ```sh
   tools/validate_dav2d.sh out.ivf
   ```
   Require exit 0. `U2` tools are tested in isolation before they are enabled in a
   preset. If a tool fails, record `K`, disable it, and continue.
2. **Sub-pass integrated gate.** After M4a, after M4b, and after M4c, encode the
   full corpus and decode every stream. No stage-level BD-rate is reported until
   the sub-pass passes.
3. **Reconstruction gate.** Exact recon match on every integrated stream. For film
   grain, compare with grain disabled; measure grain separately.
4. **Quality gate.** Only after decode/recon, measure BD-rate versus M3 on stills
   and video. Require a conservative **≥15%** integrated improvement; report M4a/b/c
   ablations separately so gains are attributable.
5. **Rate-control gate.** For CBR/VBR streams, decoded bitrate within ±5% of target,
   no buffer underflow/overflow, and no decode failures. Rate control is not a
   BD-rate claim.

**Exit criteria.** Every enabled tool has a passing isolated stream; integrated
corpus decodes; recon matches; ≥15% vs M3; rate control meets target; decoder-risk
list has no unresolved `K` entry in a default preset.

---

## 8. Cross-stage validation harness

Build this once, use it for every stage. Harness details are in
[`validation.md`](validation.md); metric tooling is in
[`quality-metrics.md`](quality-metrics.md).

### 8.1 Corpus (keep it tiny)

- Stills: 64x64, 256x256, 480x270, one 720p frame. Keep total test media well under
  100 MB.
- Video: 480x270 and 720p, 60 frames, mixed motion (pan, zoom, static, detail).
- Screen content: a few synthetic frames with text/flat colors, only if palette/
  IntraBC are shipped.
- Grain: one short grainy clip, only for film-grain evaluation.

### 8.2 Per-stream checks

1. Encode to `out.ivf`; also dump encoder reconstruction to `recon.y4m`.
2. Run the exact decoder gate:
   ```sh
   tools/validate_dav2d.sh out.ivf
   ```
3. Compare `recon.y4m` with the decoder output; fail on >1 LSB.
4. Record frame count, resolution, bitrate, and MD5 in the gate log.
5. Only then compute PSNR/SSIM/BD-rate from the decoded Y4M.

### 8.3 Gate log template

| Feature / stage | Minimal stream | Decoder result | Recon match | dav2d commit | Date | Notes |
|---|---|---|---|---|---|---|
| M0 baseline | `out_m0_64.ivf` | pending | pending | `f9dafbc` | — | first gate |
| M1 partition | `out_m1_part.ivf` | pending | pending | `f9dafbc` | — | — |
| M1 CfL | `out_m1_cfl.ivf` | pending | pending | `f9dafbc` | — | — |
| M2 NEWMV | `out_m2_newmv.ivf` | pending | pending | `f9dafbc` | — | — |
| M3 CDEF | `out_m3_cdef.ivf` | pending | pending | `f9dafbc` | — | — |
| M4 compound | `out_m4_comp.ivf` | pending | pending | `f9dafbc` | — | — |

Update this table in the repo as gates pass. A `U1`/`U2` row with no stream is not
evidence of support.

### 8.4 BD-rate protocol

- Use at least four QPs (for example 22, 27, 32, 37) for a BD-rate fit; three
  (22, 32, 42) is the minimum for a sanity check.
- Compute BD-rate on decoded output against the previous stage's anchor.
- Report still-image and video results separately. Do not mix them into one number.
- Keep the same corpus, QPs, and metric across stages; otherwise the numbers are
  not comparable.
- For film grain, report a perceptual metric or a visual A/B, not PSNR alone.

---

## 9. Decoder-risk register

These tools are most likely to fail or behave differently in dav2d. Gate them
before building dependencies on them. None are currently recorded as failing; they
are unverified, not assumed safe.

| Tool group | Gate class | Why risky | First test |
|---|---|---|---|
| DIP, MHCCP, MRL secondary, IBP | U2 | New AV2 intra tools; unusual predictors/least-squares paths | Minimal intra-only stream per tool |
| IST, CCTX, DDT, TCQ, parity hiding, FSC | U2 | New transform/quant paths with large tables or state | One block/frame per tool; TCQ+parity together and separately |
| BAWP, TIP, OPFL, local warp, ref MV bank, BRU | U2 | New inter refinements/motion fields; encoder must mirror decoder | Tiny inter stream with one tool enabled |
| Compound wedge/diffwtd/CWP | U2 | Mask generation and weighting edge cases | Two-reference stream with each compound type |
| CDEF, CCSO, non-separable Wiener, GDF, PC-Wiener, film grain | U2 | New filter/synthesis processes; in-loop or output interactions | One frame per filter; film grain with `--filmgrain` on/off |
| SDP, extended 3/4-way, 128/256 superblocks | U2 | Tree/geometry bookkeeping; many boundary cases | One frame per partition shape and superblock size |
| Tile groups, temporal layers, SVC | U2 | Multi-OBU/layer extraction and reference interactions | Two-tile/two-layer stream; `--oppoint`/`--alllayers` |
| Square/rect partition, directional intra, CfL, variable TX, transform types, deblock, single-ref inter, skip, multiple refs, B-frames | U1 | Established syntax families, but still unverified for our streams | Normal stage gates; no special isolation beyond the per-tool stream |

If a gate fails, add a row with `K`, the minimal stream path, the dav2d commit, and
the error. Do not work around a decoder failure by silently changing the bitstream;
that produces a non-conforming stream that happens to decode locally.

---

## 10. Post-M4 backlog (optional, not required for "competitive")

- 256x256 superblocks on inter content (only after 128x128 is rock solid).
- SVC/multistream/atlas delivery features, if a product use case appears.
- BRU and advanced multi-pass reference updates.
- Perceptual RD and film-grain model estimation beyond the baseline.
- 10-bit, 4:2:2/4:4:4, lossless presets.
- GPU kernel fusion, cudaGraph, multi-GPU tile parallelism.

Do not pull these into M1–M3. They add decoder risk and implementation cost without
improving the minimum competitive result. Every backlog item still gets its own
decoder gate before it is considered done.
