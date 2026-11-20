# Fast-decode CPU RD mode (`--fast-decode`)

`--fast-decode` (renamed from `--nightly` on 2026-11-20; the old spelling is an
exact alias and expands the same flag set) is the latest measured CPU-only
algorithmic RD aggregate. It is
**opt-in and CPU-only**: the default stream, the stable `quality` preset and
every GPU path keep their historical decisions and byte-identical streams.

```sh
av2enc testdata/nasa_480x240_30fps_60f.y4m -o /tmp/nightly.ivf \
    --video --frames 60 --qp 106 --cpu-threads 8 --fast-decode
```

Expansion (the gate asserts this is byte-identical to typing the flags out):

```text
--video-rd-preset quality
--video-ccso-rd --video-ccso-bands 6 --video-ccso-lambda 22 \
  --video-ccso-refit 2 --video-ccso-unit-bits 2
--video-edge-inter --video-edge-mask
--video-partition-rd
--video-key-qp-offset -27
--video-rdoq
--video-cdef
--video-cdef-rd
--video-bawp --video-bawp-carrier
--video-leaf-rd
--video-subpel-eighth
--video-key-intra-txpart
--video-qp-lambda
--avg-cdf --adaptive-cdf --cdf-continuity
--video-drift-refresh
```

`--video-qp-lambda` scales five RD lambdas with QP (leaf `step^0.72`, TX
`step^0.34`, RDOQ `step^1.5`, partition `step^2.22`, key TX constant 8). The
expansion above therefore omits `--video-tx-lambda` and
`--video-partition-rd-lambda`: those set one site by hand, and the two modes are
mutually exclusive by name (an explicit numeric lambda used to silently re-pin a
site, which made the typed-out expansion differ from `--fast-decode` by 724 B).
See `docs/qp-scaled-lambda.md` and `bench/space8k_720p/README.md`.

`--fast-decode` rejects `--video-gpu` and `--still`. The bottom-edge path requires
the single-tile low-delay P chain (no B/reorder/temporal-layer modes, no
`--video-ref-mvs`); the normal combination validation enforces that.

`--video-rd-preset best` (`docs/video-rd-preset.md`) is the opt-in **lead
stack** measured on top of this aggregate: the nine env-only RD tools
(eff-perc, UVFULL/UVLEVEL, key TX, the tail band descent), the RD driver, the
head/tail exact-rate TX windows, the drift-refresh 1.5 dB / 3250 B point and
`--video-key-qp-offset -39`. It reproduces the explicit env command byte for
byte at QP 114/116/118, measures **K 86.33 at 100 kbps (+0.097 over the 86.232
flat aggregate)**, and is documented as a **LEAD, not a KEEP** (inside the
adjacent-QP ripple and sharing the driver headroom with the head/tail TX).

`--nightly-b` (`docs/tlayers-chain.md`) is a separate **research** preset, not
part of this aggregate: the chained single-key hierarchical-B stream with
AVERAGE compound and TIP output on the dead-end leaves measures **K 86.805 at
100 kbps (+0.573 over the 86.232 flat aggregate, +0.198 over its own 86.607
compound control, +1.333 over the 85.472 single-prediction control)**. Its
60-picture chains are `avmdec`-validated only (the pinned `dav2d` has the
§7.21.6 drain defect), so it stays opt-in and the shipped flat decisions here
are unchanged. The compound control is `AV2ENC_TIP_LEAF=0`; the
single-prediction control is `--video-compound-off`. See the same doc.

## Current pins

| Purpose | QP | Bytes | Rate | CVVDP | PSNR-Y | K | MD5 |
|---|---:|---:|---:|---:|---:|---:|---|
| **rate-matched headline** | **106** | **25,008** | **100.03 kbps** | **9.7334** | **44.67 dB** | **82.2** | `0639b436…` |
| gate regression | 110 | 23,690 | 94.76 kbps | 9.7228 | 44.38 dB | 81.4 | `1b99127a…` |

Cross-encoder comparisons use the QP106 point so both sides are measured at the
nominal 100 kbps; the gates pin QP110 as a fixed regression configuration. Tables
below state the QP they were measured at, and predate the overhead and QP-scaled
lambda changes where noted.

## Measured RD

Host: Apple M4, `AV2ENC_CPU_FAST=ON`, `--cpu-threads 8`. Clip:
`testdata/nasa_480x240_30fps_60f.y4m`, 60 frames. Quality: `tools/quality.py`
(CVVDP 0.5.7, `standard_fhd`, MPS) with `dav2d-macos`.

The pin is QP110 with the `-25` reference-intro offset; the offset is the
freedom that the adaptive-CDF saving buys, so the sweep below is in offset
rather than QP (bytes and rate both move with it):

| key offset (QP110, carried CDFs) | Bytes | kbps | CVVDP | PSNR-Y |
|---:|---:|---:|---:|---:|
| -26 | 24,579 | 98.316 | 9.7102 | 43.9231 |
| **-27** | **24,822** | **99.288** | **9.7177** | **43.9358** |
| -28 | 25,220 | 100.880 | 9.7108 | 43.8504 |

The -26/-28 rows are the flag-off sweep; the pinned -27 row includes
`--video-cdef-rd` (below), which removes 63 B without touching a pixel, so its
CVVDP and PSNR are the flag-off values at a lower rate.

At the established 100 kbps point:

| Stream | Rate | CVVDP | PSNR-Y |
|---|---:|---:|---:|
| default QP189 | 100.036 kbps | 4.8021 | 24.4667 dB |
| `quality` QP190 | 100.564 kbps | 6.3021 | 27.5816 dB |
| previous nightly QP117 (no intra TX part) | 100.088 kbps | 9.5467 | 40.8670 dB |
| previous nightly QP110 (no CDF/drift) | 99.976 kbps | 9.6802 | 42.9799 dB |
| **`--fast-decode` QP110, key -25 + CDF + drift** | **99.616 kbps** | **9.7069** | **43.8667 dB** |
| at iso-rate 99.976 kbps | 99.976 kbps | **≈9.707** | **≈43.88 dB** |

The keyframe intra luma TX partition with per-part prediction/recon adds
**+0.093 JOD / +1.28 dB** at iso-rate: the three bad 64×16 keyframe leaves gain
~9 dB and the first P frame loses most of its repair cost. The adaptive-CDF
chain plus a lower key offset add a further **+0.028 JOD / +0.90 dB** at
iso-rate (it is one move: the entropy saving pays for the key bits), and the
drift refresh forces the inter residual on the frames whose reference has
slipped. Nightly is **+4.90 JOD** over the default and **+3.40 JOD** over the
stable `quality` preset at the same rate. The stream is 24,822 B, MD5
`41d90595489d4aa9a86a6b3e9132a84f`; `dav2d-macos --strict 1` decodes it and the
60-frame encoder recon payload equals the decoded payload byte for byte
(10,368,000 B).

Encode speed is **7.6 fps** (7.87 s for 60 frames, best of 3) on the M4 at the
QP110 operating point; the CDF chain is the cost, the refresh adds ~2%.

## Per-tool contributions

The aggregate was built by measuring each change at iso-rate and keeping only
the ones that stacked. Approximate contributions at or near the 100 kbps point:

| Change | Contribution |
|---|---|
| Bottom-edge inter (`--video-edge-inter`) | +2.39 JOD over the old quality preset |
| Measured partition probe (`--video-partition-rd`) | +0.41 JOD over edge+CCSO |
| Reference-intro key/first-P QP offset | +0.22 JOD |
| Luma inter RDOQ (`--video-rdoq`) | +0.20 JOD |
| CDEF (bounded search) | +0.20 JOD |
| 64-band/refit/priced-unit CCSO | +0.15 JOD |
| CCSO row priced (`--video-ccso-header-rd`, opt-in, not in the aggregate) | −455 B of row (7.6 B/frame), +0.015-0.02 JOD at iso-rate |
| Bottom residual mask (`--video-edge-mask`) | +0.07 JOD |
| BAWP carrier over inter residual | +0.04 JOD |
| Calibrated leaf RD cost (`--video-leaf-rd`) | +0.05 JOD |
| 1/8-pel ME + TX-search λ 136 | +0.04 JOD |
| Keyframe intra luma TX partition | +0.09 JOD |
| Adaptive coefficient CDFs (per-frame reset) | −13.9% rate at identical decisions |
| Key -16 → -25 offset (paid by the CDF saving) + drift refresh | +0.03 JOD |
| Cross-frame CDF carry (`--cdf-continuity`) | −2.77% rate at identical decisions, key -27 retune +0.01 JOD |
| CDEF params-row pricing (`--video-cdef-rd`) | −0.25% rate (63 B) at identical decisions |

The CCSO row itself is 1,055 B of the pin (17.6 B/frame, 62.8% of it the
all-zero `tu(7)` floor of the 64-band table); `--video-ccso-header-rd` prices
the whole row and picks `ccso_max_band_log2` with the table, cutting it to
600 B and the stream by 594 B at QP 110 for a measured
**+0.015-0.02 JOD at iso-rate**. It is an opt-in because that gain sits inside
the QP ripple; `docs/video-ccso-overhead.md` has the bit split, the reuse
measurement and the QP 108-112 curves.

Per-superblock adaptive quantisation (`--video-aq`, `docs/video-aq.md`) was
re-measured on the composite K and is **not** included. At iso-rate 100.0 kbps
the best strength (3.0) is +0.05 K (JOD −0.003, BA −0.090, SS2 −0.530), inside
the control curve's own ±0.2–0.5 K QP ripple; a denser QP scan flips the best
strength. It also cannot carry the CDF image: delta-Q requires
`--video-cdf-per-frame-qp`, which costs 2.9% rate, so enabling AQ on this
aggregate is −0.47 K versus the pin. The same work made delta-Q compose with
the chain (RDOQ, the measured partition probe, the per-superblock plan and
`--video-drift-refresh-qp 0`); with a zero plan the stream is recon-identical
to `--fast-decode --video-cdf-per-frame-qp`, and `tools/gate_video_aq.sh` checks
that composition.

Positive changes that did **not** stack and are therefore explicit opt-ins,
not part of nightly: the §8.3.2 TX context rate (it needs
`--video-tx-lambda 48` and then loses to this aggregate), the MV mode-RD
candidate term on its own (+0.004 JOD at the operating point; it is one part of
the leaf RD cost below), and the keyframe intra RD re-rank (+0.007 JOD).
Chroma-aware intra/inter RD is a measured null on this achromatic clip.
Per-frame drift refresh (`--video-drift-refresh`,
`docs/video-drift-refresh.md`) is now part of the aggregate: on its own it is
**+0.022 JOD / +0.12 dB at iso-rate 99.976 kbps** (+0.89 dB mean luma on the
refreshed frames 32-40) for +2.6% rate at the same QP, and it stacks with the
CDF/key-QP move above. It is CPU-only, single-tile, cqp and 8-bit 4:2:0, and it
composes with the CDF chain only while the refresh keeps every frame's QP
(`--video-drift-refresh-qp 0`, the default).

Cross-frame CDF carry is now decoder-gated and opt-in: `--fast-decode
--cdf-continuity` keeps every decision (identical `--video-recon`, 60/60
pictures under `dav2d --strict 1`) and cuts the pin from 24,904 B / 99.616 kbps
to 24,214 B / 96.856 kbps, i.e. −2.77% rate. Spending it on the reference
intro (`--video-key-qp-offset -27`) puts the iso-rate point at 24,885 B /
99.540 kbps / 9.7177 JOD / 43.9358 dB before the CDEF params-row pricing below
(24,822 B / 99.288 kbps, same quality): **+0.01 JOD / +0.07 dB** at 99.616
kbps. The carried form also composes with `--video-drift-refresh` (the image
follows the coded symbols, so a refresh frame needs no §9.3 re-derivation) and
replaces the reset-per-frame opt-in inside `--fast-decode`; the reset form stays
the fallback. See `docs/video-cdf-carry.md` and
`src/pipeline/video/gate_video_cdf_carry.sh`.

## CDEF params-row pricing (`--video-cdef-rd`)

`cdef_params()` (§5.18.7.10) is **frame-local**: the pinned decoder reads
`cdef_frame_enable`, then damping, `CdefStrengths`, the on-skip flag and every
strength set, and installs exactly that row. There is no "same as the previous
frame" form and nothing in the sequence header (§5.4.10 signals only
`enable_cdef`) or the tile group (§5.19) can carry the strengths, so every frame
that filters re-signals the whole row. At `CdefStrengths == 1` (the nightly
setting) that is 16 bits (`y_sec == 0`, so the chroma primary is the one-bit
"zero" form) or 20 bits (any chroma primary), against 1 bit for
`cdef_frame_enable = 0`.

Measured on the 60-frame nightly stream (`AV2ENC_TRACE_CDEF=1`, per-frame
`cdef frame=` / `cdef-final frame=` lines):

| Property | Value |
|---|---|
| Frames that signal the row | 60/60 (1,012 bits = 126.5 B) |
| Distinct strength sets | 4 (`4/2/4/2`, `1/0/0/0`, `2/0/0/0`, `12/4/8/4`) |
| Consecutive pairs where the set changes | 21/59 (35.6%) |
| Longest run of one set | 31 frames (`1/0/0/0`, frames 1-31) |
| Frames whose filter changes no sample at all | 33/60 |
| `CdefStrengths > 1` frames | 0 (no `cdef_idx` symbol is ever coded) |

So the strengths are re-signalled **60 times while they change 21 times**, and
33 of those 60 rows describe a filter that does not modify a single sample of
the padded reconstruction. `--video-cdef-rd` prices that row: after the filter
runs, the post-CDEF planes are compared with the pre-CDEF ones, and a frame that
is bit-identical codes `cdef_frame_enable = 0` plus keeps its `cdef_idx` grid
zeroed. The pixels — and therefore the decoded pictures — are unchanged.

Result on the pin: **24,885 → 24,822 B (−63 B, −0.25%)**, 33 frames disabled,
no decision changes (every frame size stays the same; only the params rows
shrink), decoded pictures byte-identical (60/60 under `dav2d --strict 1`). The
63 B is the 33 × 15 bits the rows carried (495 bits = 61.9 B) plus byte-alignment
rounding. This is a pure entropy move: CVVDP and PSNR-Y are the flag-off values
at 99.288 instead of 99.540 kbps.

`--video-cdef-rd-lambda <L>` is the opt-in rate-priced arm: it also keeps the
all-zero set as a candidate at `CdefStrengths == 1` and drops frames whose
measured visible-area gain cannot pay `L × bits`. That arm **changes the padded
reconstruction** (the gain metric only covers the visible area, but the padded
rows are reference data), so it is not in `--fast-decode`: L=200 costs +3 B versus
the exact form and already changes samples, L=1000 saves a further 116 B
(24,706 B) but would need an iso-rate CVVDP sweep to justify. The exact form is
the shipped default.

### Multi-strength cost (`CdefStrengths > 1`)

The `cdef_idx` path is not used by the nightly pin, but the debug form
(`--video-cdef-strength 4`, gated by `gate_video_filters.sh`) shows its shape on
the same clip: the search prunes to 2.08 sets on average (24/60 frames keep more
than one), the header costs 1,522 bits = 190.2 B over 60 frames (25.4 vs 16.9
bits/frame) and the per-64×64 index payload adds 172 B, so the row plus indices
cost ~6.4 B/frame instead of ~2.1 B. Each extra set is +14 bits in the frame
header (7 luma + 7 chroma), and the per-block symbol is a §8.2-coded
`cdef_index0` plus the `cdef_idx_minus_1` tail. Any future multi-strength mode
must price at least those two rows per frame.

### Header overhead

The per-frame `tile_group_obu()` header costs 1,773 B (7.1%) of the pin: 1,634 B
of header payload (74-449 bits/frame) plus 139 B of tile-group OBU framing. The
220 bits of §5.19 `byte_alignment()` padding it contains are mandatory and cost
zero bytes, so they cannot be recovered. The only removable field is the 3-bit
`tile_params()` grid that every frame repeats; `--video-tile-group-compact`
(§5.4.2 `seq_tile_info_present_flag = 1` + `allow_tile_info_change = 0`) moves it
into the sequence header and codes no per-frame tile bits: **24,885 → 24,863 B**
at 60 frames with a byte-identical decode. It is an opt-in and is **not** part of
`--fast-decode`, so the pins above are unchanged. Measurements, the alignment
distribution and the decoder evidence are in `docs/video-tile-group-header.md`.

## Inter residual RD (`--video-leaf-rd-lambda`)

`--video-leaf-rd` prices a residual block as `coded SSE + lambda(qindex) *
bits` with the anchor constant `c = 125` at QP104. The trace of frame 40 shows
that constant, not the entropy coder, is what makes the drifted P frames skip
every leaf: at QP80 the frame codes 0 of 58 leaves while the best leaf's coded
SSE is 6x smaller than its prediction SSE. `--video-leaf-rd-lambda` exposes the
anchor (default 125, byte-identical). Lowering it to 80 at matched rate gives
+0.019 JOD at -0.13 dB PSNR - a real direction but inside the ripple, so it is
measurement evidence, not a shipped default. Analysis and the per-leaf tables:
`docs/video-inter-residual-rd.md`.

## Leaf RD cost (`--video-leaf-rd`)

One cost replaces the three historical leaf heuristics (inter-candidate
SSE-only choice, the fixed `inter_rd_lambda = 4.0` skip/residual decision, and
the `3.0 + …` / `8.0 + …` intra/inter comparison):

```text
J = coded SSE + lambda(qindex) * bits
lambda(qindex) = 125 * sqrt((qlookup(qindex)/8) / (qlookup(104)/8))
```

The bit terms mirror the signalled codeword shape: YMode set/index/escape, MRL
index + secondary flag, DIP, UV mode and CfL/CCTX, `single_mode`, the NEARMV
DRL chain and the shell-coded NEWMV MVD. The reference lambda was re-measured
on this aggregate (partition probe + RDOQ already price coefficients, so the
leaf needs less rate pressure): 125 beats the standalone 250 by ≈0.04 JOD at
iso-rate, and the measured optimum is flat from 0.4× to 0.6× of the current
value. A `step^2` lambda collapses the P chain into skip-only frames above
QP 110, so the fit follows `sqrt(step)` instead.

## Adaptive CDFs with per-frame QP (`--video-cdf-per-frame-qp`)

`--avg-cdf --adaptive-cdf` used to require one fixed `base_q_idx`: the §8.2.2
CDF image was seeded once from that frame-level quantizer. `--fast-decode` codes a
per-frame QP plan (key `--qp - 16`, P `--qp`), so the aggregate could not turn
adaptation on.

`--video-cdf-per-frame-qp` opts into a reset-per-frame image instead of a
carry: every frame starts from the §9.3 defaults, and
`disable_cross_frame_cdf_init` stays 1. §8.3.1 selects those defaults from the
**current frame's** `base_q_idx`, so the encoder re-derives the flat default
image for every frame and a plan that crosses a qctx slice is legal. Pinned
`dav2d` accepts that form: it installs exactly the same per-frame §9.3 image
the encoder seeded. The carry form (`disable_cross_frame_cdf_init = 0`,
`--cdf-continuity`) is now fixed and decoder-gated: the pinned decoder does
implement §7.4.4, and the old "our §7.5 image diverges" verdict came from the
encoder seeding the first P frame from the keyframe's context, which §5.18.2
forces back to the §9.3 image (that frame has no INTER_FRAME reference). The
nightly aggregate uses the carry when it is passed and the reset form
otherwise; `docs/video-cdf-carry.md` has the measurements. Note `dav2d` exits 0
after failing pictures, so check the decoded frame count, not the exit status.

### Same quality, fewer bytes

At a fixed QP the flag changes no decision: the decoded output is
byte-identical to the frozen stream (MD5 `64be64f05c659f86b52456519ec65330`
for both at QP 110), so CVVDP and PSNR are unchanged while the entropy layer
shrinks. On the QP110 pin the plan is key 94 / P 110, which crosses slices 2
and 3, i.e. exactly the case that used to be rejected.

| Stream | Bytes | kbps | CVVDP | PSNR-Y |
|---|---:|---:|---:|---:|
| `--fast-decode` QP110 (key 94) | 24,994 | 99.976 | 9.6802 | 42.9799 |
| same, `+ --video-cdf-per-frame-qp` | 21,530 | 86.120 | 9.6802 | 42.9799 |

That is −13.86% rate at identical reconstruction. The QP110 CDF stream is
21,530 B, MD5 `b6220e473ef46b367d27c7925b1e8c23`.

### Spending the saving: the key QP, not the global QP

Freeing 13.9% of the stream does not help by itself, because lowering the
*global* QP buys almost nothing on this clip: the CVVDP loss is carried by the
drifted tail of the P chain (f33-f59 at 41-42 dB), which barely responds to
QP. Measured at iso-rate: QP99 with the CDF is 24,965 B / 99.860 kbps /
9.6703 JOD / 43.4072 dB, i.e. **-0.010 JOD** against the QP110 pin despite
+0.43 dB PSNR.

The saving does pay when it is spent on the **keyframe**, which is 70% of the
bits and the reference for the whole chain. Fixing P at QP110 and moving the
reference-intro offset down:

| Plan (QP110, adaptive CDF) | Bytes | kbps | CVVDP | PSNR-Y |
|---|---:|---:|---:|---:|
| key -16 (no change) | 21,530 | 86.12 | 9.6802 | 42.98 |
| key -24 | 23,981 | 95.92 | 9.6908 | 43.38 |
| **key -26** | **24,516** | **98.06** | **9.7200** | **43.83** |
| key -27 | 24,749 | 99.00 | 9.6927 | 43.57 |
| key -28 | 25,249 | 101.00 | 9.6955 | 43.45 |

The QP-to-QP CVVDP ripple on this clip is ±0.03 JOD, so read the gain as
**+0.01 to +0.02 JOD at iso-rate** (9.694-9.704 interpolated to 99.976 kbps),
with a firmer **+0.6 to +0.9 dB PSNR-Y**. For reference, dropping the key QP by
26 without the CDF costs ~27.9 kB and does not fit the point at all.

Recommended nightly composition: the flag and a `--video-key-qp-offset -26`
retune move together, since the flag is what pays for the lower key QP. It is
not folded into the QP110 pin yet so the in-flight frame-level levers keep a
comparable baseline.

## Where the bits and the loss are
## Where the bits and the loss are

Bit split of the 24,994-byte nightly stream (`--qp 110`), from the encoder's
tile accounting plus the IVF frame sizes:

| Layer | Bytes | Share |
|---|---:|---:|
| Keyframe tile | 17,623 | 70.5% |
| P1 tile (frame 1) | 59 | 0.2% |
| P2-P59 tiles | 4,740 | 19.0% |
| Per-frame OBU headers (60 frames) | 1,820 | 7.3% |
| IVF + sequence header | 752 | 3.0% |

P1 is cheap because its source is **byte-identical to frame 0**, so the
keyframe is its own perfect predictor: the reversed skip decision costs 59
tile bytes (47 inter blocks, 0 intra) and its reconstruction is a copy of the
keyframe's. Frame 1 therefore scores 44.75 dB, the best frame in the stream.

That also bounds how much a P1 refresh could ever be worth. Replacing the
decoded frame with the source (i.e. spending infinite bits on it) and
re-scoring with the same CVVDP model:

| Oracle (decoded frame -> source) | CVVDP | delta |
|---|---:|---:|
| baseline stream | 9.6802 | - |
| frame 1 | 9.6827 | +0.0025 |
| frame 2 | 9.6827 | +0.0025 |
| keyframe | 9.6857 | +0.0055 |
| keyframe + frame 1 | 9.6865 | +0.0063 |
| 10 worst frames | 9.7292 | +0.0490 |
| 30 worst frames | 9.8515 | +0.1713 |

One frame of 60 is worth about +0.0025 JOD no matter which frame it is, so the
duplicated opening pair is not a lever; a real refresh (residual refinement,
938-1,911 B) captures strictly less than its oracle. The loss is spread
uniformly over the whole sequence instead, which is why the remaining gap to
SVT-AV1 shows up on **every** frame at the same rate and byte count:

| Frame | AV2 nightly | SVT preset 0 / CRF 36.5 | delta |
|---|---:|---:|---:|
| 0 (key) | 44.71 dB | 50.15 dB | +5.44 |
| 1 | 44.75 | 50.27 | +5.52 |
| 5 | 43.70 | 50.18 | +6.48 |
| 20 | 43.99 | 50.17 | +6.18 |
| 30 (worst region) | 43.40 | 49.20 | +5.81 |
| 40 (worst) | 41.14 | 47.46 | +6.32 |
| 50 | 42.21 | 48.20 | +5.99 |
| 59 | 43.96 | 47.01 | +3.05 |
| mean Y | 43.11 | 48.98 | +5.87 |

Both encoders spend ~70% of the stream on the keyframe and code the P chain in
about 4.7-4.9 kB; SVT converts the same bytes into ~6 dB on each frame. The
work is per-frame coding efficiency (intra RD and coefficient coding in the
keyframe, residual coding in P), not frame-level rate allocation.

## Verification

- `dav2d-macos --strict 1` decodes the final stream (60/60 pictures).
- 60-frame `--video-recon` payload equals the decoded payload byte for byte;
  the hidden padded rows are outside the Y4M payload.
- Default 480x240 QP189 stays **25,009 B / `a1d0dba1...`**.
- `quality` QP190 stays **25,141 B / `3984677b...`**.
- `tools/gate_video_nightly.sh` checks the expansion, both pins and the nightly
  pin (**QP110, key -27, 24,822 B / `41d90595...`**), strict decode, visible
  recon equality, BAWP carrier activity, the `--video-cdef-rd` A/B control
  (smaller stream, byte-identical decoded pictures, 60/60) and the
  GPU/`--still` rejections.
- `src/pipeline/video/gate_avg_cdf.sh` adds the `--video-cdf-per-frame-qp`
  case (strict decode + visible recon at key 174 / P 190) and the named
  rejections: a per-frame plan without the
  opt-in, and the opt-in without the CDF chain.
- `tools/gate_video_edge_inter.sh`, `tools/gate_video_rd_preset.sh` and
  `gate_video_any_dims.sh` (including the MI-edge 100x70/640x362 cases) pass.
- The per-frame reference/order-hint syntax is 948 bits over 60 frames
  (2.0 B/frame) and is cut to 417 bits by `--video-ref-signal-compact`
  (`NumRefFrames = 1`, derived §7.7 map, no primary-ref block): the same 60
  pictures decode byte-identically at 24,824 B (−61 B), so it is available as
  an opt-in but is not part of this pin. Field table and the rejected
  `OrderHintBits` form: `docs/video-ref-signalling.md`.

- `tools/gate_video_tile_group.sh` covers the header surface:
  `--video-tile-group-compact` removes exactly 3 bits per frame header, nets
  −22 B at 60 frames, decodes 60/60 under `--strict 1` with a byte-identical
  payload, and is rejected with `--tiles`/`--b-frames`.

## The static floor

`--video-skip-collapse` (opt-in, not part of `--fast-decode`) rewrites a P-frame
superblock whose leaves are all plain zero-motion inter skips into one
superblock-sized skip leaf: the reconstruction is unchanged, the partition tree
bits disappear. On an all-identical 60-frame clip it takes the steady-state P
frame from 21 B to 17 B (7-byte tile to 3-byte tile; 32.4 information bits to
7.4) and the whole stream from 20,915 B to 20,695 B. On this clip it is 24,539 B
/ 9.6973 JOD at the same QP, a wash, which is why it is not in the aggregate.
The symbol table and the §8.2.4 2-byte floor: `docs/video-skip-collapse.md`.

## Caveats

This is one clip at one quality point on one host. The bottom-edge and BAWP
gains are content-dependent; the partition probe, RDOQ, CDEF and CCSO
contributions should be re-measured on other content and rates. `--fast-decode` is
a CPU development mode, not the GPU product path, and it is deliberately not
the default.

## Extended partitions

`--video-ext-partitions` composes with this aggregate: the extended solver uses
the same leaf cost as the rect tree (the keyframe estimate and the measured
P-frame probe), and its partition rows ride the §8.2 CDF image. At QP110 the
composed stream is 23,587 B / 94.348 kbps / 9.6991 JOD / 44.2385 dB; at the pin's
96.452 kbps the interpolation is 9.6949 JOD / 44.315 dB, i.e. +0.43 dB at
-0.018 JOD (inside the ±0.03 ripple). See `docs/video-ext-nightly.md`.
