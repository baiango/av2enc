# Chained single-key hierarchical GOPs (P and B)

**Verdict.** A single-key chained hierarchical stream is *expressible* and
decoder-exact for both the P and the B schedule: one key, then N chained
8-frame blocks, with the first frame of every later block a regular anchor that
references the previous block (never a CLK). The 16-picture proofs decode
16/16 with recon equality under both `avmdec` and the pinned `dav2d`; the
60-picture chains decode 60/60 with recon equality under `avmdec` (the pinned
`dav2d` has a known §7.21.6 drain defect and stops at 17/60).

**The chained path now carries the bottom-edge inter path.**
`--video-edge-inter --video-edge-mask` is legal for the chained form: the
reference bank stores the padded `Plane` reconstruction instead of the cropped
`VideoReconFrame`, so the rows past the coded MI edge survive into the next
picture's MC. The 16-picture B proof drops from 25,662 B to **13,040 B** at
the same QP (intra leaves 120 → 75; the bottom band now codes inter) and stays
16/16 recon-exact on both decoders. At 100 kbps the B chain goes from
**K 43.360 (no edge) to 74.637 (edge)** — a **+31.28 K** recovery.

**The B port adds partition-RD and the anchor refresh.** The partition probe
now prices its NEWMV + quantised residual against each reference the leaf could
use and keeps the cheaper one (the single-reference path is byte-identical:
the nightly pins reproduce). Forcing the inter residual on the chained
layer-0 anchors repairs the reference drift the flat chain calls the drift
refresh. The measured arm becomes **K 81.111 at 100.0 kbps**, wrapped as the
single opt-in **`--nightly-b`**.

The continuation adds the **sequential key-frame RD driver** to the one B key
(the key is the only picture whose layout every block anchor predicts from),
then re-tunes the **key-QP share** (−50 is the measured optimum: −45/−50/−57/
−62 give 82.424/82.495/82.158/82.121 K), layers the **exact-rate BA-shaped
pipe + tail TX**, and finally ports the remaining flat-chain tools: the
**key TX/chroma level RD**, the **UVFULL+UVLEVEL pair**, and the **band-gated
TX descent** (`TX_DETAIL_SEARCH_TAIL`, min share 0.50). The aggregate now
measures **K 85.472 at 100.0 kbps**:

```sh
av2enc testdata/nasa_480x240_30fps_60f.y4m -o /tmp/b.ivf \
  --video --frames 60 --qp 131 --cpu-threads 8 --video-gops 7 --nightly-b
```

That is **+8.51 K** over the 76.962 edge+key+BAWP arm, **+6.43 K** over the
flat edge control (79.04), and **−0.76 K** against the 86.232 nightly target.
The remaining gap is the flat-chain-only tool set (drift refresh scheduling,
ref-signal-compact, tile-group-compact) and the chained structure itself.

Default-off env gates (the raw spike flags; `--nightly-b` sets the same
options):

```sh
AV2ENC_TLAYERS_CHAIN=1                      # chained form of --video-tlayers
AV2ENC_TLAYERS_B=1                          # dyadic B block instead of P
```

With the gates off, `--video-tlayers 8 --video-gops 2` is rejected by name and
the nightly pins are byte-identical (QP106 25,008 B `0639b436…`, QP110
23,690 B `1b99127a…`; both validated with `tools/validate_dav2d.sh`).

```sh
# 16-picture proofs: one key + 2 chained 8-frame blocks
AV2ENC_TLAYERS_CHAIN=1 av2enc testdata/nasa_480x240_30fps_60f.y4m \
  -o /tmp/chain16p.ivf --video --frames 16 --qp 116 --cpu-threads 8 \
  --video-tlayers 8 --video-gops 2 --video-recon /tmp/chain16p.y4m $B_LEGAL
AV2ENC_TLAYERS_CHAIN=1 AV2ENC_TLAYERS_B=1 av2enc \
  testdata/nasa_480x240_30fps_60f.y4m \
  -o /tmp/chain16b.ivf --video --frames 16 --qp 116 --cpu-threads 8 \
  --video-tlayers 8 --video-gops 2 --video-recon /tmp/chain16b.y4m $B_LEGAL

# the edge path on the chained B carrier (the result above)
AV2ENC_TLAYERS_CHAIN=1 AV2ENC_TLAYERS_B=1 av2enc \
  testdata/nasa_480x240_30fps_60f.y4m \
  -o /tmp/chain16be.ivf --video --frames 16 --qp 116 --cpu-threads 8 \
  --video-tlayers 8 --video-gops 2 --video-edge-inter --video-edge-mask \
  --video-recon /tmp/chain16be.y4m $B_LEGAL

# 60-frame clips: one key + 7 full blocks + a 4-frame tail block
AV2ENC_TLAYERS_CHAIN=1 av2enc testdata/nasa_480x240_30fps_60f.y4m \
  -o /tmp/chain60p.ivf --video --frames 60 --qp 116 --cpu-threads 8 \
  --video-tlayers 8 --video-gops 7 --video-recon /tmp/chain60p.y4m $B_LEGAL
AV2ENC_TLAYERS_CHAIN=1 AV2ENC_TLAYERS_B=1 av2enc \
  testdata/nasa_480x240_30fps_60f.y4m \
  -o /tmp/chain60b.ivf --video --frames 60 --qp 116 --cpu-threads 8 \
  --video-tlayers 8 --video-gops 7 --video-recon /tmp/chain60b.y4m $B_LEGAL
AV2ENC_TLAYERS_CHAIN=1 AV2ENC_TLAYERS_B=1 av2enc \
  testdata/nasa_480x240_30fps_60f.y4m \
  -o /tmp/chain60be.ivf --video --frames 60 --qp 117 --cpu-threads 8 \
  --video-tlayers 8 --video-gops 7 --video-edge-inter --video-edge-mask \
  --video-key-qp-offset -57 --video-bawp --video-bawp-carrier \
  --video-recon /tmp/chain60be.y4m $B_LEGAL
```

`$B_LEGAL` is the measured reordering-legal set from `docs/bframes.md` (the
nightly set minus the flags rejected by every reordering schedule; see "Tool
set" below). `src/pipeline/video/gate_video_tlayers_chain.sh` runs both
schedules, both picture counts, the edge and key-offset/BAWP forms, and the
gate-off pins.

## Carrier choice

Two candidates were available:

* **`inter::bframes` (`build_sequence`, `kIpbb`)** always emits a key at the
  start of each unit (`build_unit`: the local display-0 frame is the key and
  the P anchor references it), and its reference pattern is the kIpbb
  key + P-anchor pair. A single-key chain would need a new unit builder, a new
  reference model and a new output plan — i.e. exactly the work already done
  for the temporal-layer plan.
* **`tl::Schedule::kHierarchicalB`** already produces the dyadic B block with
  explicit past/future anchors, a layer assignment and a validated coding
  order. Chaining needed one new edge per block boundary: the block's past
  anchor (local 0) references the previous block's future anchor (local N−1),
  and the future anchor references the past anchor. The existing
  `encode_temporal_sequence` carries it unchanged: a two-entry explicit ref map
  per picture, `reference_select = 0`, and the block layer selects one of the
  two references per leaf with single-reference prediction. No compound
  prediction is used (the owner's initial constraint).

The temporal-layer B plan is therefore the carrier; `inter::bframes` is not
extended.

## Design

### Schedules

P block (the historical `--video-tlayers` schedule):

```text
display:  0   1   2   3   4   5   6   7
coding:   0   4   2   6   1   3   5   7
layer:    0   2   1   2   0   2   1   2
refs:     -   0,4 0,4 4,0 0   4,0 4,0 4,0   (two-reference carrier)
```

B block (dyadic, `Schedule::kHierarchicalB`):

```text
display:  0   1    2    3    4    5    6    7
coding:   0   7    3    1    5    2    4    6
layer:    0   2    3    1    3    2    3    0
refs:     -   0,3  1,3  0,7  3,5  3,7  5,7  0
```

Every reference has `layer(ref) <= layer(frame)`, so the default §5.4.1
`TLayerDependencyMap` (`ref <= curr`) admits it. In a chained block with display
offset `O`, the past anchor `O` references `O−1` (the previous block's future
anchor) instead of being a key; the future anchor `O+7` references `O`; the
interior B pictures keep the block pattern shifted by `O`. The only key is
display 0. The layer-0 chain is
`key → future0 → past1 → future1 → past2 → …`, so every block anchor is a
regular P picture that refreshes one slot.

### Slots and references

`NumRefFrames = min(16, display_count)` (the syntax maximum). The key writes
`refresh_frame_flags = (1 << NumRefFrames) − 1`; every other picture writes one
bit. The encoder mirrors §7.23: a key makes only the first refreshed slot
valid, and only that slot is used as a reference.

For a 16-picture plan the store holds every picture. For 60 pictures the plan
recycles slots with the display-order frontier rule: only a picture strictly
below the smallest not-yet-coded display, and no longer referenced by a
later-coded picture, may be reused. That lets the §7.21 output process drain
every lower picture in display order before the overwrite.

**Edge-capable reference bank.** The cropped `VideoReconFrame` (width × height)
is not a reference: it drops the SB-aligned hidden bottom rows. The chained
path stores the encoder's full `Plane` reconstruction per display, and after
`encode()` (which has already run deblock/CDEF/CCSO/LR) calls
`InterFrameEncoder::pad_edge_recon()` — the same edge replication the flat
chain applies to its whole-frame buffer, from `MiRows*4` to the SB-aligned
extent. The `RefFrame` then advertises `mi_rows = plane_height/4`, so MC bounds
and the bottom-straddling leaves read the same padded samples the decoder
keeps. `--video-edge-inter` and `--video-edge-mask` are legal only on the
chained form; `--video-ref-mvs` and tiled writing still reject it.

### Order hints and output

`OrderHintBits` is 5 for the 16-picture proofs and 6 for the 60-picture
clips — the smallest span that covers the plan without a half-span edge. Five
bits are required on the 16-picture plan: the P carrier's cross-block
`d8 → d0` edge is 8 pictures, and the encoder's `relative_dist()` is modulo the
hint span while the decoder's §5.18.3.1 `get_relative_dist()` is the absolute
difference, so at 4 bits the reference changes sign for the entropy contexts.
With 4 bits the no-edge P proof happened to stay exact, but the edge-mask
bottom-inter decisions exposed it: every picture from d8 on drifted (4,328-
4,370 luma samples, max 38). Five bits make the P and B edge proofs exact.
`build_plan` rejects an ambiguous hint.

`monotonic_output_order_flag = 0`. Display 0 is `immediate_output_frame = 1`;
every later picture is implicit. The plan validator simulates the §7.21 output
process and rejects a plan that loses a picture or emits it out of display
order.

## Decoder validation regime

Chained streams are validated with AVM's reference decoder, as authorized in
`docs/validation.md`:

```sh
tools/validate_avmdec.py out.ivf --ref source.y4m --recon encoder.recon.y4m
# avmdec: 60 frame(s), recon=ok, OK
```

Flat key+P streams and the QP106/QP110 pins stay on `tools/validate_dav2d.sh`.
The pinned `dav2d` `f9dafbc` has a §7.21.6 drain defect: it does not flush an
implicit picture when a **non-key single-slot refresh** overwrites its slot, so
the 60-picture chains emit only the final store contents plus display 0
(**17/60**, exit 0). The 16-picture proofs fit the store and decode 16/16 on
both decoders. The Nov 19 `dav2d` nightly `4d6266a` behaves identically on the
unconstrained chain (17/60), so moving the pin would not remove the defect.

## Tool set and the handicap

Every reordering schedule (B, temporal layers, TIP, random access) rejects the
flat-chain-only flags by name. The chained form now legalizes the edge path and
the single-key allocation tools:

| nightly flag | status on the chained form |
|---|---|
| `--video-edge-inter --video-edge-mask` | **legal** (single tile, no ref-mvs, CPU) |
| `--video-key-qp-offset` | **legal** (the key is display 0; the plan is the existing display-indexed `frame_base_q_idx`) |
| `--video-bawp --video-bawp-carrier` | **legal** (single-prediction leaves; compound stays off) |
| `--video-partition-rd` | rejected: `--video-partition-rd does not compose with the B/temporal-layer/random-access/TIP frame schedules` |
| `--video-drift-refresh` | rejected: wired for the flat key+P chain |
| `--video-ref-signal-compact` (nightly default) | rejected: gated on the plain key+P chain |
| `--video-tile-group-compact` (nightly default) | rejected: the wrappers build their own sequence header |

The maximal legal set is `B_LEGAL` (`--video-rd-preset quality`, CCSO-RD +
header RD, RDOQ, CDEF/CDEF-RD, leaf-RD, eighth-pel, key intra TX partition,
QP-scaled lambdas, avg/adaptive/CDF-continuity, tile-flush-min) plus
`--video-edge-inter --video-edge-mask`, optionally the key-QP offset and
BAWP/carrier. `--video-ref-mvs` and tiled writing still reject.

**The +34.36 K edge handicap is recovered.** `docs/bframes.md` measured
`B_LEGAL` alone at K 44.68 and `B_LEGAL` + edge at K **79.04** at 100 kbps.
The chained edge B reaches 74.64 (junction with the key offset + BAWP:
76.96). The remaining gap to the flat edge control is the chained structure
itself (a distant-anchor reference pattern and a second reference map), not a
missing decoder path.

## Proof streams

16 pictures, QP116, `B_LEGAL`, 480x240, first 16 frames of the NASA clip
(chained plans use `OrderHintBits = 5`):

| schedule | bytes | MD5 | avmdec | pinned dav2d |
|---|---:|---|---|---|
| P chain | 25,827 | `f5485b57748a0a1afc2057f223269fb1` | 16/16, recon ok | 16/16, recon byte-exact |
| B chain | 25,662 | `d5af8679202a00fafa3fd135569a28c1` | 16/16, recon ok | 16/16, recon byte-exact |
| P chain + edge | 13,165 | `982712c2611c5bfc709e9a34ad5c3450` | 16/16, recon ok | 16/16, recon byte-exact |
| **B chain + edge** | **13,040** | `b269ceff4b2c4d3d94c35cba42d92455` | 16/16, recon ok | 16/16, recon byte-exact |
| B + edge + key-QP −57 | 27,491 | `4b72029de6ea916eb288101aeb959a49` | 16/16, recon ok | 16/16, recon byte-exact |
| B + edge + BAWP | 13,184 | `f8c6e722de0158e5e9e3499e78886126` | 16/16, recon ok | 16/16, recon byte-exact |

The edge stream halves at the same QP (25,662 → 13,040 B; intra leaves
120 → 75) because the bottom band now codes inter instead of intra. There is
no recon drift: the padded reference extent is the only change.

60 pictures, QP116, `B_LEGAL`:

| schedule | bytes | MD5 | avmdec | pinned dav2d |
|---|---:|---|---|---|
| P chain | 76,569 | `25b1d7b0b86035656ac43529d9932370` | 60/60, recon ok | 17/60 (§7.21.6 defect) |
| B chain | 75,872 | `061c0bd81c88d85e229c26f7061a10dd` | 60/60, recon ok | 17/60 (§7.21.6 defect) |
| **B chain + edge** | **25,582** | `da7bef30adba309dabb66f4104a7ebb0` | 60/60, recon ok | 17/60 (§7.21.6 defect) |

The edge B chain at QP116 is **102.3 kbps** — it lands on the 100 kbps
operating point directly, against 303 kbps for the no-edge chain. The pinned
dav2d stops at 17/60 on every 60-picture chain (the §7.21.6 defect), so all
60-picture numbers are `avmdec`-validated with recon equality.

The `--nightly-b` aggregate (the final arm) at the operating point:

| case | bytes | MD5 | decoders |
|---|---:|---|---|
| 16 f, QP116, `--video-gops 2` | 19,759 | `8720425796fe80f3ddce067643adfec2` | avmdec 16/16 recon ok; pinned dav2d 16/16 recon exact |
| 60 f, QP131, `--video-gops 7` | 24,911 | `a4c8853e3abd752503db17be769b80a4` | avmdec 60/60 recon ok; pinned dav2d 17/60 (known defect) |

The 60-picture case is 99.644 kbps, so the matched 100.0 point interpolates
against QP130 (25,322 B / 101.288 kbps): **K 85.472**, JOD 9.8353, BA 2.316,
SS2 83.112, PSNR-Y 45.6 dB; key share 64.9%. Against the references:
**+1.574** over the 83.898 arm, **+6.43** over the flat edge control 79.04,
**−0.76** vs 86.232.

## Rate-matched measurements (QP116 is not comparable)

The no-edge chain at QP116 is 303-306 kbps, so it was bracketed at QP186/187.
The edge chain at QP116 is already 102.3 kbps, so it was bracketed at
QP117/118. Every stream was validated with `tools/validate_avmdec.py` before
scoring; K is `tools/oracle.py --target-kbps 100`, which decodes with `avmdec`.

**No-edge control ladders** (the earlier B/P comparison):

| schedule | QP | bytes | kbps | PSNR-Y | JOD | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P | 185 | 26,733 | 106.932 | 27.873 | 6.4913 | 21.9873 | 1.583 | 44.055 |
| P | 186 | 25,661 | 102.644 | 27.673 | 6.3918 | 20.9857 | −0.207 | 43.385 |
| P | 187 | 24,931 | 99.724 | 27.539 | 6.3480 | 21.3756 | −1.144 | 42.959 |
| P | 188 | 24,003 | 96.012 | 27.382 | 6.2825 | 21.4338 | −2.986 | 42.318 |
| **B** | 185 | 26,553 | 106.212 | 27.899 | 6.5239 | 20.3830 | 1.819 | 44.463 |
| **B** | 186 | 25,626 | 102.504 | 27.674 | 6.4109 | 19.7720 | −0.364 | 43.621 |
| **B** | 187 | 24,704 | 98.816 | 27.540 | 6.3790 | 20.4296 | −1.157 | 43.237 |
| **B** | 188 | 24,048 | 96.192 | 27.394 | 6.3150 | 20.7336 | −2.546 | 42.644 |

**Edge ladders** (the B chain + `--video-edge-inter --video-edge-mask`, plus
one tool at a time and the combination):

| arm | QP | bytes | kbps | PSNR-Y | JOD | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| edge | 116 | 25,582 | 102.328 | 39.026 | 9.4186 | 5.5440 | 68.708 | 75.656 |
| edge | 117 | 25,085 | 100.340 | 38.395 | 9.3542 | 6.1079 | 68.282 | 74.719 |
| edge | 118 | 24,653 | 98.612 | 38.287 | 9.3459 | 6.4142 | 67.630 | 74.301 |
| edge | 119 | 23,828 | 95.312 | 38.234 | 9.3352 | 5.7795 | 67.457 | 74.795 |
| edge + key-QP −57 | 144 | 25,896 | 103.584 | 41.549 | 9.4886 | 6.2786 | 74.639 | 76.299 |
| edge + key-QP −57 | 145 | 25,319 | 101.276 | 41.419 | 9.4682 | 6.1728 | 73.903 | 76.168 |
| edge + key-QP −57 | 146 | 24,813 | 99.252 | 41.266 | 9.4438 | 6.9300 | 73.271 | 75.317 |
| edge + BAWP | 106 | 25,822 | 103.288 | 40.625 | 9.5494 | 6.6346 | 70.709 | 75.650 |
| edge + BAWP | 108 | 24,999 | 99.996 | 39.772 | 9.4643 | 6.1887 | 69.006 | 75.318 |
| edge + BAWP | 110 | 23,632 | 94.528 | 39.669 | 9.4619 | 6.3649 | 70.126 | 75.339 |
| edge + key-QP −57 + BAWP | 140 | 25,437 | 101.748 | 42.480 | 9.5594 | 6.2876 | 75.431 | 76.777 |
| edge + key-QP −57 + BAWP | 142 | 24,646 | 98.584 | 42.365 | 9.5713 | 5.9719 | 75.375 | 77.112 |
| edge + key-QP −57 + BAWP | 144 | 23,994 | 95.976 | 42.247 | 9.5552 | 6.2206 | 75.780 | 76.873 |

Matched to 100.000 kbps (linear in kbps over the bracketing pair):

| arm | QP bracket | matched kbps | JOD | BA | SS2 | **K** | Δ vs edge | vs 86.232 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| chained P, no edge | 186/187 | 100.000 | 6.3521 | 21.339 | −1.055 | 43.000 | — | −43.23 |
| chained B, no edge | 186/187 | 100.000 | 6.3892 | 20.219 | −0.902 | 43.360 | — | −42.87 |
| **chained B + edge** | 117/118 | 100.000 | 9.3526 | 6.168 | 68.154 | **74.637** | — | **−11.60** |
| B + edge + key-QP −57 | 145/146 | 100.000 | 9.4528 | 6.650 | 73.504 | **75.631** | **+0.994** | −10.60 |
| B + edge + BAWP | 108/106 | 100.000 | 9.4643 | 6.189 | 69.009 | **75.319** | **+0.682** | −10.91 |
| **B + edge + key-QP + BAWP** | 140/142 | 100.000 | 9.5660 | 6.113 | 75.400 | **76.962** | **+2.325** | **−9.27** |
| flat edge control (`B_LEGAL` + edge) | 86/87 | 100.0 | — | — | — | **79.04** | — | −7.19 |
| flat opt (nightly + head/tail TX) | — | 95.19 | 9.8237 | 2.0786 | 83.647 | **86.16** | — | 0 (target) |

* **Edge recovery:** 43.360 → **74.637**, **+31.28 K**. This is the single
  largest move in the spike and it comes entirely from the reference-bank
  padding, not from a new syntax.
* **Key-QP offset −57:** +0.99 K at iso-rate. The key is 48% of the no-edge
  payload at QP117; the offset spends the key better and the bracket moves to
  base QP 145 (key QP 88).
* **BAWP:** +0.68 K at iso-rate. It saves ~17% rate at fixed QP (QP116:
  25,582 → 21,211 B), which buys a lower base QP at 100 kbps.
* **Combination:** +2.33 K, **76.962**. The two are nearly additive.
* **Against the flat edge control (79.04):** −2.08 K with both tools. The
  remainder is the chained B structure itself (distant-anchor coding and the
  two-entry reference map), not a missing tool path.
* **Against 86.232:** −9.27 K. The missing levers are the flat-chain-only
  partition-RD, drift refresh, ref-signal-compact and tile-group-compact.

The old closed-GOP B negative (b4, 36.85 K at 100 kbps) is **not** the
single-key result: the key overhead is gone, and with the edge path the chain
reaches 74.64-76.96 K.

## B port arms (the `--nightly-b` continuation)

Each arm is added to the best previous arm and rate-matched to 100.0 kbps with
the same oracle. The gate is **+0.15 K**; an arm below it is dropped.

| arm | change | QP bracket | kbps | JOD | BA | SS2 | K | Δ vs 76.962 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| start | edge + key-QP −57 + BAWP | 140/142 | 100.000 | 9.5660 | 6.113 | 75.400 | 76.962 | — |
| **1. partition-RD** | probe per reference, min | 139/140 | 100.000 | 9.6168 | 4.775 | 77.514 | **78.983** | **+2.021** |
| 2a. layer QP `0,2,4,6` | frame-level offset | 139/141 | 100.000 | 9.5947 | 4.678 | 77.296 | 78.958 | +1.996 (Δ arm1 −0.025) |
| 2b. layer QP `0,4,8,12` | frame-level offset | 139/140 | 100.000 | 9.5959 | 4.845 | 77.180 | 78.738 | +1.776 (Δ arm1 **−0.245**) |
| 3. tile-group compact | §5.4.2 seq tile grid | — | — | — | — | — | (≈0) | −1 B at 16 f; legal, immaterial |
| **4. anchor residual** | `force_inter_residual` on layer-0 anchors | 142/143 | 100.000 | 9.6278 | 3.510 | 78.701 | **81.111** | **+4.149** |

* **Partition-RD (+2.02 K).** `partition_probe_leaf_cost()` runs the NEWMV +
  quantised-residual model for every reference in the frame's map and keeps the
  minimum. A single-reference frame takes the same ref-0 path, so the flat
  pins are byte-identical. The preset uses the nightly's base lambda
  (`2.75e-6`) without the explicit flag, so the measured QP fit applies.
* **Layer QP allocation (failed, dropped).** `--video-tlayer-qp-offsets
  L0,L1,L2,L3` is implemented and exact, but neither `0,2,4,6` (−0.03 K) nor
  `0,4,8,12` (−0.25 K) clears the +0.15 gate on top of partition-RD; the
  probe already reshapes the per-layer spend. Not in `--nightly-b`.
* **Tile-group compact (legal, immaterial).** The chained writer owns a
  single-tile sequence header, so `seq_tile_info_present_flag = 1` is legal and
  saves 1 B on the 16-picture proof. Below the gate; not in `--nightly-b`.
  `--video-ref-signal-compact` is structurally inapplicable: its form is
  `NumRefFrames = 1` with no explicit map and one refresh bit, while the
  chained plan needs 16 slots and a two-entry map.
* **Anchor residual (+2.13 K).** `--video-tlayer-anchor-residual` sets
  `force_inter_residual` on every layer-0 non-key anchor (the past and future
  anchors of each block). It costs +10.7% at a fixed QP (13,040 → 14,433 B on
  the 16-picture proof) and buys **+2.13 K** at iso-rate by repairing the
  reference drift at the natural refresh points. The flat drift refresh's PSNR
  trigger and QP drop are not ported; the unconditional anchor form is the
  measured arm.

### Continuation arms (baseline 81.111, then 82.158)

| arm | change | QP bracket | kbps | JOD | BA | SS2 | K | Δ | decision |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| **5. key RD driver** | `--video-rd-driver` on the one key | 135/136 | 100.000 | 9.6735 | 3.219 | 80.382 | **82.158** | **+1.047 vs 81.111** | in |
| 6a. selective anchor 1.5 dB / 3,250 B | drift trigger, 14/15 anchors | 135/136 | 100.000 | 9.6733 | 3.157 | 80.335 | 82.271 | +0.113 vs 82.158 | dropped |
| 6b. selective anchor 2.0 dB / 3,250 B | drift trigger, 13/15 anchors | 134/135 | 100.000 | 9.6860 | 3.281 | 80.585 | 82.135 | −0.023 | dropped |
| 7a. key-QP −45 | key share 67.6% | 126/127 | 100.000 | 9.7131 | 3.234 | 80.960 | 82.424 | +0.266 | — |
| **7b. key-QP −50** | key share 68.8% | 129/130 | 100.000 | 9.7075 | 3.198 | 81.110 | **82.495** | **+0.337** | **in** |
| 7c. key-QP −57 (previous) | key share 70.1% | 135/136 | 100.000 | 9.6735 | 3.219 | 80.382 | 82.158 | — | previous |
| 7d. key-QP −62 | key share 72.6% | 138/139 | 100.000 | 9.6733 | 3.254 | 80.564 | 82.121 | −0.037 | — |
| **8a. eff-perc pipe** | `--video-eff-perc` | 136/137 | 100.000 | 9.7301 | 2.913 | 81.093 | **83.186** | **+0.691 vs −50** | in |
| 8b. pipe + head TX | `--video-head-tx` | 133 (nearest) | 103.044 | 9.7620 | 2.817 | 82.229 | (83.741 at 103.0) | not iso-rate | not in preset |
| **8c. pipe + tail TX** | `--video-tail-tx` | 132/133 | 100.000 | 9.7557 | 2.734 | 82.234 | **83.898** | **+0.712 vs pipe** | **in** |
| 8d. pipe + head + tail | both windows | 132/133 | 100.000 | 9.7493 | 2.783 | 82.314 | 83.728 | +0.542 vs pipe | dominated by 8c |

* **Key driver (+1.05 K).** The chained B key routes through the keyframe
  `seq_driver` (`rd-driver: sbs=32 layouts=32 failed=0 leaves=19,594`). It
  already fires with `AV2ENC_RD_DRIVER=1`; `--video-rd-driver` is the CLI form
  and `--nightly-b` sets it. The 16-picture proof drops from 27,118 to 23,292 B
  at the same QP and stays recon-exact on both decoders.
* **Selective anchor trigger (failed).** `--video-tlayer-refresh-threshold`
  and `--video-tlayer-refresh-budget` port the flat drift trigger (PSNR vs the
  key anchor, extra-payload budget). On this clip the drift is nearly uniform:
  1.5 dB / 3,250 B forces **14/15** anchors (3,131 extra payload bytes) and is
  only +0.113 K; 2.0 dB forces 13/15 and is −0.023 K. Both are below the
  +0.15 gate, so `--nightly-b` keeps the unconditional anchor form.
* **Key-QP sweep (+0.34 K).** The B key share falls from 72.6% at −62 to 67.6%
  at −45; the optimum is **−50** (82.495), and the previous −57 point is
  82.158. The preset now uses −50; the operating bracket is base QP 129/130
  before the pipe and 133/134 after it.
* **Pipe and tail TX (+0.69 and +0.71 K).** The exact-rate / BA-shaped
  perceptual pipe (`AV2ENC_EFF_PERC`, CLI `--video-eff-perc`) is worth +0.691 K
  on the chained B. On top of it the display-order head/tail TX windows give
  **+0.712 (tail)** and +0.542 (both); the head window alone is not rate-matched
  at 100 kbps (its nearest point is 103.0 kbps). Tail alone is the measured
  winner and is what `--nightly-b` sets. The earlier "byte-identical"
  no-op was a zsh word-splitting artifact in the test command (`env $e` does
  not split); explicit `env A=1 B=2` reproduces the expected commit, and the
  display-order ordinal is correct (2,730 head leaves vs 2,642 tail leaves).

### Remaining flat-chain tools (baseline 83.898)

Each arm is added to the current aggregate and matched to 100.0 kbps. Env-only
gates are set with explicit `env A=1 B=1` words; `--nightly-b` now sets the
winners for the user unless an explicit env value overrides them.

| arm | QP bracket | kbps | JOD | BA | SS2 | K | Δ | decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **1. key TX / chroma level RD** | 132/133 | 100.000 | 9.7657 | 2.631 | 82.417 | **84.218** | **+0.320** | in |
| 2. UV level RD alone | 132/133 | 100.000 | — | — | — | 84.161 | −0.057 vs arm 1 | dropped |
| 3. UV full search alone | 132/133 | 100.000 | — | — | — | 84.106 | −0.112 vs arm 1 | dropped |
| 3b. UVFULL + UVLEVEL (the legal pair) | 130/131 | 100.000 | 9.7633 | 2.564 | 82.657 | **84.417** | **+0.199** vs arm 1 | in |
| **4. band-gated TX descent** | 130/131 | 100.000 | 9.8353 | 2.316 | 83.112 | **85.472** | **+1.055** | in |
| 5. head TX on top | — | — | — | — | — | — | not forced | skipped |

* **Key TX (+0.32 K).** `AV2ENC_RD_KEY_TX=1` prices the key's TX plan and chroma
  level with the measured key lambda. The key is 65.7% of the payload at
  QP132; this is the most direct payer and it clears the gate.
* **UV level / UV full (separate arms fail).** `UVLEVEL` alone changes only
  ~25 B and is −0.057 K; `UVFULL` alone is −0.112 K. The pair is the flat
  chain's legal composition (`UVLEVEL` requires the full mode search for its
  prediction set) and is +0.199 K, so the pair is in the aggregate.
* **Band-gated descent (+1.06 K).** `AV2ENC_TX_DETAIL_SEARCH_TAIL=1
  AV2ENC_TX_DETAIL_MIN=0.50` lets the partition tree reach a 16x16 coding block
  where the 2-4 px source energy is high and prices that band in the luma TX
  search. It is not P-window-only: the B leaves use it and the decoded picture
  gains +0.8 dB PSNR-Y at iso-rate. The arming window now uses the display
  ordinal (`set_display_ordinal` / `arming_ordinal()`): the armed display
  frames are 33..59, and the flat chain keeps the coding-order ordinal when no
  display ordinal is set. For the 60-frame B schedule the coding-order window
  33..59 selects exactly the same display pictures (coding 32 is display 32,
  coding 33..39 are displays 33..39, and coding 40..59 are displays 40..59), so
  the port is byte-identical (delta 0); it is correctness for other frame
  counts/block shapes, not a rate lever on this clip.
* **Head TX.** The previous arm measured head+tail as dominated by tail alone
  (83.728 vs 83.898 on the pipe base); the owner's instruction is not to force
  it, so it stays out.

### Final re-sweeps (baseline 85.472)

* **Display-order band arming:** ported; armed display frames 33..59; the
  stream is byte-identical on this schedule (delta 0.000, below the gate).
  Flat pins unchanged.
* **Key-QP re-sweep on the new aggregate** (key TX + UV pair + band descent):

| offset | QP bracket | K | Δ vs 85.472 |
|---|---|---:|---:|
| **−45** | 127/128 | 85.492 | **+0.020** |
| **−50 (current)** | 130/131 | **85.472** | — |
| −55 | 132/133 | 85.328 | −0.144 |
| −57 | 134/135 | 85.066 | −0.406 |

  The key TX/UV/band tools flatten the offset curve: −45 is +0.020, well
  below the +0.15 gate, so −50 is kept. The final B point remains
  **K 85.472, −0.76 vs the 86.232 flat aggregate**; the two arms do not cross
  it.

`--nightly-b` expands to the winning set (B_LEGAL + edge + key-QP −50 + BAWP +
partition-RD + anchor residual + key RD driver + eff-perc pipe + tail TX +
key TX + UVFULL/UVLEVEL + band-gated descent + the chained 8-frame B carrier).
It requires `--video-gops M > 1`; `--frames` must be `M*8` or `M*8 + 4`.

## What would change the answer

1. **A content-aware anchor selector.** The ported PSNR trigger forces 14/15
   anchors and is only +0.113 K: the drift on this clip is nearly uniform, so
   "is this anchor drifted?" does not discriminate. The flat chain's +1.00 K
   selector prize needs the BA-correlated masking model, not a PSNR threshold
   (`docs/negative-results.md`). The trigger plumbing
   (`--video-tlayer-refresh-threshold/-budget`) is in place for it.
2. **Allocation.** The per-layer QP vector was neutral/negative on top of
   partition-RD, but a per-frame or per-superblock allocation (the display-
   indexed `frame_base_q_idx` and delta-Q machinery exist) is still open. The
   key-QP offset is the one allocation lever that measured positive.
3. **Compound prediction.** The owner kept it off for this proof. The
   `docs/bframes.md` compound recon bugs were fixed on 2026-11-17
   (BUG-INTER-003), so a gated `--video-compound` arm could use both references
   jointly instead of selecting one per leaf. The earlier closed-GOP B
   measurement was inert (+0.04 K), so this is a low expectation.
4. **Reference selection.** The P chained plan's `kTwo` carrier is
   anchor-biased; the B plan already uses the true past/future pair. A local
   predictor for the P plan would make the P control fairer, not the B arm
   better.

## Caveats

* One clip, one operating point, CPU-only. The streams are bit-exact with the
  release build (the gate pins the 16-picture streams).
* The 60-picture chains use `--video-gops 7` plus a 4-frame tail block; only a
  0- or 4-frame tail is accepted today.
* `docs/bframes.md`'s ±1 K interpolation ripple applies. The edge recovery is
  +31 K, far outside it; the partition-RD (+2.0), anchor-residual (+2.1) and
  total (+4.1) arms are also outside it, but should be re-measured on a second
  clip. The layer-QP vectors were inside the ripple (negative) and dropped.
* The chained 16-picture plans use `OrderHintBits = 5` (the cross-block
  `d8 → d0` edge is a half-span at 4 bits); the 60-picture plans use 6. The
  no-edge pins changed by ~5 B from the earlier 4-bit report.
* The edge chain's 100 kbps bracket is QP117/118, not the expected 124-132:
  the edge path removes the bottom-row bill, so the rate-vs-QP curve returns to
  the nightly-like scale. The no-edge bracket is still QP186/187. The final
  `--nightly-b` arm is bracketed at base QP 135/136 (key QP 78/79).
