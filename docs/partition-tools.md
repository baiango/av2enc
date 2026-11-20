# Unused partition / block-structure tools at 240p30 / 100 kbps

This records what the four unmeasured block-structure tools are worth at the
project's 240p30 / 100 kbps operating point, on the CPU `--video` path:

| Tool | Flag | Status before this study |
|---|---|---|
| Separate luma/chroma partition tree | `--video-sdp` | opt-in, never scored |
| Inter TX direction transform | `--video-ddt` | opt-in, never scored |
| Extended (3-way / uneven 4-way) partitions | `--video-ext-partitions` | scored only as a rate/PSNR Pareto win, no composite K |
| 128/256 superblock trees | `--video-sb128` / `--video-sb256` | gated, never scored |
| Sub-8 keyframe leaves | `--video-min-block 4` | measured only inside AQ work |

Clip `testdata/nasa_480x240_30fps_60f.y4m`, 60 frames, 30 fps, 480x240, 8-bit
4:2:0, single tile, flat key+P chain, `--cpu-threads 8`, `nice -n 10`. Quality:
`tools/quality.py` (CVVDP 0.5.7, `standard_fhd`, MPS) against the pinned
`dav2d-macos`, plus `tools/extra_metrics.py` (Butteraugli, SSIMULACRA2) and
`tools/kpi.py` (composite K). Decode check: `dav2d --strict 1`, 60/60 pictures,
on every stream quoted below.

Reference points: **nightly QP106 = 100.032 kbps / K 82.15** (reproduced
exactly: 25,008 B, MD5 `0639b436…`) and the tuned drift recipe
(`--fast-decode --video-drift-refresh-threshold 1.5 --video-drift-refresh-budget
3250`) **= 83.80 K at 100.0 kbps** (QP110 100.584 / 83.94, QP111 99.800 /
83.75, reproduced exactly).

## Why the comparison sets are split

`--video-partition-rd` is rejected together with `--video-sdp`
(`src/cli/validate.cpp:2467`: "the SDP tree has a separate cost model"). The
probe is therefore removed for set A, so each partition tool is compared
against the same flags minus the probe, not against nightly.

* **A** — explicit `--fast-decode` expansion (all flags byte-identical to
  `--fast-decode` when `--video-partition-rd` is added back; verified on a 4-frame
  clip) with `--video-partition-rd` removed. Control **A0**.
* **B** — `--video-rd-preset quality` plus the tool.
* **C** — a stripped chain for the superblock arms. `--video-sb128/256` reject
  every loop filter, `--video-drift-refresh` and `--video-partition-rd`, so they
  cannot be measured on A or B at all; C drops those tools from A and is only
  comparable within itself.

The SDP arms additionally require dropping `--video-intra-full`
("`--video-sdp does not compose with --video-intra-full yet`",
`src/pipeline/video/video_frame.cpp:884`) and, in practice, CCSO (see the
constraint log). Their controls are measured with exactly the same flags minus
the tool.

## Set A — explicit nightly expansion minus the probe

A0 (control, QP 108–112):

| QP | kbps | JOD | BA | SS2 | PSNR-Y | K |
|---:|---:|---:|---:|---:|---:|---:|
| 108 | 102.672 | 9.7083 | 4.090 | 80.010 | 44.307 | 80.79 |
| 109 | 103.000 | 9.6950 | 4.185 | 80.147 | 44.447 | 80.60 |
| **110** | **100.828** | **9.6859** | **3.730** | **79.794** | **44.132** | **81.21** |
| **111** | **99.992** | **9.6936** | **3.719** | **79.888** | **44.254** | **81.28** |
| 112 | 98.392 | 9.6817 | 3.832 | 79.263 | 44.035 | 80.93 |

Iso-rate 100.0 kbps: **K 81.28** (JOD 9.6935, BA 3.72, SS2 79.89). Removing
the probe costs 0.87 K versus nightly; that is the reference every arm below
must beat.

Arms on A0:

| Arm | QP | kbps | JOD | BA | SS2 | PSNR-Y | K |
|---|---:|---:|---:|---:|---:|---:|---:|
| `--video-ext-partitions` | 109 | 101.392 | 9.6578 | 3.514 | 79.395 | 44.307 | 81.37 |
| | **110** | **99.752** | **9.6564** | **4.124** | **79.444** | **44.249** | **80.38** |
| | 111 | 98.556 | 9.6547 | 3.969 | 79.552 | 44.152 | 80.63 |
| `--video-min-block 4` | 108 | 101.704 | 9.6717 | 4.504 | 79.965 | 44.303 | 80.01 |
| | **109** | **99.320** | **9.6628** | **4.841** | **79.141** | **44.302** | **79.40** |
| | **110** | **97.956** | **9.6872** | **3.841** | **79.864** | **44.296** | **81.04** |
| | 111 | 97.088 | 9.6920 | 3.679 | 79.587 | 44.172 | 81.29 |
| | 112 | 94.760 | 9.7051 | 3.761 | 79.814 | 44.148 | 81.25 |
| `--video-ddt` | 110 | 100.872 | 9.6591 | 4.394 | 79.332 | 44.134 | 79.99 |
| | 111 | 101.820 | 9.6555 | 4.577 | 79.161 | 44.124 | 79.70 |

Iso-rate 100.0 kbps: ext **80.53**, min-block-4 **79.6–80.2** (the two
bracketing segments disagree: 108/109 gives 79.57, 109/110 gives 80.22; quote
≈79.9), DDT **−1.22 K at the same rate** (100.87 vs 100.83 kbps).

Reading:

* **Ext partitions lose on the composite** even though they save 1.1% of the
  bytes at the same QP: JOD −0.030, BA +0.39, SS2 −0.35 versus A0 at QP110.
  The rate/PSNR Pareto win recorded in `docs/video-ext-nightly.md` is real,
  but the saved bits buy PSNR the perceptual metrics do not value here.
* **Sub-8 keyframe leaves** lower the rate by ~3% and are neutral-to-positive
  in JOD, but Butteraugli is worse at the two brackets that straddle the target
  (4.50/4.84 at QP108/109 vs 4.09/4.19 for the control), SSIMULACRA2 is mixed
  and the K curve is dominated by the QP ripple.
* **DDT alone is negative** (−1.2 K at matched rate). It is an inter TX tool,
  not a partition tool, and it does not pay for its extra mode bits on this
  clip. It does recover most of what SDP loses (below), which is why the pair
  was worth measuring.

## SDP arms

SDP needs `--video-intra-full` off. The control (**A-noif**) is A0 minus that
one flag:

| QP | kbps | JOD | BA | SS2 | PSNR-Y | K |
|---:|---:|---:|---:|---:|---:|---:|
| 110 | 100.996 | 9.6925 | 4.029 | 79.788 | 44.203 | 80.76 |
| 111 | 100.428 | 9.6859 | 4.086 | 79.472 | 44.172 | 80.59 |

Iso-rate 100.0 kbps: **K 80.5**. Dropping `--video-intra-full` alone costs
0.8 K; that penalty belongs to the SDP constraint, not to SDP.

`--video-sdp` on the same flags (CCSO kept, which limits the decodable QPs):

| QP | kbps | JOD | BA | SS2 | PSNR-Y | K | decodes |
|---:|---:|---:|---:|---:|---:|---:|---|
| 109 | 107.232 | 9.7043 | 4.971 | 79.250 | 44.243 | 79.47 | 60/60 |
| 110 | 106.072 | 9.7000 | 4.899 | 79.217 | 44.141 | 79.53 | 60/60 |
| 111 | 104.300 | 9.6803 | 5.185 | 78.823 | 44.028 | 79.04 | 60/60 |
| 112-114 | — | — | — | — | — | — | **fails** |

At matched QP SDP is 1.2-1.6 K lower and 4-5% larger, and its keyframe is
5.4% larger (QP110, 1 frame: 20,320 B / 172 leaves vs 19,278 B / 192 leaves).
It never reaches a decodable 100 kbps point with CCSO: every QP from 112 up
fails `dav2d --strict 1`. Even if the QP111 point is used as an upper bound, it
is 4.3 kbps above the target and 2.2 K below A0.

Without CCSO the whole range is decodable, so the pair was bracketed at ~100
kbps with the SDP-compatible control **AN** (A0 minus `--video-intra-full`
minus CCSO):

| Arm | QP | kbps | JOD | BA | SS2 | PSNR-Y | K |
|---|---:|---:|---:|---:|---:|---:|---:|
| AN (control) | 119 | 97.932 | 9.4566 | 6.226 | 72.512 | 40.596 | 75.83 |
| | **120** | **99.192** | **9.4050** | **5.742** | **71.768** | **40.419** | **75.90** |
| `--video-sdp` | **120** | **95.480** | **9.0935** | **8.084** | **62.532** | **38.840** | **70.99** |
| | 121 | 99.064 | 9.0173 | 11.816 | 62.003 | 37.451 | 68.82 |
| `--video-sdp --video-ddt` | 120 | 102.892 | 9.3896 | 6.423 | 71.212 | 40.344 | 75.11 |
| | **121** | **101.020** | **9.3325** | **6.775** | **69.558** | **39.978** | **74.26** |

SDP alone is **−6 to −7 K** at ~99 kbps (68.82 vs 75.90; at QP120 it is
70.99 at 95.5 kbps) and its BA doubles. DDT recovers
about +4 K over SDP at the same QP (for 7.4 kbps more), but interpolated to
100 kbps it is ~73.8, about 2 K below the no-SDP control's 75.9. So the SDP arm is dominated three ways.

## Set B — the `quality` preset

At QP190/191 the preset's tree is already in the collapsed regime (the
historical variance/SAD cost makes every optional split unaffordable above
q≈163), so the partition tools cannot change a decision:

| Arm | QP | Bytes | kbps | JOD | BA | SS2 | PSNR-Y | K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `quality` (B0) | 190 | 25,141 | 100.564 | 6.3021 | 19.731 | −4.071 | 27.582 | 42.46 |
| | 191 | 25,351 | 101.404 | 6.2466 | 20.087 | −5.469 | 27.435 | 41.90 |
| `+ --video-ext-partitions` | 190 | 25,141 | 100.564 | 6.3021 | 19.731 | −4.071 | 27.582 | 42.46 |
| | 191 | 25,351 | 101.404 | 6.2466 | 20.087 | −5.469 | 27.435 | 41.90 |
| `+ --video-min-block 4` | 190 | 25,141 | 100.564 | 6.3021 | 19.731 | −4.071 | 27.582 | 42.46 |

Both arms are **byte-identical** to the preset at both QPs: no partition
decision, no leaf, no byte changes. The composite cannot distinguish them.

SDP does change the preset keyframe, so it was measured against the preset with
the two incompatible tools removed (`--video-intra-full`, CCSO) at the ~100 kbps
point (~QP193):

| Arm | QP | kbps | JOD | BA | SS2 | PSNR-Y | K |
|---|---:|---:|---:|---:|---:|---:|---:|
| B control (no intra-full, no CCSO) | 193 | 101.120 | 5.3117 | 22.448 | −27.479 | 26.074 | 33.26 |
| | **194** | **98.912** | **5.2873** | **23.344** | **−27.495** | **25.971** | **33.03** |
| `+ --video-sdp` | 193 | 101.564 | 5.3147 | 22.330 | −28.664 | 26.060 | 33.09 |
| | **194** | **99.108** | **5.2992** | **23.225** | **−27.211** | **25.971** | **33.15** |

Iso-rate 100.0 kbps: 33.13 vs 33.14 — **a wash**, and both sit 9 K below the
preset-with-filters context. SDP is not where the preset loses quality.

## Set C — 128/256 superblocks (stripped chain)

`--video-sb128/256` are rejected with the loop filters, with drift refresh and
with the probe, so C = A0 minus `--video-partition-rd`, `--video-cdef`,
`--video-cdef-rd`, `--video-ccso`, `--video-ccso-rd`,
`--video-ccso-header-rd`, `--video-ccso-bands/lambda/refit/unit-bits` and
`--video-drift-refresh`:

| Arm | QP | kbps | JOD | BA | SS2 | PSNR-Y | K |
|---|---:|---:|---:|---:|---:|---:|---:|
| C (control, 64×64) | 114 | 103.376 | 9.4255 | 6.623 | 70.993 | 40.615 | 75.09 |
| | **115** | **99.828** | **9.4416** | **6.847** | **71.224** | **40.470** | **75.03** |
| `--video-sb128` | 116 | 103.260 | 9.4074 | 7.298 | 69.368 | 40.008 | 74.21 |
| | **117** | **99.464** | **9.4162** | **7.436** | **69.605** | **40.146** | **74.20** |
| `--video-sb256` | **116** | **100.900** | **9.4212** | **6.966** | **69.654** | **40.138** | **74.57** |
| | 117 | 97.988 | 9.4118 | 7.330 | 69.909 | 40.248 | 74.30 |

Iso-rate 100.0 kbps: sb128 **74.20 (−0.83 K)**, sb256 **74.49 (−0.54 K)**
versus the control's 75.03. Both lose on BA and SSIMULACRA2 while CVVDP is
roughly flat (−0.02 JOD). The 128 root is 58% slower and the 256 root 59% slower on this
clip (see encode cost), so neither is a candidate.

## Retuned drift recipe

The two arms that compose with the tuned recipe (`--fast-decode
--video-drift-refresh-threshold 1.5 --video-drift-refresh-budget 3250`) are
ext partitions and min-block-4. The SDP and superblock arms cannot be stacked
(constraint log), and the standalone numbers above are 7-10 K below this
context.

| Arm | QP | kbps | JOD | BA | SS2 | PSNR-Y | K |
|---|---:|---:|---:|---:|---:|---:|---:|
| **tuned baseline** | **110** | **100.584** | **9.7605** | **2.656** | **81.250** | **44.789** | **83.94** |
| | **111** | **99.800** | **9.7513** | **2.731** | **81.422** | **44.703** | **83.75** |
| `+ ext-partitions` | 109 | 102.808 | 9.7221 | 2.918 | 80.268 | 44.683 | 82.99 |
| | 110 | 101.148 | 9.7500 | 2.835 | 80.936 | 44.768 | 83.43 |
| | **111** | **100.332** | **9.7333** | **2.862** | **81.019** | **44.679** | **83.30** |
| | 112 | 97.824 | 9.7489 | 2.805 | 80.972 | 44.614 | 83.49 |
| `+ --video-min-block 4` | 109 | 100.812 | 9.7420 | 3.033 | 80.967 | 44.833 | 82.97 |
| | 110 | 98.968 | 9.7354 | 2.906 | 80.771 | 44.586 | 83.17 |
| | 111 | 98.432 | 9.7552 | 2.610 | 81.568 | 44.695 | 84.07 |
| | 112 | 96.132 | 9.7527 | 2.878 | 81.238 | 44.570 | 83.40 |

Iso-rate 100.0 kbps: baseline **83.80**, ext **83.33 (−0.47)**, min-block-4
**83.06 (−0.74)**. The min-block arm's QP111 point (98.43 kbps, 84.07) is
0.32 K above the baseline at a lower rate, but its neighbouring QPs are
82.97 (100.81 kbps) and 83.17 (98.97 kbps), so it is the ±1.6 K ripple, not a
reproducible gain. Both arms lose on BA and SS2 at matched QP; neither is a
candidate.

## Constraint log

| # | Combination | Result | Evidence |
|---|---|---|---|
| 1 | `--video-partition-rd` + `--video-sdp` | rejected by name | `validate.cpp:2467`, "the SDP tree has a separate cost model" |
| 2 | `--video-sdp` + `--video-intra-full` | rejected by name | `video_frame.cpp:884`; the SDP writer's YMode inversion drifts at the luma/chroma boundary (`docs/negative-results.md`) |
| 3 | `--video-sdp` + `--video-ccso` | **new: decode failure** | in isolation fails `dav2d --strict 1` at every QP tried (108-120); in the full nightly-minus-probe composition it decodes at QP 109-111 and fails at QP 112+. Every failure is "Invalid argument" / "No data decoded". No decodable ~100 kbps SDP+CCSO point exists |
| 4 | `--video-sdp` + `--video-ext-partitions` | **new: encode-time failure** | validation accepts it; the encode stops with `keyframe encode failed: node 0: node has a stale allowed_mask` on both 4-frame and 60-frame clips |
| 5 | `--video-sb128/256` + loop filter | rejected by name | `video_frame.cpp:734`, "the CDEF/CCSO/GDF/LR unit grids are 64-based" |
| 6 | `--video-sb128/256` + `--video-drift-refresh` | rejected by name | `video_frame.cpp:979`, "wired for the 64x64 root partition search" |
| 7 | `--video-sb128/256` + `--video-partition-rd` | rejected by name | `validate.cpp:2502`, "does not compose with the 128/256 superblock trees" |
| 8 | `--video-min-block 4` + the A0 chain | accepted | P-frame search floor stays 8, so the flag only reaches the keyframe |
| 9 | `--video-ext-partitions` + the A0 chain | accepted | probe + ext + loop filters + drift refresh all compose |
| 10 | `--video-ddt` + A0 | accepted | only TIP/SVC/lossless/GPU reject it |

Constraint 3 is the sharpest: it is not in `validate.cpp`, and it is the
reason the SDP+CCSO bracket stops at 104.3 kbps (QP111). It is recorded here
rather than gated because a gate would freeze a decode failure that is not
understood; the combination is only reachable by typing the two flags
together, since no preset enables SDP.

## Can the partition-RD probe be applied to the SDP tree?

**No, and it does not need to be.** The two tools never price the same node:

* `InterFrameEncoder::decide()` enters `decide_sdp()` only for
  `keyframe_ && options_.enable_sdp` (`video_frame_decision.cpp:79`). P frames
  always use the rect (or extended) solver.
* The probe is predicated on `options_.partition_rd && !keyframe_ && …`
  (`video_frame_decision.cpp:143`), and prices leaves with one NEWMV search
  plus a quantised luma residual. It has no intra leaf model.
* `decide_sdp()` prices its luma leaves with `block_variance_cost_at()` plus a
  bare `2 × 0.85 × 2^((q−12)/6)` signal cost (`video_frame_decision.cpp:24-25,
  40`). That is the *older* of the two keyframe costs: it does not even use the
  rate-aware `partition_model_price_at()` + 6-leaf-bit estimate that the rect
  keyframe path uses. This is the "separate cost model" the rejection names.

So enabling both flags means "SDP keyframe (legacy cost) + probed P frames",
which is exactly what a relaxed validation would give. That was tested with a
local three-line validation relaxation (reverted, not in this branch): the
result is the `AN_sdp_probe` rows below, all 60/60 under `--strict 1`. The arm
does not win, so there is nothing to carry.

| Arm | QP | kbps | JOD | BA | SS2 | PSNR-Y | K |
|---|---:|---:|---:|---:|---:|---:|---:|
| `AN --video-sdp --video-partition-rd` | 118 | 98.924 | 9.2928 | 6.950 | 69.178 | 39.867 | 73.86 |
| | 119 | 98.324 | 9.4208 | 5.596 | 72.095 | 41.084 | 76.18 |
| | 120 | 94.968 | 9.4397 | 6.832 | 70.521 | 40.479 | 74.91 |
| | 121 | 97.132 | 9.2730 | 6.994 | 66.998 | 39.123 | 73.37 |

The probe recovers about +7 K over SDP-without-probe in that context (it
prevents a P-chain Butteraugli collapse), but the arm is still ~1 K below the
same-flags control (AN QP120: 99.19 kbps, K 75.90) and the context itself is
7 K below the nightly headline because CCSO is gone.

Making the probe price the SDP keyframe would be a different feature: an intra
probe needs a mode search plus a quantised intra residual, not one NEWMV
prediction, and `decide_sdp()` also has to be reconciled with the rect
keyframe's rate-aware fallback. That is not a small change, and SDP is not
promising enough to justify it.

## Verdict

| Claim | Result |
|---|---|
| Any partition tool beats nightly 82.15 at 100 kbps | **No.** Best arm is ext partitions at 80.53 on A0. The probe itself is worth +0.87 K over A0 (81.28 → 82.15), more than any tool in this study. |
| Any partition tool beats the tuned drift recipe 83.80 | **No.** Best stacked arm is ext at 83.33 (−0.47 K); min-block-4 is 83.06 (−0.74 K). |
| Superblock trees help at 240p | **No.** −0.83 K (128) and −0.54 K (256) on a stripped chain, at 1.6× encode cost. |
| SDP helps | **No.** −1.2 to −1.6 K at matched QP on A; −5 K at ~100 kbps on the CCSO-free control; a wash on the quality preset. |
| DDT helps partition decisions | **No.** −1.2 K alone; it only recovers part of SDP's collapse. |
| The probe can be reused for the SDP tree | **No need.** The probe is inter-only and SDP is keyframe-only; the SDP keyframe keeps its own (older) cost model. |

Nightly stays at **82.15 K**, the tuned drift recipe stays at **83.80 K**, and
no flag was added.

## Encode cost

Best of 3 encodes, 60 frames, `--cpu-threads 8`, `nice -n 10`, dev build
(`-O1`; absolute values are not release timings, only the ratios are
meaningful):

| Configuration | QP | Wall | Ratio |
|---|---:|---:|---:|
| tuned baseline (D0) | 110 | 7.04 s | 1.00× |
| + ext partitions | 110 | 7.82 s | 1.11× |
| + min-block 4 | 110 | 7.16 s | 1.02× |
| A0 | 110 | 7.08 s | 1.00× |
| A0 + ext | 110 | 8.71 s | 1.23× |
| A0 + min-block 4 | 110 | 7.08 s | 1.00× |
| A0 + DDT | 110 | 5.86 s | 0.83× |
| A-noif | 110 | 5.08 s | 1.00× |
| A-noif + SDP | 110 | 5.89 s | 1.16× |
| C control | 116 | 5.23 s | 1.00× |
| C + sb128 | 116 | 8.26 s | 1.58× |
| C + sb256 | 116 | 8.31 s | 1.59× |
| `quality` preset | 190 | 2.82 s | 1.00× |
| `quality` + ext | 190 | 2.97 s | 1.05× |

## Verification

* Every stream in every table decoded 60/60 under
  `tools/bin/dav2d-macos --strict 1`; the SDP+CCSO QPs 112+ are the only
  failures and are marked as such.
* The worktree's only source edit (the temporary validation relaxation for the
  probe+SDP experiment) was reverted; `git status` is clean apart from this
  document.
* Pins re-checked after the revert and rebuild:
  * default QP189: 25,009 B, MD5 `a1d0dba1…`
  * `quality` QP190: 25,141 B, MD5 `3984677b…`
  * `--fast-decode` QP110: 23,690 B, MD5 `1b99127a…`
  * `--fast-decode` QP106: 25,008 B, MD5 `0639b436…`, 60/60 decode
* `tools/gate_video_nightly.sh --bin build-dev/av2enc`: OK (expansion, pins,
  strict decode, visible recon, BAWP).

## Reproduction

```sh
REF=testdata/nasa_480x240_30fps_60f.y4m
BIN=build-dev/av2enc
A0="--video-rd-preset quality --video-ccso-rd --video-ccso-bands 6 \
    --video-ccso-lambda 22 --video-ccso-refit 2 --video-ccso-unit-bits 2 \
    --video-edge-inter --video-edge-mask --video-key-qp-offset -27 \
    --video-rdoq --video-cdef --video-cdef-rd --video-bawp \
    --video-bawp-carrier --video-leaf-rd --video-subpel-eighth \
    --video-key-intra-txpart --video-qp-lambda --avg-cdf --adaptive-cdf \
    --cdf-continuity --video-drift-refresh --video-tile-flush-min \
    --video-ref-signal-compact --video-tile-group-compact \
    --video-ccso-header-rd"

# control, ext partitions, sub-8 keyframe leaves
$BIN "$REF" -o /tmp/a0.ivf --video --frames 60 --cpu-threads 8 --qp 110 $A0
$BIN "$REF" -o /tmp/ae.ivf --video --frames 60 --cpu-threads 8 --qp 110 $A0 \
     --video-ext-partitions
$BIN "$REF" -o /tmp/am.ivf --video --frames 60 --cpu-threads 8 --qp 110 $A0 \
     --video-min-block 4

# tuned recipe + the two arms that stack
$BIN "$REF" -o /tmp/d0.ivf --video --frames 60 --cpu-threads 8 --qp 110 \
     --fast-decode --video-drift-refresh-threshold 1.5 \
     --video-drift-refresh-budget 3250
$BIN "$REF" -o /tmp/de.ivf --video --frames 60 --cpu-threads 8 --qp 110 \
     --fast-decode --video-drift-refresh-threshold 1.5 \
     --video-drift-refresh-budget 3250 --video-ext-partitions
```

`A0` above is the typed nightly expansion; the probe is omitted on purpose.
Adding `--video-partition-rd` reproduces `--fast-decode` byte for byte (verified on
`testdata/pub_640x360_4f.y4m`: MD5 `cd4b0f24…` both ways).
