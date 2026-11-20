# K-optimal operating point at 240p30 / 100 kbps (`--fast-decode`)

Status: **measured; one flag pair moves K by +1.6 at iso-rate and nothing else
outside the metric's own QP ripple does.** No source changes. This document is
the record of a flag-only sweep of the 60-frame 240p / 100 kbps operating point
(`testdata/nasa_480x240_30fps_60f.y4m`, `--fast-decode`, `--qp` bracketing 100 kbps).

`K = 100 * (0.5*clamp(JOD/10) + 0.25/(1+BA/3) + 0.25*clamp((SS2+50)/150))`
(`tools/kpi.py`). Every number below is K, and every comparison is at a matched
rate: configs were run at two or three QPs that bracket 100.0 kbps and K was
interpolated linearly in rate.

## Verdict

| Arm | QP | Bytes | kbps | JOD | Butteraugli | SSIMULACRA2 | PSNR-Y | K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline `--fast-decode` | 106 | 25,008 | 100.03 | 9.7334 | 3.420 | 80.84 | 44.67 | **82.16** |
| **best: + threshold 1.5 / budget 3250** | 110 | 25,146 | 100.58 | 9.7605 | 2.656 | 81.25 | 44.79 | **83.94** |
| same arm at QP111 | 111 | 24,950 | 99.80 | 9.7513 | 2.731 | 81.42 | 44.70 | 83.75 |
| **best arm, iso-rate 100.0** | — | 25,000 | 100.00 | 9.7536 | 2.712 | 81.38 | 44.73 | **83.80** |
| runner-up: threshold 1.5 / budget 3000 | 110 | 24,996 | 99.98 | 9.7474 | 2.775 | 81.33 | 44.71 | 83.61 |

Deltas of the best arm at iso-rate vs the baseline anchor:

| | baseline | best (iso-rate) | Δ |
|---|---:|---:|---:|
| rate | 100.03 kbps | 100.00 kbps | −0.03 |
| bytes (60 f) | 25,008 | 25,000 | −8 |
| CVVDP JOD | 9.7334 | 9.7536 | **+0.020** |
| Butteraugli | 3.420 | 2.712 | **−0.708** |
| SSIMULACRA2 | 80.84 | 81.38 | **+0.54** |
| PSNR-Y | 44.67 dB | 44.73 dB | +0.06 dB |
| K | 82.155 | **83.795** | **+1.64** |

K decomposition of that +1.64: **+1.45 from Butteraugli** (q_ba 0.4673 → 0.5252),
+0.10 from CVVDP (q 0.9733 → 0.9754), +0.09 from SSIMULACRA2 (q 0.8722 → 0.8759).
The gain lands almost entirely on the metric that lags SVT, which is the point
of scoring K instead of JOD.

Recipe (`--fast-decode` plus two overrides; byte-identical to typing the whole
expansion out with those values):

```sh
build-dev/av2enc testdata/nasa_480x240_30fps_60f.y4m -o /tmp/x.ivf \
  --video --frames 60 --cpu-threads 8 --qp 110 --fast-decode \
  --video-drift-refresh-threshold 1.5 --video-drift-refresh-budget 3250
```

Stream: 25,146 B, MD5 `c60250c02a315d725af19c74c4d1a5df`. `dav2d-macos
--strict 1` decodes 60/60 pictures and the decoded payload equals the
`--video-recon` payload byte for byte (10,368,360 B).

## Method

- Worktree `exp/k-sweep` (`git worktree add ... main`), `cmake --preset dev`,
  `cmake --build build-dev -j2`; `testdata` and `tools/bin` symlinked from the
  main worktree. Host: Apple M4, `--cpu-threads 8`, single tile, 8-bit 4:2:0.
- Per point: encode → `dav2d-macos --strict 1` → `tools/quality.py`
  (CVVDP 0.5.7, `standard_fhd`, MPS) → `tools/extra_metrics.py` → `tools/kpi.py`.
  One metric job at a time, `nice -n 10`.
- Validity gate: the decoded Y4M must hold 60 frames (payload 10,368,000 B). A
  point that fails is recorded as INVALID and never scored.
- Rate matching: QP is chosen per arm so that 100.0 kbps lies between two
  measured points; K (and each component) is interpolated linearly in rate. A
  point within ±0.6% of 100 kbps is quoted directly.

### Baseline reproduction

QP106 `--fast-decode` reproduces the pin exactly — 25,008 B, 100.032 kbps,
9.7334 JOD, 3.4202 BA, 80.8366 SS2, K 82.155, MD5
`0639b43630abb454f7be5ac494d2efb4` — so the harness and the tool versions match
the recorded baseline.

### The expansion actually used

`docs/video-nightly.md` lists the expansion without the four overhead levers, so
typing that list out is **not** byte-identical to `--fast-decode` any more (it is
25,756 B vs 25,008 B). The current expansion, which reproduces the pin byte for
byte, is:

```text
--video-rd-preset quality
--video-ccso-rd --video-ccso-bands 6 --video-ccso-lambda 22
  --video-ccso-refit 2 --video-ccso-unit-bits 2
--video-edge-inter --video-edge-mask --video-partition-rd
--video-key-qp-offset -27 --video-rdoq --video-cdef --video-cdef-rd
--video-bawp --video-bawp-carrier --video-leaf-rd --video-subpel-eighth
--video-key-intra-txpart --video-qp-lambda --avg-cdf --adaptive-cdf
--cdf-continuity --video-drift-refresh
--video-tile-flush-min --video-ref-signal-compact
--video-tile-group-compact --video-ccso-header-rd
```

`--video-rd-preset quality` is `--video-inter-residual --video-deblock
--video-ccso --video-tx-search --video-intra-full`. A flag sweep on top of
`--fast-decode` is equivalent to editing this list, with one trap: an explicit
`--video-partition-rd-lambda` also re-pins the partition fallback price to the
legacy 700 SSE/bit (the QP-scaled aggregate uses ~6), so that flag's rows below
mix two changes.

### The baseline curve, and the ripple

| QP | kbps | JOD | BA | SS2 | K |
|---:|---:|---:|---:|---:|---:|
| 104 | 104.63 | 9.7185 | 4.172 | 80.11 | 80.73 |
| 105 | 101.37 | 9.7216 | 3.541 | 80.40 | 81.81 |
| **106** | **100.03** | **9.7334** | **3.420** | **80.84** | **82.16** |
| 107 | 99.54 | 9.7113 | 4.323 | 80.26 | 80.51 |
| 108 | 97.16 | 9.7216 | 3.682 | 80.63 | 81.60 |

One QP step moves K by up to **1.9** (106 → 107) at essentially the same rate,
because Butteraugli swings between 3.42 and 4.32 while JOD barely moves. Any
difference below ~0.3 K at this operating point is not resolvable, and a single
point 2% off 100 kbps can look 0.4 K better or worse purely from that ripple.
This is why the arms that mattered were bracketed with two QPs.

## Sweep

### Loop filters (ablation and GDF)

| Arm | kbps | JOD | BA | SS2 | K | Read |
|---|---:|---:|---:|---:|---:|---|
| baseline | 100.03 | 9.7334 | 3.420 | 80.84 | 82.16 | — |
| − `--video-cdef --video-cdef-rd` (105/106) | 99.66 / 98.53 | 9.721 / 9.730 | 4.034 / 3.498 | 80.35 / 80.49 | 80.99 / 81.94 | the 105 point is 0.3% off the target and 1.2 K down, so CDEF is worth **≈+0.9 K** at iso-rate |
| − all `--video-ccso*` | 113.25 | 9.3941 | 6.067 | 74.13 | 75.93 | CCSO is structural: without it the same QP costs 13% more bits and loses 6 K |
| − `--video-deblock` | 99.95 | 9.7232 | 3.561 | 80.44 | 81.79 | deblock +0.37 K |
| − `--video-ccso-header-rd` only (106/107) | 102.22 / 99.88 | 9.724 / 9.722 | 3.263 / 3.676 | 81.04 / 80.17 | 82.44 / 81.54 | iso-rate 81.59 → the row pricing is worth **+0.57 K** |
| + `--video-gdf` | 101.17 | 9.6932 | 4.273 | 79.58 | 80.38 | clearly negative (consistent with `docs/negative-results.md`) |
| + `--video-wiener` / `--video-pc-wiener` / `--video-chroma-lr` | — | — | — | — | INVALID at sweep time | At sweep time **the pinned decoder rejected the stream** (2/60 frames decode, `Error decoding frame: Invalid argument`); CVVDP on the single decoded frame reported a bogus 9.92 JOD, so these arms were excluded. Root cause (found on `fix/lr-decode`): the six §6.17.2 loop-restoration CDF rows were missing from the carried §8.2 image, so `--cdf-continuity` desynchronised frame 3. The 60-frame streams now decode 60/60; re-measure these arms before quoting a K. |

CDEF strength/damping variants (both change the filtered picture):

| Arm | kbps | JOD | BA | SS2 | K | Read |
|---|---:|---:|---:|---:|---:|---|
| `--video-cdef-strength 2` | 99.90 | 9.7069 | 3.691 | 80.75 | 81.53 | worse |
| `--video-cdef-strength 4` | 100.12 | 9.7319 | 3.353 | 80.60 | 82.23 | +0.08 at the point; not a robust gain |
| `--video-cdef-damping 4` | 100.65 | 9.7413 | 3.300 | 80.60 | 82.38 | +0.22 at 0.6% above the target; its QP105 neighbour is 80.87, so unbracketed — not a robust gain |

### RD lambdas

| Arm | kbps | JOD | BA | SS2 | K | Read |
|---|---:|---:|---:|---:|---:|---|
| `--video-leaf-rd-lambda 100` | 102.20 | 9.7160 | 3.596 | 80.82 | 81.75 | worse |
| `--video-leaf-rd-lambda 115` | 100.53 | 9.7114 | 3.546 | 80.68 | 81.80 | worse |
| `--video-leaf-rd-lambda 145` | 99.44 | 9.6922 | 3.435 | 80.31 | 81.84 | worse |
| `--video-leaf-rd-lambda 200` (105/106) | 99.15 / 97.67 | 9.679 / 9.680 | 3.557 / 3.339 | 79.56 / 79.56 | 81.43 / 81.83 | worse; the SS2 loss dominates |
| `--video-tx-lambda 48` | 99.81 | 9.7113 | 3.908 | 80.10 | 81.10 | −1.1 K |
| `--video-tx-lambda 64` | 99.47 | 9.7196 | 4.268 | 80.29 | 80.63 | −1.5 K |
| `--video-rdoq-scale 0.05` | 100.52 | 9.7235 | 3.271 | 80.50 | 82.33 | +0.17 raw, ≈+0.08 at iso-rate |
| `--video-rdoq-scale 0.2` | 100.23 | 9.7111 | 3.915 | 80.63 | 81.17 | worse |
| `--video-partition-rd-lambda 1.375e-6` | 102.58 | 9.7011 | 4.199 | 80.12 | 80.61 | worse (also re-pins the fallback) |
| `--video-partition-rd-lambda 5.5e-6` | 102.73 | 9.7169 | 3.418 | 80.42 | 82.01 | worse at iso-rate |
| `--video-partition-rd-depth 2` | 102.39 | 9.7301 | 3.273 | 80.44 | 82.35 | +0.19 raw, ≈−0.1 at iso-rate |

The QP-scaled defaults (`leaf` ≈130, `tx` ≈46, `rdoq` 0.107 at QP106) are on
their plateau; every explicit value is neutral-to-worse.

### Key/P allocation (`--video-key-qp-offset`)

| offset | key QP | QP | kbps | JOD | BA | SS2 | K | Read |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| −21 | 85 | 106 | 92.98 | 9.7423 | 3.862 | 80.05 | 81.31 | fewer key bits; at ≈97 kbps it measures 80.45–81.49, so at 100 kbps ≈81.0 — worse |
| −21 | 81 | 102 | 97.48 | 9.6961 | 4.288 | 80.06 | 80.45 |  |
| −27 | 79 | **106** | **100.03** | **9.7334** | **3.420** | **80.84** | **82.16** | the pin |
| −35 | 71 | 106 | 110.64 | 9.6954 | 4.034 | 80.09 | 80.82 | needs a coarser P QP to hit 100 → worse |
| −45 | 61 | 106 | 125.22 | 9.7245 | 3.639 | 81.39 | 81.82 | vastly over budget at 100 kbps |

The key frame is ~79% of the stream at this point, so the offset is the main
rate knob. Trading key bits for P bits at a fixed total rate loses: the finer
key propagates through the whole P chain. −27 is confirmed as the optimum.

### Partition / tree

| Arm | kbps | JOD | BA | SS2 | K | Read |
|---|---:|---:|---:|---:|---:|---|
| `--video-ext-partitions` (105/106) | 100.64 / 98.99 | 9.719 / 9.713 | 3.338 / 3.388 | 80.63 / 80.49 | 82.20 / 82.05 | iso-rate 82.14 — a wash |
| `--video-partition-rd-depth 2` | 102.39 | 9.7301 | 3.273 | 80.44 | 82.35 | see above, wash |
| `--video-sb128` | — | — | — | — | rejected | `--video-partition-rd does not compose with the 128/256 superblock trees` |
| `--video-max-depth 6` | 100.03 | 9.7334 | 3.420 | 80.84 | 82.16 | byte-identical (depth 4 is never reached) |
| `--video-min-block 4` | 97.53 | 9.7015 | 4.077 | 80.09 | 80.79 | worse |

### Coding tools not in the aggregate

| Arm | kbps | JOD | BA | SS2 | K | Read |
|---|---:|---:|---:|---:|---:|---|
| `--video-cfl` | 99.18 | 9.7031 | 3.463 | 80.23 | 81.83 | worse |
| `--video-ist` | 100.08 | 9.7334 | 3.420 | 80.84 | 82.16 | never selected (stream differs by 11 header bits only) |
| `--video-idtx` | 99.67 | 9.7125 | 3.436 | 80.44 | 81.96 | worse |
| `--video-ibp` | 98.59 | 9.6985 | 3.787 | 80.22 | 81.25 | worse |
| `--video-dip` | 102.50 | 9.7195 | 3.472 | 80.80 | 81.98 | worse at iso-rate |
| `--video-ddt` | 98.15 | 9.6739 | 3.851 | 79.52 | 80.90 | worse |
| `--video-cctx` (104/105/106) | 102.76 / 99.44 / 97.56 | 9.703 / 9.714 / 9.723 | 4.315 / 4.124 / 3.380 | 80.08 / 80.63 / 80.65 | 80.45 / 80.87 / 82.15 | iso-rate 80.5; the QP106 point only looks good because of the BA ripple |
| `+ --video-drl-reorder` | 100.03 | 9.7334 | 3.420 | 80.84 | 82.16 | neutral (syntax bit only) |
| `- --video-intra-full` | 100.00 | 9.7221 | 3.879 | 80.63 | 81.29 | intra-full is +0.87 K |
| `- --video-tx-search --video-key-intra-txpart` | 111.75 | 9.7156 | 4.007 | 80.45 | 81.02 | TX search is structural (saves 12% rate) |
| `- --video-inter-residual` | — | — | — | — | rejected | needs `--video-rdoq` off; not separable without disabling the aggregate's trellis |
| `--video-tcq` | — | — | — | — | rejected | `--video-rdoq ... does not compose with --video-tcq` |
| `--video-qm` | — | — | — | — | rejected | `--video-qm is gated at --qp 90`; unavailable at 100 kbps |
| `--video-flex-mvres` | — | — | — | — | rejected | does not compose with `--video-bawp` |
| `--video-drift-refresh-qp` | — | — | — | — | rejected | only accepts 0..64; it cannot make a refresh finer than the frame QP |

### Drift refresh: the one lever that moves K

The aggregate caps the per-sequence refresh payload at 1,200 B (9 frames,
1,201 B used at QP106) and triggers on a 2.5 dB drop below the key's PSNR.
Raising the cap and letting the same budget cover more frames is the win:

| threshold | budget | QP | bytes | kbps | JOD | BA | SS2 | K |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2.5 | 1200 (default) | 106 | 25,008 | 100.03 | 9.7334 | 3.420 | 80.84 | 82.16 |
| 2.5 | 2000 | 110 | 24,381 | 97.52 | 9.7396 | 4.234 | 81.06 | 80.91 |
| 2.5 | 2500 | 110 | 24,471 | 97.88 | 9.7584 | 3.286 | 81.79 | 82.69 |
| 2.5 | 3000 | 110 | 24,903 | 99.61 | 9.7633 | 3.160 | 81.92 | 82.98 |
| 2.5 | 3200 | 110 | 25,248 | 100.99 | 9.7586 | 3.149 | 82.02 | 82.99 |
| 2.5 | 3500 | 110 | 25,460 | 101.84 | 9.7715 | 3.083 | 82.58 | 83.28 (≈82.8 iso) |
| 2.5 | 4500 | 110 | 26,278 | 105.11 | 9.7795 | 2.949 | 83.06 | 83.68 (≈82.5 iso) |
| 2.5 | 6000 | 115/116 | 24,791 / 24,276 | 99.16 / 97.10 | 9.752 / 9.755 | 3.316 / 3.299 | 82.35 / 82.45 | 82.69 / 82.75 (≈82.9 iso) |
| 1.0 | 1200 | 106 | 25,242 | 100.97 | 9.7001 | 3.822 | 79.54 | 81.08 |
| 1.0 | 3000 | 110 | 25,129 | 100.52 | 9.7194 | 2.870 | 80.35 | 83.10 |
| 1.5 | 2750 | 110 | 24,893 | 99.57 | 9.7485 | 3.080 | 81.13 | 82.93 |
| **1.5** | **3000** | **110** | **24,996** | **99.98** | **9.7474** | **2.775** | **81.33** | **83.61** |
| 1.5 | 3250 | 110 | 25,146 | 100.58 | 9.7605 | 2.656 | 81.25 | **83.94** |
| 1.5 | 3250 | 111 | 24,950 | 99.80 | 9.7513 | 2.731 | 81.42 | 83.75 |
| 1.5 | 3500 | 110 | 25,440 | 101.76 | 9.7520 | 3.023 | 81.73 | 83.17 |
| 1.7 | 3000 | 110 | 24,896 | 99.58 | 9.7409 | 2.860 | 81.41 | 83.41 |
| 2.0 | 3000 | 110 | 25,044 | 100.18 | 9.7604 | 2.976 | 81.43 | 83.26 |
| 2.0 | 4000 | 111 | 25,912 | 103.65 | 9.7557 | 2.986 | 82.01 | 83.31 (≈82.6 iso) |
| 3.0 | 3000 | 110 | 24,923 | 99.69 | 9.7636 | 3.475 | 82.13 | 82.42 |

Threshold 1.0 and 0.5 are byte-identical (the trigger floors at 1.0 dB). The
mechanism: with a larger cap the refresh residual is no longer truncated, so the
same payload repairs the drifted tail instead of half-repairing more frames.
At 1.5/3000, 29 of 60 frames refresh with 3,048 B; at 1.0/3000, 35 frames share
2,983 B and every one of them is under-repaired (SS2 80.35 vs 81.33).

The optimum is a narrow plateau: budget 3,000–3,250 with threshold 1.5–1.7
all measure 83.4–83.9 at 100 kbps. 2,500 and below leave drift on the table;
3,500 and above buy quality that the rate no longer pays for.

### Combinations

The three plausible stacks of the small non-negative knobs were measured at
QP106; none beat the plain aggregate at iso-rate. Stacking onto the refresh win
(bottom three rows) is exactly neutral:

| Arm | kbps | JOD | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|
| ext-partitions + cdef-strength 4 + rdoq-scale 0.05 | 99.20 | 9.7145 | 3.364 | 80.31 | 82.08 |
| ext-partitions + cdef-damping 4 | 99.27 | 9.7034 | 3.630 | 80.18 | 81.53 |
| ext-partitions + cdef-strength 4 + cdef-damping 4 | 99.35 | 9.6990 | 3.623 | 80.35 | 81.54 |
| threshold 2.5 + budget 3000 + ext-partitions | 99.68 | 9.7665 | 3.171 | 82.04 | 82.99 |
| threshold 2.5 + budget 3000 + cdef-damping 4 | 100.16 | 9.7673 | 3.172 | 82.04 | 83.00 |
| threshold 2.5 + budget 3000 + cdef-strength 4 | 100.43 | 9.7599 | 3.729 | 81.82 | 81.92 |

Adding ext-partitions or cdef-damping to the refresh win is exactly neutral
(82.99–83.00 vs 82.98), so the K-optimal recipe keeps the plain aggregate plus
the two refresh flags.

## Full measurement record

Every scored point (one row per encode; `INVALID` and CLI rejects are listed in
the text above). ΔK against the baseline anchor is only indicative for rows more
than ~1% away from 100 kbps.

| config | change vs `--fast-decode` | QP | bytes | kbps | JOD | BA | SS2 | K | PSNR-Y |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `base` | `--fast-decode` (the pin) | 104 | 26,158 | 104.63 | 9.7185 | 4.172 | 80.11 | 80.73 | 44.49 |
| `base` | `--fast-decode` (the pin) | 105 | 25,342 | 101.37 | 9.7216 | 3.541 | 80.40 | 81.81 | 44.54 |
| `base` | `--fast-decode` (the pin) | 106 | 25,008 | 100.03 | 9.7334 | 3.420 | 80.84 | 82.16 | 44.67 |
| `base` | `--fast-decode` (the pin) | 107 | 24,885 | 99.54 | 9.7113 | 4.323 | 80.26 | 80.51 | 44.55 |
| `base` | `--fast-decode` (the pin) | 108 | 24,291 | 97.16 | 9.7216 | 3.682 | 80.63 | 81.60 | 44.60 |
| `cctx` | + `--video-cctx` | 104 | 25,690 | 102.76 | 9.7029 | 4.315 | 80.08 | 80.45 | 44.43 |
| `cctx` | + `--video-cctx` | 105 | 24,861 | 99.44 | 9.7139 | 4.124 | 80.63 | 80.87 | 44.61 |
| `cctx` | + `--video-cctx` | 106 | 24,390 | 97.56 | 9.7234 | 3.380 | 80.65 | 82.15 | 44.61 |
| `cdefd4` | `--video-cdef-damping 4` | 105 | 25,464 | 101.86 | 9.7049 | 3.988 | 79.70 | 80.87 | 44.25 |
| `cdefd4` | `--video-cdef-damping 4` | 106 | 25,163 | 100.65 | 9.7413 | 3.300 | 80.60 | 82.38 | 44.61 |
| `cdefs2` | `--video-cdef-strength 2` | 106 | 24,976 | 99.90 | 9.7069 | 3.691 | 80.75 | 81.53 | 44.70 |
| `cdefs4` | `--video-cdef-strength 4` | 105 | 25,520 | 102.08 | 9.7213 | 3.617 | 80.47 | 81.69 | 44.69 |
| `cdefs4` | `--video-cdef-strength 4` | 106 | 25,029 | 100.12 | 9.7319 | 3.353 | 80.60 | 82.23 | 44.69 |
| `cfl` | + `--video-cfl` | 106 | 24,794 | 99.18 | 9.7031 | 3.463 | 80.23 | 81.83 | 44.43 |
| `combo1` | ext-partitions + cdef-strength 4 + rdoq-scale 0.05 | 106 | 24,801 | 99.20 | 9.7145 | 3.364 | 80.31 | 82.08 | 44.62 |
| `combo2` | ext-partitions + cdef-damping 4 | 106 | 24,817 | 99.27 | 9.7034 | 3.630 | 80.18 | 81.53 | 44.68 |
| `combo3` | ext-partitions + cdef-strength 4 + cdef-damping 4 | 106 | 24,837 | 99.35 | 9.6990 | 3.623 | 80.35 | 81.54 | 44.61 |
| `d275th15` | threshold 1.5 + budget 2750 | 110 | 24,893 | 99.57 | 9.7485 | 3.080 | 81.13 | 82.93 | 44.74 |
| `d325th15` | threshold 1.5 + budget 3250 | 110 | 25,146 | 100.58 | 9.7605 | 2.656 | 81.25 | 83.94 | 44.79 |
| `d325th15` | threshold 1.5 + budget 3250 | 111 | 24,950 | 99.80 | 9.7513 | 2.731 | 81.42 | 83.75 | 44.70 |
| `d35th15` | threshold 1.5 + budget 3500 | 110 | 25,440 | 101.76 | 9.7520 | 3.023 | 81.73 | 83.17 | 44.73 |
| `d3cdefd4` | threshold 2.5 + budget 3000 + cdef-damping 4 | 110 | 25,040 | 100.16 | 9.7673 | 3.172 | 82.04 | 83.00 | 44.72 |
| `d3cdefs4` | threshold 2.5 + budget 3000 + cdef-strength 4 | 110 | 25,108 | 100.43 | 9.7599 | 3.729 | 81.82 | 81.92 | 44.79 |
| `d3ext` | threshold 2.5 + budget 3000 + ext-partitions | 110 | 24,921 | 99.68 | 9.7665 | 3.171 | 82.04 | 82.99 | 44.92 |
| `d3th05` | threshold 0.5 + budget 3000 | 110 | 25,129 | 100.52 | 9.7194 | 2.870 | 80.35 | 83.10 | 44.44 |
| `d3th10` | threshold 1.0 + budget 3000 | 110 | 25,129 | 100.52 | 9.7194 | 2.870 | 80.35 | 83.10 | 44.44 |
| `d3th15` | threshold 1.5 + budget 3000 | 110 | 24,996 | 99.98 | 9.7474 | 2.775 | 81.33 | 83.61 | 44.71 |
| `d3th17` | threshold 1.7 + budget 3000 | 110 | 24,896 | 99.58 | 9.7409 | 2.860 | 81.41 | 83.41 | 44.70 |
| `d3th20` | threshold 2.0 + budget 3000 | 110 | 25,044 | 100.18 | 9.7604 | 2.976 | 81.43 | 83.26 | 44.71 |
| `d3th30` | threshold 3.0 + budget 3000 | 110 | 24,923 | 99.69 | 9.7636 | 3.475 | 82.13 | 82.42 | 44.74 |
| `d4th20` | threshold 2.0 + budget 4000 | 111 | 25,912 | 103.65 | 9.7557 | 2.986 | 82.01 | 83.31 | 44.66 |
| `ddt` | + `--video-ddt` | 106 | 24,538 | 98.15 | 9.6739 | 3.851 | 79.52 | 80.90 | 44.10 |
| `dip` | + `--video-dip` | 106 | 25,624 | 102.50 | 9.7195 | 3.472 | 80.80 | 81.98 | 44.52 |
| `drift1` | threshold 1.0 (budget 1200) | 106 | 25,242 | 100.97 | 9.7001 | 3.822 | 79.54 | 81.08 | 44.09 |
| `drift1b` | threshold 1.0 + budget 3000 | 106 | 26,840 | 107.36 | 9.7227 | 3.513 | 80.26 | 81.84 | 44.58 |
| `drift25k` | budget 2500 (threshold 2.5) | 110 | 24,471 | 97.88 | 9.7584 | 3.286 | 81.79 | 82.69 | 44.74 |
| `drift2k` | budget 2000 (threshold 2.5) | 110 | 24,381 | 97.52 | 9.7396 | 4.234 | 81.06 | 80.91 | 44.63 |
| `drift32k` | budget 3200 (threshold 2.5) | 110 | 25,248 | 100.99 | 9.7586 | 3.149 | 82.02 | 82.99 | 44.80 |
| `drift35k` | budget 3500 (threshold 2.5) | 110 | 25,460 | 101.84 | 9.7715 | 3.083 | 82.58 | 83.28 | 44.88 |
| `drift45k` | budget 4500 (threshold 2.5) | 110 | 26,278 | 105.11 | 9.7795 | 2.949 | 83.06 | 83.68 | 44.97 |
| `driftb` | budget 3000 (threshold 2.5) | 106 | 26,380 | 105.52 | 9.7503 | 3.107 | 81.98 | 83.03 | 44.96 |
| `driftb` | budget 3000 (threshold 2.5) | 107 | 26,105 | 104.42 | 9.7644 | 3.247 | 81.99 | 82.83 | 44.90 |
| `driftb` | budget 3000 (threshold 2.5) | 108 | 25,653 | 102.61 | 9.7636 | 3.046 | 82.23 | 83.26 | 44.94 |
| `driftb` | budget 3000 (threshold 2.5) | 109 | 25,529 | 102.12 | 9.7646 | 3.007 | 82.28 | 83.36 | 44.91 |
| `driftb` | budget 3000 (threshold 2.5) | 110 | 24,903 | 99.61 | 9.7633 | 3.160 | 81.92 | 82.98 | 44.76 |
| `driftb` | budget 3000 (threshold 2.5) | 111 | 24,753 | 99.01 | 9.7570 | 3.260 | 82.05 | 82.77 | 44.77 |
| `driftb` | budget 3000 (threshold 2.5) | 112 | 24,537 | 98.15 | 9.7612 | 3.133 | 81.63 | 82.97 | 44.64 |
| `driftb6k` | budget 6000 (threshold 2.5) | 112 | 26,344 | 105.38 | 9.7773 | 2.723 | 83.23 | 84.20 | 44.88 |
| `driftb6k` | budget 6000 (threshold 2.5) | 115 | 24,791 | 99.16 | 9.7522 | 3.316 | 82.35 | 82.69 | 44.45 |
| `driftb6k` | budget 6000 (threshold 2.5) | 116 | 24,276 | 97.10 | 9.7547 | 3.299 | 82.45 | 82.75 | 44.36 |
| `drlreorder` | + `--video-drl-reorder` | 106 | 25,008 | 100.03 | 9.7334 | 3.420 | 80.84 | 82.16 | 44.67 |
| `extpart` | + `--video-ext-partitions` | 105 | 25,159 | 100.64 | 9.7191 | 3.338 | 80.63 | 82.20 | 44.68 |
| `extpart` | + `--video-ext-partitions` | 106 | 24,748 | 98.99 | 9.7126 | 3.388 | 80.49 | 82.05 | 44.75 |
| `gdf` | + `--video-gdf` | 106 | 25,292 | 101.17 | 9.6932 | 4.273 | 79.58 | 80.38 | 43.49 |
| `ibp` | + `--video-ibp` | 106 | 24,647 | 98.59 | 9.6985 | 3.787 | 80.22 | 81.25 | 44.57 |
| `idtx` | + `--video-idtx` | 106 | 24,918 | 99.67 | 9.7125 | 3.436 | 80.44 | 81.96 | 44.48 |
| `ist` | + `--video-ist` | 106 | 25,019 | 100.08 | 9.7334 | 3.420 | 80.84 | 82.16 | 44.67 |
| `key21` | `--video-key-qp-offset -21` | 102 | 24,371 | 97.48 | 9.6961 | 4.288 | 80.06 | 80.45 | 44.42 |
| `key21` | `--video-key-qp-offset -21` | 103 | 24,222 | 96.89 | 9.7226 | 3.723 | 80.30 | 81.49 | 44.49 |
| `key21` | `--video-key-qp-offset -21` | 106 | 23,244 | 92.98 | 9.7423 | 3.862 | 80.05 | 81.31 | 44.46 |
| `key35` | `--video-key-qp-offset -35` | 106 | 27,660 | 110.64 | 9.6954 | 4.034 | 80.09 | 80.82 | 44.47 |
| `key45` | `--video-key-qp-offset -45` | 106 | 31,304 | 125.22 | 9.7245 | 3.639 | 81.39 | 81.82 | 44.97 |
| `leaf100` | `--video-leaf-rd-lambda 100` | 106 | 25,551 | 102.20 | 9.7160 | 3.596 | 80.82 | 81.75 | 44.66 |
| `leaf115` | `--video-leaf-rd-lambda 115` | 106 | 25,133 | 100.53 | 9.7114 | 3.546 | 80.68 | 81.80 | 44.65 |
| `leaf145` | `--video-leaf-rd-lambda 145` | 106 | 24,861 | 99.44 | 9.6922 | 3.435 | 80.31 | 81.84 | 44.46 |
| `leaf200` | `--video-leaf-rd-lambda 200` | 105 | 24,788 | 99.15 | 9.6790 | 3.557 | 79.56 | 81.43 | 44.12 |
| `leaf200` | `--video-leaf-rd-lambda 200` | 106 | 24,417 | 97.67 | 9.6802 | 3.339 | 79.56 | 81.83 | 44.16 |
| `maxdepth6` | `--video-max-depth 6` | 106 | 25,008 | 100.03 | 9.7334 | 3.420 | 80.84 | 82.16 | 44.67 |
| `minblock4` | `--video-min-block 4` | 106 | 24,382 | 97.53 | 9.7015 | 4.077 | 80.09 | 80.79 | 44.56 |
| `noccso2` | − all five `--video-ccso*` flags | 106 | 28,313 | 113.25 | 9.3941 | 6.067 | 74.13 | 75.93 | 41.49 |
| `noccsord` | − `--video-ccso-header-rd` only | 106 | 25,555 | 102.22 | 9.7238 | 3.263 | 81.04 | 82.44 | 44.71 |
| `noccsord` | − `--video-ccso-header-rd` only | 107 | 24,969 | 99.88 | 9.7218 | 3.676 | 80.17 | 81.54 | 44.52 |
| `nocdef` | − `--video-cdef --video-cdef-rd` | 105 | 24,914 | 99.66 | 9.7213 | 4.034 | 80.35 | 80.99 | 44.35 |
| `nocdef` | − `--video-cdef --video-cdef-rd` | 106 | 24,632 | 98.53 | 9.7302 | 3.498 | 80.49 | 81.94 | 44.49 |
| `nodeblock` | − `--video-deblock` | 106 | 24,988 | 99.95 | 9.7232 | 3.561 | 80.44 | 81.79 | 44.70 |
| `nointrafull` | − `--video-intra-full` | 106 | 25,001 | 100.00 | 9.7221 | 3.879 | 80.63 | 81.29 | 44.65 |
| `notxsearch` | − `--video-tx-search --video-key-intra-txpart` | 106 | 27,937 | 111.75 | 9.7156 | 4.007 | 80.45 | 81.02 | 44.41 |
| `part05` | `--video-partition-rd-lambda 1.375e-6` | 106 | 25,644 | 102.58 | 9.7011 | 4.199 | 80.12 | 80.61 | 44.62 |
| `part2` | `--video-partition-rd-lambda 5.5e-6` | 106 | 25,683 | 102.73 | 9.7169 | 3.418 | 80.42 | 82.01 | 44.62 |
| `partd2` | `--video-partition-rd-depth 2` | 106 | 25,597 | 102.39 | 9.7301 | 3.273 | 80.44 | 82.35 | 44.73 |
| `rdoq005` | `--video-rdoq-scale 0.05` | 106 | 25,131 | 100.52 | 9.7235 | 3.271 | 80.50 | 82.33 | 44.62 |
| `rdoq02` | `--video-rdoq-scale 0.2` | 106 | 25,058 | 100.23 | 9.7111 | 3.915 | 80.63 | 81.17 | 44.66 |
| `tx48` | `--video-tx-lambda 48` | 106 | 24,953 | 99.81 | 9.7113 | 3.908 | 80.10 | 81.10 | 44.59 |
| `tx64` | `--video-tx-lambda 64` | 106 | 24,868 | 99.47 | 9.7196 | 4.268 | 80.29 | 80.63 | 44.58 |
| `win` | threshold 1.5 + budget 3000 | 109 | 25,804 | 103.22 | 9.7038 | 3.253 | 80.59 | 82.28 | 44.47 |
| `win` | threshold 1.5 + budget 3000 | 111 | 24,802 | 99.21 | 9.7342 | 3.679 | 80.97 | 81.73 | 44.63 |
| `win3250` | threshold 1.5 + budget 3250 | 110 | 25,146 | 100.58 | 9.7605 | 2.656 | 81.25 | 83.94 | 44.79 |

## 720p recipe check

Same clip family as `bench/space8k_720p` (`testdata/space8k_1440x720_60f.y4m`,
60 frames, `--fast-decode`, QP bracketing 300 kbps). At this rate the refresh budget
is only ~3% of the stream, so the 240p optimum is not expected to transfer
unchanged — it does transfer, but through the *threshold*, not the budget.

| Arm | QP | kbps | JOD | BA | SS2 | PSNR-Y | K |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 199 | 300.63 | 7.1793 | 25.293 | −9.470 | 31.40 | 45.30 |
| baseline | 200 | 294.50 | 7.1331 | 25.918 | −10.239 | 31.32 | 44.89 |
| budget 3000 only | 199 | 304.32 | 7.1794 | 25.157 | −9.377 | 31.41 | 45.33 (≈45.1 iso) |
| budget 3000 only | 200 | 298.14 | 7.1333 | 25.679 | −10.194 | 31.32 | 44.92 |
| **threshold 1.5 + budget 3000** | 199 | 301.13 | 7.1907 | 25.696 | −7.767 | 31.51 | **45.61** (≈45.6 iso) |
| threshold 1.5 + budget 3000 | 200 | 294.75 | 7.1485 | 25.536 | −8.997 | 31.42 | 45.21 |

At a matched 294.5 kbps: baseline 44.89, candidate 45.21 (**+0.32 K**). At 300.6
kbps: 45.30 vs ≈45.58 (**+0.28 K**). The budget alone is negative at 720p
(≈45.08 at 300.6), so the two knobs are not interchangeable: the threshold is
what selects the frames, and the budget must be large enough to finish them.

## Caveats

- **The K-vs-QP curve is rippled (±1.9 K).** Butteraugli moves by ±0.9 between
  adjacent QPs at this point while JOD barely moves, so differences under ~0.3 K
  are not resolvable and a single-point comparison 2% off the target rate can
  point the wrong way (see `cctx`, `drift45k`, `combo2`). Every claim here is
  either at ≤0.6% of 100 kbps or interpolated between two measured brackets.
- **One clip, one rate, one host.** The 240p clip is a slow fade with a drifting
  P tail; the refresh win is a P-tail repair, which is why it is inert on the
  keyframe-dominated part of the budget. Re-measure before quoting it on real
  detail (the 720p row above is the first check: +0.3 K, not +1.6).
- **The recipe moves the matched-rate QP, not just the flags.** The refresh
  payload costs ~0.5% more rate at a fixed QP, so the 100 kbps point moves from
  QP106 (baseline) to QP110 (candidate). Quote the pair as an operating point,
  not as "QP106 plus two flags". The flags are also `cqp`-only: `--fast-decode`
  applies the refresh (and the carried CDFs) only in `--rc cqp`, so a `crf`/`vbr`
  arm would measure a different aggregate.
- **JOD is nearly saturated** at this point (q_cvvdp 0.973 baseline → 0.975
  candidate). CVVDP alone cannot see the difference; the win is Butteraugli.
- **`--video-qm` cannot be evaluated here** (gated at `--qp 90`).  The
  `--video-wiener` / `--video-pc-wiener` / `--video-chroma-lr` arms were
  blocked by an encoder bug, not by the tools: the six §6.17.2 LR CDF rows were
  missing from the carried §8.2 image.  Fixed on `fix/lr-decode`; the 60-frame
  nightly streams now decode 60/60 (`gate_video_cdf_carry.sh`).  They still
  need a measurement pass.

## Reproduction

```sh
git worktree add <worktree-root>/k-sweep -b exp/k-sweep main
cd <worktree-root>/k-sweep
ln -s <repo>/testdata testdata
ln -s <repo>/tools/bin tools/bin
cmake --preset dev && cmake --build build-dev -j2

REF=testdata/nasa_480x240_30fps_60f.y4m
build-dev/av2enc "$REF" -o /tmp/x.ivf --video --frames 60 --cpu-threads 8 \
  --qp 110 --fast-decode --video-drift-refresh-threshold 1.5 \
  --video-drift-refresh-budget 3250
tools/bin/dav2d-macos --strict 1 -i /tmp/x.ivf -o /tmp/x.y4m --muxer yuv4mpeg2
DAV2D=tools/bin/dav2d-macos python3 tools/quality.py --ref "$REF" --ivf /tmp/x.ivf --json /tmp/q.json
python3 tools/extra_metrics.py --ref "$REF" --test /tmp/x.y4m --json /tmp/e.json
python3 tools/kpi.py --quality-json /tmp/q.json --extra-json /tmp/e.json
```
