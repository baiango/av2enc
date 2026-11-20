# Leave-one-out marginals of the nightly composition

Status: **every feature of the current nightly aggregate has a non-positive
marginal K at the operating point. There is no bottleneck to swap out.** The
largest marginals are the loop-filter stack (K −5.85 if removed), the CDF carry
(−4.51), the drift refresh (−4.46), the bottom-edge path (−4.10) and RDOQ
(−3.68). The smallest are the header/overhead levers: tile-group compact
(−0.04), compact reference signalling (−0.07), minimum tile flush (−0.11) and
CCSO header RD (−0.25).

The only sign flip found is BAWP at the *plain* `--fast-decode` control: removing
it there measures +0.29 K, but the plain control's own QP106→107 segment moves
1.65 K over 0.49 kbps, so the sign is not resolvable. At the measured operating
point (the tuned drift recipe) BAWP is −0.36 K. A reduced preset built from
that one positive point (`--fast-decode --video-bawp-off`) is 82.34 K at 100 kbps,
1.45 K below the tuned recipe; adding `--video-refs 3` to it does not change
the result (82.34 K). No swap wins.

Branch `exp/u-refstruct` merged to `main` at `3dcd151` before measuring. Build
`cmake --preset dev` (`build-dev/av2enc`, `-O1`), Apple M4, `--cpu-threads 8`,
`testdata/nasa_480x240_30fps_60f.y4m` (60 frames, 240p30). One CVVDP job at a
time under `nice -n 10`.

## 1. Method

* **Baseline**: `--fast-decode --video-drift-refresh-threshold 1.5
  --video-drift-refresh-budget 3250`. Re-verified on the merged tree: QP110 is
  25,146 B, 100.584 kbps, JOD 9.7605, BA 2.656, SS2 81.25, K 83.937 (MD5
  `c60250c0…`, unchanged), and the four-point curve gives **K = 83.795 at
  100.000 kbps**. `ΔK = K(ablation) − 83.795`; a positive ΔK means the feature
  is a net cost, a negative ΔK is its marginal value.
* **Protocol**: identical to `docs/reference-structure.md` — encode-only rate
  scans, then 2–4 quality QPs bracketing 100 kbps, linear interpolation of K vs
  kbps. Every scored stream decodes 60/60 under `dav2d --strict 1`.
* **Sign check**: the plain `--fast-decode` control was measured at QP105/106/107
  (K@100 = 82.048; QP106 is the familiar 82.155 at 100.032 kbps), and the
  near-zero ablations were repeated there.
* **No-op check**: the three overhead-only ablations (tile-group, compact
  ref signal, minimum tile flush) decode to a **byte-identical Y4M** to the
  baseline at the same QP; they change header bits and rate only.

## 2. Experiment A — leave-one-out marginals

All arms are the tuned drift recipe plus one switch. Negative ΔK = the feature
is worth that much; positive = removing it improves matched-rate quality.

| Feature removed | Switch | K@100 | ΔK | Bracket points (QP: kbps / K) | Encode cost |
|---|---|---:|---:|---|---:|
| loop filters (CDEF+CCSO+deblock) | `--video-loop-filters-off` | 77.948 | **−5.85** | 115: 100.256 / 77.944 · 116: 97.708 / 77.977 | −42% |
| CDF chain | `--video-cdf-chain-off` | 79.287 | **−4.51** | 114: 101.136 / 79.312 · 115: 98.596 / 79.257 | −24% |
| drift refresh | `--video-drift-refresh-off` | 79.331 | **−4.46** | 102: 101.040 / 79.568 · 103: 99.440 / 79.203 | −37% |
| edge inter + edge mask | `--video-edge-inter-off --video-edge-mask-off` | 79.696 | **−4.10** | 129: 101.912 / 79.722 · 130: 99.992 / 79.696 | −2% |
| RDOQ | `--video-rdoq-off` | 80.119 | **−3.68** | 112: 100.204 / 80.039 · 113: 99.420 / 80.348 | −13% |
| key/first-P QP offset | `--video-key-qp-offset-off` | 80.535 | **−3.26** | 86: 101.372 / 80.490 · 88: 101.888 / 80.563 · 90: 97.964 / 80.602 | −12% |
| partition probe | `--video-partition-rd-off` | 82.287 | **−1.51** | 115: 100.968 / 82.159 · 116: 98.752 / 82.453 | −5% |
| edge mask only | `--video-edge-mask-off` | 82.801 | **−1.00** | 111: 100.728 / 82.593 · 112: 99.216 / 83.024 | −4% |
| TX search | `--video-tx-search-off` | 82.810 | **−0.99** | 116: 100.504 / 82.992 · 117: 99.116 / 82.490 | −27% |
| deblock | `--video-deblock-off` | 83.269 | **−0.53** | 111: 100.176 / 83.184 · 112: 99.192 / 83.657 | −14% |
| BAWP | `--video-bawp-off` | 83.441 | **−0.36** | 111: 102.068 / 83.210 · 112: 99.472 / 83.499 | −9% |
| CCSO header RD | `--video-ccso-header-off` | 83.546 | **−0.25** | 112: 100.276 / 83.654 · 113: 99.728 / 83.438 | −8% |
| minimum tile flush | `--video-tile-flush-off` | 83.682 | **−0.11** | 111: 100.228 / 83.747 · 112: 99.108 / 83.426 | −2% |
| compact ref signalling | `--video-ref-signal-off` | 83.728 | **−0.07** | 111: 100.068 / 83.747 · 112: 98.920 / 83.426 | +1% |
| tile-group compact | `--video-tile-group-off` | 83.757 | **−0.04** | 110: 100.736 / 83.937 · 111: 99.960 / 83.747 | +11% |

Notes:

* `--video-loop-filters-off` is the combined deblock + CDEF + CCSO switch (the
  merged switch sets all four toggles). `--video-deblock-off` isolates deblock
  at −0.53; the CDEF+CCSO block is therefore worth about −5.3 on its own.
* `--video-edge-inter-off` alone is rejected on the tuned recipe:

  ```text
  av2enc: --video-edge-mask needs --video-edge-inter: only that path reconstructs and reference-pads the hidden bottom rows
  ```

  so the edge row is the combined edge-inter + edge-mask removal, and the
  mask-only row isolates the mask.
* The three recon-identical rows (tile-group, ref signal, tile flush) were
  verified at the same QP: decoded Y4M MD5 equal to the baseline
  (`de32b6e4…` at QP110 for tile-group, `d0333e3f…` at QP111 for ref signal and
  tile flush). Their ΔK is the rate cost of the dropped overhead bits.
* Encode cost is the median of 3 encodes at the arm's lower bracket QP against
  the baseline at the same QP. The host's repeat-to-repeat spread is up to
  ±15%, so the two +11%/+1% rows are noise; the large negatives are real.

### Detail appendix (all scored points)

| Arm | QP | Bytes | kbps | JOD | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|---:|---:|
| loop-filters off | 115 | 25,064 | 100.256 | 9.4737 | 4.636 | 74.52 | 77.944 |
| loop-filters off | 116 | 24,427 | 97.708 | 9.5108 | 4.657 | 73.77 | 77.977 |
| CDF chain off | 114 | 25,284 | 101.136 | 9.7045 | 5.124 | 79.34 | 79.312 |
| CDF chain off | 115 | 24,649 | 98.596 | 9.6962 | 5.050 | 78.75 | 79.257 |
| drift off | 102 | 25,260 | 101.040 | 9.6918 | 4.921 | 79.84 | 79.568 |
| drift off | 103 | 24,860 | 99.440 | 9.6824 | 5.154 | 79.56 | 79.203 |
| edge off | 129 | 25,478 | 101.912 | 9.6124 | 4.015 | 75.81 | 79.722 |
| edge off | 130 | 24,998 | 99.992 | 9.6046 | 3.973 | 75.51 | 79.696 |
| RDOQ off | 112 | 25,051 | 100.204 | 9.6566 | 4.151 | 77.61 | 80.039 |
| RDOQ off | 113 | 24,855 | 99.420 | 9.6681 | 4.069 | 78.38 | 80.348 |
| key QP offset off | 86 | 25,343 | 101.372 | 9.6641 | 3.920 | 77.98 | 80.490 |
| key QP offset off | 88 | 25,472 | 101.888 | 9.6755 | 3.946 | 78.33 | 80.563 |
| key QP offset off | 90 | 24,491 | 97.964 | 9.6959 | 4.058 | 78.98 | 80.602 |
| partition off | 115 | 25,242 | 100.968 | 9.7083 | 3.305 | 80.34 | 82.159 |
| partition off | 116 | 24,688 | 98.752 | 9.6971 | 3.119 | 80.27 | 82.453 |
| edge mask off | 111 | 25,182 | 100.728 | 9.7351 | 3.191 | 80.82 | 82.593 |
| edge mask off | 112 | 24,804 | 99.216 | 9.7453 | 3.015 | 80.97 | 83.024 |
| TX search off | 116 | 25,126 | 100.504 | 9.7070 | 2.972 | 81.38 | 82.992 |
| TX search off | 117 | 24,779 | 99.116 | 9.6911 | 3.118 | 80.66 | 82.490 |
| deblock off | 111 | 25,044 | 100.176 | 9.7384 | 2.928 | 81.05 | 83.184 |
| deblock off | 112 | 24,798 | 99.192 | 9.7509 | 2.753 | 81.19 | 83.657 |
| BAWP off | 111 | 25,517 | 102.068 | 9.7161 | 2.886 | 81.33 | 83.210 |
| BAWP off | 112 | 24,868 | 99.472 | 9.7235 | 2.724 | 80.67 | 83.499 |
| CCSO header off | 112 | 25,069 | 100.276 | 9.7356 | 2.712 | 81.08 | 83.654 |
| CCSO header off | 113 | 24,932 | 99.728 | 9.7551 | 2.888 | 81.55 | 83.438 |
| tile flush off | 111 | 25,057 | 100.228 | 9.7513 | 2.731 | 81.42 | 83.747 |
| tile flush off | 112 | 24,777 | 99.108 | 9.7363 | 2.816 | 81.10 | 83.426 |
| ref signal off | 111 | 25,017 | 100.068 | 9.7513 | 2.731 | 81.42 | 83.747 |
| ref signal off | 112 | 24,730 | 98.920 | 9.7363 | 2.816 | 81.10 | 83.426 |
| tile-group off | 110 | 25,184 | 100.736 | 9.7605 | 2.656 | 81.25 | 83.937 |
| tile-group off | 111 | 24,990 | 99.960 | 9.7513 | 2.731 | 81.42 | 83.747 |

### Sign check at the plain `--fast-decode` point

Plain control curve: QP105 101.368 / 81.806, QP106 100.032 / 82.155,
QP107 99.540 / 80.508 → **K@100 = 82.048**.

| Ablation at plain nightly | QP(s) | kbps | K | K@100 | ΔK vs plain control |
|---|---:|---:|---:|---:|---:|
| `--video-bawp-off` | 105/106/107/108 | 102.444/100.204/97.948/97.684 | 82.412/82.357/82.110/82.034 | 82.335 | **+0.29** |
| `--video-ref-signal-off` | 106 | 100.272 | 82.155 | — | ≈0 (recon identical) |
| `--video-tile-flush-off` | 106 | 100.448 | 82.155 | — | ≈0 (recon identical) |
| `--video-tile-group-off` | 106 | 100.184 | 82.155 | — | ≈0 (recon identical) |
| `--video-ccso-header-off` | 107/108 | 99.876/98.456 | 81.539/81.908 | 81.507 | −0.54 |
| `--video-deblock-off` | 106 | 99.952 | 81.787 | — | −0.10 |
| `--video-edge-mask-off` | 106 | 100.512 | 81.496 | — | −0.53 |
| `--video-drift-refresh-off` | 102/103 | 101.040/99.440 | 79.568/79.203 | 79.331 | **−2.72** |

BAWP is the only positive sign, and it is not trustworthy: at QP106 the pair
is 82.357 (off) vs 82.155 (control) for +0.17 kbps; correcting by the control's
local slope gives anything from +0.05 to +0.58 because the control's
QP106→107 segment drops 1.65 K over 0.49 kbps. The four-point re-bracket gives
+0.29 but spans that same ripple. At the actual operating point BAWP is
−0.36 K. Treated as no actionable positive marginal.

## 3. Verdict — where the value is

The nightly composition is **locally optimal at this operating point**. There
is no feature whose removal frees quality at matched rate; the rank order of
value is:

```text
loop filters 5.85 > CDF carry 4.51 > drift refresh 4.46 > edge path 4.10
> RDOQ 3.68 > key-QP offset 3.26 > partition probe 1.51 > edge mask 1.00
> TX search 0.99 > deblock 0.53 > BAWP 0.36 > CCSO header 0.25
> tile flush 0.11 > ref signal 0.07 > tile group 0.04
```

The reference/compound tools measured as losses in
`docs/reference-structure.md` were locked behind four of the largest-value
features (CDF carry, drift refresh, edge path, BAWP, 13.4 K of measured
marginal in total). There is no budget to buy them: the only reference tool that beat its control
when its blocker was removed was ref-MVs, by +0.09 K, against a control 4.7 K
below the bar; the only near-bar arm remains refs 3 on top of the drift recipe.

## 4. Experiment B — spending the budget

### 4.1 Reduced preset

**None.** No feature has a positive marginal at the tuned-drift operating
point, so the reduced preset is the empty set. The one positive measurement
(BAWP at the plain nightly point) was built and measured as a bounded check:

| Preset | QP(s) | kbps | K | K@100 | vs tuned recipe |
|---|---|---:|---:|---:|---:|
| plain nightly − BAWP (`--video-bawp-off`) | 105/106/107/108 | 102.44/100.20/97.95/97.68 | 82.41/82.36/82.11/82.03 | **82.335** | **−1.46** |
| plain nightly − BAWP + refs 3 (`--video-bawp-off --video-ref-signal-off --video-refs 3`) | 106/107 | 100.552/99.068 | 82.183/82.614 | **82.343** | **−1.45** |
| tuned drift recipe (baseline) | 109–112 | — | — | **83.795** | — |

The refs-3 add-back buys +0.01 over the BAWP-off preset, i.e. nothing. Even
granting the entire ambiguous BAWP positive, the resulting preset is 1.45 K
below the current best.

### 4.2 Reverse direction — only the blocking feature removed

These arms remove exactly the features that locked each tool out and keep the
rest of the tuned aggregate, so the residual is the tool's own value plus the
cost of the blocker.

| Tool | Blockers removed | Control K@100 | Tool K@100 | Tool Δ vs control | Blocker cost (ΔK) |
|---|---|---:|---:|---:|---:|
| `--video-refs 2` | compact ref signal only | 83.728 | 81.377 | **−2.35** | −0.07 |
| `--video-refs 3` | compact ref signal only | 83.728 | 81.191 | **−2.54** | −0.07 |
| `--video-refs 4` | compact ref signal only | 83.728 | 80.978 | **−2.75** | −0.07 |
| `--video-compound` | BAWP + drift + CDF chain | 80.147 | 79.977 | **−0.17** | −0.36/−4.46/−4.51 (combined −3.65) |
| `--video-ref-mvs` | edge inter + edge mask | 79.082 | 79.168 | **+0.09** | −4.10/−1.00 (combined −4.71) |
| `--video-refs 2 --video-ref-mvs` | edge inter + edge mask | 79.082 | 76.870 | **−2.21** | −4.10/−1.00 |

Read this table as two separate costs:

* **refs 2/3/4**: the blocker is effectively free (compact ref signalling is
  worth 0.07 K), so this is a clean tool measurement — extra references cost
  2.35–2.75 K against the blocker-only control (2.42–2.82 K against the bar).
* **compound**: no single-blocker arm exists (BAWP and drift are validated
  out; the CDF chain decodes 3/60 with compound), so all three must go
  together. The combined blocker cost measured as the control is 3.65 K, and
  the tool itself is −0.17 K. Both the blocker and the tool point the same
  way.
* **ref-MVs**: the tool is +0.09 K over its control, but the blocker removal
  (edge path) costs 4.7 K. This is the one case where the tool is not the
  problem; the blocker is simply worth more than the tool.

The reverse direction therefore does not rescue any reference tool. The
best reference arm remains `--video-refs 3` on the tuned drift recipe at
K 83.77, a −0.03 tie inside the ripple, and it requires the same
`--video-ref-signal-off` that costs 0.07 K.

## 5. Cost and verification

* 33 quality points for Experiment A (encode + strict decode + CVVDP +
  Butteraugli + SSIMULACRA2), 14 at the plain-nightly control, 2 for the
  bounded preset check, plus ~90 encode-only rate scans and 24 three-encode
  timing runs.
* Every scored stream decoded **60/60** under `dav2d --strict 1`. No ablation
  failed to encode.
* The three overhead ablations are reconstruction no-ops, verified by decoded
  Y4M MD5 equality against the baseline at the same QP.
* `--video-edge-inter-off` cannot be applied alone (`--video-edge-mask needs
  --video-edge-inter`, quoted above), so the edge row is the pair; the
  mask-only row isolates the mask.
* Baseline pins are byte-identical on the merged tree (nightly QP106 25,008 B
  `0639b436…`; tuned drift QP110 25,146 B `c60250c0…`), and
  `tools/gate_video_ref_signalling.sh` and
  `src/pipeline/video/video_drift_refresh_gate.sh` both pass on `build-dev`.

## 6. Reproduction

```sh
# baseline
build-dev/av2enc testdata/nasa_480x240_30fps_60f.y4m -o /tmp/b.ivf \
  --video --frames 60 --cpu-threads 8 --qp 110 --fast-decode \
  --video-drift-refresh-threshold 1.5 --video-drift-refresh-budget 3250
# an ablation (example: loop-filter stack)
build-dev/av2enc testdata/nasa_480x240_30fps_60f.y4m -o /tmp/a.ivf \
  --video --frames 60 --cpu-threads 8 --qp 115 --fast-decode \
  --video-drift-refresh-threshold 1.5 --video-drift-refresh-budget 3250 \
  --video-loop-filters-off
```

## 7. Limits

* One clip, one resolution, one operating point. The marginals are properties
  of this 240p30 point, not of the tools in general; the ranking could move at
  another QP or resolution.
* The QP ripple is of the same order as the small marginals: the plain control
  moves 1.65 K between QP106 and QP107, and the key-QP-off arm is flat within
  0.11 K over QPs 86–90. The rows below 0.5 K are not distinguishable from the
  ripple; the negative sign of every row is, however, consistent.
* Encode timings are dev-build `-O1` on a noisy M4; only the large deltas
  (drift, loop filters, TX search, CDF) are stable.
