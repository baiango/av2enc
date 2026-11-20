# Reference-structure tools at the 240p30 / 100 kbps point

Status: **negative. No reference-structure tool, alone or on top of the tuned
drift-refresh recipe, beats the current best of K = 83.80 at 100.0 kbps.** The
closest arm is `--video-refs 3` plus the tuned drift refresh at K = 83.77
(−0.03), inside a QP ripple that is larger than the difference; its bracket
points disagree in sign (+0.13 at 100.44 kbps, −0.44 at 98.64 kbps). Every
other arm is 0.2–11 K below.

Branch `exp/u-refstruct`, based on `main` at `a7ecfa2`. Measurement build
`cmake --preset dev` (CPU reference, `build-dev/av2enc`, `-O1`), Apple M4,
`--cpu-threads 8`, `testdata/nasa_480x240_30fps_60f.y4m` (60 frames, 240p30).
All quality points use one CVVDP job at a time under `nice -n 10`.

## 1. What was measured

Exact per-point pipeline:

```sh
REF=testdata/nasa_480x240_30fps_60f.y4m
BIN=build-dev/av2enc
"$BIN" "$REF" -o /tmp/x.ivf --video --frames 60 --cpu-threads 8 --qp Q <flags>
tools/bin/dav2d-macos --strict 1 -i /tmp/x.ivf -o /tmp/x.y4m --muxer yuv4mpeg2
DAV2D=tools/bin/dav2d-macos python3 tools/quality.py --ref "$REF" --ivf /tmp/x.ivf --json /tmp/q.json
python3 tools/extra_metrics.py --ref "$REF" --test /tmp/x.y4m --json /tmp/e.json
python3 tools/kpi.py --quality-json /tmp/q.json --extra-json /tmp/e.json
```

`kbps = IVF bytes * 8 / 2 s / 1000` (the IVF container, as in the baselines).
Each arm was rate-scanned with encode-only runs, then 2–4 QPs were quality
measured around 100.0 kbps. `K@100` is a linear interpolation; bracketing
points are listed because the QP ripple on this clip is larger than several of
the deltas.

`K` is `tools/kpi.py`: `100 * (0.50*JOD/10 + 0.25/(1+BA/3) + 0.25*(SS2+50)/150)`.

### Baselines reproduced (exact)

| Stream | QP | Bytes | kbps | JOD | BA | SS2 | K | Decode |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `--fast-decode` | 106 | 25,008 | 100.032 | 9.7334 | 3.4202 | 80.84 | **82.155** | 60/60 |
| tuned drift | 110 | 25,146 | 100.584 | 9.7605 | 2.6562 | 81.25 | **83.937** | 60/60 |
| tuned drift | 111 | 24,950 | 99.800 | 9.7513 | 2.7310 | 81.42 | **83.747** | 60/60 |

The tuned recipe is `--fast-decode --video-drift-refresh-threshold 1.5
--video-drift-refresh-budget 3250`. Two extra bar points were measured to
characterise the curve:

| QP | kbps | K |
|---:|---:|---:|
| 109 | 104.020 | 83.214 |
| 110 | 100.584 | 83.937 |
| 111 | 99.800 | 83.747 |
| 112 | 98.672 | 83.426 |

At 100.000 kbps the piecewise-linear bar is **K = 83.795** (the quoted 83.80).

Representative measured streams (the nightly pin, the tuned drift stream,
refs 3 + drift, compound, ref-mvs and short refresh) also pass
`tools/validate_dav2d.sh` with exit 0. The compound + carried-CDF stream does
not (3/60 pictures), which is the one decode failure recorded below.

## 2. Constraint log (verbatim)

Every rejection below was reproduced with `build-dev/av2enc` on this branch.

### Reference signalling

`--fast-decode` expands `--video-ref-signal-compact`, which pins `NumRefFrames = 1`.
Any `--video-refs > 1` is therefore rejected:

```text
av2enc: --video-ref-signal-compact needs --video-refs 1: the §7.7 derived map must hold exactly the one live slot
```

**Switch found: `--video-ref-signal-off`.** It suppresses only the nightly
expansion element; `--video-refs 2/3/4` then encode and decode. The cost is the
larger explicit reference map (the compact form saves 61 B / 0.245% over 60
frames, see `docs/video-ref-signalling.md`), which is part of every multi-ref
number here.

### Compound

`--video-compound` requires two references and is rejected by two nightly
elements:

```text
av2enc: --video-compound needs --video-refs 2 (the (0,1) pair is implied by NumTotalRefs == 2)
```

```text
av2enc: --video-bawp is gated on the plain single-prediction P chain: it cannot be combined with --b-frames, --video-tlayers, a frame-type wrapper, --video-compound or --video-gme/--video-warp (and --video-inter-residual needs --video-bawp-carrier)
```

With BAWP suppressed (`--video-bawp-off`) the next rejection is the drift
refresh:

```text
av2enc: error: --video-drift-refresh is wired for single-reference inter prediction
```

There is no drift-refresh switch that keeps it on; the compound arms therefore
run with `--video-bawp-off --video-drift-refresh-off`.

**New decoder failure (not a named rejection): `--video-compound` + the carried
CDF chain.** The nightly aggregate uses `--avg-cdf --adaptive-cdf
--cdf-continuity`. A 60-frame stream with `--video-refs 2 --video-compound` and
the carried chain encodes, but the pinned decoder outputs **3 of 60 pictures**
and then prints `Error decoding frame: Invalid argument` on every remaining
frame (the process exits 0, so only the picture count catches it). Isolation:

| Configuration | Decoded |
|---|---:|
| `--video-refs 2 --video-compound` | 60/60 |
| `+ --avg-cdf --adaptive-cdf` | 60/60 |
| `+ --cdf-continuity` | **3/60** |
| `--video-refs 2 --cdf-continuity` (no compound) | 60/60 |
| `--video-compound --cdf-continuity` without avg/adaptive | 60/60 |

So it is the combination of per-frame CDF adaptation **and** the cross-frame
carry with compound syntax. This is a latent encoder/decoder desync, separate
from the questions in this document. All compound arms here were therefore
measured with `--video-cdf-chain-off` (per-frame §9.3 derivation), against a
control with the same flag. That control is already 3.6 K below the bar.

### OPFL / refine-MV

```text
av2enc: --video-opfl/--video-refine-mv need --video-compound-near-near: the temporal refinement tools only run on a NEAR_NEARMV compound block
```

Adding the carrier:

```text
av2enc: --video-compound-near-near is gated on the two-reference B-frame path (one past and one future reference): pass --b-frames
```

These two tools cannot be measured on a P chain at all. B frames are another
agent's scope; no stream was produced.

### Long-term

`--video-long-term` contradicts almost every part of the nightly aggregate, in
this order: `--video-edge-inter`, `--video-deblock`, `--video-tile-group-compact`,
`--video-tx-search`, `--video-rdoq`, `--video-bawp`, the loop filters, and
`--video-partition-rd`. Representative messages:

```text
av2enc: --video-edge-inter is measured on the low-delay P chain; drop the B-frame / reorder / temporal-layer mode
av2enc: --video-deblock does not compose with --video-long-term/--video-olk, --video-leading, --video-sef, --video-switch or --video-ras: those header writers do not signal deblocking_filter_params()
av2enc: --video-tx-search does not compose with the frame-type wrapper modes (their header writers do not signal tx_mode_select)
av2enc: --video-rdoq is wired on the single-tile plain P chain only
av2enc: --video-partition-rd does not compose with the B/temporal-layer/random-access/TIP frame schedules
```

A stripped composition (edge/mask, deblock, loop filters, tx search, RDOQ,
partition RD, BAWP, CDF chain, key-QP offset and tile-group compact all off)
passes validation but then hits the hard format limit:

```text
av2enc: error: --video-long-term codes OrderHint with OrderHintBits = 4, so it supports at most 16 coded frames (requested 60)
```

So long-term **cannot be measured at the 60-frame point**. On a 15-frame
prefix it was checked as an auxiliary: with `--video-cdf-per-frame-qp` added it
decodes 15/15 (without that flag it decodes 2/15 — a second latent header/CDF
interaction), and the encoded bytes are **identical to the same stripped
control at every QP from 106 to 118**, with byte-identical decoded output. On
this flat chain the flag is a header-only LongTermId tag; it changes no
decision and no picture. It is inert, not a lever.

### Short refresh

```text
av2enc: error: reference-management tools are not wired with the average-CDF/CDF-continuity chain
```

With `--video-cdf-chain-off`, the compact reference signal then owns the same
field:

```text
av2enc: error: --video-ref-signal-compact owns refresh_frame_flags; drop --video-short-refresh
```

Both are avoided with `--video-ref-signal-off --video-cdf-chain-off`. In that
configuration the short-refresh stream is **the same size as the control at
every QP tested and decodes to a byte-identical Y4M** (checked at QP114 and
QP115). `enable_short_refresh_frame_flags` at `NumRefFrames = 1` saves at most
one bit per frame, and the header alignment absorbs it.

### Ref-MVs

The known edge-inter lock is accurate, and it has a second consequence:

```text
av2enc: --video-edge-inter cannot carry the temporal MV field: the bottom SVB rows have no coded MI grid
```

With `--video-edge-inter-off`:

```text
av2enc: --video-edge-mask needs --video-edge-inter: only that path reconstructs and reference-pads the hidden bottom rows
```

Both `--video-edge-inter-off --video-edge-mask-off` are required. That control
is already 4.7 K below the bar (the edge path is a nightly win). An extra
reference (`--video-refs 2 --video-ref-mvs`) composes and also decodes 60/60.

### Same-reference compound

Same-ref compound is a `reference-management tool`, so it inherits the
`--avg-cdf/--adaptive-cdf/--cdf-continuity` rejection on top of the compound
constraints; it needs `--video-cdf-chain-off` as well.

## 3. Arm results

Bracketing points are `QP: kbps / K`. `Δbar` is against the drift-refresh bar at
that same rate, so it removes the ripple from the rate axis.

### 3.1 Reference count, no compound (single prediction)

| Arm | Bracket points (QP: kbps / K) | K@100 | Δbar@100 | Decode | Enc. cost |
|---|---|---|---:|---:|---|---|
| `--video-refs 2` | 106: 100.728 / 82.008 · 107: 99.216 / 80.697 | 81.377 | **−2.42** | 60/60 | 6.64 s (+14%) |
| `--video-refs 3` | 106: 101.628 / 81.350 · 107: 99.992 / 81.190 | 81.191 | **−2.60** | 60/60 | 6.57 s (+13%) |
| `--video-refs 4` | 106: 101.412 / 82.228 · 107: 100.104 / 80.911 · 108: 98.560 / 81.916 | 80.978 | **−2.82** | 60/60 | 6.71 s (+15%) |

Full bracket detail:

| Arm | QP | Bytes | kbps | JOD | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|---:|---:|
| refs 2 | 106 | 25,182 | 100.728 | 9.7323 | 3.463 | 80.45 | 82.008 |
| refs 2 | 107 | 24,804 | 99.216 | 9.7080 | 4.187 | 80.32 | 80.697 |
| refs 3 | 106 | 25,407 | 101.628 | 9.7062 | 3.777 | 80.51 | 81.350 |
| refs 3 | 107 | 24,998 | 99.992 | 9.7158 | 3.900 | 80.45 | 81.190 |
| refs 4 | 106 | 25,353 | 101.412 | 9.7306 | 3.369 | 80.80 | 82.228 |
| refs 4 | 107 | 25,026 | 100.104 | 9.6902 | 3.928 | 79.80 | 80.911 |
| refs 4 | 108 | 24,640 | 98.560 | 9.7231 | 3.531 | 80.90 | 81.916 |

Multi-reference prediction is not neutral: every count loses 2.4–2.8 K at
matched rate, and the loss grows with the count. The extra reference does not
buy better leaf decisions; it changes the reference list enough to disturb the
skip/residual balance the QP-ripple already sits on. Refs 4 is also visibly
non-monotone (QP106 is the best point of the arm at a higher rate).

### 3.2 Compound family (refs 2, BAWP and drift off, CDF chain off)

The control is `--video-ref-signal-off --video-refs 2 --video-bawp-off
--video-drift-refresh-off --video-cdf-chain-off` (no compound).

| Arm | QP | Bytes | kbps | JOD | BA | SS2 | K | Δ vs control (same QP) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 116 | 25,038 | 100.152 | 9.6862 | 4.352 | 79.20 | 80.165 | — |
| control | 117 | 24,691 | 98.764 | 9.6755 | 4.397 | 78.91 | 80.003 | — |
| `--video-compound` | 116 | 25,073 | 100.292 | 9.6779 | 4.472 | 79.38 | 79.990 | −0.18 |
| `--video-compound` | 117 | 24,724 | 98.896 | 9.6759 | 4.445 | 78.85 | 79.929 | −0.07 |
| `+ --video-compound-diffwtd` | 115 | 25,538 | 102.152 | 9.6591 | 5.210 | 78.10 | 78.781 | — |
| `+ --video-compound-diffwtd` | 116 | 25,063 | 100.252 | 9.6799 | 4.544 | 79.19 | 79.872 | −0.29 |
| `+ --video-compound-diffwtd` | 117 | 24,895 | 99.580 | 9.6777 | 4.543 | 78.86 | 79.807 | −0.20 |
| `+ --video-compound-wedge` | 119 | 25,160 | 100.640 | 9.5681 | 6.030 | 76.21 | 77.181 | ≈−3.0* |
| `+ --video-compound-wedge` | 120 | 24,749 | 98.996 | 9.6118 | 4.841 | 76.13 | 78.646 | ≈−1.4* |
| `+ --video-same-ref-compound` | 116 | 25,040 | 100.160 | 9.3921 | 8.848 | 67.64 | 72.897 | −7.27 |
| `+ --video-same-ref-compound` | 117 | 24,791 | 99.164 | 9.3212 | 9.733 | 65.81 | 71.798 | −7.21 |

\* The control was not quality-measured at QP119/120. The delta uses the
control's K at the same rate: 80.17 at the wedge's 100.64 kbps point and 80.03
at its 99.00 kbps point (interpolated from QP116/117: 80.165 at 100.152 kbps →
80.003 at 98.764 kbps). The wedge's distortion is unambiguous (BA 6.0, SS2 76.2).

Iso-rate: control 80.147, compound 79.977, diffwtd 79.848, wedge 77.751,
same-ref 72.721. **The compound tool itself is ≤ 0 at matched rate even against
its own handicapped control; the control is already 3.65 K below the bar, so
the family starts 3.6 K behind before measuring.** The masked variants are worse
than plain AVERAGE, and same-reference compound destroys quality.

### 3.3 Short refresh

| Arm | QP | Bytes | kbps | JOD | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|---:|---:|
| control (`--video-cdf-chain-off`) | 114 | 25,334 | 101.336 | 9.7045 | 5.124 | 79.34 | 79.312 |
| control | 115 | 24,699 | 98.796 | 9.6962 | 5.050 | 78.75 | 79.257 |
| `--video-short-refresh` | 114 | 25,334 | 101.336 | 9.7045 | 5.124 | 79.34 | 79.312 |
| `--video-short-refresh` | 115 | 24,699 | 98.796 | 9.6962 | 5.050 | 78.75 | 79.257 |

Iso-rate 79.283 (−4.51 vs bar). The arm is **byte-identical in size and
decoded output** to the control at the same QP; the 4.5 K deficit is the price
of dropping the carried-CDF chain, not of short refresh. The tool itself is a
no-op at `NumRefFrames = 1`.

### 3.4 Ref-MVs

The control has `--video-edge-inter-off --video-edge-mask-off`.

| Arm | QP | Bytes | kbps | JOD | BA | SS2 | K | Δ vs control |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 128 | 25,138 | 100.552 | 9.6178 | 4.309 | 74.51 | 79.102 | — |
| control | 129 | 24,141 | 96.564 | 9.6061 | 4.354 | 74.35 | 78.954 | — |
| `--video-ref-mvs` | 128 | 25,131 | 100.524 | 9.6178 | 4.303 | 74.52 | 79.111 | +0.01 |
| `--video-ref-mvs` | 129 | 24,639 | 98.556 | 9.6112 | 4.224 | 75.31 | 79.324 | +0.37 |
| `--video-refs 2 --video-ref-mvs` | 128 | 25,403 | 101.612 | 9.5779 | 4.779 | 73.53 | 78.119 | −0.98 |
| `--video-refs 2 --video-ref-mvs` | 129 | 24,539 | 98.156 | 9.4129 | 5.908 | 69.74 | 75.440 | −3.51 |

Iso-rate: control 79.082, ref-mvs 79.168 (+0.09), refs2+ref-mvs 76.870
(−2.21 vs control, −6.93 vs bar). Temporal MVs are a small positive over their
own control at one reference and a large negative with two. Neither is close to
the bar: removing edge inter + mask costs 4.7 K before the tool is even
switched on.

### 3.5 Drift-refresh stack (the only arms near the bar)

Recipe unchanged from the baseline: `--video-drift-refresh-threshold 1.5
--video-drift-refresh-budget 3250`, with `--video-ref-signal-off` to make the
reference count legal.

| Arm | Bracket points (QP: kbps / K) | K@100 | Δbar@100 | Decode | Enc. cost |
|---|---|---|---:|---:|---|---|
| refs 2 + drift | 111: 101.544 / 82.971 · 112: 99.500 / 83.751 | 83.560 | **−0.24** | 60/60 | 7.37 s (+38%) |
| refs 3 + drift | 111: 101.204 / 83.582 · 112: 100.444 / 84.028 · 113: 98.644 / 82.979 · 114: 97.592 / 84.093 | **83.769** | **−0.03** | 60/60 | 8.05 s (+66%) |
| refs 4 + drift | 112: 100.400 / 83.494 · 113: 99.452 / 83.646 | 83.558 | **−0.24** | 60/60 | 8.35 s (+72%) |

Full bracket detail:

| Arm | QP | Bytes | kbps | JOD | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|---:|---:|
| refs 2 + drift | 111 | 25,386 | 101.544 | 9.7384 | 3.069 | 81.53 | 82.971 |
| refs 2 + drift | 112 | 24,875 | 99.500 | 9.7580 | 2.748 | 81.47 | 83.751 |
| refs 3 + drift | 111 | 25,301 | 101.204 | 9.7635 | 2.850 | 81.67 | 83.582 |
| refs 3 + drift | 112 | 25,111 | 100.444 | 9.7590 | 2.642 | 81.64 | 84.028 |
| refs 3 + drift | 113 | 24,661 | 98.644 | 9.7503 | 3.079 | 81.34 | 82.979 |
| refs 3 + drift | 114 | 24,398 | 97.592 | 9.7597 | 2.625 | 81.76 | 84.093 |
| refs 4 + drift | 112 | 25,100 | 100.400 | 9.7622 | 2.869 | 81.43 | 83.494 |
| refs 4 + drift | 113 | 24,863 | 99.452 | 9.7613 | 2.813 | 81.63 | 83.646 |

The refs-3 curve is the decisive counter-example to a clean win: its four
measured points alternate sign against the bar (−0.23, +0.13, −0.44, +0.97)
with no monotone trend. A 2-point interpolation puts it at 83.77 (−0.03); the
single point nearest 100 kbps is +0.13. That is a tie inside the ripple, not a
reproducible gain. Refs 2 and 4 are consistently ~0.24 K below.

Reconstructed output check: `--video-recon` equals the `dav2d --strict 1` Y4M
byte for byte for refs3+drift QP112 (10,368,399 B, MD5 `a1c31e31…`), for the
compound arm QP116 (`92763046…`) and for ref-mvs QP128 (`d3618250…`).

## 4. Verdict

**Nothing beats 83.80.** Re-confirmed with the AVM oracle on `exp/luma-multiref`
(`docs/video-multiref-p.md`): refs2/refs3 interpolate to **81.377 / 81.191** at
100.0 kbps and reproduce this document's QP107 points exactly; the nightly
matched K is 82.048 and the eff-perc drift recipe loses 0.26-0.32 K when
references are added.

* Multi-reference single prediction loses **2.4–2.8 K**. The reference tools
  are not being held back by a missing tune; at this point extra references
  change the decisions for the worse.
* Compound prediction is ≤ 0 against its own control and 3.6–3.9 K below the
  bar. The control's deficit is structural: BAWP, the drift refresh and the
  carried-CDF chain are all incompatible with compound (the last one is a
  decoder desync, not a syntax rule).
* Wedge compound (−6.0 K) and same-reference compound (−11.1 K) are
  catastrophic. Ref-MVs is a small positive with one reference (+0.09) and a
  large negative with two (−2.21), but its control is 4.7 K down because edge
  inter and the edge mask must be dropped.
* Short refresh is a no-op at this reference count, and its control loses
  4.5 K because the carried CDF chain must be dropped.
* Long-term cannot be coded at 60 frames at all (OrderHintBits = 4 caps it at
  16), and on a 15-frame prefix it is a header-only no-op.
* The only arm that reaches the bar is `--video-refs 3` on top of the tuned
  drift recipe, at **83.77 vs 83.80** — a tie inside the ripple. Its cost is
  also the highest of the plausible arms (+66% encode time).

The family-level conclusion is consistent with `docs/negative-results.md`:
reference *structure* is not the missing lever at 240p30 / 100 kbps. The flat
P chain plus the drift refresh already spends the available rate on the
pictures that matter; adding reference slots or compound syntax moves the leaf
decisions, and the leaf allocator is what limits the trough.

## 5. What it cost

* **Measurement time**: ~40 quality points (encode + strict decode + CVVDP +
  Butteraugli + SSIMULACRA2), each 30–60 s, plus ~90 encode-only rate scans and
  3× encode-cost repeats. One CVVDP job at a time.
* **Encode cost** (median of 3, dev `-O1`, same-QP nightly reference; the host's
  repeat-to-repeat spread is up to ±40%, so treat sub-15% deltas as noise):

| Arm | QP | Arm | Nightly @QP | Delta |
|---|---:|---:|---:|---:|
| refs 2 | 106 | 6.64 s | 5.83 s | +14% |
| refs 3 | 106 | 6.57 s | 5.83 s | +13% |
| refs 4 | 106 | 6.71 s | 5.83 s | +15% |
| refs 2 + drift | 111 | 7.37 s | 5.33 s | +38% |
| refs 3 + drift | 112 | 8.05 s | 4.85 s | +66% |
| refs 4 + drift | 112 | 8.35 s | 4.85 s | +72% |
| compound | 116 | 5.14 s | 4.63 s | +11% |
| diffwtd | 116 | 5.93 s | 4.63 s | +28% |
| wedge | 119 | 8.04 s | 4.55 s | +77% |
| same-ref | 116 | 4.94 s | 4.63 s | +7% |
| CDF-off control | 114 | 4.38 s | 5.07 s | −14% |
| short refresh | 114 | 4.83 s | 5.07 s | −5% |
| ref-mvs | 128 | 5.26 s | 4.45 s | +18% |
| refs 2 + ref-mvs | 128 | 6.22 s | 4.45 s | +40% |

A second interleaved pass put refs 2/3/4 at +21/+15/+11%; the sign and rough
size are stable, the exact percentage is not.

* **New measurement switches on this branch** (`src/cli/options.h`,
  `src/cli/parse.cpp`, `src/cli/nightly.cpp`): `--video-edge-inter-off`,
  `--video-edge-mask-off`, `--video-bawp-off`, `--video-drift-refresh-off`,
  `--video-cdf-chain-off`, `--video-key-qp-offset-off`,
  `--video-deblock-off`, `--video-loop-filters-off`,
  `--video-tx-search-off`, `--video-rdoq-off`, `--video-partition-rd-off`.
  They only suppress the matching `--fast-decode` expansion element, are off by
  default, and leave the pinned streams byte-identical (nightly QP106 is still
  25,008 B MD5 `0639b43630abb454f7be5ac494d2efb4`; tuned drift QP110 is still
  25,146 B MD5 `c60250c02a315d725af19c74c4d1a5df`). They exist so the arms are
  reproducible; they are not proposed for `main`.
* **Two latent decode failures surfaced** and isolated, not fixed here:
  1. compound + `--avg-cdf --adaptive-cdf --cdf-continuity` desynchronises
     `dav2d` at frame 3 (3/60 pictures);
  2. the stripped long-term composition decodes 2/15 pictures unless
     `--video-cdf-per-frame-qp` is added.
  Both are header/CDF-image interactions worth a separate bug report.

## 6. Reproduction

```sh
# baseline
build-dev/av2enc testdata/nasa_480x240_30fps_60f.y4m -o /tmp/n.ivf \
  --video --frames 60 --cpu-threads 8 --qp 106 --fast-decode
# bar
build-dev/av2enc testdata/nasa_480x240_30fps_60f.y4m -o /tmp/d.ivf \
  --video --frames 60 --cpu-threads 8 --qp 110 --fast-decode \
  --video-drift-refresh-threshold 1.5 --video-drift-refresh-budget 3250
# best reference arm
build-dev/av2enc testdata/nasa_480x240_30fps_60f.y4m -o /tmp/r3.ivf \
  --video --frames 60 --cpu-threads 8 --qp 112 --fast-decode \
  --video-ref-signal-off --video-refs 3 \
  --video-drift-refresh-threshold 1.5 --video-drift-refresh-budget 3250
# compound control (also the CDF-workaround A/B)
build-dev/av2enc testdata/nasa_480x240_30fps_60f.y4m -o /tmp/c.ivf \
  --video --frames 60 --cpu-threads 8 --qp 116 --fast-decode \
  --video-ref-signal-off --video-refs 2 --video-bawp-off \
  --video-drift-refresh-off --video-cdf-chain-off
```

## 7. Limits

* One clip, one resolution, one operating point. The drift trigger is known to
  be content-dependent (`docs/video-drift-refresh.md`); the reference-count
  conclusion is measured on the same clip as every baseline, not generalized.
* The drift threshold/budget were **not re-tuned per reference count**. Refs 3
  at the shared recipe ties the bar; a per-count retune might move it inside
  the ripple, but the curve above shows no monotone direction to tune toward,
  and tuning one point against a ±1 K ripple would not be evidence.
* The bar itself is a 2-point interpolation over a ripple of ~0.35 K; the
  refs-3 curve has a ripple of ~1.1 K. The ±0.03 K tie is below both.
* Encode timings are from the dev `-O1` build on an M4 with 8 threads and are
  noisy; only the large effects (drift stack, wedge) are stable.
