# Loop restoration at 240p30 / 100 kbps and the 720p pin

Status: **negative. No loop-restoration arm is worth shipping as an opt-in at
any of the three measured points.**

* The **luma Wiener NS** arm is a tie at the plain 240p point
  (**K@100 = 82.08 vs 82.05, +0.03**) — inside the ±1.6 K QP ripple and not a
  gain. At the tuned-drift point it is **−1.35 K**. At 720p it cannot code the
  pin's rate at all (floor ~474 kbps against a 146 kbps pin).
* **Chroma LR** costs 14.3 kB of syntax over 60 frames at 240p and 216 kB at
  720p. At matched rate it scores **−25.6 K** / **−26.1 K** / **−11.0 K**.
* **PC-Wiener** is almost rate-free but scores **−1.4 K** / **−3.1 K** /
  **−2.8 K**, and its decoded output is not the encoder reconstruction: at the
  720p pin 56 of 60 frames differ, 36% of samples, max 81 LSB.
* The composed **Wiener + chroma** arm is the sum of the two costs; at 240p its
  rate floor is **125.4 kbps**, so it cannot reach 100 kbps at all.

Branch `exp/lr-score`, based on `main` at `3dcd151` (the merge that added the
§6.17.2 LR CDF rows to the carried image, `gate_video_cdf_carry.sh` 2b).
Measurement build `cmake --preset dev` (CPU reference, `build-dev/av2enc`,
`-O1`), Apple M4, `--cpu-threads 8`. All quality points use one CVVDP job at a
time under `nice -n 10`. The host was shared with other worktrees, so encode
*timings* are noisy; they were re-measured interleaved (section 8).

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

`kbps = IVF bytes * 8 / 2 s / 1000`. `K@100` is a linear interpolation on kbps
between the two bracketing QPs. `Δbar` is against the bar interpolated to the
same rate, so it removes the rate axis from the QP ripple.

Points A and B use `testdata/nasa_480x240_30fps_60f.y4m`; point C uses
`testdata/space8k_1440x720_60f.y4m`.

### The three points and their bars

| Point | Recipe | Bar QP | Bytes | kbps | JOD | BA | SS2 | K |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A: 240p plain | `--nightly` | 106 | 25,008 | 100.032 | 9.7334 | 3.4202 | 80.84 | **82.155** |
| B: 240p tuned drift | `--nightly --video-drift-refresh-threshold 1.5 --video-drift-refresh-budget 3250` | 110 | 25,146 | 100.584 | 9.7605 | 2.6562 | 81.25 | **83.937** |
| C: 720p pin | `--nightly --video-key-qp-offset -57 --video-leaf-rd-lambda 250 --video-ext-partitions` | 217 | 36,572 | 146.288 | 7.3236 | 28.8713 | −10.146 | **45.6135** |

All three reproduce byte- and metric-exactly (A MD5 `0639b436…`, B
`c60250c…`, C `c4ba2813…`). The quoted 82.15 / 83.80 / 45.61 are the verdict
numbers; at exactly 100.000 kbps the local bars interpolate to 82.048 (A) and
83.796 (B). Point C is JOD-dominated: 36.6 of its 45.6 K is the JOD term, so
the 720p judgement is made on JOD first.

Local bar curves used for `Δbar` (measured here unless marked):

| Point | QP | kbps | JOD | K |
|---|---:|---:|---:|---:|
| A | 105 | 101.368 | 9.7216 | 81.806 |
| A | 106 | 100.032 | 9.7334 | 82.155 |
| A | 107 | 99.540 | 9.7113 | 80.508 |
| A | 88 | 126.880 | 9.6965 | 80.978 |
| B | 109\* | 104.020 | — | 83.214 |
| B | 110 | 100.584 | 9.7605 | 83.937 |
| B | 111\* | 99.800 | — | 83.747 |
| B | 95 | 123.604 | 9.7012 | 80.732 |
| C | 217 | 146.288 | 7.3236 | 45.614 |
| C | 216 | 150.196 | 7.3657 | 45.894 |
| C | 161 | 474.368 | 8.5018 | 57.901 |
| C | 160 | 481.236 | 8.5021 | 57.948 |
| C | 125 | 974.832 | 8.4678 | 56.825 |
| C | 120 | 1,092.152 | 8.4366 | 56.696 |

\* Carried from `docs/reference-structure.md` (same recipe, same clip); the
QP110 point reproduced exactly at 83.937, so the two extra points are used only
to extend the B curve above 101 kbps.

### The arms

All arms are the point's aggregate plus one or two flags:

| Arm | Flags |
|---|---|
| wiener | `--video-wiener` (luma §7.20.3 non-separable Wiener NS) |
| chroma | `--video-chroma-lr` (U/V §7.20.3) |
| pc | `--video-pc-wiener` (luma §7.20.4 PC-Wiener) |
| wc | `--video-wiener --video-chroma-lr` (they compose) |

## 2. Constraint log

Nothing had to be dropped for the LR flags — unlike the reference-structure
tools, the loop filters compose with the nightly aggregate, the drift refresh,
the carried CDF chain, the QP-scaled lambdas and `--video-ext-partitions`:

```text
A: --nightly                        + wiener / chroma / pc / wiener+chroma  rc 0
B: + drift threshold 1.5 budget 3250 + wiener / chroma / pc / wiener+chroma  rc 0
C: --nightly --video-key-qp-offset -57 --video-leaf-rd-lambda 250
   --video-ext-partitions            + wiener / chroma / pc / wiener+chroma  rc 0
```

The only named rejection in the family makes `pc` and `wiener` mutually
exclusive (`src/cli/validate.cpp:1257`):

```text
av2enc: --video-pc-wiener replaces the luma Wiener NS decision; do not combine the two luma restoration tools
```

That is why `pc` is a separate arm and the composition arm is
`--video-wiener --video-chroma-lr`. `--video-pc-wiener --video-chroma-lr` also
composes and was probed, but it is dominated by both parents and is not
reported. The pre-existing `--tiles > 1x1` rejection for
`--video-wiener/--video-chroma-lr` (`validate.cpp:1035`) does not fire: every
point is single-tile.

## 3. Point A — 240p plain nightly (bar 82.15)

### 3.1 Arm tables

| Arm | QP | Bytes | kbps | JOD | BA | SS2 | K | Decode |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| wiener | 119 | 25,568 | 102.272 | 9.7389 | 3.945 | 80.04 | 81.167 | 60/60 |
| wiener | 120 | 25,138 | 100.552 | 9.7501 | 3.598 | 80.34 | 81.840 | 60/60 |
| wiener | 121 | 25,070 | 100.280 | 9.7410 | 3.353 | 80.04 | 82.185 | 60/60 |
| wiener | 122 | 24,677 | 98.708 | 9.7385 | 3.707 | 80.15 | 81.565 | 60/60 |
| chroma | 200 | 25,668 | 102.672 | 7.8488 | 10.200 | 40.23 | 59.965 | 60/60 |
| chroma | 210 | 24,778 | 99.112 | 7.3462 | 12.280 | 31.75 | 55.265 | 60/60 |
| pc | 105 | 25,641 | 102.564 | 9.6933 | 4.780 | 79.83 | 79.744 | 60/60 |
| pc | 106 | 25,078 | 100.312 | 9.7169 | 3.925 | 80.31 | 81.133 | 60/60 |
| pc | 107 | 24,704 | 98.816 | 9.6322 | 5.191 | 77.94 | 78.641 | 60/60 |
| wc | 240 | 31,339 | 125.356 | 5.8945 | 19.762 | −5.47 | 40.189 | 60/60 |

| Arm | Brackets (QP: kbps / K) | K@100 | Δbar@100 | Rate floor |
|---|---|---|---:|---:|---|
| wiener | 121: 100.280 / 82.185 · 122: 98.708 / 81.565 | **82.075** | **+0.03** | ~60 kbps |
| chroma | 200: 102.672 / 59.965 · 210: 99.112 / 55.265 | **56.44** | **−25.61** | 99 kbps |
| pc | 106: 100.312 / 81.133 · 107: 98.816 / 78.641 | **80.61** | **−1.44** | ~85 kbps |
| wc | 240: 125.356 / 40.189 | — | **−40.84 @125.4** | **125.4 kbps** |

### 3.2 The Wiener arm is a tie, not a gain

The arm's four points are non-monotone (82.19 at 100.28 kbps, 81.84 at
100.55), and the bar is non-monotone over 1.8 kbps (82.16 → 80.51). The
point-by-point `Δbar` is −0.61 / −0.18 / +0.10, and the iso-rate interpolation
is **+0.03**. The encoder log shows why: at its matched QP121 the arm has
**23,096 B of tile payload against the baseline's 23,022 B at QP106**. The
matched-rate stream is the same size, with the residual bits traded for LR
coefficient bits at a 15-QP higher quantiser. There is no net quality gain to
pay for the trade.

The PC-Wiener arm is the same trade with a worse split: its best point is
100.312 kbps / 81.133 (−0.95 against the bar at that rate), and its lower
bracket falls off the clip's content-change cliff (BA 5.19 at QP107) even
harder than the bar's does.

### 3.3 Chroma LR and the composed arm

Chroma LR is not a quality problem at iso-QP; it is a rate problem. The arm at
QP200 measures 102.672 kbps while the plain baseline at the *same* QP is
24.896 kbps — a 4.1× stream. To return to 100 kbps the arm must run at QP210,
and the resulting JOD is 7.35 against the bar's 9.73. The composed arm never
returns: its rate floor is 125.356 kbps at QP240 (the maximum QP), where the
stream is 28,427 B of tile payload and scores K 40.19.

## 4. Point B — 240p tuned drift (bar 83.80)

| Arm | QP | Bytes | kbps | JOD | BA | SS2 | K | Decode |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| bar | 110 | 25,146 | 100.584 | 9.7605 | 2.656 | 81.25 | 83.937 | 60/60 |
| wiener | 123 | 25,338 | 101.352 | 9.7455 | 2.939 | 80.34 | 83.079 | 60/60 |
| wiener | 125 | 24,994 | 99.976 | 9.7230 | 3.130 | 79.49 | 82.431 | 60/60 |
| chroma | 205 | 25,172 | 100.688 | 7.6118 | 10.538 | 35.28 | 57.812 | 60/60 |
| pc | 111 | 25,401 | 101.604 | 9.6768 | 4.941 | 77.78 | 79.126 | 60/60 |
| pc | 112 | 24,838 | 99.352 | 9.6826 | 3.592 | 79.21 | 81.326 | 60/60 |
| wc | 240 | 31,339 | 125.356 | 5.8945 | 19.762 | −5.47 | 40.189 | 60/60 |

| Arm | Brackets (QP: kbps / K) | K@100 | Δbar@100 | Rate floor |
|---|---|---|---:|---:|---|
| wiener | 123: 101.352 / 83.079 · 125: 99.976 / 82.431 | **82.44** | **−1.35** | ~60 kbps |
| chroma | 205: 100.688 / 57.812 | **56.9** | **−26.10 @100.7** | 99 kbps |
| pc | 111: 101.604 / 79.126 · 112: 99.352 / 81.326 | **80.69** | **−3.10** | ~85 kbps |
| wc | 240: 125.356 / 40.189 | — | **−40.5 @123.6** | **125.4 kbps** |

The Wiener arm is *worse* at the tuned point than at the plain point (−1.35
instead of +0.03). Both the drift refresh and the Wiener filter spend bits on
the same frames, and the arm's refresh count moves (the measured streams log
18 refresh frames at QP125 and 28 at QP110 for the bar, so the comparison is
not like-for-like in refresh count). The chroma and composed arms are the same
collapse; `wc` at QP240 is byte-identical to the point-A arm because the drift
refresh is inert that high.

## 5. Point C — 720p pin (bar 45.61)

| Stream | QP | Bytes | kbps | JOD | BA | SS2 | K | Decode |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **pin (bar)** | 217 | 36,572 | 146.288 | 7.3236 | 28.871 | −10.146 | **45.614** | 60/60 |
| pc | 218 | 37,522 | 150.088 | 7.2551 | 32.411 | −12.924 | 44.573 | 60/60 |
| pc | 219 | 36,765 | 147.060 | 7.2091 | 34.746 | −17.767 | 43.405 | 60/60 |
| pc | 220 | 35,751 | 143.004 | 7.0882 | 68.673 | −26.641 | 40.381 | 60/60 |
| wiener | 217 | 122,900 | 491.600 | — | — | — | — | 60/60 |
| wiener | 224 | 118,534 | 474.136 | 7.1461 | 28.893 | −15.487 | 43.834 | 60/60 |
| chroma | 217 | 252,384 | 1,009.536 | 7.3267 | 29.611 | −9.057 | 45.757 | 60/60 |
| chroma | 226 | 246,565 | 986.260 | — | — | — | — | 60/60 |
| wc | 240 | 319,033 | 1,276.132 | — | — | — | — | 60/60 |
| *bar @150.2* | 216 | 37,549 | 150.196 | 7.3657 | 30.600 | −9.001 | 45.894 | 60/60 |
| *bar @474.4* | 161 | 118,592 | 474.368 | 8.5018 | 21.923 | 24.296 | 57.901 | 60/60 |
| *bar @974.8* | 125 | 243,708 | 974.832 | 8.4678 | 28.239 | 22.510 | 56.825 | 60/60 |

The italic rows are plain-pin streams at the arms' rates (no LR), for the
matched-rate comparison. The rate story is decided before any quality
measurement:

* **Wiener** is 3.36× the pin at the same QP (491.6 kbps) and only 17.5 kbps
  smaller seven QPs later (474.1 at QP224). Its floor is ~470 kbps: **the arm
  cannot code the pin's 146.3 kbps at any QP.**
* **Chroma** is 6.9× the pin at the same QP (1,009.5 kbps, 252 kB); its floor
  is ~950 kbps.
* **`wc`** (1,276 kbps at QP240) was encoded and decoded but not quality
  scored: it is 8.7× the pin and dominated by both parents.
* **PC-Wiener** is the only arm at the pin's rate: 147.060 kbps at QP219
  against the pin's 146.288.

### 5.1 JOD at matched rate (JOD first)

| Arm | Arm point | Bar at that rate | ΔJOD | ΔK |
|---|---|---|---:|---:|
| pc | 147.060 / JOD 7.2091; at 146.288: JOD 7.1861 | 146.288 / 7.3236 | **−0.138** | −2.78 |
| wiener | 474.136 / JOD 7.1461 | 474.368 / 8.5018 | **−1.356** | −14.06 |
| chroma | 1,009.536 / JOD 7.3267 | 1,009.5 / 8.4586 | **−1.132** | −11.03 |

The QP brackets make the JOD interpolation stable: pc is interpolated between
147.060 and 143.004, wiener is matched almost exactly (474.136 against a
474.368 bar point), and the chroma bar is interpolated between 974.832 and
1,092.152 kbps. The pc *K* interpolation is less stable than its JOD because
the second bracket QP220 hits the clip's content-change cliff (BA 34.7 → 68.7,
SS2 −17.8 → −26.6); that is why the 720p judgement is on JOD.

Two facts are worth stating on their own:

* Wiener at 474 kbps scores **JOD 7.146, below the 146 kbps pin's 7.324**.
  Spending 3.2× the rate plus restoration makes the picture worse than not
  restoring at all.
* Chroma at 1,010 kbps scores **JOD 7.327, +0.003 over the 146 kbps pin**.
  Six point nine times the rate moves the JOD by three thousandths.

## 6. Decode and reconstruction evidence

Every measured stream decodes with `dav2d-macos --strict 1`; the picture count
is read from the decoder log (`Decoded 60/60 frames`) because the decoder exits
0 even after failed pictures. Representative streams (A-wiener QP121, A-pc
QP106, A-chroma QP210, B-wiener QP125, C-wiener QP224, C-pc QP219, C-chroma
QP217) also pass `tools/validate_dav2d.sh` with exit 0.

**Wiener and chroma LR are reconstruction-exact** where checked with
`--video-recon` (encoder reconstruction vs decoded Y4M, sample by sample):

| Stream | QP | Differing samples | Max │diff│ |
|---|---:|---:|---:|
| A-wiener | 121 | **0** of 10,368,000 | 0 |
| B-chroma | 205 | **0** of 10,368,000 | 0 |
| A-wc | 240 | **0** of 10,368,000 | 0 |
| C-wiener | 224 | **0** of 93,312,000 | 0 |

**PC-Wiener is not reconstruction-exact, and the previously recorded ±1 LSB
on 5 frames is the best case, not the typical one.** It reproduces exactly at
QP110, then grows as the quantiser drops, and compounds on the drift chain and
at 720p:

| Stream | QP | Frames differing | Differing samples | Max │diff│ |
|---|---:|---:|---:|---:|
| A-pc | 110 (plain nightly) | 5 (55–59) | 52 | 1 |
| A-pc + `--video-key-qp-offset -25` | 110 | 1 (59) | 13 | 1 |
| A-pc | 106 | 28 (32–59) | 27,002 | 13 |
| A-pc | 104 | 54 (6–59) | 57,793 | 8 |
| B-pc | 112 (drift) | 28 (32–59) | 245,840 | 32 |
| default `--video` | 189 | 58 (1–59) | 13,344 | 9 |
| **C-pc** | **219 (720p pin)** | **56 (4–59)** | **33,806,745 of 93,312,000 (36.2%)** | **81** |

The QP110 row is the exact case in `docs/negative-results.md`; it is the
*mildest* measured case. At the operating points used for scoring, the decoder
output differs from the encoder reconstruction on 28/60 frames at 240p
(QP106/QP112) and on 56/60 frames at the 720p pin, where more than a third of
all samples differ. The PC-Wiener scores in sections 3–5 are measured on those
decoded pictures (i.e. what a viewer gets), but the encoder is optimising a
different picture. That is a second, independent reason not to ship the tool;
the mechanism remains the §7.20.4 feature-window border already recorded in
`docs/negative-results.md`, and the pinned decoder is a black box for that path.

## 7. Why the rate cost is what it is

Both video LR decisions are **rate-blind**. The Wiener/chroma search is called
with `lr_options.lambda_bits = 0.0`
(`src/pipeline/video/post/video_loop_filters.cpp:1366`), so a unit is enabled
whenever its filter reduces SSE at all; `wiener_decision.h` says so in its own
header comment: *"With lambda_bits == 0 the search is pure distortion … a
production encoder should set a nonzero lambda."* PC-Wiener uses the same 0.0
in `DecidePerUnitPcWiener`, but its syntax is one use bit per unit instead of
coefficients, which is why it is nearly rate-free.

Measured LR syntax, from the encoder's own tile accounting at the point QP:

| Plane | Units/frame | LR syntax, 60 frames | Share of the point |
|---|---:|---:|---|
| 480×240 luma Wiener | 32 | +4,416 B | 17.7% of 25,008 B (A) |
| 480×240 chroma | 64 | +14,285 B | 57.1% of 25,008 B |
| 1440×720 luma Wiener | 276 | +86,447 B | 2.36× the 36,572 B pin |
| 1440×720 chroma | 828 | +215,803 B | 5.90× the pin |

Per unit-frame the coefficient cost is roughly constant — 2.3 B for 240p luma,
3.7 B for 240p chroma, 5.2 B for 720p luma, 4.3 B for 720p chroma — so the
overhead scales with frame area while the rate budget at a fixed quality point
does not. At 240p the luma filter is 18% of the stream and exactly cancels its
own gain; at 720p it is more than twice the entire stream.

A rate-aware decision (or a frame-level filter, which the decision module
supports but the video path deliberately disables: `allow_frame_filters =
false`) could reduce the syntax, but neither exists on this tree and neither
was measured here.

## 8. Encode cost

Median of three interleaved runs on the shared host (`build-dev`, M4,
`--cpu-threads 8`, 60 frames). Absolute times are only comparable within a
row; the host load from other worktrees moved repeat-to-repeat by up to ±25%.

### 240p

| Arm | QP | Arm | Control at the same QP | Delta |
|---|---:|---:|---:|---:|
| pc | 106 | 7.87 s | 5.55 s | **+42%** |
| wiener | 121 | 11.71 s | 5.30 s | **+121%** |
| chroma | 205 (drift) | 13.58 s | 4.31 s | **+215%** |
| wc | 240 | 21.40 s | 4.56 s | **+369%** |

### 720p

| Stream | QP | Time | Runs |
|---|---:|---:|---|
| pin (control) | 217 | 40.87 s | 40.87 / 41.22 / 40.74 |
| pc | 217 | 55.09 s | 53.66 / 59.95 / 55.09 |
| pc | 219 | 54.43 s | 54.43 / 54.35 / 57.39 |
| base (control @224) | 224 | 39.37 s | 40.00 / 36.13 / 39.37 |
| wiener | 224 | 229.2–234.6 s | two runs (scan points) |
| chroma | 217 | 286.9–354.4 s | two runs (scan points) |
| base (control @240) | 240 | 47.16 s | 51.26 / 45.62 / 47.16 |
| wc | 240 | 465.0 s | one run |

Same-QP ratios: pc +35%, wiener **×5.8**, chroma **×7.0**, `wc` **×9.9**. The
cost is the per-unit least-squares fit: the chroma arm fits 828 units per
frame, with the 32-tap luma / 12+6-tap chroma search, over 60 frames.

## 9. Verdict

**Nothing here is worth shipping as an opt-in.**

Against **82.15** (240p plain): the best arm is Wiener at **82.08 (+0.03)** —
a tie inside the ±1.6 K QP ripple, with no bracket point more than +0.10.
PC-Wiener is −0.95 at its nearest point (−1.44 at exactly 100 kbps), chroma LR
−25.61, and `wc` −40.84 while failing to reach 100 kbps at all.

Against **83.80** (240p tuned drift): the best arm is Wiener at **82.44
(−1.35)**; PC-Wiener −3.10, chroma −26.10, `wc` −40.5.

Against **45.61** (720p pin): PC-Wiener is the only arm that codes the pin rate
and it is **−0.138 JOD / −2.78 K**. Wiener is **−1.356 JOD** at its own
(3.2×) rate and scores *below the pin's JOD*; chroma is **−1.132 JOD** at 6.9×
the rate and moves JOD by +0.003 against the pin.

The family-level conclusion is that loop restoration is a high-rate tool on
this encoder: it buys distortion with coefficient bits, and at 100 kbps (240p)
or 146 kbps (720p) those bits are not available. The one arm that is
rate-neutral (PC-Wiener) loses on distortion and does not decode to the
encoder's reconstruction. Shipping any of them would also carry a 35–886%
encode-time penalty at the measured points.

## 10. Reproduction

```sh
REF=testdata/nasa_480x240_30fps_60f.y4m
# point A arm
build-dev/av2enc "$REF" -o /tmp/a.ivf --video --frames 60 --cpu-threads 8 \
  --qp 121 --nightly --video-wiener
# point B arm
build-dev/av2enc "$REF" -o /tmp/b.ivf --video --frames 60 --cpu-threads 8 \
  --qp 125 --nightly --video-drift-refresh-threshold 1.5 \
  --video-drift-refresh-budget 3250 --video-wiener
# point C arm (720p)
build-dev/av2enc testdata/space8k_1440x720_60f.y4m -o /tmp/c.ivf --video \
  --frames 60 --cpu-threads 8 --qp 224 --nightly \
  --video-key-qp-offset -57 --video-leaf-rd-lambda 250 \
  --video-ext-partitions --video-wiener
# recon equality (Wiener/chroma) and divergence (PC-Wiener)
build-dev/av2enc "$REF" -o /tmp/r.ivf --video --frames 60 --cpu-threads 8 \
  --qp 121 --nightly --video-wiener --video-recon /tmp/r.y4m
tools/bin/dav2d-macos --strict 1 -i /tmp/r.ivf -o /tmp/r.dec.y4m \
  --muxer yuv4mpeg2
```

## 11. Limits

* One clip per point, one operating point per clip, one host. The rate-blind
  decision is the measured implementation; a nonzero `lambda_bits` would
  change every arm and was not measured (no CLI knob exists).
* The 240p bars are two-point interpolations over a ripple of 0.35 K (bar) to
  1.65 K (bar curve), and the arms' own curves ripple by ~1 K. The Wiener tie
  (+0.03, +0.10 at its best point) and the point-A PC-Wiener loss (−0.95 to
  −1.44) are inside the clip's QP ripple in magnitude; the point-B PC-Wiener
  loss (−3.10) is outside it. The sign was checked at every bracket point and
  no LR point was positive by more than +0.10, but a per-QP retune would not be
  evidence at this ripple size.
* At 720p the arms cannot be rate-matched to the pin (except PC-Wiener), so
  the comparison is made at the arm's own rate against a plain-pin stream at
  the same rate; those plain streams are 3–7× the pin and near the top of the
  clip's JOD curve, which is the honest comparison for "would spending these
  bits on LR beat spending them on the encoder".
* Encode timings come from a shared host at load average ~9–10 on 10 cores;
  the ratios are stable to ~±25%, the absolute seconds are not.
