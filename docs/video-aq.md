# Per-superblock adaptive quantisation (`--video-aq`)

`--video-aq` is an opt-in CPU-only `--video` flag that turns the §5.18.7.8
delta-Q syntax into a content-driven plan. Every superblock's qindex offset
comes from its **source luma variance** relative to the frame mean, and the
plan is coded through the existing `--video-delta-q` machinery (so
`rc::plan_delta_q()`, the §5.20.5.11 full-SB-skip rewrite and the writer's
`write_superblock_delta_q()` verification all apply unchanged).

It composes with the whole `--fast-decode` aggregate when the CDF chain uses the
per-frame reset (`--video-cdf-per-frame-qp`), including `--video-rdoq`, the
measured partition probe and the decision-level drift refresh.

**Measured verdict: null on the composite K.** At iso-rate 100.0 kbps the best
strength moves K by **+0.05** (JOD −0.003, Butteraugli −0.090, SSIMULACRA2
−0.530) — and that +0.05 is inside the control curve's own QP-to-QP K ripple,
which is ±0.2–0.5. The BA gain is real but small; SSIMULACRA2 gives back more
than BA gains. The flag stays an opt-in with rate-neutral defaults and is
**not** part of `--fast-decode`.

A second, independent cost is the CDF composition: delta-Q cannot carry the
§8.2.4 image, so an AQ stream must use `--video-cdf-per-frame-qp`. That mode
costs 2.9% rate at the same QP; at iso-rate the matched AQ-off control is
**0.52 K lower** than the plain `--fast-decode` pin (81.63 vs 82.15). Even if AQ
were +0.05 in isolation, turning it on under `--fast-decode` is **−0.47 K** versus
the pin.

```sh
av2enc testdata/nasa_480x240_30fps_60f.y4m -o /tmp/aq.ivf \
    --video --frames 60 --cpu-threads 8 --qp 106 --fast-decode \
    --video-cdf-per-frame-qp --video-aq --video-aq-strength 3
```

## Flags

| Flag | Meaning |
|---|---|
| `--video-aq` | Enable the activity plan (implies `delta_q_present = 1`). Replaces `--video-delta-q`. |
| `--video-aq-strength <S>` | Multiplier on the normalised activity deviation, in reduced-delta steps. Default `1.0`. |
| `--video-aq-clamp <N>` | Clamp on the activity offset, in reduced-delta steps (`1..8`). Default `3`. |
| `--video-aq-bias <N>` | Constant offset added to every superblock (`-8..8`). Default: auto (rate-neutral centring, see below). |
| `--video-delta-q-res <N>` | Reuses the delta-Q granularity. `--video-aq` defaults it to `1` (one coded step = 2 qindex). |

## Plan shape

For each superblock `i` of the frame's superblock grid, from the source luma
plane (not the reconstruction):

```text
act_i  = variance of the superblock's source luma samples (integer, as already
         computed for the segmentation split)
mean   = frame mean of act over the tile
rel_i  = (mean - act_i) / mean                      (0 when mean == 0)
raw_i  = clamp(round(strength * rel_i), -clamp, +clamp)
bias   = -round( sum(act_i * raw_i) / sum(act_i) )  (auto; 0 when sum == 0)
target = clamp(base_q_idx + (raw_i + bias) << delta_q_res, 1, 255)
```

* Textured superblocks (`act_i > mean`) go **finer** (lower qindex, more bits),
  flat ones go coarser. The `mean` reference is per frame and `rel_i` is
  dimensionless, so one strength behaves the same on a textured keyframe and a
  nearly black tail frame.
* `raw_i` is an integer number of reduced-delta steps, so `target` is exactly
  reachable by `rc::plan_delta_q()`; nothing is lost to rounding.
* **Rate-neutral centring.** A pure activity map spends rate: on this clip the
  uncentred textbook shape (`strength 3`, `clamp 3`, `delta_q_res 2`) measured
  **+20.4% rate**. The default bias centres the plan on the activity-weighted
  mean offset, with variance as the per-superblock bit proxy, so the textured
  superblocks that go finer are paid for by the flat ones that go coarser.
* `--video-aq-bias` overrides the centring with a constant. A constant bias is
  a global QP move, so it is a rate knob, not a quality knob.
* Below `strength 0.5` every `rel_i` rounds to zero: the plan is all-zero and
  the stream differs from AQ off only by the delta-Q syntax floor (+0.2–0.3%
  rate at QP106).

## Measured RD

Host: Apple M4, `--cpu-threads 8`, clip
`testdata/nasa_480x240_30fps_60f.y4m`, 60 frames. Every row is
`--fast-decode --video-cdf-per-frame-qp` with the listed AQ strength. The protocol
grid is QP 104/106/108; because the plan spends rate, most strengths put QP108
above 100 kbps, so QP110 is added where needed to bracket the target. Each
config is interpolated linearly in rate to **100.0 kbps**. Quality via
`tools/quality.py` (CVVDP 0.5.7, `standard_fhd`, MPS) with `dav2d-macos`, then
`tools/extra_metrics.py` (Butteraugli, SSIMULACRA2) and `tools/kpi.py`.

Axes: strength × {QP 104, 106, 108, 110}; `clamp 3`, `delta_q_res 1`, bias
auto. `K` uses `0.50·CVVDP/10 + 0.25/(1+BA/3) + 0.25·(SS2+50)/150`.

### Iso-rate (100.0 kbps)

| strength | bracketing QPs | JOD | BA | SS2 | K | dJOD | dBA | dSS2 | dK |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **0 (AQ off)** | 106–108 | 9.7222 | 3.670 | 80.643 | **81.63** | — | — | — | — |
| 0.25 (all-zero plan) | —\* | — | — | — | — | — | — | — | ≈−0.03 |
| 0.5 | 108–110 | 9.7104 | 4.046 | 80.194 | 80.90 | −0.0118 | +0.376 | −0.449 | −0.73 |
| 0.75 | 108–110 | 9.7184 | 3.703 | 80.371 | 81.51 | −0.0038 | +0.034 | −0.271 | −0.12 |
| 1.0 | 106–108 | 9.7229 | 3.712 | 80.164 | 81.48 | +0.0007 | +0.042 | −0.479 | −0.15 |
| 1.5 | 108–110 | 9.7040 | 4.442 | 79.828 | 80.24 | −0.0182 | +0.772 | −0.815 | −1.39 |
| 2.0 | 108–110 | 9.7068 | 3.591 | 80.151 | 81.60 | −0.0154 | −0.078 | −0.492 | −0.02 |
| **3.0 (best)** | **108–110** | **9.7195** | **3.580** | **80.113** | **81.68** | **−0.0026** | **−0.090** | **−0.530** | **+0.05** |
| 4.0 | 108–110 | 9.7206 | 3.829 | 80.387 | 81.32 | −0.0016 | +0.159 | −0.256 | −0.31 |
| 6.0 | 108–110 | 9.7310 | 4.003 | 80.417 | 81.10 | +0.0088 | +0.333 | −0.226 | −0.53 |

\* strength 0.25 is a no-op: its decoded pictures are identical to AQ off at
every QP, so the row is the QP108 point (100.19 kbps) and its true iso-rate
delta is the delta-Q syntax floor, about −0.03 K.

The best row is strength **3.0**: **+0.05 K**, **−0.003 JOD**, **−0.090 BA**,
**−0.530 SS2**. The K decomposition is +0.16 from BA, −0.09 from SS2 and −0.01
from CVVDP. BA — the component that carries the K gap — does improve, but by
0.09, about 5% of the 1.91 BA gap to SVT-AV1 preset 0; SSIMULACRA2 loses 0.53,
7% of that gap.

### Raw K by QP

Cells are `kbps / K`.

| strength | QP104 | QP106 | QP108 | QP110 |
|---:|---|---|---|---|
| 0 | 107.84 / 80.73 | 102.95 / 82.15 | 99.85 / 81.60 | 97.58 / 81.41 |
| 0.5 | 108.35 / 80.37 | 104.15 / 81.70 | 100.99 / 80.73 | 97.86 / 81.26 |
| 0.75 | 108.51 / 79.74 | 104.59 / 81.48 | 102.36 / 81.27 | 99.45 / 81.57 |
| 1.0 | 106.20 / 80.71 | 103.44 / 81.30 | 99.88 / 81.49 | — |
| 1.5 | 108.77 / 81.03 | 102.95 / 81.07 | 101.22 / 79.85 | 97.85 / 80.98 |
| 2.0 | 108.88 / 80.45 | 105.64 / 80.93 | 101.87 / 81.26 | 99.08 / 81.79 |
| 3.0 | 108.80 / 81.34 | 105.00 / 80.70 | 101.89 / 81.77 | 99.21 / 81.65 |
| 4.0 | — | — | 100.33 / 81.48 | 98.75 / 80.72 |
| 6.0 | — | — | 100.52 / 81.22 | 98.03 / 80.66 |

The control's own K moves **1.64** between QP106 (82.15) and QP107 (80.51);
the AQ rows move similarly (strength 3.0: 82.14 at QP107, 80.23 at QP109).
This is the drift-refresh/key-QP decision flipping, not measurement noise: the
streams are deterministic and the decoded output is recon-exact. A denser scan
(adding QP 105/107/109) flips the best strength to 0.75 (+0.20 to +0.30 K) and
puts strength 3.0 at −0.26 to −0.73 K. **No strength is reproducible outside
the ripple**, which is why the headline is a null rather than +0.05.

## Why it does not pay

The activity map wants a large offset range on this clip (frame 0 variance:
min 0, mean 1057, max 3001 over 32 superblocks), but

* the flat superblocks are mostly already coded as **full-SB skips**: 1,512 of
  3,904 planned steps in a 60-frame QP110 run are unreachable and are rewritten
  to the carried qindex by `rc::plan_delta_q()`. Making a skipped superblock
  coarser saves nothing, so the "cheap" side of the trade has very little to
  give;
* the textured side is expensive: +2 qindex on a coded superblock costs far
  more than −2 saves on a skipped one, which is why the uncentred map overspent
  by 20% rate;
* with the plan centred, the remaining redistribution is ±2–4 qindex. At that
  amplitude CVVDP is flat (AQ shuffles error rather than removing it), BA
  gains a few hundredths and SSIMULACRA2 loses half a point;
* every strength that touches the plan also changes the drift-refresh
  decisions (7 frames instead of 9 at QP106), which is where the large
  QP-to-QP K jumps come from.

## Composition notes (what the measurement needed)

`--video-aq` is delta-Q, so it inherited four blanket rejections that were
re-validated and relaxed (see `docs/negative-results.md`):

1. **Adaptive-CDF chain.** §5.18.7's delta/segment CDFs are not in the §8.2.4
   image, so the encoder and decoder can only start from the same values when
   the image is re-derived per frame. That is exactly
   `--video-cdf-per-frame-qp`; the carried form (`--cdf-continuity`) stays
   rejected by name. The reset costs 2.9% rate at QP106 (25,737 B vs the pin's
   25,008 B), which is most of the 0.52 K difference between the matched
   control and the plain pin at iso-rate.
2. **RDOQ.** The trellis quantises against `params.quant_factor`, which already
   carries the §7.14.2 per-superblock step, and prices coefficients in the raw
   `base_q_idx` §8.3.1 slice. Only the segmentation `SEG_LVL_ALT_Q` and
   per-plane UV variants stay rejected.
3. **Measured partition probe.** It was disabled whenever a delta-Q plan was
   active, which changed the partition decisions for reasons unrelated to AQ.
   The probe now prices each node with its own superblock qindex
   (`probe_syntax_price()`), so an all-zero plan reproduces the non-delta-Q
   stream exactly.
4. **Drift refresh.** `--video-drift-refresh` only conflicts when it changes
   the frame `base_q_idx` (`--video-drift-refresh-qp != 0`); at the default 0
   the refresh is decision-level and composes.

With the plan at zero, `--fast-decode --video-cdf-per-frame-qp --video-delta-q
--video-delta-q-res 1 --video-seg-delta 0` is recon-identical to
`--fast-decode --video-cdf-per-frame-qp` (the control that proves the
probe/refresh/CDF relaxations do not move pixels).

## Limits

* CPU 8-bit 4:2:0 `--video`, one tile, `--rc cqp`. GPU, `--tiles > 1x1`,
  10-bit/4:2:2/4:4:4 and `--rc crf/vbr` are named rejections.
* Rejected by name with the segmentation surfaces (`--video-segmentation`,
  `--video-per-segment-q`, `--video-seg-skip/globalmv/temporal`,
  `--video-segments`), `--video-delta-q`, `--video-uv-delta-q`, TCQ/CCTX, SDP,
  `--video-qm` and `--cdf-continuity`.
* `--video-aq-strength/clamp/bias` require `--video-aq`.
* Plan targets are clamped to `[1, 255]`; a frame whose `base_q_idx` is near a
  bound can clip the finest/coarsest superblocks.

## Verification

* `dav2d --strict 1` decodes all 60 pictures of the measured streams; the
  encoder `--video-recon` payload equals the decoded payload byte for byte
  (10,368,000 B) on the full/2/6-frame prefixes.
* Two encodes of the same settings are byte-identical.
* `--video-aq-strength 0` is byte-identical to the neutral delta-Q ramp
  (`--video-delta-q --video-delta-q-res 1 --video-seg-delta 0`), so the
  activity plan is the only thing the flag changes.
* `--video-aq-strength 0.25` decodes identically to AQ off at QP104/106/108
  (the plan is all-zero).
* Flags off: the pins are byte-identical (default QP189 25,009 B `a1d0dba1…`,
  quality QP190 25,141 B `3984677b…`, nightly QP110 23,690 B `1b99127a…`,
  explicit-λ control 24,113 B `7f03eb6e…`, 720p 36,572 B `c4ba2813…`).
* `tools/gate_video_aq.sh` covers the rejections, the neutral-ramp control,
  determinism, strict decode + recon on the full/2/6-frame streams and the
  nightly CDF-chain composition; `gate_video_nightly` and
  `gate_video_qp_lambda` stay green.
