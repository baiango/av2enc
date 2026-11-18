# Extended perceptual RD core (`AV2ENC_EFF_PERC`)

Status: **complete, env-gated, default off; no CLI flag.** Branch
`exp/eff-perc`, worktree `eff-perc`, built on the merged pipe-perc pipeline
(`docs/pipe-perc.md`). Gate on: `AV2ENC_EFF_PERC=1`. The tuned point is λ
scale `0.18`, error CSF gain `1.5`, rate term `1` (R19 interval width) on the
measured drift recipe. Gate off is byte-identical to the three pinned 240p
streams, the merged pipe-perc point and the 720p pin.

**K oracle result: the luma-only achievable envelope is 85.18 at 100.92 kbps
(85.10 interpolated to 100.0), below the 86 target by 0.90 K.** Per the
oracle-first rule the pipeline stops here; no further tuning was done and no
realised >86 claim is made. The wall comparison is against the internal
84.85 pipe-perc point, not the 82.15 pin.

## Why this gate exists, and what the ceiling says

pipe-perc reached 84.85 with an exact inter-syntax probe, top-N real-path
re-scoring and one λ for D and R. It lost to SVT-AV1 mainly on the luma
detail/body and the fade. This branch attacked the three remaining luma RD
terms:

| Term | pipe-perc | eff-perc |
|---|---|---|
| D | 8x8 source gradient cell weight, chroma plain SSE | error-domain low/mid-band CSF + optional per-pixel multi-scale/edge source weight; chroma plane weight |
| R | R09 live syntax + static §8.3 coefficients | same syntax probe + selectable coefficient term: static §9.3, R19 interval width, or live frame-start CDF rows |
| λ | 0.2 | refit to 0.18 with the interval coefficient term |

The envelope sweep below shows the surviving gain is the R+λ refit plus a
small error-CSF effect. A decoded-domain hybrid from the chroma workstream
(`exp/eff-chroma`, commit `706899d`) puts **0.61 K of the 3.93 K SVT gap on
chroma and 3.32 K on luma**; the former "SS2 correlates with chroma MSE"
reading was not causal. This branch therefore spends no more budget on chroma.

## K oracle (required before tuning)

Definition from `docs/k-oracle.md`: `K*_P = sup_θ K_P(θ)` at 240p/100 kbps.
The comparison wall is **84.85** (the merged pipe-perc point at 99.976 kbps)
and the target is 86.0.

**Method.** Every arm was encoded on `testdata/nasa_480x240_30fps_60f.y4m`
with 60 frames, 8-bit 4:2:0, `--nightly --cpu-threads 8`, the measured drift
recipe (`--video-drift-refresh-threshold 1.5 --video-drift-refresh-budget
3250`), and two or three QPs that bracket 100.0 kbps. Streams were decoded
with `dav2d --strict 1` and scored with `tools/quality.py` (CVVDP 0.5.7,
`standard_fhd`, MPS), `tools/extra_metrics.py` (Butteraugli, SSIMULACRA2) and
`tools/kpi.py`. K is linearly interpolated in rate to 100.0 kbps. The sweep is
D shape ∈ {0 neutral, 1 pipe gradient, 2 multi-scale, 3 multi-scale+edge,
4 variance+edge, 5 edge-only}, error-CSF gain ∈ {0, 0.5, 1, 1.25, 1.5, 2,
2.5, 3}, λ ∈ {0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30}, chroma weight ∈
{1, 2, 4}, coefficient rate ∈ {static, interval, live slice, both} and top-N
∈ {2, 4, 8}.

**Achievable envelope (primary, a lower bound on K*):**

| Arm | Best K at 100.0 kbps | Rate | Δ vs 84.85 |
|---|---:|---:|---:|
| **shipped (CSF 1.5, rate 1, λ 0.18)** | **85.10** (85.18 realised) | 100.92 measured | **+0.25** |
| CSF 1.5, static coefficients, λ 0.18 | 84.76 | 97.78–100.43 | −0.09 |
| CSF 1.5, interval + live slice, λ 0.18 | 84.95 | 99.13–101.19 | +0.10 |
| CSF 1.0, interval, λ 0.20 | 85.04 | 100.4 | +0.19 |
| best D-shape with a non-neutral source weight (shape 1, detail 0.1) | 85.06 | 100.0 | +0.21 |
| top-N 4 or 8 (same knobs) | 84.99 | 100.0 | +0.14 |
| chroma weight 2 (same knobs) | 85.01 | 100.0 | +0.16 |

The envelope is **85.18 measured / 85.10 at matched rate**, i.e. **0.90 K
short of 86**. No allocation bound is reported: the pipeline has no
frame-level reallocation knob and the drift recipe is fixed. No decision bound
is reported: an exact-bit decision oracle (R05) was not built for this
pipeline. The shipped luma-only envelope is therefore the reported ceiling.

### K vs λ (rate 1, CSF 1.5, QP bracket)

| λ scale | QP points (kbps) | K @ 100.0 | Δ vs 84.85 |
|---:|---|---:|---:|
| 0.15 | 119/120 (103.0 / 99.8) | 84.86 | +0.01 |
| **0.18** | **119/120 (100.9 / 98.7)** | **85.10** | **+0.25** |
| 0.20 | 119/120 (99.8 / 97.8) | 84.80 | −0.05 |
| 0.25 | 118/121 (100.3 / 95.2) | 84.86 | +0.01 |
| 0.30 | 118/121 (98.6 / 93.1) | 84.85 | 0.00 |

The curve peaks at 0.18 and falls on both sides. The plateau is narrow, and
the QP-to-QP ripple on this clip is ±1.5 K, so the adjacent-QP points (not a
single bracket) are the evidence for the peak.

### D-shape ablation at the tuned rate/λ

| D shape | Meaning | K @ 100.0 | Note |
|---|---|---:|---|
| **0** | neutral source map + error CSF | **85.10** | shipped |
| 1 | pipe-perc 8x8 gradient, detail 0.1 | 85.06 | neutral once the gain is small |
| 1 | pipe-perc 8x8 gradient, detail 1.0 | 84.97 (measured 103.8 kbps) | ≈84.7 matched |
| 2 | per-pixel multi-scale, detail 0.1 | 84.55 | texture up-weight loses |
| 3 | multi-scale + coherent edge, detail 0.1 | 84.85 | edge term does not recover it |
| 4 | 8x8 variance + edge, CSF 0 | 84.96 at 104.6 kbps | ~84.7 matched |
| 5 | coherent edge only, detail 0.25 | 84.8 (101.3 kbps) | edges alone do not help |

**The source-activity detail weight is not the lever.** At matched rate every
source-weight shape is neutral or negative; the 8x8 gradient at full strength
is the pipe baseline, and the finer per-pixel shapes are worse. This is the
same "shifting error, not removing it" result as the earlier dead-zone and AQ
experiments (`docs/negative-results.md`). The surviving D change is the
**error-domain CSF**, which prices the low/mid-frequency error band (where the
measured Butteraugli gap lives) above the high-frequency tail. The
detail/edge source models remain in the tree as measured ablations, but the
default is shape 0.

### Chroma coupling

Chroma is coupled into the same D (`AV2ENC_EFF_PERC_CHROMA`, default 1.0) at
every leaf, skip/residual and intra/inter comparison, and the chroma CSF uses
the same low/mid-band term. Weighting it 2x moved BA by +0.04 and left SS2
flat at matched rate; the quantiser zeroes most inter chroma coefficients at
QP 119, so the weight cannot buy chroma detail. This agrees with the
decoded-domain hybrid: chroma is only 0.61 K of the gap, and the chroma
workstream owns it. The default stays 1.0 (neutral).

### Coefficient term

The default is the R19 range-coder interval width (`AV2ENC_RATE_INTERVAL`
family, cherry-picked as `video_rate_interval.h`): the exact §8.3 symbol
sequence of the decided plan is walked through the §8.2.6 interval arithmetic
against a scratch copy of the live frame-start §8.2 CDF image. It replaces the
static §9.3 `scan_coefficient_bits` estimate at every leaf plan and inside the
`coeff_bits` lambda (luma searched plan, intra luma, chroma).

| Coefficient term | K @ 100.0 | Note |
|---|---:|---|
| static §9.3 context model | 84.77 (100.43 kbps) | no interval |
| **R19 interval width** | **85.10** | shipped |
| interval + live CDF rows in TX search | 84.95 | TX search now prices live rows |
| live CDF rows only | 84.5 (unstable, 104.0 kbps) | QP sampling flips decisions |

The interval estimate is not a free win: on the plain and tuned drift bars it
was negative (`docs/rate-model-r19-interval.md`). It becomes positive only
with the λ refit (0.2 → 0.18) and the error CSF, because the better estimate
was previously priced by a lambda calibrated to the old bias. That is the
same finding as pipe-perc: R and λ must move together.

## D model

For each coded sample, with `e = src - recon`:

```text
low3(x,y) = mean of e over the 3x3 window (edge-clamped inside the leaf)
D         = Σ w(x,y) * (e^2 + 1.5 * low3^2) / 256
```

`w` is Q8, 256 neutral. `AV2ENC_EFF_PERC_CSF2` adds a 7x7 low-pass term; it was
measured and is 0 by default (it over-emphasised slow error and lost about
0.4 K). The keyframe intra TX search uses the same CSF; switching it off for
the key alone (`AV2ENC_EFF_PERC_KEY_CSF=0`) costs 0.26 K at 100.012 kbps, so
the key keeps it. The same D is handed to `search_luma_tx` as `distortion_weight` plus
`lowpass_gain`, so the TX size/type/partition choice minimizes the same J as
the leaf. Chroma uses the same CSF with `AV2ENC_EFF_PERC_CHROMA` scaling.

## R term

Unchanged from pipe-perc for syntax: the R09 live-CDF probe mirrors
`is_inter`, `skip_flag`, `comp_mode`, `single_ref`, `single_mode`, `drl_idx`,
the eighth-pel shell MVD and BAWP, advances in coded order and prices
`-log2(p)`. The coefficient term is selectable as above. The probe cannot
model compound / TIP / AMVD / flex-precision leaves; the measured point has
`fallbacks=0`.

## λ

One weight prices the leaf and the TX candidate:

```text
lambda = leaf_rd_lambda(qindex, options) * AV2ENC_EFF_PERC_SCALE
J      = D + lambda * R
```

The shipped scale is 0.18. `AV2ENC_EFF_PERC_SCALE` reproduces the sweep.

## Search

The cheap inter candidates keep the historical SSE+static-syntax ranking; the
proxy winner plus the next cheapest distinct candidate (top-2) are re-scored
through the real path (predict → BAWP/inter-intra → TX search → chroma →
reconstruct → rate). Top-4 and top-8 produce identical streams to top-2 on
this clip (the distinct-candidate set is exhausted at two), so the extra cost
buys nothing. The intra candidate enters the same J. The drift refresh is
unchanged.

## Results

Host: Apple M4, dev `-O1` build, `--cpu-threads 8`, 60 frames. All measured
streams decoded 60/60 under `dav2d --strict 1`; the shipped point's
`--video-recon` payload equals the decoded payload byte for byte
(10,368,000 B).

| Point | QP | Bytes | kbps | JOD | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|---:|---:|
| nightly pin | 106 | 25,008 | 100.032 | 9.7334 | 3.4202 | 80.837 | 82.15 |
| drift wall | 110 | 25,146 | 100.584 | 9.7605 | 2.6562 | 81.250 | 83.94 |
| drift wall @100.0 | — | — | — | — | — | — | 83.80 |
| pipe-perc shipped | 119 | 24,994 | 99.976 | 9.7584 | 2.3425 | 82.106 | 84.85 |
| **eff-perc shipped** | **119** | **25,230** | **100.920** | **9.7713** | **2.2630** | **82.441** | **85.18** |
| eff-perc QP120 | 120 | 24,674 | 98.696 | 9.7625 | 2.2874 | 81.924 | 84.98 |
| **eff-perc @100.0** | — | — | — | — | — | — | **85.10** |

At the shipped point the component deltas vs pipe-perc are JOD +0.013,
Butteraugli −0.079 and SS2 +0.335. The 500 B rate delta is the price of the
better decisions at the same 100 kbps budget; the matched-rate comparison is
the 85.10 row.

## Encode cost

On the dev build the shipped point costs about the same as pipe-perc (top-2,
wavefront off, interval model on the decided plan). The interval estimate adds
one model walk per scored leaf plan; it is not a hot-path change relative to
the second full predict/TX/recon pass. This is a research gate, not a
production speed claim.

## Verification

* Gate off: default QP189 25,009 B `a1d0dba1…`, quality QP190 25,141 B
  `3984677b…`, nightly QP110 23,690 B `1b99127a…`, 720p pin 36,572 B
  `c4ba2813…` — all byte-identical.
* `AV2ENC_EFF_PERC=0` is byte-identical to unset.
* The merged pipe-perc point still reproduces: 24,994 B
  `650f2e3514978257c78f373f7bf99a50`.
* Gate on, shipped point: 25,230 B `7e8ab1c19d37a72cfd932e3ec42277a6`,
  `dav2d --strict 1` 60/60, encoder `--video-recon` payload equals the decoded
  payload byte for byte (10,368,000 B).
* `src/pipeline/video/gate_video_eff_perc.sh --bin build-dev/av2enc`: OK
  (pins, gate-off identity, pipe-perc, tuned point, 60/60 decode, recon,
  counters, env-only). Registered as `gate_video_eff_perc` in
  `tools/run_gates.sh`.
* `tools/test.sh --build build-dev`: 10/10.
* `tools/gate_video_nightly.sh`: OK.
* `src/pipeline/video/gate_video_qp_lambda.sh`: OK.

## Reproduce

```sh
REF=testdata/nasa_480x240_30fps_60f.y4m
BIN=build-dev/av2enc

# shipped point (defaults: scale 0.18, shape 0, CSF 1.5, rate 1, chroma 1)
AV2ENC_EFF_PERC=1 "$BIN" "$REF" -o /tmp/ep.ivf --video --frames 60 \
    --cpu-threads 8 --qp 119 --nightly \
    --video-drift-refresh-threshold 1.5 --video-drift-refresh-budget 3250

# lambda sweep
for s in 0.15 0.18 0.2 0.25 0.3; do
  AV2ENC_EFF_PERC=1 AV2ENC_EFF_PERC_SCALE=$s "$BIN" "$REF" ...
done

# D-shape ablation
AV2ENC_EFF_PERC=1 AV2ENC_EFF_PERC_DSHAPE=1 AV2ENC_EFF_PERC_DETAIL=0.1 ...
AV2ENC_EFF_PERC=1 AV2ENC_EFF_PERC_CSF=0 ...

# score
DAV2D=tools/bin/dav2d-macos python3 tools/quality.py --ref "$REF" \
    --ivf /tmp/ep.ivf --json /tmp/q.json
tools/bin/dav2d-macos --strict 1 -i /tmp/ep.ivf -o /tmp/ep.y4m
python3 tools/extra_metrics.py --ref "$REF" --test /tmp/ep.y4m --json /tmp/e.json
python3 tools/kpi.py --quality-json /tmp/q.json --extra-json /tmp/e.json
```

The README speed point uses `build-release-timing` (the `release-timing`
preset) with the same command: **7.5 fps** at QP119 on Apple M4 with 8 threads
(three runs 7.3-7.6 fps; stream 25,230 B / 100.92 kbps).

That is not an iso-QP comparison. At the same QP the gate is about **2.2x
slower** than the control: 6.8 vs 14.6 fps at QP106, and 7.6 vs 16.5 fps at
QP119. It re-scores the top-N inter candidates and prices the interval rate;
the QP119 point is faster than QP106 only because the quantiser is coarser.

## Env knobs (no CLI flags)

| Variable | Default | Meaning |
|---|---:|---|
| `AV2ENC_EFF_PERC` | off | master gate; any non-zero value enables |
| `AV2ENC_EFF_PERC_SCALE` | 0.18 | λ multiplier |
| `AV2ENC_EFF_PERC_DSHAPE` | 0 | 0 neutral source map + CSF, 1 pipe gradient, 2 multi-scale, 3 multi-scale+edge, 4 variance+edge, 5 edge-only |
| `AV2ENC_EFF_PERC_DETAIL` | 1.0 | source detail/edge gain multiplier |
| `AV2ENC_EFF_PERC_CSF` | 1.5 | 3x3 low/mid-band error gain |
| `AV2ENC_EFF_PERC_KEY_CSF` | =CSF | keyframe TX-search gain (0 = plain-SSE key ablation) |
| `AV2ENC_EFF_PERC_CSF2` | 0.0 | 7x7 lower-band error gain |
| `AV2ENC_EFF_PERC_CHROMA` | 1.0 | chroma distortion weight |
| `AV2ENC_EFF_PERC_TOP` | 2 | cheap inter candidates re-scored |
| `AV2ENC_EFF_PERC_RATE` | 1 | bit 0 interval width; bit 1 live CDF rows |

## Limits

* **The envelope is below 86; do not tune this mechanism further.** The
  luma-only D/λ/R refinement moves 84.85 → 85.10 at matched rate. The SVT gap
  is 3.32 K luma + 0.61 K chroma, and this pipeline changes only which
  existing leaves/modes/transforms the encoder emits; it cannot change the
  prediction class, the frame allocation or the quantiser floor.
* Measured on the CPU single-tile, low-delay flat-P `--nightly` surface.
  Tiles/B frames/TIP/compound/AMVD/flex-precision leaves fall back to the
  historical constants (the measured point has zero fallbacks). The CUDA
  device path is not wired.
* The interval coefficient model is exact arithmetic over the §8.3 symbol
  sequence but its CDF state is the frame-start image, not the leaf-start live
  state; the live-slice variant is available (`RATE` bit 1) and measured.
* No 720p tuning is claimed, and the 240p gain does not transfer: at QP225 the
  gate measures 293.7 kbps, JOD 7.1183, BA 32.4072, SS2 -14.899, **K 43.56**
  against the 45.61 pin; QP224 (299.2 kbps) is 44.07. The 720p pin is a
  gate-off identity check only.
