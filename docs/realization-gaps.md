# Realization-gap loop: eff-perc and filter/allocation at 240p/100 kbps

Status: **measured ledger, no kept patch.** Branch `exp/loop-eff`, base
`832e1a8b` (main). Every attempt is one contained, default-off
implementation change (env gate or an explicit `eff_perc::enabled()` branch);
all negative/flat patches were reverted, and the tree at the end of the loop is
byte-identical to the base for every gated-off stream.

21 attempts were run against the two measured realization gaps. None reached
the keep bar (sign-consistent **>= +0.15 K** at matched rate). K is
`tools/kpi.py` through `tools/oracle.py` with the AVM decoder, interpolated
linearly in rate to 100.000 kbps over the attempt's own bracket. Fixture:
`testdata/nasa_480x240_30fps_60f.y4m`, 60 frames, 8-bit 4:2:0, single tile,
`--cpu-threads 8`.

## Baselines

| recipe | QPs | bytes / kbps / K | matched K @100 |
|---|---|---|---:|
| eff-perc (`AV2ENC_EFF_PERC=1`, drift 1.5/3250) | 118 | 25,789 / 103.156 / 85.046 | — |
| | 119 | 25,230 / 100.920 / 85.180 | — |
| | 120 | 24,674 / 98.696 / 84.985 | **85.099** (3-pt) |
| eff-perc, 2-point method used by every attempt (118/120) | | | **85.0025** |
| best filter/allocation arm (`--video-key-qp-offset -26`, drift 2.0/2500) | 106 | — / 103.160 / — | — |
| | 108 | 25,028 / 100.112 / 82.996 | — |
| | 110 | 24,457 / 97.828 / 82.191 | **82.9568** (108/110) |

The 3-point and 2-point eff baselines differ by 0.10 K because the 2-point
bracket skips QP119; every delta below is computed against the same 2-point
method as its arm, so the deltas are comparable to each other.

## Diagnosis

### eff-perc: the 5 K is the f0-32 head, and it is not a keyframe loss

Source-replacement segment ceilings for the shipped QP119 stream
(`tools/oracle.py`, avmdec + CVVDP/BA/SS2):

| segment | K with source replacement | dK |
|---|---:|---:|
| key f0 | 85.305 | **+0.125** |
| f0-32 | 90.197 | **+5.017** |
| f33-43 | 87.300 | +2.119 |
| f44-59 | 88.995 | +3.815 |

The 90.12 ceiling is the f0-32 row. The key picture is 1/60 of the metric and
its own replacement is +0.125, so the gap is not keyframe quality. The head
P frames are: they inherit the key's QP92 reconstruction (46.05 dB) and track
it to 44.92 dB with 1,433 B over 32 frames, while SVT reaches 49.78 dB with
2,619 B. The tail is the second-largest row (+3.8) and 69% of its error is
already coded residual (`docs/luma-audit.md` §4.3), i.e. rate-bound.

Four decision families were tested against that attribution.

* **Head/allocation.** Charging the zero-MV skip when the reference is wrong
  (unsigned RMS gate) spends rate in the head and loses 0.50 K. Reserving
  drift budget for the late frames (linear pacing) loses 0.21 K even though it
  moves 2 of the 3.25 kB into f46-59; the greedy refresh spend is locally
  optimal. Restricting the refresh frame's forced residual to one MV class
  loses 1.4-2.3 K. The head's problem is that its 45 B/frame cannot repair a
  4.9 dB deficit, not that the budget is in the wrong frames.
* **Coefficient/TX residual.** High-frequency weighting in the inter trellis,
  trellis lambda refits (0.4×, 2.5×), a detail-aware large-TX penalty
  (0.5/2.0 bits per 32×32), and a measured rate-model bias correction are all
  flat or negative. The one positive signal is the cheaper trellis lambda
  (0.4×) at **+0.12 K**, below the bar and inside the bracket.
* **Rate model.** Using the static §8.3 model for intra/key decisions (the R19
  interval model over-prices key residual by ~190 bits/leaf) saves 117 B on
  the key and is **+0.06 K** at matched rate — real but small.
* **Leaf/partition consistency.** Pricing the key TX partition with the same
  error CSF as its parts loses 0.19 K; scaling the key TX lambda onto the
  eff λ point loses 0.13 K. The shipped plain-SSE key partition is better on
  this clip.

**Verdict.** The remaining 5 K is per-frame coding efficiency in the head
P frames (prediction + coefficient/transform granularity), not a decision
or allocation error. The decision-level queue for this mechanism is
exhausted at the measured λ/QP point.

### filter/allocation: the ceiling is not reachable by filter or refresh decisions

The best arm's 88.964 ceiling is the f44-59 source replacement — the same
tail that the arm leaves as uncoded skip error (nightly tail: 91% of error in
skip leaves, `docs/luma-audit.md` §4.3). Filters and refresh placement cannot
remove prediction error; they can only re-shape it.

Measured, all negative against the 82.9568 base:

* low/mid-band (CSF) drift trigger: −0.80; it fires earlier and spends
  2 kbps more at QP108.
* predictive triggers (reference vs current / next source): −0.83 / −1.20.
* two-consecutive-frame run requirement: −1.14 (delays the first repair).
* strict budget reserve (no last-frame overrun): −0.42.
* refresh residual restricted to one MV class: −2.28 (zero-MV leaves must be
  repaired).
* refresh-frame residual skip threshold: −0.73.
* per-frame refresh λ: −0.04 (eff side).

Diagnostic composition: `AV2ENC_EFF_PERC=1` on the best arm's own flags
(key `-26`, drift 2.0/2500) is **84.84 K** at matched rate, *below* the eff
recipe's 85.00 — the arm's +0.80 does not stack with the decision metric,
because eff-perc's own CSF/λ already makes a better version of the same
reallocation (and keeps the 1.5/3250 refresh spend).

**Verdict.** The filter decision space is closed (perceptual CDEF negative,
CDEF/CCSO/Wiener/GDF parameter sets negative or inert, deblock has no decision,
CDEF-on-skip is rejected by the tile writer). The refresh planner is at a local
optimum at this budget (five placement variants, all negative). The 6.0 K
realization gap on this lane is the same head/tail coding-efficiency gap that
eff-perc measures; it is not a filter or allocator decision.

## Ledger

`ΔK` is matched-rate K minus the same-method baseline (eff 85.0025 / combo
82.9568). "flat" = |ΔK| < 0.15 or sign-inconsistent.

| # | target | patch (default off) | ΔK | verdict |
|---:|---|---|---:|---|
| 1 | eff | zero-MV skip charged on an unsigned reference deficit (`AV2ENC_SKIP_DRIFT_REF`, RMS ≥ 2 LSB) | −0.50 | negative |
| 2 | eff | inter RDOQ distortion weight on ring ≥6 (`AV2ENC_RDOQ_HF=0.5`) | −0.05 | negative |
| — | eff | calibration: `RDOQ_HF=0` is effectively gate-off (no weight applied) | +0.02 | bracket check |
| 3 | eff | drift trigger on the low/mid-band CSF PSNR (`AV2ENC_DRIFT_CSF=1.5`) | −0.15 | negative |
| 4 | eff | linearly paced drift budget (`AV2ENC_DRIFT_PACE`) | −0.21 | negative |
| 5 | eff | key TX-partition trial cost on the error CSF (same D as the parts) | −0.19 | negative |
| 6 | eff | static §8.3 rate for intra/key leaves, interval for inter (`EFF_PERC_KEY_RATE`) | +0.06 | flat |
| 7 | eff | refresh frames force residual only above a leaf RMS (`AV2ENC_REFRESH_SKIP_T=1.5`) | +0.06 | flat |
| 8 | eff | key TX/partition lambda on the eff λ scale | −0.13 | negative |
| 9 | eff | refresh-frame λ weight (`EFF_PERC_REFRESH_SCALE=0.7`) | −0.04 | flat |
| 10 | eff | coefficient trellis lambda ×0.4 (`AV2ENC_RDOQ_LAMBDA_SCALE`) | +0.12 | flat (near miss) |
| 11 | eff | coefficient trellis lambda ×2.5 | −0.19 | negative |
| 12 | eff | detail-aware penalty on parts ≥32×32 (`AV2ENC_TX_DETAIL=0.5`) | +0.06 | flat |
| 13 | eff | same, 2.0 | −0.08 | negative |
| 14 | eff | measured inter rate-model bias correction, 13 bits (`EFF_PERC_RATE_BIAS`) | −0.26 | negative |
| 15 | filter/allocation | CSF drift trigger on the best arm | −0.80 | negative |
| 16 | filter/allocation | refresh residual skip threshold on the best arm | −0.73 | negative |
| 17 | filter/allocation | predictive trigger, reference vs current source | −0.83 | negative |
| 18 | filter/allocation | predictive trigger, reference vs next source | −1.20 | negative |
| 19 | filter/allocation | require two consecutive over-threshold frames | −1.14 | negative |
| 20 | filter/allocation | refresh residual only on non-zero-MV leaves | −2.28 | negative |
| 21 | filter/allocation | reserve the last refresh cost in the budget check | −0.42 | negative |

Each patch was built in `build-dev`, measured on the attempt's own QP ladder,
and reverted (`git checkout -- <files>`). No preset, default, CLI flag, test
or syntax changed; no third-party source was read.

## Verification (after reverting every patch)

* `src/pipeline/video/gate_video_eff_perc.sh --bin build-dev/av2enc`: OK
  (pins, pipe-perc, tuned point, 60/60 decode, recon).
* `tools/gate_video_nightly.sh --bin build-dev/av2enc`: OK (expansion, pins,
  strict decode, visible recon, BAWP).
* `tools/test.sh --build build-dev`: 10/10.
* Every scored stream decoded through `tools/oracle.py` (avmdec).
* 4:2:0 only; no push/merge.

## Blockers

* No kept patch: the keep bar is not met by any decision-level change tested.
* Both gaps need per-frame coding efficiency (prediction class, coefficient
  granularity, transform/entropy), not another tree/refresh/rate decision. The
  measured rank-1/rank-2 mechanisms of `docs/luma-audit.md` are the only
  remaining class, and their decision-level forms measured negative here and in
  `docs/tool-improvement-queue-2.md` §2/§9.
* The filter/allocation lane cannot use its 88.964 ceiling as a target: the
  ceiling is a pooling bound on tail frames whose error is uncoded prediction
  error, and the only measured mechanism that reduces it is the eff-perc
  decision metric, which has its own (85.0) point.

## Reproduce

```sh
REF=testdata/nasa_480x240_30fps_60f.y4m
BIN=build-dev/av2enc

# eff baseline (3 points)
for q in 118 119 120; do
  AV2ENC_EFF_PERC=1 "$BIN" "$REF" -o /tmp/eff$q.ivf --video --frames 60 \
    --cpu-threads 8 --qp $q --nightly \
    --video-drift-refresh-threshold 1.5 --video-drift-refresh-budget 3250
done
python3 tools/oracle.py --ref "$REF" \
  --ivf eff@118=/tmp/eff118.ivf --ivf eff@119=/tmp/eff119.ivf \
  --ivf eff@120=/tmp/eff120.ivf --target-kbps 100 --json /tmp/eff.json

# filter/allocation base
for q in 108 110; do
  "$BIN" "$REF" -o /tmp/d$q.ivf --video --frames 60 --cpu-threads 8 --qp $q \
    --nightly --video-key-qp-offset -26 \
    --video-drift-refresh-threshold 2.0 --video-drift-refresh-budget 2500
done
python3 tools/oracle.py --ref "$REF" --ivf d@108=/tmp/d108.ivf \
  --ivf d@110=/tmp/d110.ivf --target-kbps 100 --no-segments --json /tmp/d.json
```
