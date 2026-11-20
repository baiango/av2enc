# Rate-model protocol (RD search)

Shared rules for the rate-model branches. Every model is env-gated, default
off, and judged only on K at matched rate. No model ships on a single QP point.

## Bars

| Point | Flags | QPs | Bar |
|---|---|---|---|
| 240p plain | `--fast-decode` | 105/106/107 | **K 82.15** at 100.0 kbps |
| 240p drift | `--fast-decode --video-drift-refresh-threshold 1.5 --video-drift-refresh-budget 3250` | 109/110/111 | **K 83.80** |
| 720p pin | pin recipe | 216/217/218 | **K 45.61**, JOD 7.3236 |

## Measurement

```sh
REF=testdata/nasa_480x240_30fps_60f.y4m
BIN=build-dev/av2enc
"$BIN" "$REF" -o /tmp/x.ivf --video --frames 60 --cpu-threads 8 --qp Q <flags>
tools/bin/dav2d-macos --strict 1 -i /tmp/x.ivf -o /tmp/x.y4m --muxer yuv4mpeg2
DAV2D=tools/bin/dav2d-macos python3 tools/quality.py --ref "$REF" --ivf /tmp/x.ivf --json /tmp/q.json
python3 tools/extra_metrics.py --ref "$REF" --test /tmp/x.y4m --json /tmp/e.json
python3 tools/kpi.py --quality-json /tmp/q.json --extra-json /tmp/e.json
```

- Bracket every arm; report both points and the interpolation. The 240p curve
  ripples by ±1.6 K per adjacent QP.
- One CVVDP job at a time per agent, `nice -n 10`. The host runs many branches;
  metric jobs serialize, so wall time is not a result.
- A gain must clear the bar by more than the ripple to be considered.

## Model error first

Every branch must instrument predicted vs **actual** range-coder bits per leaf
(by mode and block size) and report median/p90/signed bias before changing the
decision. A model that does not reduce the error is not interesting even if K
moves; a model that reduces error but not K is a finding about the decisions.

## Rules

- Gate off: pins byte-identical — 25,009 `a1d0dba1`, 25,141 `3984677b`,
  23,690 `1b99127a`, 24,113 `7f03eb6e`, 720p 36,572 `c4ba2813`.
- Gate on: 60/60 decode, `--video-recon` `cmp`-identical, `tools/test.sh` 10/10,
  `gate_video_nightly`, `gate_video_qp_lambda`.
- Do not add a new tuning constant to make a model fit; remove or replace the
  existing ones (`inter_syntax_bits`, `kModelRateScale`, partition constants).
- No CLI flags; env gate only (`AV2ENC_RATE_*`).
- One branch per model, no push, no merge. Commit the code and a doc under
  `docs/`.

## Models

| ID | Branch | Gate | Model |
|---|---|---|---|
| R01 | `exp/rate-model` | `AV2ENC_ACTUAL_RATE` | exact `-log2(p)` from the live adaptive CDF |
| R02 | `exp/rate-ctx` | `AV2ENC_RATE_CTX` | static/carried CDF with real contexts, no 0.4 |
| R03 | `exp/rate-adapt` | `AV2ENC_RATE_ADAPT` | per-frame actual/predicted calibration by class |
| R04 | `exp/rate-learned` | `AV2ENC_RATE_LEARNED` | offline-fit table/regression |
| R05 | `exp/rate-exact` | `AV2ENC_RATE_EXACT` | scratch entropy-encode, exact bits |
| R06 | `exp/rate-laplace` | `AV2ENC_RATE_LAPLACE` | analytic Laplacian coefficient rate |
| R07 | `exp/rate-eob` | `AV2ENC_RATE_EOB` | EOB/nonzero/magnitude parametric model |
| R08 | `exp/rate-geom` | `AV2ENC_RATE_GEOM` | per-context geometric fit, closed form |
| R09 | `exp/rate-hybrid-s` | `AV2ENC_RATE_HYBRID_S` | exact syntax + analytic coefficients |
| R10 | `exp/rate-hybrid-c` | `AV2ENC_RATE_HYBRID_C` | exact coefficients + constant syntax |
| R11 | `exp/rate-mv` | `AV2ENC_RATE_MV` | MVD entropy model (Laplacian per precision) |
| R12 | `exp/rate-refskip` | `AV2ENC_RATE_REFSKIP` | priced reference index and skip flag |
| R13 | `exp/rate-partctx` | `AV2ENC_RATE_PARTCTX` | partition cost from split context |
| R14 | `exp/rate-sb` | `AV2ENC_RATE_SB` | per-superblock spatial calibration |
| R15 | `exp/rate-2pass` | `AV2ENC_RATE_2PASS` | fast first pass measures class averages |
| R16 | `exp/rate-rls` | `AV2ENC_RATE_RLS` | online recursive least squares |
| R17 | `exp/rate-mlp` | `AV2ENC_RATE_MLP` | tiny fixed-point MLP |
| R18 | `exp/rate-kl` | `AV2ENC_RATE_KL` | price the CDF adaptation cost |
| R19 | `exp/rate-interval` | `AV2ENC_RATE_INTERVAL` | range-coder interval-width estimate |
| R20 | `exp/rate-ensemble` | `AV2ENC_RATE_ENSEMBLE` | conservative max of two models |

## Second wave (R21–R40)

| ID | Branch | Gate | Model |
|---|---|---|---|
| R21 | `exp/rate-bitplane` | `AV2ENC_RATE_BITPLANE` | significance/refinement bit-plane probabilities |
| R22 | `exp/rate-runlength` | `AV2ENC_RATE_RUNLENGTH` | zero-run + level run-length model |
| R23 | `exp/rate-syntaxtree` | `AV2ENC_RATE_SYNTAXTREE` | binary sig/gt1/gt2/parity decision tree |
| R24 | `exp/rate-entropy` | `AV2ENC_RATE_ENTROPY` | per-block empirical entropy + overhead |
| R25 | `exp/rate-kt` | `AV2ENC_RATE_KT` | Krichevsky–Trofimov universal code |
| R26 | `exp/rate-mixture` | `AV2ENC_RATE_MIXTURE` | two-component Laplacian mixture |
| R27 | `exp/rate-ggd` | `AV2ENC_RATE_GGD` | generalized Gaussian with fitted shape |
| R28 | `exp/rate-dr` | `AV2ENC_RATE_DR` | rate-distortion bound at the quantiser step |
| R29 | `exp/rate-perceptual` | `AV2ENC_RATE_PERC` | CSF/BA-weighted coefficient rate |
| R30 | `exp/rate-statemachine` | `AV2ENC_RATE_SM` | §8.3 symbol state machine, small tables |
| R31 | `exp/rate-neighbor` | `AV2ENC_RATE_NEIGHBOR` | left/above coefficient context |
| R32 | `exp/rate-frame` | `AV2ENC_RATE_FRAME` | frame complexity -> per-SB lambda allocation |
| R33 | `exp/rate-mode` | `AV2ENC_RATE_MODE` | mode cost only, measured table |
| R34 | `exp/rate-skip` | `AV2ENC_RATE_SKIP` | skip vs residual binary cost only |
| R35 | `exp/rate-txtype` | `AV2ENC_RATE_TXTYPE` | per-TX-type symbol statistics only |
| R36 | `exp/rate-chroma` | `AV2ENC_RATE_CHROMA` | separate U/V rate model |
| R37 | `exp/rate-partdepth` | `AV2ENC_RATE_PARTDEPTH` | partition cost by depth/node count |
| R38 | `exp/rate-residual` | `AV2ENC_RATE_RESIDUAL` | regressor on SATD/energy/sparsity |
| R39 | `exp/rate-colocated` | `AV2ENC_RATE_COLOC` | previous frame co-located bits scaled by activity |
| R40 | `exp/rate-bma` | `AV2ENC_RATE_BMA` | Bayesian model averaging, online weights |

## Finding from R03 (online adaptive)

The measured actual/predicted ratio is **≈0.88**, not the shipped `0.4`.
Calibrating the model to that accuracy loses K (82.05 -> 81.10 plain,
83.80 -> 83.14 with drift) while reducing the model error. The 0.4 scale is an
RD-tradeoff calibration that compensates the distortion term/lambda, not a
bit-bias correction.

Consequence for the other branches: a model that changes the **absolute** rate
scale must recalibrate lambda jointly, or it will move the operating point and
lose. Models that improve the **relative ordering** of candidates (mode-only,
skip-only, ref/skip, partition context, TX-type) are the ones that can help
without a joint lambda change.

## Finding from R13 (partition context)

The partition constants are **8 bits deliberately**: they are rate-pressure
terms, not bit estimates. The measured cost is ~0.7 bits (held-out MAE 7.30 ->
0.63, bias -0.009). Replacing the constants makes splits cheaper: 5.4% of nodes
change, fixed-QP rate +0.95%, and K falls -0.78 plain / -1.51 drift / -0.91 at
720p.

Together with R03 (rate ratio 0.88 vs shipped 0.4) and R12 (skip 3.5 -> 1.6
bits), the pattern is consistent: the shipped constants are RD controls
calibrated against the distortion term. Absolute-accuracy models move the
operating point and lose. Only relative-ordering corrections, or a joint
rate+lambda recalibration, can use a better rate model without that penalty.
