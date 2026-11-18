# Rate-model results (R01–R40 wave)

Status: diagnostic wave closed. None of the completed models beats the bars.
The purpose of this document is to stop the next person from repeating the
same experiment: the shipped rate constants are **RD controls**, not estimates
of the bit count.

Bars: 240p `--nightly` **82.15** at 100.0 kbps; tuned drift **83.80**;
720p pin JOD 7.3236 / **45.61**.

## Completed models

| ID | Model | Model error | K result |
|---|---|---|---|
| R03 | online adaptive, per-class actual/predicted | true ratio **0.88** vs shipped 0.4; syntax bias −25.7% → +1.1%; tsc2/3 +3.9/+2.4%; tsc4 +14.2% | **−0.95** plain, **−0.65** drift |
| R09 | exact syntax probe + analytic coefficients | syntax **+2.68 → −0.02 bits/leaf**; coefficients +13.02 → **+16.28** | **−2.06** plain, −0.56 drift, +0.31 720p (ripple) |
| R11 | MVD Laplacian per precision | MV-only error **−40…−60%**, incl. AMVD and flex precision | +0.27 plain (ripple), **−0.63** drift, +0.32 720p (ripple) |
| R12 | reference/skip pricing | skip median 3.52 → **1.60 bits**, bias +2.85 → +0.95; refs2 over-selection 4–6× removed | **−0.50** plain, −0.54 drift; refs2 81.38 → **82.06** (tie with refs1) |
| R13 | partition cost from split context | held-out MAE **7.30 → 0.63 bits**, bias −0.009 | **−0.78** plain, **−1.51** drift, −0.91 720p |
| R10 | exact coefficient syntax mirror (live CDF, real contexts) | p90 **19.5 → 3.87 bits**, bias **−7.09 → +0.09** | **−1.79** plain, **−1.29** drift; 720p flat |
| R19 | range-coder interval width (near-exact, 2.3–2.6× cheaper than trial coding) | inter residual bias +27.2 → +12.9 bits, p90 58.6 → 26.8; keyframe +232 bits (CDF restart) | **−1.14** plain, **−0.59** drift, −0.14 720p |
| R17 | 13→6→1 MLP, 85 Q16 weights (340 B) | p90 30.9 → 17.4 bits on the fitted QPs; dense intra 127 → 17; but the held-out 720p clip gets **no p90 gain** (7.50 → 8.88) | **+0.77** plain (ripple), **−0.45** drift, 720p +0.07 |
| R07 | EOB/nonzero/magnitude parametric fit per QP/mode | inter median 10.03 → **2.88 bits**; intra 138 → **16.75**; encode CPU −5.9% | **−0.48** plain, **−1.05** drift, 720p +0.10 |
| R24 | per-block empirical entropy + histogram overhead | inter median **1.49 b** (bias +0.5%), intra 23.3 b (+15%) because of the histogram cost; static §8.3 is better on intra (3.99 b) | **−0.97** plain, **−1.90** drift; 720p +0.11 (ripple) |
| R26 | two-component Laplacian mixture per plane/mode/TX size | P-frame p90 28.0 → 23.8 bits, bias 12.1 → 9.6; 7–12× tail ratio | −0.04 plain, **−0.95** drift, 0.00 720p |
| R35 | TX-type/size statistics per class | inter bias +9.53 → **+2.03 bits**, p90 32.2 → 10.0; 37–57% of plans change | **−0.49** plain, **−1.51** drift, −0.20 720p |
| R29 | Mannos–Sakrison CSF-weighted coefficient price | a price, not a predictor: no error reduction (median 3.37 → 4.30 bits) | **−0.42** plain, **−3.10** drift (+3.7% rate); **2.33× CPU** |
| R20 | conservative max(static, Laplacian) ensemble | static is already unbiased on inter; the max inherits the worse tail (p90 36.5 → 89.3); static alone is the best estimator | **−0.77** plain, **−0.33** drift, −0.29 720p |
| R33 | mode cost only, measured (mode, bsize, qindex) table | mode error 1.50 → **0.17 bits**; GLOBALMV +1.35 → 0.00, chroma +1.63 → 0.00 | **−0.17** plain, **−1.00** drift, +0.05 720p |
| R30 | 10-state §8.3 syntax state machine (240 rows vs 724 CDF rows) | 720p inter 8.05 → **0.03 bits** median; 240p inter 3.91 → 1.86; intra median slightly worse | **−0.23** plain, **−1.25** drift, +0.10 720p; **−5% CPU** |
| R32 | frame/SB lambda allocation from source activity + zero-motion SATD | SB bit-share TV **0.447** (240p), 0.37–0.40 (720p); mode is what carries rate, not activity | **−0.67** plain, **−0.95** drift, −0.04 720p; 43% of leaves change |
| R14 | per-superblock spatial calibration (left/above ratios) | leaf error 4.35 → **0.27 bits** median, p90 36 → 1.54 | **−1.87** plain, **−1.44** drift, −1.18 720p; 1.5–1.9× CPU |
| R22 | run-length/run-level sequential code | MAE 17.6 → **6.05 bits**, p90 44.6 → 18.1; only 0.9% of skip decisions change | **−0.24** plain, **−0.66** drift, −0.07 720p; in the partition probe **−2.4** |
| R27 | generalized Gaussian, fitted shape per mode/TX size | all shapes 0.24–0.94 (heavy tails); Laplacian bias +250 bits, GGD **+19.7** (p90 490 → 54); still worse than the static CDF walk | **−0.62** plain, **−1.28** drift, +0.15 720p |
| R40 | Bayesian model averaging, class-conditioned weights | inter median −8.0 → **+1.47 bits**, p90 78 → 10.5, bias +3.7; 25.4% of leaves change | **−0.55** plain, **−0.65** drift, 720p +0.01 |
| R36 | chroma-specific U/V model (plane contexts, fitted scales) | chroma median 1.00 → **0.19 bits**, p90 5.69 → 2.13, actual/pred 0.99 | **−0.66** plain, **−0.14** drift, −0.17 720p |
| R02 | context-correct static model (real contexts, no 0.4) | coeff median −1.72 → +0.59 b, p90 31.0 → 18.6; syntax +1.27 → +0.02 b | **−0.70** plain, **−2.30** drift |
| R01 | cost-only live entropy probe (exact `-log2(p)`, no bits emitted) | inter skip −3.50 → **0.00** bits; inter residual −6.33 → **0.00 / MAE 0.04**; intra untouched (+127.65) | **−0.16** plain, **−1.12** drift; 9× wall (serial SB) |
| R05 | exact trial-coded oracle at the real writer state (0/11,568 mismatches) | total bias −4.92 / MAE 5.53 bits; residual −14.45; GLOBALMV-skip mode +3.70 → 0.98 | **82.01** plain, **81.72** drift; 5.2× encode |
| R18 | live-CDF + KL adaptation cost on the RDOQ trellis | static +8.71 b median / +40.2 p90; carried seed +1.42 / +40.2; **live CDF +0.20 / +4.19**; live+KL +0.39 / +4.81 | **−0.34** plain, **−0.10** drift, 720p flat; +27% CPU |

All gate-off pins were byte-identical; every measured stream decoded 60/60 with
recon `cmp` where applicable.

## What the pattern means

1. **Absolute accuracy is not the objective.** In every completed branch the
   model error fell and K did not rise. R13 is the extreme: an 11× better model
   costs 1.5 K with drift.
2. **The constants are calibrated controls.**
   - `kModelRateScale = 0.4` compensates the distortion term/λ. The true ratio
     is 0.88; correcting it moves the operating point.
   - The 8-bit partition constant is anti-split pressure, not an estimate of a
     0.7-bit symbol. Replacing it cheapens splits and grows the tree.
   - The flat inter base (2.0/1.7 bits) and the `4 + log2(shell)` MV term are
     **priors** toward residual/NEWMV that were worth more than the bits they
     mis-priced.
3. **Only differential errors near a decision boundary change outcomes.** R12's
   reference term is the clean example: pricing a free second reference removes
   a 4–6× over-selection and cancels the refs2 penalty, but only to a tie.
   R10 is the stronger result: its coefficient correction *is* differential
   (inter +5.18 bits vs intra −149.3), so it changes 379 block-size, 549
   mode/MV and 228 skip decisions, and still loses 1.8 K. The constants are
   partly compensating the **SSE-vs-perceptual distortion mismatch**, not just
   lambda; exactly pricing the bits removes that compensation.
4. **Accuracy becomes useful in three situations only**: pricing a *new*
   candidate (compound, extra reference) that has no calibrated constant;
   transferring to another operating point (the 0.4 fit is 240p/100 kbps);
   and a **joint λ refit** where R and λ are re-optimised together.

## The metric that ranks models

Bit error cannot rank them. Use decision regret against an exact-bit oracle:

```
oracle c* = argmin_c D(c) + λ·R_true(c)
model  ĉ  = argmin_c D(c) + λ·R_model(c)
regret    = J_true(ĉ) − J_true(c*)
```

Rank by pick accuracy and regret, then K-test only the top few. The
`rate-oracle` branch builds this dump and analyzer; without it, another model
is another calibration the pipeline will absorb.

## Consequence for the pipeline work

Do not swap R alone. A pipeline change must move **D, R, λ, candidates and the
search together**, or the calibration absorbs the difference. That is the design
of the `pipe-perc` and `pipe-ref` branches.
