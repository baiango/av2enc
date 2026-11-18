# K oracle for pipeline experiments

> The automated version of this oracle is `tools/oracle.py`; the definition,
> the standing candidate queue and the go/no-go rule are in
> `docs/oracle-search.md`.

Two walls:

- **Internal best: K 83.80** at 240p/100 kbps (tuned drift), now **85.10** with
  the perceptual RD core (`AV2ENC_EFF_PERC=1`).
- **Stage 1 wall: K 90.0.** SVT-AV1 v4.1.0 preset 0 at 100.3 kbps is
  **K 88.83** (JOD 9.830, BA 1.51, SS2 88.30). A pipeline that cannot reach 90
  does not compete, regardless of its gain over 83.80.
- **Stage 2 wall: K 95.0** (stretch).

**95 feasibility (2026-11-18).** No measured first-party mechanism exceeds
90.2 at 100 kbps; the conventional AV2 class crosses 95 only at ~0.6–0.8 Mbps
(SVT 777.6 kbps; AVM 188.9 kbps reaches 92.76). With AVM's BA 1.29, perfect
JOD/SS2 would cap at 92.49, and 95 requires BA <= 0.44. `docs/target-95.md`
concludes that 95 needs a new codec class, not another AV2 tool/flag sweep.
The immediate target is SVT parity (88.8), then AVM parity (~90).

Component requirements at SVT's JOD/SS2:

| Target | JOD | BA | SS2 |
|---|---:|---:|---:|
| SVT p0 | 9.830 | 1.51 | 88.30 |
| 90 | 9.83 | **≈1.2** (better than SVT) | 88.3 |
| 95 | 9.90 | **≈0.44** | 92 |

Stage 1 already requires beating SVT on Butteraugli; stage 2 requires a
different class of prediction/quantisation, not tuning.

Every pipeline must announce its bound against **90**, not just 83.80, before it
reports a tuned result.

## Definition

For a pipeline `P` with its knobs `θ`, `K_P(θ)` is the measured K at 100 kbps.
The oracle `K*_P = sup_θ K_P(θ)`. Exact sup is too expensive, so report the
strongest of these three bounds:

| Bound | What it answers | How |
|---|---|---|
| **Achievable envelope** | can this pipeline reach the wall at all? | best K over a coarse sweep of its continuous knobs (λ, allocation), at matched rate. This is a **lower** bound on `K*` and the primary number. |
| **Allocation bound** | could better frame/bit allocation save it? | per-frame R-D curves from several QPs, convex hull, knapsack at 100 kbps, metric measured on the assembled stream. Upper bound if frames are treated independently. |
| **Decision bound** | could better per-block decisions save it? | exact-bit oracle picks (R05 probe) at the same D; measure K of the oracle-chosen stream. Upper bound for that candidate set. |

## Required report

1. The bound value(s) at 240p/100 kbps, with the method.
2. `K*` vs the wall: **below** → stop, do not tune; **above** → state the headroom.
3. The knob sweep or allocation trace that supports the bound, so the bound is
   reproducible.

## Rules

- Do not tune inside a pipeline whose achievable envelope is below 83.80.
- Do not compare a tuned point to 82.15 if the pipeline includes drift; the wall
  is 83.80.
- The oracle is a bound on K, not on J. Decision regret in J is diagnostic, not
  a K oracle; it can say "the decisions are near-optimal" but not "the pipeline
  can reach 83.80".

## Measured pipeline oracles

| Pipeline | Gate | Envelope (240p/100 kbps) | Allocation ceiling | vs wall 83.80 |
|---|---|---:|---:|---:|
| Deep inter anchor (`docs/pipe-deep.md`) | `AV2ENC_PIPE_DEEP=1` | **84.02** (interpolated at 100.0 from depth −14: 99.576 kbps K 84.00, 103.004 kbps K 84.14) | **87.69** (all 59 inter frames at the key's QP 84, 236.6 kbps; rate-free quality ceiling, no allocation) | above 83.80 but **below the 84.81 best and the 90 wall**; do not tune |
| Extended perceptual RD core (`docs/eff-perc.md`) | `AV2ENC_EFF_PERC=1` | **85.18** (85.10 interpolated to 100.0; QP119 100.920 kbps) | not computed (no frame-allocation knob; drift recipe fixed) | **below the 86 target by 0.90**; luma-only D/R/λ envelope, stop and report the ceiling |

The deep-anchor ceiling is the strongest bound this mechanism admits: 87.69
with *every* inter frame at the key depth and 2.4x the rate. The 90 wall needs
BA ≈1.2 at SS2 88.3; the inter path measures BA 1.835 / SS2 86.814 even with
no rate constraint, so anchor depth alone cannot reach it.
