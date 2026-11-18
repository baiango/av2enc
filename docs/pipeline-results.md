# Pipeline results and K oracles

Rule: a pipeline must announce its K oracle before tuning (`docs/k-oracle.md`).
The internal wall is 83.80 at 240p/100 kbps; the updated oracle also requires a
bound against the Stage-1 90.0 target.

| Pipeline | Oracle bound | Achievable envelope | Realised | Verdict |
|---|---:|---:|---:|---|
| `pipe-metric-alloc` | **84.53** plain / **85.81** tuned (independent-frame knapsack, JOD=10); 83.19 / 84.62 with measured JOD | **82.12** / **82.97** | 81.15 / 82.89 | stopped: bound above wall, achieved below |
| `pipe-dp` | **86.53** (sequential DP, JOD=10) / **85.33** (JOD=9.76) | **84.40** | **84.59** @ 99.20 / **84.44** @ 100.93; iso 100 **84.52** | **positive vs 83.80; below Stage-1 90.0** |

## `pipe-metric-alloc` finding

The independent-frame allocation bound clears 83.80, but the real allocation
does not. Frames are not independent:

- per-frame BA curves are non-monotone on 47/60 (plain) and 44/60 (tuned) frames;
- a QP change moves the reference chain and the drift-refresh schedule
  (11/9/7 refreshes become 7);
- one P-frame step flips a near-tie (82.24 -> 81.63 K).

The next allocator must carry a **reference-quality state** through the frames
(sequential DP), not solve a knapsack over independent per-frame curves.

## `pipe-dp` finding

The sequential allocator carries a reference-quality state and a refresh
action, so the frame costs are conditioned on the chain the encoder actually
produces. At 240p/100 kbps:

* DP bound **86.53** (`JOD=10`) / **85.33** (measured max JOD);
* best whole-chain envelope **84.40** at 100 kbps;
* realised **84.52** K at the iso-rate point (84.59 at 99.20 kbps, 84.44 at
  100.93 kbps).

The state is four drift buckets times the previous refresh flag. The DP
refreshes 42/58 inter frames and raises the base QP to 116, spending the rate on
hot-band residuals instead of a lower base QP. The knapsack's tuned arm left
the hot tail at BA 5.65; the DP holds it at 3.48. The prediction/realisation
gap is 0.73 K, about half the knapsack's tuned gap. The updated K oracle lists
the perceptual RD core at 85.10; the BA refresh (84.81) was later removed with its third-party metric port, so `pipe-dp` is +0.73 over the 83.80
wall but 0.29 below the current internal best. The full design, probe counts,
state trace and reproduction are in `docs/pipe-dp.md`.

## `eff-joint` (affordable joint search)

| Budget | K @100 | CPU |
|---|---:|---:|
| shapes=1, top=1 | **84.59** | 0.94x |
| shapes=2, top=1 | 84.43 | 1.48x |
| exhaustive | 84.65 | 6.4x |
| + BA refresh placement | 84.48 | — |

Envelope 84.59 < 86: stopped. The search budget is not the lever: the cheapest
budget is within 0.06 K of exhaustive, and composition with the BA refresh is
not additive (84.48, below BA alone 84.81).

## Efficiency workstreams

| Branch | Envelope | Finding |
|---|---:|---|
| `eff-joint` | 84.59 | search budget is not the lever (0.94x CPU is within 0.06 K of exhaustive) |
| `eff-twosided` | 84.49 | predictor wins at the same QP (K 83.29 -> 84.65, BA 2.455); loses only to the reordered-anchor penalty (+6.7% stream) |
| `eff-chroma` | 84.91 | decoded hybrid: chroma is 0.61 K of the 3.93 gap, **luma is 3.32 K**; the SS2/chroma-MSE correlation is not causal |
| `eff-hierb` | 84.40 | two-sided + BAWP composes (+0.33 same-QP, +0.50 iso); the pyramid still trails the P chain by 0.45 because of anchor cost |
| `eff-perc` | **85.10** | best internal point: error-domain CSF + interval rate + λ refit; verified 85.18 at 100.92 kbps (BA 2.263) |

The hybrid method (swap one plane from the reference encoder's decoded output)
attributes K without re-encoding; it is the right tool for ranking mechanisms.
