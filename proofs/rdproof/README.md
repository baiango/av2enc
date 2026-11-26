# How far is the RD pipeline from bit-optimal? A Lean 4 proof

Machine-checked answer in `RdProof.lean`. Build:

```sh
cd proofs/rdproof
elan toolchain install     # Lean 4.36.0, no Mathlib
lake build                 # 0 errors, 0 sorry
lake env lean Check.lean   # axioms audit (see below)
```

## The honest split

**Not provable here.** Global bit-optimality of the encoder. The space is
partition tree x mode x MV x transform x level; the search is heuristic, and
proving the C++ implements these DPs would need a refinement proof of the
3,913-line leaf coder. Anyone claiming "bit-optimal" without that is claiming
something they have not shown.

**Provable, and proved.** Two things that together bound the distance:

1. **The decision core is exact inside its model.** The trellis/partition DP
   returns a value below *every* candidate path, so the gap the DP leaves is
   zero. `dp2_optimal` states this for the two-stage, two-state trellis — the
   dependency shape that matters for RDOQ, because a level's rate depends on the
   previous level.
2. **The realised stream's distance from the optimum is bounded by measurable
   quantities.** `pipeline_gap_bound`:

   ```
   realised_cost - optimum  ≤  (chosenModel - LB) + lam * n * e
   ```

   where, on a real encode, every right-hand term is measurable:
   * `chosenModel` — the cost the encoder believed it paid (decision trace),
   * `LB` — a lower bound on the optimum (relaxation),
   * `e` — the per-symbol rate-model error, from modelled-vs-actual bits,
   * `lam`, `n` — the price and the symbol count.

So the deliverable is a **proof-carrying gap report**, not a claim of
optimality: "we are at most `(chosen - LB) + lam*n*e` from optimal", with the
two inputs that are not proved named as hypotheses and measured instead.

## Theorems

| theorem | statement |
|---|---|
| `lmin_le` | the minimum over a candidate set is `≤` every evaluated candidate |
| `dp2_optimal` | the trellis DP is `≤` every admissible path → modelled gap = 0 |
| `gap_nonneg` | the gap is non-negative |
| `gap_le_of_lb` | a lower bound on the optimum bounds the gap |
| `realised_gap_bound` | in-model gap plus model error bounds the realised gap |
| `exact_model_is_optimal` | exact model + chosen attains the optimum ⟹ realised cost **is** the optimum |

That last theorem carries the subtlety worth knowing: `opt ≤ chosen` alone only
gives `chosen - opt ≥ 0`. Optimality needs the chosen plan to *attain* the
optimum (`hOpt`), which is exactly the assumption a heuristic search cannot
discharge for the full space — only per superblock, by exhaustive enumeration.
The first draft of this file stated it without `hOpt` and Lean refused to prove
it, which is the proof doing its job.

## Axioms audit

```
'RdProof.lmin_le'                depends on axioms: [propext, Quot.sound]
'RdProof.dp2_optimal'            depends on axioms: [propext, Quot.sound]
'RdProof.gap_le_of_lb'           depends on axioms: [propext, Quot.sound]
'RdProof.realised_gap_bound'     depends on axioms: [propext, Quot.sound]
'RdProof.pipeline_gap_bound'     depends on axioms: [propext, Quot.sound]
'RdProof.exact_model_is_optimal' depends on axioms: [propext, Quot.sound]
```

No `sorryAx`: every theorem is complete.

## What this does NOT give you

* It does not connect the Lean model to the C++ — a gap between the proof and
  the binary remains, and closing it would need a refinement proof or a
  differential test harness that checks the C++ realises the same recurrence.
* It does not prove the candidate set contains the true optimum. That needs the
  per-superblock exhaustive enumeration (`docs/negative-results.md`, the
  partition-probe work) to establish `hOpt` for small instances.
* `e` (rate-model error) is an input, not a result. If it is large, the bound is
  loose and honest: it says the pipeline is at most that far off, which on the
  measured 720p streams is where the real headroom lives.

## Oracle ceiling on a 60-frame stream: the gap is in the P chain

Measured 2026-11-26 with `tools/oracle.py` on `testdata/space8k_1440x720_60f.y4m`
(60 frames, 1440x720), QP ladder 196/200/204/208, matched at **294.5 kbps**.

Base **K = 44.886**. Source-segment replacement ceilings (make that range
perfect, rate-free upper bound):

| segment | K | dK |
|---|---:|---:|
| key f0 | 46.697 | +1.81 |
| f0-32 | 63.164 | **+18.28** |
| f33-43 | 50.401 | +5.52 |
| f44-59 | 52.062 | +7.18 |

**This bounds the "how far from optimal" question in metric units, and the
answer is: far, but only on the P chain.** The key is within ~1.8 JOD of its
ceiling; the inter path is ~18 JOD from it.

That reframes the Lean work honestly. `pipeline_gap_bound` is correct, but its
value depends entirely on `LB`, and the oracle shows the distance is large
enough that a tight bound is not what is needed: the P-first frames are ~18 JOD
from a rate-free ceiling, so the actionable work is inter coding, not proving
that the *key-side* decisions are within epsilon of optimal.

It also explains why ARNR, TPL, lookahead and the content-derived key split all
measured <= 0 on 12-24 frame extracts: those extracts are key-dominated
(75 % of bits in the key), so they measure the key, which is the part that is
already near its ceiling. Any future lever measurement on this encoder should
use a stream long enough for P frames to carry the bits, and should report which
segment the oracle attributes the deficit to.

Caveat: segment rows are pooling bounds, not encodable streams; they upper-bound
what any decision change in that range could recover, they do not promise it.

## Allocation model (`RdAlloc.lean`): what Lean can and cannot tell you about f0-32

`RdProof` bounds a *decision* gap. `RdAlloc` models the other candidate: how the
bits are spread **within** the chain.

The oracle put +18.28 JOD of ceiling on f0-32, but the per-frame bits showed the
P chain already holds 61.8 % of the stream — so "P needs more bits" is not the
deficit. What is *not* normally optimal is spreading: constant QP gives the same
quantiser to a pulldown near-duplicate (<1,000 bits) and to a frame with new
content (5,000 bits), whose marginal slopes differ.

| theorem | statement | axioms |
|---|---|---|
| `greedy2_optimal` | for frames with diminishing returns, no split of two bits beats the greedy (largest-marginal-first) plan | `propext, Quot.sound` |
| `split_beats_flat_right` | one-bit-each beats both-bits-on-one exactly when the other frame's first marginal exceeds this frame's second | same |
| `split_beats_flat_left` | mirror direction | same |

Lean rejected the first draft of `split_beats_flat_right`, which claimed
`B.g0 > A.g1 ⟹ split(2) < split(0)`. That implication is false (`A.g0` can
dominate the total), and the assistant caught it — the second time today a
proof refused an overclaim that looked plausible when written in prose.

**What this buys, and what it does not.** It gives the provable yardstick for
any per-frame QP planner, which is exactly what the TPL experiments lacked:
those allocated from measured *bits* rather than marginal *slopes*, and lost
0.03-0.21 JOD. A water-filling plan against measured per-frame R-D curves is the
one allocation design with a proof behind it.

It does **not** move f0-32 by itself. The measured ~1.2 JOD there is inter
residual coding at 608 B per 1440x720 frame — transform and entropy efficiency,
which is not a combinatorial object and cannot be enumerated, modelled or proven
in Lean. Anyone proposing to "model the encoder in Lean" to recover that 1.2 JOD
is proposing to formalise signal processing, not to find an optimum.

## Allocation, measured: the theorem's predicted gain is 0.026 dB

`tools/waterfill.py` builds the greedy water-filling plan over measured
per-frame R-D points (the constructive form of `greedy2_optimal`) and reports it
against constant QP at the same total bits. On `space8k_1440x720_60f`, three
QPs (194/198/202), luma SSE per frame:

| plan | bits | total SSE | PSNR |
|---|---:|---:|---:|
| constant QP198 | 34,699 | 2.8732e9 | 31.486 dB |
| **water-filled** | 34,685 | 2.8557e9 | **31.512 dB** |
| delta | -14 B | **-0.61 %** | **+0.026 dB** |

47 of 60 frames are re-quantised and the whole exercise is worth 0.026 dB.

**So the provably-optimal allocation is worth ~0.03 dB, against a ~1.2 JOD gap
to SVT.** That settles the allocation question in the negative with a proof
behind the reference point rather than a heuristic: the P chain's bits are
already spread nearly optimally, which is consistent with the per-frame shares
(P 61.8 %, key 38.2 %) and with why TPL's bit-based reallocation lost.

The remaining f0-32 deficit is therefore not allocation and not decisions. It is
the cost of coding an inter frame at ~608 B: prediction accuracy plus
residual/transform efficiency. No allocation theorem applies to it.

## Pricing.lean — is the intra gap a mispriced decision or a missing tool?

`RdProof` closes the search (the chosen tree is optimal given the candidates) and
`RdAlloc` closes the allocation (+0.026 dB). Neither explains the measured
intra deficit (0.147 JOD at 25 KB on `src720.png`). `Pricing.lean` splits what
is left into the only two remaining buckets.

**The bound.** The encoder minimises an additive price over independent leaves;
the truth is what the range coder spends. If every leaf is priced within `e`
bits of its true cost, then for the chosen coding `c` and any alternative `a`

    sumTrue c ≤ sumTrue a + 2 * nerr e c          (pricing_gap_bound)

where `nerr e l` is `l.length * e`. So a decision that is optimal under the
model can miss the true optimum by at most **twice the total mispricing** — and
`e = 0` (an exact model) makes it truly optimal
(`exact_model_is_true_optimal`), which retires tuning as a lever.

**Consequence for the decision we face.** Measure `e`. If it is small, the
deficit is *model error* (syntax/tools) and no decision change can close it; if
it is large, fixing the model is a bounded, provable gain.

### What the measurement says

`AV2ENC_RATE_HYBRID_S_MEASURE=1` records the actual range-coder bits per leaf
next to the decision's prediction. On 24 inter frames at QP198 (7,577 leaves):

| term | leaves | predicted | actual | bias | p90 abs |
|---|---:|---:|---:|---:|---:|
| mode syntax (`syntax_new`) | 7,577 | 4.199 | 4.203 | -0.00 | **0.77** |
| coefficients, residual leaves | 245 | **51.27** | **36.96** | **+14.32** | **26.57** |
| coefficients, all leaves | 7,577 | 1.658 | 1.195 | +0.46 | 0.00 |

Mode syntax is priced almost exactly. The **coefficient model is ~39% too
expensive on the leaves that actually code a residual**, with a one-sided
(positive) bias. That is the shape the bound cares about: every
residual-versus-skip decision made with this model is biased toward *not*
coding the residual.

**Correction (measured same day): the bias does *not* reach intra.** The
guess above was that `estimate_tx_bits` / the Laplacian model prices intra
coefficients too, so the bias would transfer.  Tested: `src720.png` at QP212
with `AV2ENC_RATE_HYBRID_S=1` produces a **byte-identical** stream (25,168 B,
cvvdp 7.2415, both runs), so the gate does not touch intra decisions and the
0.147 JOD intra gap is *not* this bug.  The coefficient pricing defect is an
inter-residual problem (worth +0.011 JOD at 0.4 % fewer bytes on 24 frames of
space8k), and the intra gap must be sought elsewhere.

### Intra frontier probe (same day, all at QP212 on `src720.png`)

| setting | bytes |
|---|---:|
| baseline `--nightly` | 25,168 |
| `--video-min-block 4` | 25,905 |
| `--video-max-depth 6` | 25,168 (identical — depth cap not binding) |
| `--video-ext-partitions` | 25,729 |
| both (`min-block 4` + `max-depth 6`) | 34,025 |

No partitioning setting is cheaper at the same QP, so intra is at its frontier
for the tree it searches: consistent with `RdProof`, and consistent with the
deficit coming from coefficient pricing rather than decisions.

### Scope

`Pricing.lean` is a bound, not an estimate. It says nothing about *how* to make
the coefficient model exact, and nothing about which tool SVT has that we lack.
It converts "we are 0.147 JOD behind" into "how much of that is mispricing",
which is answerable by measurement — and the measurement now says most of the
observable model error is in coefficients.

## Cause.lean — can the *cause* be proved?

**No, not positively.** A discrepancy proves the model is wrong; it cannot prove
which mechanism produced it. "The numbers fit mechanism X, therefore X caused
them" is abduction — any flexible hypothesis fits three points.

**Proved: elimination.** `no_affine_bias_fits` shows that the additive family
`price = overhead + magnitude + position·eob`, whose bias against the coder is
affine in `eob`, **cannot reproduce the three measurements at any constants**:
the secant slopes disagree (`456·4401 ≠ 1406·1619`). So the defect is not a
tuning constant — the family is misspecified, and further constant-fitting
cannot succeed by construction. (Both early "fixes" in this session were exactly
that, and both left residual error.)

With the two measured eliminations this gives:

| candidate cause | status |
|---|---|
| static model; adaptive contexts would fix it | **ruled out** (adaptive has the same bias: −7.86 vs −9.00 @QP198) |
| a wrong constant in the additive family | **ruled out by proof** (`no_affine_bias_fits`) |
| 4x4 partitions intrinsically worse | **ruled out** (`Partition.lean`) |
| residual cost is non-additive over scan positions (context state / TX shape / significance-map structure) | **surviving hypothesis** — supported by elimination, not measured directly |

**What would establish a cause positively:** a predicted-delta intervention —
change one term, predict the bias movement, measure that it moves by exactly
that. The `0.5 → 0` position change did move it by the predicted `−0.31·eob`
(16.0 → 3.0 bits mean absolute), which is causal evidence that the double-counted
position term was *a* cause. It was not the whole cause: the surviving −3.5..+4.6
bit spread is precisely what elimination 2 says no additive form can absorb.
