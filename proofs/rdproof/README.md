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
